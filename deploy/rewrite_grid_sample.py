#!/usr/bin/env python3
"""Lower static-NCHW 2D GridSample to Gather + bilinear arithmetic.

Supports opset 16..19, FLOAT input/grid, bilinear, zeros, both align_corners
settings. Finite inputs/grids are required. This is an ONNX equivalence path,
NOT a promise that RKNN maps every new operator to the NPU. Keep the original
model; re-calibrate and validate the newly compiled artifact on the board.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def rewrite_model(model):
    opset = next(x.version for x in model.opset_import if x.domain == "")
    if not 16 <= opset <= 19:
        raise ValueError(f"Only opset 16..19 is tested, got {opset}")
    model = onnx.shape_inference.infer_shapes(copy.deepcopy(model))
    graph = model.graph
    types = {v.name: v.type.tensor_type for v in
             [*graph.input, *graph.value_info, *graph.output]}
    used = {s for n in graph.node for s in [n.name, *n.input, *n.output]}
    used.update(t.name for t in graph.initializer)
    nodes, records = [], []
    for original in graph.node:
        if original.op_type != "GridSample" or original.domain:
            if any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                   for a in original.attribute):
                raise ValueError("Control-flow subgraphs are not supported by this rewriter")
            nodes.append(original)
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in original.attribute}
        if attrs.get("mode", b"bilinear") != b"bilinear" or \
                attrs.get("padding_mode", b"zeros") != b"zeros":
            raise ValueError(f"Unsupported GridSample attributes: {original.name}: {attrs}")
        align = attrs.get("align_corners", 0)
        if align not in (0, 1):
            raise ValueError(f"Invalid align_corners: {align}")
        x, grid = original.input
        xt, gt = types.get(x), types.get(grid)
        if xt is None or gt is None or xt.elem_type != TensorProto.FLOAT or gt.elem_type != TensorProto.FLOAT:
            raise ValueError(f"{original.name}: need inferred FLOAT feature/grid types")
        dims = [d.dim_value for d in xt.shape.dim]
        if len(dims) != 4 or any(d <= 0 for d in dims):
            raise ValueError(f"{original.name}: need static N,C,H,W; got {dims}")
        if len(gt.shape.dim) != 4 or gt.shape.dim[-1].dim_value not in (0, 2):
            raise ValueError(f"{original.name}: grid must have shape N,OH,OW,2")
        n, c, h, w = dims
        prefix = f"ec_bilinear_{len(records)}"
        while any(s.startswith(prefix + "/") for s in used):
            prefix += "_"
        serial = 0

        def const(label, data, dtype=np.float32):
            name = f"{prefix}/{label}"
            graph.initializer.append(numpy_helper.from_array(np.asarray(data, dtype=dtype), name))
            used.add(name)
            return name

        def op(kind, *inputs, **kwargs):
            nonlocal serial
            serial += 1
            name = f"{prefix}/{kind}_{serial}"
            nodes.append(helper.make_node(kind, list(inputs), [name], name=name, **kwargs))
            used.add(name)
            return name

        zero, one, half = const("zero", 0), const("one", 1), const("half", .5)
        minus_one = const("minus_one", -1)
        width, height = const("width", w), const("height", h)
        xmax, ymax = const("xmax", w - 1), const("ymax", h - 1)
        ix, iy = const("ix", 0, np.int64), const("iy", 1, np.int64)
        gx, gy = op("Gather", grid, ix, axis=3), op("Gather", grid, iy, axis=3)

        def coordinate(g, size, maximum):
            # Preserve GridSample's order of operations, including the half pixel.
            if align:
                p = op("Mul", op("Div", op("Add", g, one), const(f"two_{serial}", 2)), maximum)
            else:
                p = op("Mul", op("Sub", op("Mul", op("Add", g, one), size), one), half)
            # All coordinates outside [-1,size] already have zero support.
            # Clip before floor/cast to avoid overflow for very distant samples.
            return op("Clip", p, minus_one, size)

        px, py = coordinate(gx, width, xmax), coordinate(gy, height, ymax)
        x0, y0 = op("Floor", px), op("Floor", py)
        x1, y1 = op("Add", x0, one), op("Add", y0, one)
        dx, dy = op("Sub", px, x0), op("Sub", py, y0)
        wx, wy = op("Sub", one, dx), op("Sub", one, dy)
        # A flattened NHWC table permits one index per sample, not per channel.
        flat = op("Reshape", op("Transpose", x, perm=[0, 2, 3, 1]),
                  const("flat_shape", [n * h * w, c], np.int64))
        batch = const("batch_offsets", (np.arange(n, dtype=np.int64) * h * w).reshape(n, 1, 1), np.int64)
        wi = const("width_int", w, np.int64)
        axes = const("axes", [3], np.int64)

        def sample(xx, yy, weight):
            valid = op("And", op("And", op("GreaterOrEqual", xx, zero), op("Less", xx, width)),
                       op("And", op("GreaterOrEqual", yy, zero), op("Less", yy, height)))
            xx = op("Cast", op("Clip", xx, zero, xmax), to=TensorProto.INT64)
            yy = op("Cast", op("Clip", yy, zero, ymax), to=TensorProto.INT64)
            index = op("Add", op("Add", op("Mul", yy, wi), xx), batch)
            values = op("Gather", flat, index, axis=0)
            # Mask each neighbour separately: clipping alone gives border padding!
            values = op("Where", op("Unsqueeze", valid, axes), values, zero)
            return op("Mul", values, op("Unsqueeze", weight, axes))

        a = sample(x0, y0, op("Mul", wx, wy))
        b = sample(x1, y0, op("Mul", dx, wy))
        d = sample(x0, y1, op("Mul", wx, dy))
        e = sample(x1, y1, op("Mul", dx, dy))
        total = op("Add", op("Add", op("Add", a, b), d), e)
        nodes.append(helper.make_node("Transpose", [total], list(original.output),
                                      name=prefix + "/output", perm=[0, 3, 1, 2]))
        records.append({"name": original.name, "feature_shape": dims, "align_corners": align})
    del graph.node[:]
    graph.node.extend(nodes)
    onnx.checker.check_model(model)
    return model, records


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--meta-in", type=Path)
    parser.add_argument("--meta-out", type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve() or args.output.exists():
        parser.error("Use a NEW output path; original artifacts are never overwritten")
    if bool(args.meta_in) != bool(args.meta_out):
        parser.error("Pass --meta-in and --meta-out together")
    if args.meta_out and (args.meta_out.exists() or args.meta_out.resolve() == args.output.resolve()):
        parser.error("Use a separate NEW metadata path")
    model, records = rewrite_model(onnx.load(args.input))
    if not records:
        parser.error("No GridSample nodes found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    report = {"source": str(args.input.resolve()), "source_sha256": sha256(args.input),
              "output_sha256": sha256(args.output), "replaced_count": len(records),
              "nodes": records, "board_verified": False}
    args.output.with_suffix(".rewrite.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.meta_in:
        meta = json.loads(args.meta_in.read_text())
        meta["grid_sample_rewrite"] = report
        meta["onnxruntime_parity_before_rewrite"] = meta.pop("onnxruntime_parity", {})
        meta["onnxruntime_parity"] = {"pending": True}
        meta["onnx_graph_info"] = {"num_nodes": len(model.graph.node), "GridSample": 0}
        args.meta_out.parent.mkdir(parents=True, exist_ok=True)
        args.meta_out.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "replaced_count": len(records),
                      "GridSample_remaining": 0}))


if __name__ == "__main__":
    main()
