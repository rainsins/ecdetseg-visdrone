#!/usr/bin/env python3
"""Fold constant-only Sin/Cos/Equal/Where branches, preserving ONNX opset.

Only whitelist-proven, input-independent ancestors are evaluated. No image
sample is used to freeze values. Dynamic sampling masks remain dynamic.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

TARGETS = {"Sin", "Cos", "Equal", "Where"}
PURE = TARGETS | {"Constant", "ConstantOfShape", "Identity", "Add", "Sub",
    "Mul", "Div", "Pow", "Cast", "Reshape", "Transpose", "Concat",
    "Expand", "Tile", "Unsqueeze", "Squeeze", "Slice", "Gather", "Range",
    "Shape", "Floor", "Neg", "Flatten"}

def fold_model(source):
    model = onnx.shape_inference.infer_shapes(copy.deepcopy(source))
    graph = model.graph
    if any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
           for n in graph.node for a in n.attribute):
        raise ValueError("Control-flow subgraphs are not supported")
    producers = {o: i for i, node in enumerate(graph.node) for o in node.output}
    initializers = {v.name: v for v in graph.initializer}
    inputs = {v.name for v in graph.input}
    infos = {v.name: v for v in [*graph.input, *graph.value_info, *graph.output]}
    cache = {}

    def constant(name):
        if not name: return True
        if name in inputs: return False  # includes overridable initializers
        if name in initializers: return True
        if name in cache: return cache[name]
        index = producers.get(name)
        if index is None: return False
        node = graph.node[index]
        cache[name] = False
        value = (not node.domain and node.op_type in PURE and
                 all(constant(i) for i in node.input))
        cache[name] = value
        return value

    chosen, skipped = [], []
    for node in graph.node:
        if node.domain or node.op_type not in TARGETS: continue
        if len(node.output) == 1 and constant(node.output[0]):
            chosen.append(node)
        else:
            skipped.append(node.name or node.output[0])
    if not chosen: return model, {"folded": [], "skipped_dynamic_or_unproven": skipped}

    needed_nodes, needed_init = set(), set()
    def collect(name):
        if not name: return
        if name in initializers:
            needed_init.add(name); return
        index = producers[name]
        if index in needed_nodes: return
        needed_nodes.add(index)
        for i in graph.node[index].input: collect(i)
    for node in chosen: collect(node.output[0])
    outputs = [infos[n.output[0]] for n in chosen]
    sub = helper.make_model(helper.make_graph(
        [n for i,n in enumerate(graph.node) if i in needed_nodes],
        "constant_only", [], outputs,
        [t for t in graph.initializer if t.name in needed_init]),
        opset_imports=list(model.opset_import), ir_version=model.ir_version)
    onnx.checker.check_model(sub)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    values = ort.InferenceSession(sub.SerializeToString(), options,
        providers=["CPUExecutionProvider"]).run(None, {})
    replaced, records = set(), []
    for node,value in zip(chosen,values):
        if np.issubdtype(value.dtype,np.floating) and not np.isfinite(value).all():
            raise ValueError(f"Non-finite constant: {node.name}")
        name = node.output[0]
        graph.initializer.append(numpy_helper.from_array(value,name))
        replaced.add(name)
        records.append({"name":node.name, "op":node.op_type,
                        "shape":list(value.shape), "bytes":value.nbytes})
    kept = [n for n in graph.node if not any(o in replaced for o in n.output)]
    # Prune only nodes no longer reachable from graph outputs. Shared ancestors
    # remain if any live branch still needs them. No runtime input is removed.
    live = {v.name for v in graph.output}
    result = []
    for node in reversed(kept):
        if any(o in live for o in node.output):
            result.append(node); live.update(node.input)
    del graph.node[:]; graph.node.extend(reversed(result))
    tensors = [t for t in graph.initializer if t.name in live or t.name in inputs]
    del graph.initializer[:]; graph.initializer.extend(tensors)
    info = [v for v in graph.value_info if v.name in live]
    del graph.value_info[:]; graph.value_info.extend(info)
    onnx.checker.check_model(model)
    return model, {"folded":records, "skipped_dynamic_or_unproven":skipped}

def digest(path):
    with open(path,"rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    report_path = a.output.with_suffix(".static.json")
    if a.output.exists() or report_path.exists() or a.input.resolve()==a.output.resolve():
        p.error("Use new output/report paths; existing artifacts are preserved")
    source = onnx.load(a.input)
    model,report = fold_model(source)
    report.update(source=str(a.input.resolve()),source_sha256=digest(a.input),
        before=dict(Counter(n.op_type for n in source.graph.node)),
        after=dict(Counter(n.op_type for n in model.graph.node)),
        board_verified=False, quantized_accuracy_verified=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    onnx.save(model,a.output)
    report["output_sha256"] = digest(a.output)
    report_path.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
