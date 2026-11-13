#!/usr/bin/env python3
"""Build a class-stratified VisDrone calibration set for RKNN INT8.

Why stratified instead of plain random?
  VisDrone is extremely long-tailed (train: car 144,867 instances vs
  awning-tricycle 3,246).  A uniform 300-image sample can easily contain no
  awning-tricycle / bus / tricycle at all, so the activation ranges for those
  heads would be calibrated on almost no data.  We allocate calibration slots
  proportionally to per-class instance counts (D'Hondt apportionment with a
  minimum of one slot per class) and sample *within* each dominant-class
  stratum.  This preserves the dataset prior while guaranteeing that rare
  classes appear.

Why stretch the calibration images?
  ``preprocess.py`` now performs the same direct 640x640 resize as the PyTorch
  training/val transform.  RKNN's calibration reader must see that distribution,
  so the saved PNGs are 640x640 stretched images (no padding).  They remain
  uint8 RGB and still require normalization by
  ``rknn.config(mean_values/std_values)``.

Outputs:
  <artifacts>/calib_imgs/*.png
  <artifacts>/calib.txt                 # one relative path per line
  <artifacts>/calib_manifest.json       # audit trail for the paper
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deploy import (  # noqa: E402
    DEFAULT_ONNX_NAME,
    INPUT_SIZE,
    MEAN_RGB_01,
    SEED,
    STD_RGB_01,
    VISDRONE_CLASSES,
    Timer,
    default_repo_root,
    iter_coco_images,
    resolve_data_root,
    setup_logging,
    write_json,
)
from deploy.preprocess import preprocess_path  # noqa: E402


def _write_stretch_png(src: Path, dst: Path, size: int) -> None:
    """Export a calibration image exactly as the deployment preprocessor sees it.

    This must go through :func:`deploy.preprocess.preprocess_path` rather than
    ``cv2.resize``.  RKNN computes activation ranges from these files, so a
    different resampler means the calibration distribution differs from the
    inference distribution -- invisibly, and in a way that inflates the measured
    quantisation loss.  Measured on a 1360x765 frame, ``cv2.INTER_AREA`` differs
    from PIL bilinear on 75% of pixels (max 34/255).

    The tensor is RGB CHW; the PNG is written as RGB so RKNN's own reader sees
    the same bytes regardless of channel-order flags.
    """
    result = preprocess_path(src, size=size)
    rgb_hwc = np.ascontiguousarray(result.chw.transpose(1, 2, 0))
    # cv2.imwrite expects BGR; convert back so the stored PNG holds RGB values.
    bgr_for_write = cv2.cvtColor(rgb_hwc, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(dst), bgr_for_write):
        raise IOError(f"failed to write calibration image: {dst}")


def _load_coco(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"annotation file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _instance_counts(coco: dict) -> Dict[int, int]:
    counts: Dict[int, int] = collections.Counter()
    for ann in coco.get("annotations", []):
        counts[int(ann["category_id"])] += 1
    return {cid: int(counts.get(cid, 0)) for cid in range(len(VISDRONE_CLASSES))}


def _dominant_class(cats: Sequence[int]) -> int:
    """Dominant class of one image; ties resolved to the lower class id."""
    c = collections.Counter(int(x) for x in cats)
    return max(c, key=lambda k: (c[k], -k))


def allocate_quotas(weights: Sequence[float], capacities: Sequence[int], total: int) -> List[int]:
    """D'Hondt apportionment with a one-slot minimum.

    Each class first receives one slot (if it has any candidate image), then the
    remaining slots are assigned to the class with the largest
    ``weight / (already_allocated + 1)``.  This is deterministic, exact, and
    respects per-stratum capacity.
    """
    n = len(weights)
    quotas = [0] * n
    for i in range(n):
        if weights[i] > 0 and capacities[i] > 0:
            quotas[i] = 1

    if sum(quotas) > total:
        # Only possible if every class needs a minimum but total is tiny.
        for i in sorted(range(n), key=lambda j: weights[j]):
            if sum(quotas) <= total:
                break
            if quotas[i] > 0:
                quotas[i] -= 1

    def score(i: int) -> float:
        return float(weights[i]) / float(quotas[i] + 1)

    while sum(quotas) < total:
        candidates = [i for i in range(n) if quotas[i] < capacities[i]]
        if not candidates:
            break
        # max with a deterministic tie-break: higher weight, then lower id
        best = max(candidates, key=lambda i: (score(i), weights[i], -i))
        quotas[best] += 1

    # If capacity forced under-allocation, fill from any remaining stratum.
    while sum(quotas) < total:
        candidates = [i for i in range(n) if quotas[i] < capacities[i]]
        if not candidates:
            break
        quotas[max(candidates, key=lambda i: capacities[i] - quotas[i])] += 1

    return quotas


def _unique(names: List[str]) -> List[str]:
    """Deduplicate while keeping graph order."""
    out: List[str] = []
    seen = set()
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _resolve_probe_nodes(model) -> List[str]:
    """Locate every decoder GatherElements output to probe for non-finite values.

    The previous version hard-coded ``/model/decoder/GatherElements_output_0``.
    Torch's auto-numbering of same-op nodes is not stable across versions, so a
    first attempt resolved the node structurally -- but it took only the *last*
    match in graph order.  That is the wrong node: the frozen ONNX graph has two
    decoder gathers,

        /model/decoder/GatherElements_output_0     <- this one goes +inf
        /model/decoder/GatherElements_1_output_0

    and probing only the second reported "all 300 images clean" while one bad
    image stayed in the set.  A single inf image makes RKNN set the activation
    scale to inf, and the resulting INT8 model emitted near-zero scores for
    every image (measured: AP 0.00 vs 35.08 FP32).

    So probe *all* of them and treat any non-finite output as a bad image.
    Returns an empty list when no decoder gather exists, which the caller
    turns into a hard error rather than a silent skip.
    """
    matches = [
        node.output[0]
        for node in model.graph.node
        if node.op_type == "GatherElements"
        and any(o for o in node.output)
        and "decoder" in node.name
    ]
    if matches:
        return _unique(matches)
    # Fall back to any GatherElements in the graph.
    fallback = [
        node.output[0] for node in model.graph.node if node.op_type == "GatherElements" and node.output
    ]
    return _unique(fallback)


def _filter_nonfinite_calib(
    train_img_dir: Path,
    calib_img_dir: Path,
    out_dir: Path,
    calib_lines: List[str],
    records: List[dict],
    selected_ids: List[int],
    strata: Dict[int, List[int]],
    image_by_id: Dict[int, dict],
    cats_by_image: Dict[int, List[int]],
    size: int,
    seed: int,
    onnx_path: Path,
    logger: logging.Logger,
) -> tuple[List[str], List[dict], List[int], List[str]]:
    """Drop calibration images that make the query-selection gather non-finite.

    Why this exists
    ---------------
    The D-FINE encoder selects 300 queries with TopK and then gathers anchors /
    memory.  A handful of VisDrone frames (e.g. 9999999_00576_d_0000260.jpg)
    produce +inf at ``/model/decoder/GatherElements_output_0``.  A single such
    image makes RKNN's activation scale ``inf`` for the whole layer, and the
    INT8 model then emits near-zero scores for every image (sim AP ~0).  The
    fix is data curation: detect non-finite calibration activations in FP32 and
    replace those images with other images from the same dominant-class stratum.
    """
    removed: List[str] = []
    if not onnx_path.is_file():
        logger.warning("calibration non-finite filter skipped: ONNX not found at %s", onnx_path)
        return calib_lines, records, selected_ids, removed
    try:
        import onnx
        from onnx import TensorProto, helper
        import onnxruntime as ort
    except ImportError as exc:
        logger.warning("calibration non-finite filter skipped: %s", exc)
        return calib_lines, records, selected_ids, removed

    model = onnx.load(str(onnx_path))
    node_outputs = _resolve_probe_nodes(model)
    if not node_outputs:
        raise RuntimeError(
            "calibration non-finite filter could not locate any query-selection Gather node "
            "(expected */decoder/GatherElements*). Refusing to build a calibration set without "
            "the non-finite guard: a single inf activation collapses the INT8 model to ~0 mAP. "
            "Pass --no-nonfinite-filter to skip this deliberately."
        )
    logger.info("calibration non-finite probe nodes (%d): %s", len(node_outputs), node_outputs)

    existing = {o.name for o in model.graph.output}
    for name in node_outputs:
        if name not in existing:
            model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    tmp_onnx = out_dir / "_calib_nonfinite_probe.onnx"
    onnx.save(model, str(tmp_onnx))

    sess = ort.InferenceSession(str(tmp_onnx), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    resolved = [o.name for o in sess.get_outputs()]
    missing = [name for name in node_outputs if name not in resolved]
    if missing:
        tmp_onnx.unlink(missing_ok=True)
        raise RuntimeError(f"probe nodes {missing} did not survive as graph outputs")

    # The ONNX graph expects ImageNet-normalised float input in 0..1.
    mean = np.asarray(MEAN_RGB_01, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(STD_RGB_01, dtype=np.float32).reshape(1, 3, 1, 1)

    def _is_finite(rel: str) -> bool:
        pre = preprocess_path(out_dir / rel, size=size)
        x = (pre.as_batch().astype(np.float32) / 255.0 - mean) / std
        # Every gathered tensor is checked: probing a single "representative"
        # node is what let an inf image through in the first place.
        outs = sess.run(node_outputs, {input_name: x.astype(np.float32)})
        for name, out in zip(node_outputs, outs):
            if not np.isfinite(out).all():
                logger.debug("non-finite probe %s for %s", name, rel)
                return False
        return True

    bad_indices: List[int] = []
    for idx, rel in enumerate(calib_lines):
        if not _is_finite(rel):
            bad_indices.append(idx)
            logger.warning(
                "calibration image produces non-finite query-selection activations: %s", rel
            )

    if not bad_indices:
        logger.info("calibration non-finite filter: all %d images clean", len(calib_lines))
        del sess
        tmp_onnx.unlink(missing_ok=True)
        return calib_lines, records, selected_ids, removed

    selected_set = set(selected_ids)
    rng = random.Random(seed)
    for idx in bad_indices:
        old_rec = records[idx]
        cls = int(old_rec["dominant_class"])
        old_rel = calib_lines[idx]
        candidates = [iid for iid in strata.get(cls, []) if iid not in selected_set]
        rng.shuffle(candidates)

        # A replacement must itself be clean.  The previous version took the
        # first unused id without probing it, so a bad replacement could be
        # swapped in silently (and would poison the same layer).
        new_id = None
        new_rel = None
        for cand_id in candidates:
            cand_im = image_by_id[cand_id]
            cand_out = f"{cand_id:07d}_{Path(cand_im['file_name']).stem}.png"
            cand_dst = calib_img_dir / cand_out
            _write_stretch_png(train_img_dir / cand_im["file_name"], cand_dst, size)
            cand_rel = cand_dst.relative_to(out_dir).as_posix()
            if _is_finite(cand_rel):
                new_id, new_rel = cand_id, cand_rel
                break
            logger.warning("replacement candidate also non-finite, skipping: %s", cand_rel)
            cand_dst.unlink(missing_ok=True)

        if new_id is None or new_rel is None:
            # Hard failure.  Continuing would leave an image that turns the
            # whole layer's activation scale into +inf; the resulting INT8 model
            # collapses to ~0 mAP and the failure surfaces far from its cause.
            del sess
            tmp_onnx.unlink(missing_ok=True)
            raise RuntimeError(
                f"no finite replacement available for bad calibration image {old_rel} "
                f"(dominant class {VISDRONE_CLASSES[cls]}, {len(candidates)} candidates tried). "
                "Refusing to produce a calibration set known to contain non-finite activations."
            )

        selected_set.add(new_id)
        im = image_by_id[new_id]
        dst = out_dir / new_rel
        # Remove the unusable file to keep the artifact directory clean.
        old_path = out_dir / old_rel
        if old_path.is_file() and old_path != dst:
            old_path.unlink()
        calib_lines[idx] = new_rel
        records[idx] = {
            "image_id": new_id,
            "file_name": im["file_name"],
            "calib_file": new_rel,
            "dominant_class": cls,
            "classes_present": sorted(set(cats_by_image[new_id])),
        }
        removed.append(old_rel)
        logger.info("replaced bad calibration image %s -> %s", old_rel, new_rel)

    del sess
    tmp_onnx.unlink(missing_ok=True)

    # The replacement path reuses `strata`, so a duplicate could only appear via
    # a stale id; assert instead of silently producing a shorter calib.txt.
    selected_ids = [int(r["image_id"]) for r in records]
    if len(set(selected_ids)) != len(selected_ids):
        dupes = sorted({i for i in selected_ids if selected_ids.count(i) > 1})
        raise RuntimeError(f"duplicate calibration image ids after replacement: {dupes[:5]}")
    return calib_lines, records, selected_ids, removed


def build_calib(
    train_img_dir: Path,
    train_ann: Path,
    val_ann: Path,
    out_dir: Path,
    num_images: int,
    seed: int,
    size: int,
    filter_nonfinite: bool = True,
) -> dict:
    logger = logging.getLogger("calib")
    train = _load_coco(train_ann)
    val = _load_coco(val_ann)

    images = list(iter_coco_images(train))
    if len(images) < num_images:
        raise ValueError(f"train split has {len(images)} images, cannot sample {num_images}")

    image_by_id = {int(im["id"]): im for im in images}
    cats_by_image: Dict[int, List[int]] = collections.defaultdict(list)
    for ann in train.get("annotations", []):
        cats_by_image[int(ann["image_id"])].append(int(ann["category_id"]))

    inst_counts = _instance_counts(train)
    logger.info("train instances per class: %s", {
        VISDRONE_CLASSES[c]: inst_counts[c] for c in range(len(VISDRONE_CLASSES))
    })

    # Stratum = dominant class of the image.  This gives every image exactly one
    # stratum, so the final set is a true stratified sample (no duplicate image
    # can be drawn for two classes).
    strata: Dict[int, List[int]] = collections.defaultdict(list)
    for im in images:
        iid = int(im["id"])
        cats = cats_by_image.get(iid, [])
        if not cats:
            continue  # no annotations -> cannot inform any class
        strata[_dominant_class(cats)].append(iid)

    weights = [float(inst_counts.get(c, 0)) for c in range(len(VISDRONE_CLASSES))]
    capacities = [len(strata.get(c, [])) for c in range(len(VISDRONE_CLASSES))]
    quotas = allocate_quotas(weights, capacities, num_images)
    logger.info("quota per dominant class: %s", {
        VISDRONE_CLASSES[c]: quotas[c] for c in range(len(VISDRONE_CLASSES))
    })

    rng = random.Random(seed)
    selected_ids: List[int] = []
    for c in range(len(VISDRONE_CLASSES)):
        candidates = sorted(strata.get(c, []))
        take = min(quotas[c], len(candidates))
        if take:
            selected_ids.extend(rng.sample(candidates, take))
        if quotas[c] > len(candidates):
            logger.warning(
                "class %s: quota %d > available dominant images %d; extra slots will be filled from other strata",
                VISDRONE_CLASSES[c],
                quotas[c],
                len(candidates),
            )

    # Top up if capacities forced a shortfall (deterministic shuffle).
    if len(selected_ids) < num_images:
        chosen = set(selected_ids)
        pool = sorted(i for i in image_by_id if i not in chosen and cats_by_image.get(i))
        rng.shuffle(pool)
        selected_ids.extend(pool[: num_images - len(selected_ids)])

    selected_ids = sorted(set(selected_ids))[:num_images]
    if len(selected_ids) != num_images:
        raise RuntimeError(f"selected {len(selected_ids)} images, expected {num_images}")

    # Split safety: separate annotation files can reuse numeric ids, so compare
    # file names as well as ids.
    val_names = {Path(im["file_name"]).name for im in val.get("images", [])}
    train_names = [Path(image_by_id[i]["file_name"]).name for i in selected_ids]
    overlap = sorted(set(train_names) & val_names)
    if overlap:
        raise RuntimeError(
            f"calibration/val overlap detected ({len(overlap)} files), first: {overlap[:5]}. "
            "Check that the train/val annotation files belong to different splits."
        )

    calib_img_dir = out_dir / "calib_imgs"
    calib_img_dir.mkdir(parents=True, exist_ok=True)
    calib_lines: List[str] = []
    records = []
    for iid in selected_ids:
        im = image_by_id[iid]
        src = train_img_dir / im["file_name"]
        if not src.is_file():
            raise FileNotFoundError(f"train image missing: {src}")
        out_name = f"{iid:07d}_{Path(im['file_name']).stem}.png"
        dst = calib_img_dir / out_name
        _write_stretch_png(src, dst, size)
        rel = dst.relative_to(out_dir).as_posix()
        calib_lines.append(rel)
        records.append(
            {
                "image_id": iid,
                "file_name": im["file_name"],
                "calib_file": rel,
                "dominant_class": int(_dominant_class(cats_by_image[iid])),
                "classes_present": sorted(set(cats_by_image[iid])),
            }
        )

    removed: List[str] = []
    if filter_nonfinite:
        calib_lines, records, selected_ids, removed = _filter_nonfinite_calib(
            train_img_dir=train_img_dir,
            calib_img_dir=calib_img_dir,
            out_dir=out_dir,
            calib_lines=calib_lines,
            records=records,
            selected_ids=selected_ids,
            strata=strata,
            image_by_id=image_by_id,
            cats_by_image=cats_by_image,
            size=size,
            seed=seed,
            onnx_path=out_dir / DEFAULT_ONNX_NAME,
            logger=logger,
        )
    else:
        logger.warning("calibration non-finite filter disabled by --no-nonfinite-filter")

    calib_txt = out_dir / "calib.txt"
    with open(calib_txt, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(calib_lines) + "\n")

    selected_cat_counts: Dict[int, int] = collections.Counter()
    for iid in selected_ids:
        for c in set(cats_by_image[iid]):
            selected_cat_counts[c] += 1
    selected_inst_counts: Dict[int, int] = collections.Counter()
    train_annotations_by_image: Dict[int, List[int]] = cats_by_image
    for iid in selected_ids:
        for c in train_annotations_by_image[iid]:
            selected_inst_counts[c] += 1

    logger.info("calibration class coverage (images containing class / instances):")
    for c in range(len(VISDRONE_CLASSES)):
        logger.info(
            "  %-16s quota=%3d dominant=%3d present_images=%3d instances=%5d",
            VISDRONE_CLASSES[c],
            quotas[c],
            sum(1 for iid in selected_ids if _dominant_class(cats_by_image[iid]) == c),
            selected_cat_counts.get(c, 0),
            selected_inst_counts.get(c, 0),
        )

    manifest = {
        "seed": seed,
        "num_images": num_images,
        "input_size": size,
        "train_ann": str(train_ann),
        "val_ann": str(val_ann),
        "train_img_dir": str(train_img_dir),
        "class_names": list(VISDRONE_CLASSES),
        "train_instance_counts": {VISDRONE_CLASSES[c]: inst_counts[c] for c in range(len(VISDRONE_CLASSES))},
        "quotas": {VISDRONE_CLASSES[c]: quotas[c] for c in range(len(VISDRONE_CLASSES))},
        "selected_images_containing_class": {
            VISDRONE_CLASSES[c]: selected_cat_counts.get(c, 0) for c in range(len(VISDRONE_CLASSES))
        },
        "selected_instance_counts": {
            VISDRONE_CLASSES[c]: selected_inst_counts.get(c, 0) for c in range(len(VISDRONE_CLASSES))
        },
        "selection": records,
        "nonfinite_replaced": removed,
    }
    write_json(out_dir / "calib_manifest.json", manifest)
    logger.info("wrote %s (%d lines) and %s", calib_txt, len(calib_lines), out_dir / "calib_manifest.json")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Stratified VisDrone calibration set for RKNN")
    parser.add_argument("--repo-root", type=Path, default=default_repo_root())
    parser.add_argument("--data-root", type=Path, default=None, help="VisDrone root (contains train/ and annotations_*)")
    parser.add_argument("--train-images", type=Path, default=None, help="override <data-root>/train")
    parser.add_argument("--train-ann", type=Path, default=None)
    parser.add_argument("--val-ann", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <repo>/deploy/artifacts")
    parser.add_argument("--num", type=int, default=300)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--size", type=int, default=INPUT_SIZE)
    parser.add_argument("--no-nonfinite-filter", action="store_true",
                        help="skip the FP32 non-finite query-selection calibration filter")
    args = parser.parse_args()

    logger = setup_logging(name="calib")
    repo_root = args.repo_root.expanduser().resolve()
    data_root = resolve_data_root(repo_root, args.data_root)
    train_img_dir = (args.train_images or (data_root / "train")).expanduser().resolve()
    train_ann = (args.train_ann or (data_root / "annotations_VisDrone_train.json")).expanduser().resolve()
    val_ann = (args.val_ann or (data_root / "annotations_VisDrone_val.json")).expanduser().resolve()
    out_dir = (args.out_dir or (repo_root / "deploy" / "artifacts")).expanduser().resolve()

    logger.info("data_root=%s", data_root)
    logger.info("train_images=%s", train_img_dir)
    logger.info("train_ann=%s", train_ann)
    logger.info("val_ann=%s", val_ann)
    logger.info("out_dir=%s num=%d seed=%d", out_dir, args.num, args.seed)

    with Timer("stratified calibration sampling + PIL stretch PNG export", logger):
        build_calib(
            train_img_dir=train_img_dir,
            train_ann=train_ann,
            val_ann=val_ann,
            out_dir=out_dir,
            num_images=args.num,
            seed=args.seed,
            size=args.size,
            filter_nonfinite=not args.no_nonfinite_filter,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
