#!/usr/bin/env python3
"""Write a COCO json containing the first N images (and their annotations).

    python scripts/make_subset.py --ann <big.json> --out <subset.json> --n 64
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()

    d = json.load(open(a.ann))
    imgs = d.get("images", [])[: a.n]
    ids = {im["id"] for im in imgs}
    anns = [x for x in d.get("annotations", []) if x["image_id"] in ids]
    out = {
        "images": imgs,
        "annotations": anns,
        "categories": d.get("categories", []),
        "info": d.get("info", {}),
        "licenses": d.get("licenses", []),
    }
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"[make_subset] {a.out}: {len(imgs)} imgs, {len(anns)} anns")


if __name__ == "__main__":
    main()
