"""Load raw crawl data, extract features, compute aggregate statistics.

Outputs:
  data/processed/papers_features.csv   (one row per arXiv paper)
  data/processed/arxiv_stats.json      (aggregate stats for charts/report)
  data/processed/openalex_stats.json   (venue + citation stats from OpenAlex)
"""

import json
import os
from collections import Counter, defaultdict

import pandas as pd

from .features import extract_features
from crawler.config import (
    MODULE_KEYWORDS, DETECTORS, DOMAIN_KEYWORDS, DATASET_KEYWORDS,
    YEAR_START, YEAR_END,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")


def _load_json(name):
    with open(os.path.join(RAW, name), encoding="utf-8") as f:
        return json.load(f)


def build_features():
    data = _load_json("arxiv_papers.json")
    papers = data.get("papers", [])
    rows = [extract_features(p) for p in papers]
    df = pd.DataFrame(rows)
    # keep only in-range years (already filtered in crawl, double-check)
    if "year" in df:
        df = df[(df["year"] >= YEAR_START) & (df["year"] <= YEAR_END)]
    return df


def module_year_trend(df):
    """Return {module: {year: count}} for modules, ordered by total desc."""
    mod_year = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        for m in row["modules"]:
            mod_year[m][int(row["year"])] += 1
    return {m: dict(sorted(d.items())) for m, d in mod_year.items()}


def arxiv_stats(df):
    def _count_of(col):
        c = Counter()
        for _, row in df.iterrows():
            for x in row[col]:
                c[x] += 1
        return c

    year_trend = dict(sorted(Counter(df["year"]).items()))
    venue = Counter(df["venue"])
    return {
        "n_papers": int(len(df)),
        "year_trend": {int(k): int(v) for k, v in year_trend.items()},
        "venue_dist": dict(venue.most_common()),
        "module_dist": dict(_count_of("modules").most_common()),
        "module_year_trend": {k: {int(y): int(c) for y, c in v.items()}
                              for k, v in module_year_trend(df).items()},
        "detector_dist": dict(_count_of("detectors").most_common()),
        "domain_dist": dict(_count_of("domains").most_common()),
        "dataset_dist": dict(_count_of("datasets").most_common()),
        "category_dist": dict(Counter(df["primary_category"]).most_common(15)),
    }


def openalex_stats():
    top = _load_json("openalex_top_cited.json").get("works", [])
    trend = _load_json("openalex_year_trend.json")

    venue = Counter()
    for w in top:
        v = w.get("venue") or "unknown"
        venue[_norm_venue(v)] += 1

    return {
        "n_top_cited": len(top),
        "venue_dist": dict(venue.most_common(25)),
        "year_trend_pub": trend,
        "top_cited": sorted(top, key=lambda x: -(x.get("cited_by_count") or 0))[:50],
        "year_dist": dict(Counter(w.get("year") for w in top).most_common()),
    }


def _norm_venue(v):
    v = (v or "").lower()
    rules = [
        ("CVPR", ["computer vision and pattern recognition", "cvpr"]),
        ("ICCV", ["international conference on computer vision", "iccv"]),
        ("ECCV", ["european conference on computer vision", "eccv"]),
        ("NeurIPS", ["neural information processing"]),
        ("ICLR", ["learning representations"]),
        ("ICML", ["international conference on machine learning"]),
        ("AAAI", ["artificial intelligence", "aaai"]),
        ("TPAMI", ["pattern analysis and machine intelligence"]),
        ("IJCV", ["international journal of computer vision"]),
        ("TIP", ["image processing"]),
        ("PR", ["pattern recognition"]),
        ("TNNLS", ["neural networks and learning systems"]),
        ("TGRS", ["geoscience and remote sensing"]),
        ("TMM", ["multimedia"]),
        ("Sensors", ["sensors"]),
        ("Remote Sensing", ["remote sensing"]),
        ("IEEE Access", ["ieee access"]),
        ("Expert Syst. Appl.", ["expert systems with applications"]),
        ("Neurocomputing", ["neurocomputing"]),
        ("arXiv", ["arxiv"]),
    ]
    for label, subs in rules:
        for s in subs:
            if s in v:
                return label
    return (v or "unknown")[:40]


def main():
    os.makedirs(PROC, exist_ok=True)

    df = build_features()
    df.to_csv(os.path.join(PROC, "papers_features.csv"), index=False)

    s_arxiv = arxiv_stats(df)
    with open(os.path.join(PROC, "arxiv_stats.json"), "w", encoding="utf-8") as f:
        json.dump(s_arxiv, f, ensure_ascii=False, indent=2)

    s_oa = openalex_stats()
    with open(os.path.join(PROC, "openalex_stats.json"), "w", encoding="utf-8") as f:
        json.dump(s_oa, f, ensure_ascii=False, indent=2)

    print(f"[arxiv] {s_arxiv['n_papers']} papers")
    print(f"[arxiv] year trend: {s_arxiv['year_trend']}")
    print(f"[arxiv] venues: {dict(list(s_arxiv['venue_dist'].items())[:10])}")
    print(f"[arxiv] modules: {dict(list(s_arxiv['module_dist'].items())[:12])}")
    print(f"[arxiv] detectors: {dict(list(s_arxiv['detector_dist'].items())[:12])}")
    print(f"[arxiv] domains: {dict(list(s_arxiv['domain_dist'].items())[:10])}")
    print(f"[arxiv] datasets: {dict(list(s_arxiv['dataset_dist'].items())[:10])}")
    print(f"[openalex] {s_oa['n_top_cited']} top-cited works")


if __name__ == "__main__":
    main()
