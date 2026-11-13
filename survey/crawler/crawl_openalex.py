"""Crawl OpenAlex for small/tiny object detection (2022-2026).

Collects:
1. Top-cited works (venue + citations) for 3 core phrases.
2. Publication-year trend via group_by.

Usage: python -m crawler.crawl_openalex
Outputs:
  data/raw/openalex_top_cited.json
  data/raw/openalex_year_trend.json
"""

import json
import os
import sys
import time

from . import openalex_client
from .config import YEAR_START, YEAR_END

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

CORE_PHRASES = [
    "small object detection",
    "tiny object detection",
    "small target detection",
]


def _clean(w):
    src = ((w.get("primary_location") or {}).get("source") or {})
    return {
        "openalex_id": (w.get("id") or "").split("/")[-1],
        "doi": w.get("doi"),
        "title": w.get("display_name"),
        "year": w.get("publication_year"),
        "cited_by_count": w.get("cited_by_count"),
        "referenced_works_count": w.get("referenced_works_count"),
        "venue": src.get("display_name"),
        "venue_type": (w.get("type") or ""),
        "is_oa": (w.get("open_access") or {}).get("is_oa"),
    }


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    # 1. Year trend (group_by)
    year_trend = {}
    for phrase in CORE_PHRASES:
        groups = openalex_client.works_by_year(phrase, YEAR_START, YEAR_END)
        for g in groups:
            key = str(g.get("key"))
            cnt = g.get("count", 0)
            year_trend.setdefault(key, {}).setdefault(phrase, 0)
            year_trend[key][phrase] += cnt
        print(f"[trend] {phrase}: {[(g.get('key'), g.get('count')) for g in groups]}")
        time.sleep(1.0)

    with open(os.path.join(RAW_DIR, "openalex_year_trend.json"), "w", encoding="utf-8") as f:
        json.dump(year_trend, f, ensure_ascii=False, indent=2)

    # 2. Top cited works per phrase
    top_cited = {}
    seen_ids = set()
    for phrase in CORE_PHRASES:
        works = openalex_client.top_cited(phrase, YEAR_START, YEAR_END,
                                          per_page=200, max_pages=1)
        for w in works:
            c = _clean(w)
            oid = c["openalex_id"]
            if oid and oid not in seen_ids:
                seen_ids.add(oid)
                top_cited[oid] = c
        print(f"[cited] {phrase}: fetched {len(works)}")
        time.sleep(1.0)

    out = {
        "meta": {"year_start": YEAR_START, "year_end": YEAR_END,
                 "n_unique": len(top_cited)},
        "works": list(top_cited.values()),
    }
    with open(os.path.join(RAW_DIR, "openalex_top_cited.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\nDONE: {len(top_cited)} unique top-cited works -> data/raw/")


if __name__ == "__main__":
    main()
