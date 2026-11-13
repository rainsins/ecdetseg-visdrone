"""Crawl arXiv for small/tiny object detection papers (2022-2026).

- Sorts by submittedDate descending, stops at year boundary (no date filter).
- Deduplicates by arxiv_id.
- Caches incrementally so an interrupted run can be resumed.
- Robust to 429 rate limiting (backoff handled in arxiv_client).

Usage: python -m crawler.crawl_arxiv
Outputs: data/raw/arxiv_papers.json
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone

from . import arxiv_client
from .config import QUERIES, YEAR_START, YEAR_END

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
OUT_PATH = os.path.join(RAW_DIR, "arxiv_papers.json")
CACHE_PATH = os.path.join(RAW_DIR, "arxiv_cache.json")

MAX_PER_QUERY = 500


def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"papers": {}, "done_queries": []}


def save_cache(cache):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true', help='Refresh queries and archive old raw/cache files')
    args = parser.parse_args()
    os.makedirs(RAW_DIR, exist_ok=True)
    if args.refresh:
        import shutil
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        archive = os.path.join(RAW_DIR, 'archive', stamp)
        os.makedirs(archive)
        for path in (CACHE_PATH, OUT_PATH):
            if os.path.exists(path):
                shutil.copy2(path, archive)
        save_cache({'papers': {}, 'done_queries': [], 'query_stats': [],
                    'query_definitions': {}})
    cache = load_cache()
    papers = OrderedDict(cache["papers"])
    done = set(cache["done_queries"])
    stats = {s['query']: s for s in cache.get('query_stats', [])}
    definitions = cache.get('query_definitions', {})
    # A legacy cache hit must not erase its existing query statistics.
    if not args.refresh and os.path.exists(OUT_PATH):
        with open(OUT_PATH, encoding='utf-8') as f:
            for s in json.load(f).get('meta', {}).get('queries', []):
                stats.setdefault(s['query'], s)

    def checkpoint():
        save_cache({'papers': papers, 'done_queries': sorted(done),
                    'query_stats': list(stats.values()), 'query_definitions': definitions})

    for qid, base_query, kind in QUERIES:
        if qid in done:
            if definitions.get(qid) not in (None, base_query):
                raise RuntimeError(f'{qid}: query changed; use --refresh')
            print(f"[skip] {qid} (cached)")
            continue

        print(f"[{qid}] kind={kind}", flush=True)
        try:
            total, batch = arxiv_client.crawl_until_year(
                base_query, YEAR_START, max_results=MAX_PER_QUERY)
        except Exception as e:  # noqa: BLE001
            print(f"[error] {qid}: {e}", file=sys.stderr)
            checkpoint()
            sys.exit(1)

        added = 0
        for p in batch:
            if p["arxiv_id"] and p["arxiv_id"] not in papers:
                papers[p["arxiv_id"]] = p
                added += 1
        done.add(qid)
        stats[qid] = {"query": qid, 'query_text': base_query, "total": total,
                      "fetched": len(batch), "added": added, "kind": kind,
                      'fetched_at': datetime.now(timezone.utc).isoformat(),
                      'possibly_truncated': len(batch) >= MAX_PER_QUERY and
                        (total is None or total > len(batch)) and
                        not any(p.get('published', '')[:4].isdigit() and
                                int(p['published'][:4]) < YEAR_START for p in batch)}
        definitions[qid] = base_query
        checkpoint()
        print(f"    total={total} fetched={len(batch)} added={added} "
              f"unique={len(papers)}", flush=True)
        time.sleep(arxiv_client.REQUEST_INTERVAL)

    all_papers = list(papers.values())
    # Final safety: keep only papers with a v1 published year in range.
    all_papers = [p for p in all_papers
                  if p["published"] and YEAR_START <= int(p["published"][:4]) <= YEAR_END]

    out = {
        "meta": {
            "year_start": YEAR_START,
            "year_end": YEAR_END,
            "queries": list(stats.values()),
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'all_queries_have_provenance': all(q[0] in definitions and
                bool(stats.get(q[0], {}).get('fetched_at')) for q in QUERIES),
            'query_config_sha256': hashlib.sha256(json.dumps(QUERIES).encode()).hexdigest(),
            "n_unique_papers": len(all_papers),
        },
        "papers": all_papers,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\nDONE: {len(all_papers)} unique papers -> {OUT_PATH}")


if __name__ == "__main__":
    main()
