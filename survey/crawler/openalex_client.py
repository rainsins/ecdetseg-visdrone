"""OpenAlex client (free tier, polite pool).

Used to enrich the arXiv corpus with:
- accurate venue labels (primary_location.source.display_name)
- citation counts (cited_by_count)
- publication-year trend (group_by)

Budget-aware: OpenAlex free tier shares a small daily budget per IP, so we
keep request count low (per-page=200, few calls) and degrade gracefully.
"""

import os
import sys
import time

import requests

API = "https://api.openalex.org/works"
MAILTO = "research@example.com"
# Optional key via env OPENALEX_KEY (raises the daily budget).
API_KEY = os.environ.get("OPENALEX_KEY", "")

FIELDS = ",".join([
    "id", "doi", "display_name", "publication_year", "cited_by_count",
    "referenced_works_count", "primary_location", "type", "open_access",
])


def _params(filter_str, per_page=200, sort=None, group_by=None, cursor="*"):
    p = {
        "filter": filter_str,
        "per-page": per_page,
        "mailto": MAILTO,
        "cursor": cursor,
    }
    # select cannot be combined with group_by.
    if not group_by:
        p["select"] = FIELDS
    if sort:
        p["sort"] = sort
    if group_by:
        p["group_by"] = group_by
    if API_KEY:
        p["api_key"] = API_KEY
    return p


def search(filter_str, per_page=200, sort=None, group_by=None, max_pages=3,
           cursor="*"):
    """Paginate through OpenAlex results. Returns list of work dicts."""
    results = []
    for _ in range(max_pages):
        resp = requests.get(API, params=_params(filter_str, per_page, sort,
                                                group_by, cursor), timeout=60)
        if resp.status_code == 429 or resp.status_code == 403:
            print("    [openalex] rate-limited/budget exhausted "
                  f"(HTTP {resp.status_code}), stopping", file=sys.stderr)
            break
        if resp.status_code != 200:
            print(f"    [openalex] HTTP {resp.status_code}: {resp.text[:200]}",
                  file=sys.stderr)
            break
        data = resp.json()
        meta = data.get("meta", {})
        if group_by:
            # group_by responses have a flat "group_by" list, not "results".
            results.extend(data.get("group_by", []))
        else:
            results.extend(data.get("results", []))
        nxt = meta.get("next_cursor")
        if not nxt or not meta.get("results"):
            break
        cursor = nxt
        time.sleep(1.0)
    return results


def works_by_year(search_phrase, year_start, year_end):
    """Return list of {year, count} via group_by (cheap, one request)."""
    f = (f'title_and_abstract.search:"{search_phrase}",'
         f"publication_year:{year_start}-{year_end}")
    return search(f, per_page=50, group_by="publication_year", max_pages=1)


def top_cited(search_phrase, year_start, year_end, per_page=200, max_pages=1):
    """Return list of top-cited works matching a phrase, with venue + citations."""
    f = (f'title_and_abstract.search:"{search_phrase}",'
         f"publication_year:{year_start}-{year_end}")
    return search(f, per_page=per_page, sort="cited_by_count:desc",
                  max_pages=max_pages)
