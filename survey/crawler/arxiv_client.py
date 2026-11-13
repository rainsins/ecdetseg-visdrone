"""Robust arXiv API client.

- Polite: sleeps between every request (arXiv asks ~1 req / 3s).
- Retries on 429 with exponential backoff.
- No submittedDate filter (syntax was unreliable); instead we sort by
  submittedDate descending and stop once papers fall below the target year.
"""

import os
import re
import sys
import time
import xml.etree.ElementTree as ET

import requests

ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"

REQUEST_INTERVAL = 3.2   # seconds between requests
MAX_RETRIES = 3
BASE_BACKOFF = 3.2
_last_request = 0.0


def parse_entry(entry):
    def txt(path, default=""):
        el = entry.find(path)
        return el.text.strip() if el is not None and el.text else default

    paper_id = txt(f"{ATOM}id")
    arxiv_id = re.sub(r"v\d+$", "", paper_id.split("/abs/")[-1]) if paper_id else ""
    arxiv_id = arxiv_id.strip()

    categories = [c.get("term") for c in entry.findall(f"{ATOM}category")]

    authors = []
    for a in entry.findall(f"{ATOM}author"):
        name = (a.findtext(f"{ATOM}name") or "").strip()
        if name:
            authors.append(name)

    return {
        "arxiv_id": arxiv_id,
        "title": re.sub(r"\s+", " ", txt(f"{ATOM}title")),
        "abstract": re.sub(r"\s+", " ", txt(f"{ATOM}summary")),
        "published": txt(f"{ATOM}published"),
        "updated": txt(f"{ATOM}updated"),
        "authors": authors,
        "primary_category": (entry.find(f"{ARXIV}primary_category").get("term", "")
                             if entry.find(f"{ARXIV}primary_category") is not None
                             else (categories[0] if categories else "")),
        "categories": categories,
        "comment": txt(f"{ARXIV}comment"),
        "journal_ref": txt(f"{ARXIV}journal_ref"),
        "doi": txt(f"{ARXIV}doi"),
    }


def _request(params, sleep=REQUEST_INTERVAL):
    """Single GET with retry/backoff on 429/5xx."""
    global _last_request
    session = requests.Session()
    # Explicit override; otherwise requests uses HTTP(S)_PROXY / NO_PROXY.
    proxy = os.environ.get('SURVEY_PROXY')
    if proxy:
        session.trust_env = False
        if proxy.lower() != 'direct':
            session.proxies.update(http=proxy, https=proxy)
    for attempt in range(MAX_RETRIES):
        time.sleep(max(0, sleep - (time.monotonic() - _last_request)))
        try:
            _last_request = time.monotonic()
            resp = session.get(ARXIV_API, params=params, timeout=30,
                               headers={'User-Agent': 'EdgeCrafter-survey/2.0'})
            if resp.status_code == 200:
                return resp
            if resp.status_code != 429 and resp.status_code < 500:
                raise RuntimeError(f'arXiv HTTP {resp.status_code}; check proxy/network')
            # Rate-limited or server error -> back off
            wait = BASE_BACKOFF * (2 ** attempt)
            print(f"    [retry] HTTP {resp.status_code}, sleeping {wait:.0f}s "
                  f"(attempt {attempt + 1}/{MAX_RETRIES})", file=sys.stderr)
            time.sleep(wait)
        except requests.RequestException as e:
            wait = BASE_BACKOFF * (2 ** attempt)
            print(f"    [retry] {e}, sleeping {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"arXiv request failed after {MAX_RETRIES} retries")


def query(search_query, start=0, max_results=100, sort_by="submittedDate",
          sort_order="descending"):
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    resp = _request(params)
    root = ET.fromstring(resp.content)

    total = parse_total(root)

    papers = [parse_entry(e) for e in root.findall(f"{ATOM}entry")]
    return total, papers


def parse_total(root):
    # arXiv feeds use OpenSearch 1.0 as well as 1.1. Unknown is not zero.
    for el in root.iter():
        if el.tag.rsplit('}', 1)[-1] == 'totalResults':
            return int(el.text)
    return None


def crawl_until_year(search_query, year_start, max_results=500):
    """Fetch results sorted by submittedDate descending, stopping once we
    pass below year_start. Returns (total, papers)."""
    papers = []
    start = 0
    total = None
    while start < max_results:
        total, batch = query(search_query, start=start, max_results=min(100, max_results-start))
        if not batch:
            break
        seen = {p['arxiv_id'] for p in papers}
        if all(p['arxiv_id'] in seen for p in batch):
            raise RuntimeError('arXiv repeated a page; refusing incomplete cached crawl')
        papers.extend(batch)
        start += len(batch)
        # Sorted descending: once the oldest in this batch is before year_start,
        # we've collected everything >= year_start (no date filter needed).
        oldest_year = min(
            (int(p["published"][:4]) for p in batch if p["published"]),
            default=year_start,
        )
        if oldest_year < year_start:
            break
        if total is not None and start >= total:
            break
    return total, papers
