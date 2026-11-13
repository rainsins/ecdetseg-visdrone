"""Feature extraction for TOD papers: venue, modules, detectors, domains, datasets, year."""

import re

from crawler.config import (
    MODULE_KEYWORDS, DETECTORS, DOMAIN_KEYWORDS, DATASET_KEYWORDS,
    YEAR_START, YEAR_END,
)

# ---------------------------------------------------------------------------
# Venue classification (from arXiv comment + journal_ref fields only)
# ---------------------------------------------------------------------------
# Ordered: most specific / longest phrases first to avoid false positives.
_VENUE_RULES = [
    # (venue_label, [distinctive substrings])
    ("TPAMI", ["pattern analysis and machine intelligence"]),
    ("IJCV", ["international journal of computer vision"]),
    ("TNNLS", ["neural networks and learning systems"]),
    ("TGRS", ["geoscience and remote sensing"]),
    ("TMM", ["multimedia"]),
    ("TIP", ["image processing"]),
    ("NeurIPS", ["neural information processing systems", "neurips"]),
    ("ICLR", ["learning representations", "iclr"]),
    ("ICML", ["international conference on machine learning", "icml"]),
    ("AAAI", ["artificial intelligence", "aaai"]),
    ("CVPR", ["computer vision and pattern recognition", "cvpr"]),
    ("ICCV", ["international conference on computer vision", "iccv"]),
    ("ECCV", ["european conference on computer vision", "eccv"]),
    ("WACV", ["applications of computer vision", "wacv"]),
    ("BMVC", ["british machine vision conference", "bmvc"]),
    ("ACCV", ["asian conference on computer vision", "accv"]),
    ("TMLR", ["transactions on machine learning research", "tmlr"]),
    ("PR", ["pattern recognition"]),
    ("IEEE Access", ["ieee access"]),
    ("Sensors", ["sensors"]),
    ("Remote Sensing", ["remote sensing"]),
]


def classify_venue(comment, journal_ref):
    text = " ".join([comment or "", journal_ref or ""]).lower()
    for label, subs in _VENUE_RULES:
        for s in subs:
            if s in text:
                return label
    # arXiv-specific hints
    if "arxiv" in text:
        return "arXiv"
    return "arXiv (preprint)"


def classify_year(published):
    if not published:
        return None
    m = re.match(r"(\d{4})", published)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Keyword-based multi-label classification over the abstract
# ---------------------------------------------------------------------------
def _match(text, keyword_map):
    """Return list of keys whose keywords appear in text (case-insensitive).

    Single-token keywords are matched with word boundaries to avoid false
    hits (e.g. "ct" inside "object", "frame" inside "framework"). Multi-word
    phrases are matched as substrings.
    """
    low = text.lower()
    hits = []
    for key, kws in keyword_map.items():
        for kw in kws:
            if any(ch in kw for ch in (" ", "-")):
                if kw in low:
                    hits.append(key)
                    break
            else:
                pat = rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])"
                if re.search(pat, low):
                    hits.append(key)
                    break
    return hits


def extract_features(paper):
    abstract = paper.get("abstract", "") or ""
    title = paper.get("title", "") or ""
    text = f"{title}. {abstract}"

    return {
        "arxiv_id": paper.get("arxiv_id"),
        "year": classify_year(paper.get("published")),
        "venue": classify_venue(paper.get("comment"), paper.get("journal_ref")),
        "modules": _match(text, MODULE_KEYWORDS),
        "detectors": _match(text, DETECTORS),
        "domains": _match(text, DOMAIN_KEYWORDS),
        "datasets": _match(text, DATASET_KEYWORDS),
        "primary_category": paper.get("primary_category", ""),
        "n_authors": len(paper.get("authors", [])),
    }
