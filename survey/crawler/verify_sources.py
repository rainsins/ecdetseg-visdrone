"""Verify candidate metadata by arXiv ID without overwriting local records."""
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from . import arxiv_client as client
from analysis.decision import features, normalize_title
from analysis.features import extract_features

ROOT = Path(__file__).resolve().parents[1]

def compare(local, remote):
    title_equal = normalize_title(local['title']) == normalize_title(remote['title'])
    abstract_equal = ' '.join(local['abstract'].split()) == ' '.join(remote['abstract'].split())
    published_equal = local.get('published') == remote.get('published')
    return dict(status='verified_metadata' if title_equal and abstract_equal and published_equal else 'metadata_mismatch',
                local_title=local['title'], remote_title=remote['title'],
                title_equal=title_equal, abstract_equal=abstract_equal, published_equal=published_equal,
                local_published=local.get('published'),
                abstract_sha256=hashlib.sha256(local['abstract'].encode()).hexdigest(), remote=remote)

def main():
    papers = json.loads((ROOT/'data/raw/arxiv_papers.json').read_text())['papers']
    candidates = [p for p in papers if
        {'frequency/spectral', 'query/DETR'} <= set(extract_features(p)['modules']) or
        {'frequency', 'detr', 'small'} <= features(p).keys()]
    checked_at = datetime.now(timezone.utc).isoformat()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory = ROOT/'data/raw/verification'/stamp
    directory.mkdir(parents=True)
    results = []
    for start in range(0, len(candidates), 20):
        batch = candidates[start:start+20]
        try:
            response = client._request({'id_list': ','.join(p['arxiv_id'] for p in batch), 'max_results': len(batch)})
            feed = directory/f'batch-{start}.xml'
            feed.write_bytes(response.content)
            root = ET.fromstring(response.content)
            remotes = {p['arxiv_id']: p for p in map(client.parse_entry, root.findall(f'{client.ATOM}entry'))}
            for p in batch:
                remote = remotes.get(p['arxiv_id'])
                result = compare(p, remote) if remote else {'status': 'not_returned'}
                results.append(dict(arxiv_id=p['arxiv_id'], checked_at=checked_at,
                                    feed=str(feed.relative_to(ROOT)), **result))
        except (RuntimeError, ValueError, ET.ParseError) as exc:
            results.extend(dict(arxiv_id=p['arxiv_id'], status='network_or_parse_error',
                                checked_at=checked_at, error=str(exc)) for p in batch)
    data = dict(checked_at=checked_at, scope='metadata-only, not full-text review', results=results)
    dest = ROOT/'data/processed/source_verification.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    (directory/'comparison.json').write_text(json.dumps(data, ensure_ascii=False, indent=2))
    dest.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(json.dumps(dict(Counter(r['status'] for r in results)), ensure_ascii=False))

if __name__ == '__main__':
    main()
