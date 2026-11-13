import csv
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from unittest.mock import patch
from analysis.decision import build, features, yearly_share, verification_status
from crawler.arxiv_client import parse_entry, parse_total, crawl_until_year
from crawler.verify_sources import compare

def paper(aid='2501.00001', published='2025-01-01', title='Frequency-domain DETR for small object detection in UAV images'):
    return dict(arxiv_id=aid, published=published, title=title,
                abstract='Wavelet features and object queries for VisDrone.', authors=['A'])

class DecisionTests(unittest.TestCase):
    def test_atom_authors_primary_category_and_totals(self):
        entry = ET.fromstring('''<entry xmlns="http://www.w3.org/2005/Atom" xmlns:a="http://arxiv.org/schemas/atom">
        <id>http://arxiv.org/abs/2501.00001v2</id><author><name>Alice</name></author>
        <author><name>Bob</name></author><category term="cs.AI"/><a:primary_category term="cs.CV"/>
        </entry>''')
        parsed = parse_entry(entry)
        self.assertEqual(parsed['authors'], ['Alice', 'Bob'])
        self.assertEqual(parsed['primary_category'], 'cs.CV')
        self.assertEqual(parsed['arxiv_id'], '2501.00001')
        for version in ('1.0', '1.1'):
            root = ET.fromstring(f'<feed xmlns:o="http://a9.com/-/spec/opensearch/{version}/"><o:totalResults>42</o:totalResults></feed>')
            self.assertEqual(parse_total(root), 42)
        self.assertIsNone(parse_total(ET.fromstring('<feed/>')))

    def test_loose_words_do_not_establish_intersection(self):
        e = features(dict(title='Spectral measurements and query efficiency', abstract='Gaussian generative DINO features.'))
        for key in ('frequency', 'detr', 'loss', 'diffusion'):
            self.assertNotIn(key, e)
        self.assertTrue({'frequency', 'detr', 'small'} <= features(paper()).keys())

    def test_zero_years_and_partial_year(self):
        papers = [paper(), paper('2601.00001', '2026-01-01')]
        e = {p['arxiv_id']: features(p) for p in papers}
        rows = yearly_share(papers, e, ['frequency', 'detr'], date(2026, 11, 13))
        self.assertEqual([r['year'] for r in rows], [2022, 2023, 2024, 2025])
        self.assertEqual(rows[0]['hits'], 0)
        self.assertIsNone(rows[0]['share_pct'])
        self.assertEqual(rows[-1]['share_pct'], 100)

    def test_verification_detects_stale_and_mismatched_records(self):
        p = paper()
        check = compare(p, p)
        self.assertEqual(verification_status(p, {p['arxiv_id']: check}), 'verified_metadata')
        self.assertEqual(verification_status(dict(p, abstract='Changed'), {p['arxiv_id']: check}), 'stale_verification')
        self.assertEqual(compare(p, dict(p, title='Other title'))['status'], 'metadata_mismatch')
        self.assertEqual(compare(p, dict(p, published='2024-01-01'))['status'], 'metadata_mismatch')

    def test_report_filters_dates_duplicates_and_preserves_manual_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root/'data/raw'; raw.mkdir(parents=True)
            records = [paper(), paper(), paper('2612.00001', '2026-12-01'), paper('bad', 'n/a')]
            (raw/'arxiv_papers.json').write_text(json.dumps({'papers': records, 'meta': {}}))
            audit = build(root, date(2026, 11, 13))
            self.assertEqual(audit['included_count'], 1)
            self.assertEqual(len(audit['excluded']), 3)
            review = root/'output/paper_review.csv'
            with review.open(encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
            rows[0]['notes'] = 'Manual note: p.4, needs comparison'
            with review.open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            build(root, date(2026, 11, 13))
            self.assertIn('Manual note: p.4', review.read_text(encoding='utf-8-sig'))
            self.assertIn('原宽规则交集 1 篇', (root/'output/report.md').read_text())
            ranking = json.loads((root/'output/ranking.json').read_text())
            self.assertNotIn('score', ranking['directions'][0])

    def test_pagination_respects_cap_and_rejects_repeated_page(self):
        with patch('crawler.arxiv_client.query', return_value=(1, [paper()])) as query:
            total, papers = crawl_until_year('test', 2022, max_results=1)
            self.assertEqual(total, 1)
            query.assert_called_once_with('test', start=0, max_results=1)
        with patch('crawler.arxiv_client.query', return_value=(5, [paper()])):
            with self.assertRaisesRegex(RuntimeError, 'repeated a page'):
                crawl_until_year('test', 2022, max_results=3)

if __name__ == '__main__':
    unittest.main()
