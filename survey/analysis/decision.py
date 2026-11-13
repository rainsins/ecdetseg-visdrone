"""Evidence-first triage; keyword candidates are not verified contributions."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

from crawler.config import QUERIES, YEAR_START, YEAR_END
from .features import extract_features

ROOT = Path(__file__).resolve().parents[1]
RULE_VERSION = 'decision-v2'
# Narrower than the exploratory taxonomy: exclude query alone, DINO alone,
# spectral alone, Gaussian alone, and generative alone. Still requires review.
RULES = {
    'frequency': r'fourier|wavelet|\bfft\b|\bdct\b|\bdwt\b|frequency[- ](?:domain|aware|guided|decoupled|spatial|semantic)|(?:high|low)[- ]frequency|spectral (?:enhancement|routing|filter)',
    'detr': r'(?<![a-z0-9])(?:[a-z0-9]+-)?detr(?![a-z0-9])|detection transformers?|object quer(?:y|ies)|deformable detection transformer',
    'small': r'(?:small|tiny)[- ](?:object|target)|visdrone|ai[- ]?tod|tinyperson|soda-d',
    'loss': r'label assignment|hungarian|matching cost|wasserstein|\bnwd\b|\brfla\b|localization loss|localisation loss',
    'distillation': r'distillation|teacher[- ]student',
    'efficient': r'lightweight|real[- ]time|compact|edge devices?|low[- ]latency',
    'context': r'multi[- ]scale|feature pyramid|receptive field|context',
    'diffusion': r'diffusion|denoising diffusion',
    'openworld': r'open[- ](?:world|vocabulary|set)|few[- ]shot|zero[- ]shot',
    'infrared': r'infrared|thermal|dim[- ]small',
    'aerial': r'aerial|\buav\b|visdrone|ai[- ]?tod|remote sensing|soda-d',
    'other_task': r'3d object|multi[- ]view 3d|point cloud|lidar',
}
DIRECTIONS = [
    {'id': 'freq_x_detr', 'name': '频域 × 查询式小目标检测', 'keys': ['frequency', 'detr', 'small'],
     'priority': '先诊断，再决定', 'question': '背景抑制是否改善候选 query 覆盖，又是否损伤密集小目标？',
     'gate': '完成 B0/B1 控制；精读交集论文并核对训练/推理接入点；观察 query 覆盖与分组召回。'},
    {'id': 'freq_loss', 'name': '频域相关监督/分配', 'keys': ['frequency', 'loss', 'small'],
     'priority': '由诊断触发', 'question': '失败主要来自匹配/定位，还是候选特征缺失？',
     'gate': '只有匹配或定位诊断支持时才立项；少命中不代表空白。'},
    {'id': 'detr_small_query', 'name': '小目标 query 选择', 'keys': ['detr', 'small'],
     'priority': '作为诊断对象', 'question': 'top-k 前后目标覆盖是否明显变化？',
     'gate': '固定 query 数、CDN、匹配和损失计数；先定位再改结构。'},
    {'id': 'edge_tod', 'name': '轻量化/蒸馏', 'keys': ['efficient', 'distillation', 'small'],
     'priority': '第二阶段', 'question': '精度收益稳定后，能否迁移到轻量骨干？',
     'gate': '当前用户目标是精度优先；部署不参与本轮主线优先级。'},
    {'id': 'classic_context', 'name': '多尺度/上下文', 'keys': ['context', 'small'],
     'priority': '作为控制组', 'question': '简单平滑或分辨率变化能否解释收益？',
     'gate': '先做低复杂度对照，不以关键词多判定饱和。'},
    {'id': 'diffusion_sr', 'name': '扩散辅助小目标', 'keys': ['diffusion', 'small'],
     'priority': '暂缓', 'question': '是否值得新增复现和计算成本？',
     'gate': '目前缺少相对现有路线的明确问题证据。'},
    {'id': 'openworld_tod', 'name': '开放词汇/少样本小目标', 'keys': ['openworld', 'small'],
     'priority': '暂缓', 'question': '是否需要改变当前任务及评估协议？',
     'gate': '先完成现有闭集检测问题；不将 robustness 当作开放世界代理。'},
]
REVIEW_FIELDS = ['arxiv_id', 'title', 'url', 'scope_hint', 'verification', 'evidence',
                 'review_status', 'task', 'baseline', 'datasets', 'insertion_point',
                 'training_only', 'uses_gt', 'query_change', 'matching_change',
                 'fair_comparison', 'difference_to_ours', 'source_locator', 'notes']
MANUAL_FIELDS = REVIEW_FIELDS[6:]


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def features(p):
    text = p.get('title', '') + '. ' + p.get('abstract', '')
    evidence = {}
    for key, regex in RULES.items():
        match = re.search(regex, text, re.I)
        if match:
            evidence[key] = text[max(0, match.start()-70):match.end()+100].replace('\n', ' ')
    return evidence


def scope_hint(evidence):
    if 'other_task' in evidence:
        return 'other-task-review'
    if 'infrared' in evidence:
        return 'infrared-review'
    if 'aerial' in evidence:
        return 'aerial-candidate'
    return 'task-unconfirmed'


def normalize_title(title):
    return re.sub(r'[^a-z0-9]', '', title.lower())


def verification_status(paper, checks):
    check = checks.get(paper['arxiv_id'], {})
    if check.get('status') != 'verified_metadata':
        return check.get('status', 'not_checked')
    # A verification does not survive a changed local title or abstract.
    if normalize_title(check.get('local_title', '')) != normalize_title(paper.get('title', '')):
        return 'stale_verification'
    if check.get('abstract_sha256') != hashlib.sha256(paper.get('abstract', '').encode()).hexdigest():
        return 'stale_verification'
    if check.get('local_published') != paper.get('published'):
        return 'stale_verification'
    return 'verified_metadata'


def yearly_share(papers, evidence, keys, as_of):
    rows = []
    # Include zero-hit years, exclude current partial year from comparisons.
    for year in range(YEAR_START, min(YEAR_END, as_of.year-1)+1):
        group = [p for p in papers if p.get('published', '').startswith(str(year))]
        count = sum(set(keys) <= evidence[p['arxiv_id']].keys() for p in group)
        rows.append({'year': year, 'n': len(group), 'hits': count,
                     'share_pct': round(100*count/len(group), 2) if group else None})
    return rows


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def build(root=ROOT, as_of=None):
    as_of = as_of or date.today()
    raw_path = root / 'data/raw/arxiv_papers.json'
    raw = read_json(raw_path)
    out = root / 'output'
    out.mkdir(parents=True, exist_ok=True)
    papers, excluded, seen = [], [], set()
    for p in raw['papers']:
        aid = p.get('arxiv_id')
        try:
            published = date.fromisoformat(p.get('published', '')[:10])
        except ValueError:
            excluded.append({'arxiv_id': aid, 'reason': 'invalid_date'}); continue
        if not aid or aid in seen:
            excluded.append({'arxiv_id': aid, 'reason': 'missing_or_duplicate_id'}); continue
        if published > as_of or not YEAR_START <= published.year <= YEAR_END:
            excluded.append({'arxiv_id': aid, 'reason': 'outside_date_window'}); continue
        seen.add(aid)
        papers.append(p)
    evidence = {p['arxiv_id']: features(p) for p in papers}
    verification = read_json(root / 'data/processed/source_verification.json', {})
    checks = {c['arxiv_id']: c for c in verification.get('results', [])}
    broad = [p for p in papers if {'frequency/spectral', 'query/DETR'} <=
             set(extract_features(p)['modules'])]
    strict = [p for p in papers if {'frequency', 'detr', 'small'} <= evidence[p['arxiv_id']].keys()]
    union = {p['arxiv_id']: p for p in broad + strict}
    review_path = out / 'paper_review.csv'
    old = {}
    if review_path.exists():
        with review_path.open(encoding='utf-8-sig', newline='') as stream:
            old = {r['arxiv_id']: r for r in csv.DictReader(stream)}
    reviews = []
    for aid, p in sorted(union.items()):
        e = evidence[aid]
        row = dict(arxiv_id=aid, title=p['title'], url=f'https://arxiv.org/abs/{aid}',
                   scope_hint=scope_hint(e), verification=verification_status(p, checks),
                   evidence=json.dumps(e, ensure_ascii=False))
        row.update({k: old.get(aid, {}).get(k, '') for k in MANUAL_FIELDS})
        row['review_status'] = row['review_status'] or 'unreviewed'
        reviews.append(row)
    # Retain human annotations even if future rules stop matching that paper.
    for aid, row in old.items():
        if aid not in union:
            row['scope_hint'] = 'outside-current-candidates'
            reviews.append(row)
    write_csv(review_path, REVIEW_FIELDS, reviews)
    directions = []
    for d in DIRECTIONS:
        matched = [p for p in papers if set(d['keys']) <= evidence[p['arxiv_id']].keys()]
        directions.append({**d, 'keyword_candidates': len(matched),
            'aerial_candidates': sum(scope_hint(evidence[p['arxiv_id']]) == 'aerial-candidate' for p in matched),
            'ids': [p['arxiv_id'] for p in matched],
            'full_year_sample_trend': yearly_share(papers, evidence, d['keys'], as_of),
            'novelty': 'not_established', 'priority_basis': 'user_goal_and_experiment_dependencies'})
    meta = raw.get('meta', {})
    query_stats = meta.get('queries', [])
    warnings = [
        '关键词匹配仅产生候选，不证明采用该模块、提出贡献或实验可比。',
        '技术词参与检索：样本有选择偏差；占比也不能直接当作总体趋势。',
        'OpenAlex 为每短语 top-200 高被引样本，不能代表全部 venue 分布。',
        '训练期使用 GT 本身不是缺陷；推理零开销必须核对实际推理图。',
        '未完成逐篇全文差异核查，禁止输出“空白/新颖性极高/必测/饱和”等确定结论。',
    ]
    if not meta.get('fetched_at') and not meta.get('all_queries_have_provenance'):
        warnings.append('历史语料缺少抓取时间；文件修改时间不能代替来源时间。')
    unknown = [q['query'] for q in query_stats if q.get('total') in (None, 0) and q.get('fetched', 0)>0]
    if unknown:
        warnings.append(f'{len(unknown)} 条历史检索有结果但 total 未知/错误；覆盖率与截断状态不可判定，需 refresh。')
    capped = [q['query'] for q in query_stats if q.get('possibly_truncated')]
    if capped:
        warnings.append(f'{len(capped)} 条检索可能达到上限，不能用于总体增长估计。')
    missing_authors = sum(not p.get('authors') for p in papers)
    if missing_authors:
        warnings.append(f'{missing_authors} 篇缺作者；已修复解析器，但历史记录不会自动变完整。')
    audit = dict(as_of=as_of.isoformat(), rule_version=RULE_VERSION,
        input_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        raw_count=len(raw['papers']), included_count=len(papers), excluded=excluded,
        configured_queries=len(QUERIES), recorded_queries=len(query_stats),
        broad_intersection=len(broad), strict_intersection=len(strict),
        strict_scope_counts=dict(Counter(scope_hint(evidence[p['arxiv_id']]) for p in strict)),
        source_verification_counts=dict(Counter(verification_status(p, checks) for p in union.values())),
        warnings=warnings)
    (out/'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    (out/'ranking.json').write_text(json.dumps({'method': 'decision-gates-not-scores',
        'as_of': as_of.isoformat(), 'directions': directions}, ensure_ascii=False, indent=2)+'\n')
    write_csv(out/'decision_matrix.csv', ['id', 'name', 'priority', 'keyword_candidates',
              'aerial_candidates', 'question', 'gate', 'novelty'], directions)
    matches = [dict(arxiv_id=p['arxiv_id'], title=p['title'], published=p['published'],
        url=f"https://arxiv.org/abs/{p['arxiv_id']}", scope_hint=scope_hint(evidence[p['arxiv_id']]),
        verification=verification_status(p, checks), evidence=evidence[p['arxiv_id']]) for p in papers]
    (root/'data/processed').mkdir(parents=True, exist_ok=True)
    (root/'data/processed/decision_evidence.json').write_text(json.dumps(matches, ensure_ascii=False, indent=2)+'\n')
    lines = ['# 小目标检测：证据与决策报告（v2）',
        f'> 分析日期 {as_of}；语料 SHA256 `{audit["input_sha256"]}`；规则 {RULE_VERSION}',
        '## 结论先行',
        '**建议继续 ECDet 精度诊断和 SET 迁移验证，但暂不认定“频域×DETR”是新颖选题。**',
        '以下优先级来自用户当前精度优先目标及实验依赖，不是论文热度评分或发表概率。',
        '## 语料与可靠性',
        f'- 输入 {len(raw["papers"])} 篇；日期/去重筛查后 {len(papers)} 篇；排除 {len(excluded)} 篇。',
        f'- 配置 {len(QUERIES)} 组检索；历史文件记录 {len(query_stats)} 组。',
        *['- '+w for w in warnings],
        '## 频域 × DETR：真实交集',
        f'- 原宽规则交集 {len(broad)} 篇；收紧规则并要求小目标证据后 {len(strict)} 篇。',
        f'- 收紧交集任务提示：{json.dumps(audit["strict_scope_counts"], ensure_ascii=False)}。分类仍需人工确认。',
        f'- 候选来源核验：{json.dumps(audit["source_verification_counts"], ensure_ascii=False)}。核验仅比较元数据，不代表阅读全文。',
        '| arXiv ID | 标题 | 任务提示 | 元数据核验 |', '|---|---|---|---|']
    for row in reviews:
        lines.append(f'| {row["arxiv_id"]} | {row["title"].replace("|", "/")} | {row["scope_hint"]} | {row["verification"]} |')
    lines += ['## 决策表', '| 方向 | 优先级 | 严格候选/航空提示 | 下一道证据门槛 |', '|---|---|---|---|']
    for d in directions:
        lines.append(f'| {d["name"]} | {d["priority"]} | {d["keyword_candidates"]}/{d["aerial_candidates"]} | {d["gate"]} |')
    lines += ['## 完整年份的样本占比（描述性，不作排名）',
        f'{as_of.year} 为未完成年度，单独保留在原统计中，不参与首末年比较。零命中年份保留；分母为当年本地语料。',
        '| 年份 | 样本数 | 频域×DETR×小目标命中 | 样本占比 % |', '|---|---|---|---|']
    for row in directions[0]['full_year_sample_trend']:
        lines.append(f'| {row["year"]} | {row["n"]} | {row["hits"]} | {row["share_pct"]} |')
    lines += ['## 下一轮实验建议（待执行）',
        '1. B0 原模型与 B1 无增强辅助监督先跑通：同初始化、输入尺度、协议、seed，记录额外训练成本。',
        '2. 固定覆盖定义，检查 encoder 候选到 top-k 的 GT 覆盖变化、decoder 定位误差、按尺度/密度分组召回。',
        '3. B2 全图平滑与 B3 HBS 对齐辅助路径及损失权重，排除仅增加监督或平滑的解释。',
        '4. 若背景误检下降却损伤极小目标召回，再验证尺度/密度感知保护；若瓶颈是 query 入选，再研究相关训练约束。',
        '5. 短程筛选不作为最终排名；同正式协议多 seed 验证后，再增加第二数据集和第二检测器。',
        '6. 如果收益无法超过重复实验波动，或不能区别于相关工作的机制，就回到诊断，暂不扩大模块组合。',
        '## 人工核查入口',
        '`paper_review.csv` 保存原句证据及人工字段；重新生成不会覆盖人工填写内容。',
        '逐篇记录任务、基线、数据集版本、接入点、是否只用于训练、GT 依赖、query/匹配变化、公平性、与本方案区别及页码。',
        '源记录不一致或核验失败的论文必须先复核；未写全文定位的区别不作为新颖性证据。',
        '本报告不会从关键词频次推出“即插即用、推理零开销、跨架构通用”；这些都需实现与实验验证。']
    (out/'report.md').write_text('\n'.join(('\n'+line if line.startswith('## ') else line) for line in lines)+'\n', encoding='utf-8')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--as-of', type=date.fromisoformat, default=date.today())
    args = parser.parse_args()
    print(json.dumps(build(as_of=args.as_of), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
