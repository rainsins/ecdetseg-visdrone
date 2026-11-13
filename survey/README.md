# 小目标检测(TOD)调研爬虫与分析

用爬虫对小目标检测(Small/Tiny Object Detection)领域做**数据驱动的综述分析**，
为课题组(EdgeCrafter / SET 频域增强)选定下一个小目标检测选题方向。

## 数据来源

| 来源 | 作用 | 备注 |
|------|------|------|
| arXiv API | 主语料(标题+摘要)，本次刷新后 924 篇去重论文(2022–2026) | 26 组检索式，数量由配置动态记录 |
| OpenAlex API | venue 标签 + 引用数 + 出版年趋势 | 免费额度共享，请求数已尽量压缩 |

> 本次已验证 `SURVEY_PROXY=http://127.0.0.1:7890` 可访问 arXiv。支持代理环境变量；设置 `SURVEY_PROXY=direct` 强制直连。请求间隔至少 3.2s，限流/服务故障有界重试。
> Semantic Scholar 被限流、DBLP 有反爬、OpenReview 需验证，均未采用。

## 目录结构

```
survey/
├── pixi.toml                 # pixi 环境(python 3.11 + requests/pandas/matplotlib)
├── crawler/
│   ├── config.py             # 检索式 + 关键词/模块/检测器/领域/数据集 taxonomy
│   ├── arxiv_client.py       # arXiv API 客户端(限速 + 退避 + 年份截断)
│   ├── crawl_arxiv.py        # -> data/raw/arxiv_papers.json
│   ├── openalex_client.py    # OpenAlex 客户端
│   └── crawl_openalex.py     # -> data/raw/openalex_*.json
├── analysis/
│   ├── features.py           # venue/模块/检测器/领域/数据集 特征抽取(词边界匹配)
│   ├── analyze.py            # -> data/processed/*.csv + *.json
│   ├── plot.py               # -> output/figures/*.png
│   └── rank.py               # 选题评分排序 + 报告 -> output/report.md + ranking.json
├── data/
│   ├── raw/                  # 爬虫原始 JSON
│   └── processed/            # papers_features.csv + 聚合统计 JSON
└── output/
    ├── report.md             # ★ 最终中文报告(现状/趋势/基线/模块/选题/风险/排序)
    ├── ranking.json          # v2 决策条件（对象格式，不再是 v1 分数数组）
    ├── audit.json            # 语料指纹、筛选数量、数据质量问题
    ├── decision_matrix.csv   # 按问题及实验依赖排列的决策表
    ├── paper_review.csv      # 候选原句证据 + 人工全文对照字段
    └── figures/              # 8 张图表
```

## 复现步骤

```bash
cd survey
pixi install                      # 安装环境
pixi run crawl                    # 爬 arXiv(约 6–8 分钟，注意 3.2s 限速)
pixi run crawl-openalex           # 爬 OpenAlex(约 1 分钟)
pixi run analyze                  # 特征抽取 + 聚合统计
pixi run plot                     # 生成图表
pixi run report                   # 生成评分排序 + report.md
```

v2 推荐流程（在 survey 下）：

```bash
export SURVEY_PROXY=http://127.0.0.1:7890
pixi run python -m crawler.crawl_arxiv --refresh
pixi run verify
pixi run analyze
pixi run plot
pixi run report
pixi run test
```

`--refresh` 先把旧原始语料/缓存复制至 `data/raw/archive/<timestamp>/`；中断后用不带
`--refresh` 的 crawl 继续。旧缓存缺乏来源信息时，报告保留警告，不把重生成时间当作抓取时间。
`verify` 按 ID 比较候选标题、摘要和首次提交日期，保留返回 XML 及核验时间；仅为元数据核验。
`report` 只读已有语料，重新生成不会覆盖 `paper_review.csv` 的人工字段。

`analysis/rank_legacy.py` 归档原评分实现，供追溯；不再由 report 调用。
历史图表仍采用宽关键词分类，仅作样本浏览；选题依据使用 v2 的严格交集及人工核查。

## 决策结论与边界

本轮分析：宽规则频域×DETR 交集 15 篇；严格规则并要求小目标线索后 14 篇，
其中 11 篇有航空场景提示。15 篇候选元数据已通过 arXiv API 核验。
这些是阅读候选数，不是已确认竞争方法数，也不构成“空白点”或“饱和”证明。

优先完成 ECDet 的 B0/B1、query 覆盖和分组召回诊断，再决定 HBS/API 适配方向。
不以摘要频次、引用数或预设的新颖性分数替代全文对照与实验。

完整结果见 `output/report.md`，针对项目的决策建议见 `output/decision_notes.md`。
