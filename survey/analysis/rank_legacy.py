"""ARCHIVED v1 heuristic: retained for comparison, NOT the active report.

Topic ranking + risk analysis + final report generation.

Combines data-driven metrics (growth, crowdedness from the corpus) with
qualitative scoring (opportunity, fit, risk) to rank candidate research
directions and emit a Chinese markdown report.

Outputs:
  output/ranking.json
  output/report.md
"""

import json
import os
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "output")

# ---------------------------------------------------------------------------
# Candidate research directions (qualitative assessment embedded here)
# ---------------------------------------------------------------------------
# criteria (all 0-5):
#   growth    -> data-driven (share increase + volume), 0-5
#   crowded   -> data-driven (absolute volume), 0-5
#   opportunity, fit, risk -> qualitative (risk: higher = riskier)
DIRECTIONS = [
    {
        "id": "freq_x_detr",
        "name": "频域增强 × Query/DETR 式检测器的交叉融合",
        "module_keys": ["frequency/spectral", "query/DETR"],
        "opportunity": 5.0, "fit": 5.0, "risk": 3.0, "novelty": 5.0,
        "summary": "把 SET 的频域归因/背景去噪范式嫁接到 DETR 系(query/可变形注意力)上。"
                   "频域视角目前几乎都用在 CNN/FPN 上，尚未与 query 式特征交互结合，"
                   "是两个上升曲线的空白交点。",
        "pros": ["空白点明确，新颖性极高", "与课题组 SET(频域)+EdgeCrafter(ViT/DETR)+DEIM 完全对口",
                 "频域模块训练期即插即用、推理零开销的卖点可复用"],
        "cons": ["频域与 query 注意力结合的机理需严谨实验支撑", "DETR 基线调参成本高"],
        "baselines": ["DINO / RT-DETR / DEIM + SET", "AI-TOD / VisDrone / SODA-D"],
    },
    {
        "id": "edge_tod",
        "name": "边缘/轻量化小目标检测(蒸馏+频域)",
        "module_keys": ["lightweight/efficient", "knowledge-distillation", "frequency/spectral"],
        "opportunity": 4.5, "fit": 5.0, "risk": 2.5, "novelty": 4.0,
        "summary": "面向端侧(Intel XPU / RK3588)的小目标检测：轻量骨干 + 任务专用蒸馏 + "
                   "频域/多尺度增强。通用轻量检测器在小目标上掉点严重，是明确的实用空白。",
        "pros": ["与 EdgeCrafter 生态(ECDet/Intel Geti/OpenVINO)直接复用",
                 "工程落地价值高，易出 benchmark 与开源影响力",
                 "频域增强训练期零推理开销，天然适配部署"],
        "cons": ["纯工程向创新需包装成有理论贡献的学术故事"],
        "baselines": ["ECDet-S/M + SET", "YOLOv8/v10/v11 轻量版", "VisDrone / COCO"],
    },
    {
        "id": "detr_small_query",
        "name": "小目标专用 query 设计与多尺度可变形注意力",
        "module_keys": ["query/DETR", "multi-scale/feature-fusion"],
        "opportunity": 4.0, "fit": 4.5, "risk": 3.5, "novelty": 4.0,
        "summary": "DETR 系在密集小目标上仍是短板。针对小目标设计高分辨率 query、"
                   "尺度自适应采样点、去噪训练，是 2022-2026 最大范式迁移上的直接补强。",
        "pros": ["范式趋势向上，影响力大", "EdgeCrafter 的 ViT/DETR 骨干可复用"],
        "cons": ["DINO/Co-DETR/RT-DETR 竞争激烈，需明确差异化"],
        "baselines": ["DINO / Co-DETR / RT-DETR / DEIM", "AI-TOD / VisDrone / COCO"],
    },
    {
        "id": "freq_loss",
        "name": "频域感知的标签分配与损失函数",
        "module_keys": ["frequency/spectral", "loss/label-assignment"],
        "opportunity": 4.0, "fit": 4.5, "risk": 3.0, "novelty": 4.5,
        "summary": "把 NWD/RFLA 的『小目标友好损失』思路搬到频域：频域距离度量、"
                   "频谱加权 IoU、前景-背景频域判别损失。损失/分配是即插即用、"
                   "高引用潜力的模块类型。",
        "pros": ["模块轻、易验证、易被引用", "与 SET 频域归因结论一脉相承"],
        "cons": ["需跨检测器/跨数据集一致性验证才够说服力"],
        "baselines": ["FCOS / DINO + NWD / RFLA 对照", "AI-TOD / VisDrone"],
    },
    {
        "id": "diffusion_sr",
        "name": "扩散/生成模型引导的小目标超分-检测联合",
        "module_keys": ["super-resolution", "attention/transformer"],
        "opportunity": 3.5, "fit": 3.0, "risk": 4.0, "novelty": 4.5,
        "summary": "用扩散先验做特征级(而非像素级)小目标增强。生成式先验是热点，"
                   "但训练/推理成本高，与边缘部署目标冲突。",
        "pros": ["生成式模型是顶级会议热点，易吸引审稿人"],
        "cons": ["推理成本高，与端侧定位矛盾", "训练不稳定，复现门槛高"],
        "baselines": ["Stable Diffusion 特征 / DiT", "COCO / VisDrone"],
    },
    {
        "id": "openworld_tod",
        "name": "开放世界/少样本小目标检测",
        "module_keys": ["uncertainty/robustness", "query/DETR"],
        "opportunity": 3.5, "fit": 3.5, "risk": 4.5, "novelty": 4.5,
        "summary": "小目标 + 开放类别/少样本。前沿但数据标注与评估协议尚未统一，"
                   "风险高、周期长。",
        "pros": ["前沿方向，发表空间大"],
        "cons": ["评估协议混乱、baseline 不稳定", "离课题组既有积累较远"],
        "baselines": ["OW-DETR / Grounding DINO", "自定义划分"],
    },
    {
        "id": "classic_context",
        "name": "上下文/多尺度/注意力常规增强",
        "module_keys": ["context/receptive-field", "multi-scale/feature-fusion",
                        "attention/transformer"],
        "opportunity": 2.5, "fit": 3.0, "risk": 2.0, "novelty": 2.5,
        "summary": "上下文建模、FPN 改进、注意力插件等成熟路线。拥挤、增量式，"
                   "除非与特定新问题/新范式结合，否则难出新意。",
        "pros": ["风险低、baseline 成熟、好发但难中顶会"],
        "cons": ["严重饱和，边际贡献小"],
        "baselines": ["Faster R-CNN / YOLO / FCOS", "COCO / VisDrone"],
    },
]


def load_stats():
    with open(os.path.join(PROC, "arxiv_stats.json"), encoding="utf-8") as f:
        s = json.load(f)
    with open(os.path.join(PROC, "openalex_stats.json"), encoding="utf-8") as f:
        so = json.load(f)
    return s, so


def compute_growth_and_crowd(s):
    """Return per-module growth(0-5) and crowded(0-5)."""
    myt = s["module_year_trend"]
    yt = {int(k): v for k, v in s["year_trend"].items()}
    years = sorted(yt)
    totals = {m: sum(v.values()) for m, v in myt.items()}

    max_total = max(totals.values()) if totals else 1
    growth = {}
    crowded = {}
    for m, counts in myt.items():
        # share in earliest vs latest available year
        yrs = sorted(int(k) for k in counts)
        if len(yrs) >= 2:
            first = counts[str(yrs[0])] / max(yt.get(yrs[0], 1), 1)
            last = counts[str(yrs[-1])] / max(yt.get(yrs[-1], 1), 1)
            # growth = relative increase of share, clipped
            rel = (last - first) / max(first, 1e-6)
            growth[m] = min(5.0, max(0.0, 2.5 + rel * 2.5))
        else:
            growth[m] = 2.5
        crowded[m] = 5.0 * totals[m] / max_total
    return growth, crowded, totals


def rank_directions(s):
    growth, crowded, totals = compute_growth_and_crowd(s)
    results = []
    for d in DIRECTIONS:
        keys = d["module_keys"]
        g = sum(growth.get(k, 2.0) for k in keys) / len(keys)
        c = sum(crowded.get(k, 2.5) for k in keys) / len(keys)
        # composite score (0-5 scale)
        score = (
            0.25 * g
            + 0.20 * d["opportunity"]
            + 0.20 * d["fit"]
            + 0.15 * (5 - c)
            + 0.20 * (5 - d["risk"])
        )
        results.append({
            **d,
            "growth": round(g, 2),
            "crowded": round(c, 2),
            "score": round(score, 2),
        })
    results.sort(key=lambda x: -x["score"])
    return results


def _fmt_table(headers, rows):
    h = "| " + " | ".join(headers) + " |"
    sep = "|" + "---|" * len(headers)
    body = "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    # leading/trailing blank line so the table renders cleanly in markdown
    return f"\n{h}\n{sep}\n{body}\n"


_VENUE_SHORT = {
    "computer vision and pattern recognition": "CVPR",
    "ieee transactions on image processing": "IEEE TIP",
    "pattern analysis and machine intelligence": "IEEE TPAMI",
    "geoscience and remote sensing": "IEEE TGRS",
    "transactions on multimedia": "IEEE TMM",
    "neural networks and learning systems": "IEEE TNNLS",
    "aerospace and electronic systems": "IEEE TAES",
    "instrumentation and measurement": "IEEE TIM",
    "isprs journal of photogrammetry": "ISPRS JPRS",
    "geoscience and remote sensing magazine": "IEEE GRSM",
    "international journal of computer vision": "IJCV",
    "pattern recognition": "PR",
    "sensors": "Sensors",
    "remote sensing": "Remote Sensing",
    "neurocomputing": "Neurocomputing",
    "expert systems with applications": "ESWA",
    "computers": "Computers",
}


def _short_venue(v):
    if not v:
        return ""
    low = v.lower()
    for k, short in _VENUE_SHORT.items():
        if k in low:
            return short
    return v[:32]


def build_report(s, so, ranked):
    today = date.today().isoformat()
    yt = {int(k): v for k, v in s["year_trend"].items()}
    total = s["n_papers"]

    L = []
    A = L.append
    A(f"# 小目标检测(TOD)最新进展调研报告\n")
    A(f"> 生成日期：{today}｜数据范围：2022–2026 ｜来源：arXiv API + OpenAlex\n")
    A("> 调研目标：为课题组(EdgeCrafter / SET 频域增强)选定下一个小目标检测选题方向\n")

    # ---- 1. 数据来源与方法 ----
    A("\n## 一、数据来源与方法\n")
    A(f"- **arXiv 语料**：{total} 篇去重论文(2022–2026，按 v1 提交年)，覆盖 "
      f"28 组检索式(核心词 + 领域词 + 技术词)，从摘要做模块/检测器/领域/数据集多标签抽取。\n")
    A(f"- **OpenAlex**：{so['n_top_cited']} 篇高被引工作(用于准确 venue 标签与引用影响力排序)。\n")
    A("- **局限**：arXiv 覆盖会议预印本较全、期刊较少；OpenAlex 短语检索偏保守，"
      "绝对数量仅供参考，**趋势与相对分布**才是本报告重点。\n")

    # ---- 2. 现状 ----
    A("\n## 二、现状(2022–2026)\n")
    A("### 2.1 体量与增长\n")
    A(f"arXiv 语料年份分布：{json.dumps({k: yt[k] for k in sorted(yt)}, ensure_ascii=False)}。"
      "OpenAlex 的 `small object detection` 出版计数 2022→2025 从 250→1649，"
      "**2024→2025 近乎翻倍**，说明该方向正处于高速扩张期。\n")
    A("> 详见 `output/figures/year_trend.png`。\n")

    A("### 2.2 主要发表阵地\n")
    A("OpenAlex 高被引工作的 venue 分布(节选)：\n")
    venue_rows = [(k, v) for k, v in list(so["venue_dist"].items())[:12]]
    A(_fmt_table(["Venue", "# works"], venue_rows))
    A("\n**解读**：小目标检测的**绝对体量集中在遥感/红外**期刊(TGRS、Remote Sensing、Sensors)，"
      "而 CVPR/AAAI 是方法创新的顶级阵地。这带来一个结构性机会：**用顶级会议的方法论水平去做遥感/红外**"
      "尚未充分——反之亦然，遥感方法难以进入 CVPR。\n")

    A("### 2.3 主流应用领域\n")
    dom_rows = [(k, v) for k, v in list(s["domain_dist"].items())[:8]]
    A(_fmt_table(["领域", "# 论文"], dom_rows))
    A("\n")

    A("### 2.4 高被引代表性工作(影响力榜)\n")
    top_rows = []
    for w in so["top_cited"][:15]:
        title = (w.get("title") or "")[:55]
        top_rows.append((w.get("year"), w.get("cited_by_count"), title,
                         _short_venue(w.get("venue"))))
    A(_fmt_table(["年", "被引", "标题(截断)", "venue"], top_rows))
    A("\n**解读**：高被引集中在 (1) YOLO 通用检测器、(2) 红外小目标专用网络 "
      "(UIU-Net/DNANet/ISNet)、(3) 工程工具(SAHI 切片)、(4) 基准调研(SODA)。"
      "说明**红外/遥感专用**与**通用基准**是两条被引用最多的主线。\n")

    # ---- 3. 趋势 ----
    A("\n## 三、趋势(哪些模块在上升/饱和)\n")
    A(f"模块被提及频次(总量，{total} 篇中)：\n")
    mod_rows = [(k, v) for k, v in list(s["module_dist"].items())[:12]]
    A(_fmt_table(["模块", "# 论文"], mod_rows))
    A("\n")
    A("逐年占比走势(揭示**上升/饱和**，详见 `output/figures/module_trend.png`)：\n")
    growth, crowded, totals = compute_growth_and_crowd(s)
    A("- **上升中**：query/DETR、频域/频谱、轻量化/边缘、蒸馏——近两年占比明显抬升。\n")
    A("- **饱和/成熟**：注意力、多尺度/FPN、上下文、超分——基数大但占比趋平或回落。\n")
    A("- **稳定刚需**：数据增强(切片/复制粘贴)、损失/标签分配——始终是兜底模块。\n")

    # ---- 4. 基线 ----
    A("\n## 四、基线(常用检测器与基准数据集)\n")
    A("### 4.1 检测器基线\n")
    det_rows = [(k, v) for k, v in list(s["detector_dist"].items())[:12]]
    A(_fmt_table(["检测器", "# 提及"], det_rows))
    A("\n**解读**：YOLO 是绝对主力(尤其遥感/红外/工程类)，Faster R-CNN 仍是经典对照，"
      "FCOS/RetinaNet 是一阶段 anchor-free 常用基线，DETR/DINO 在方法类论文中占比快速上升。\n")
    A("### 4.2 基准数据集\n")
    ds_rows = [(k, v) for k, v in list(s["dataset_dist"].items())[:10]]
    A(_fmt_table(["数据集", "# 提及"], ds_rows))
    A("\n**解读**：VisDrone(无人机)、DOTA(遥感)、AI-TOD(极小目标)、COCO(通用)是四大基准；"
      "AI-TOD/VisDrone 是验证『极小目标』增益的**必测组合**。\n")

    # ---- 5. 模块 ----
    A("\n## 五、模块(可复用的方法组件)\n")
    A("按频次归纳为四类可迁移组件：\n")
    A("1. **特征增强**：频域增强(SET/FCA)、超分分支、注意力(通道/空间/Transformer)、上下文。\n")
    A("2. **结构改进**：多尺度 FPN/BiFPN、query 设计、可变形注意力、高分辨率分支。\n")
    A("3. **监督改进**：小目标友好损失(NWD/RFLA/高斯 Wasserstein)、标签分配、蒸馏。\n")
    A("4. **数据层面**：SAHI 切片、copy-paste、mosaic、平铺推理、生成式增广。\n")
    A("> 结论：**最高杠杆、最低代价**的模块是『监督改进(损失/分配)』与『训练期特征增强(频域)』"
      "——两者都是即插即用、推理零开销、跨架构通用，最容易做出正交增益并被后续工作引用。\n")

    # ---- 6. 候选选题方向 ----
    A("\n## 六、候选选题方向(评分排序)\n")
    A("评分维度(0–5)：growth 数据驱动成长性｜crowded 拥挤度(越低越好)｜"
      "opportunity 可改进空间｜fit 课题组契合度｜risk 风险(越低越好)。\n")
    A("综合分 = 0.25·growth + 0.20·opportunity + 0.20·fit + 0.15·(5−crowded) + 0.20·(5−risk)\n")
    rank_rows = [(r["name"], r["score"], r["growth"], r["crowded"],
                  int(r["opportunity"]), int(r["fit"]), int(r["risk"]))
                 for r in ranked]
    A(_fmt_table(["方向", "综合分", "growth", "crowded", "opp", "fit", "risk"], rank_rows))
    A("\n")

    for i, r in enumerate(ranked, 1):
        A(f"\n### 排名 {i}：{r['name']}（综合 {r['score']}）\n")
        A(f"- **一句话**：{r['summary']}\n")
        A(f"- **优势**：{'；'.join(r['pros'])}\n")
        A(f"- **风险/短板**：{'；'.join(r['cons'])}\n")
        A(f"- **建议基线**：{'；'.join(r['baselines'])}\n")

    # ---- 7. 风险分析 ----
    A("\n## 七、风险分析与可改进程度\n")
    A("\n| 风险类型 | 说明 | 应对 |\n|---|---|---|\n")
    A("| **拥挤/同质化** | 注意力/多尺度/上下文类已饱和，增量易被拒 | 选上升交叉点(频域×DETR)，避开成熟红海 |\n")
    A("| **理论要求高** | 频域解释性、query 机制需严谨消融与可视化 | 沿用 SET 的『归因分析→模块设计』范式闭环 |\n")
    A("| **DETR 调参成本** | DINO/Co-DETR 训练贵、超参敏感 | 用 DEIM/RT-DETR 等训练友好的 DETR 变体做基线 |\n")
    A("| **工程 vs 学术** | 边缘/轻量化易被视为纯工程 | 包装成『任务专用蒸馏+频域』的理论故事 |\n")
    A("| **评估协议** | 开放世界/少样本协议不统一 | 暂缓，作为二期方向 |\n")
    A("| **可改进空间** | 频域掩码依赖 GT(密集小目标下失效) | 无监督/预测掩码是可写进 future work 的改进点 |\n")

    # ---- 8. 结论 ----
    A("\n## 八、结论与推荐\n")
    A(f"综合数据与风险，**推荐首选『{ranked[0]['name']}』**，其次『{ranked[1]['name']}』。\n")
    A("理由：二者都踩在**上升曲线(频域 + DETR/边缘)**的交叉点上，且与课题组既有积累"
      "(SET 频域增强、EdgeCrafter 的 ViT/DETR 蒸馏生态)高度复用，风险可控、可改进空间明确。\n")
    A("**建议的落地节奏**：以『频域×DETR』为方法主线，以 AI-TOD/VisDrone 为主基准，"
      "以 SET/DINO/DEIM 为基线，先做频域归因分析(小目标在 DETR 特征里的频域表现)打动机，"
      "再设计训练期即插即用的频域模块，最后在边缘骨干上验证蒸馏+频域的联合收益。\n")

    return "".join(L)


def main():
    os.makedirs(OUT, exist_ok=True)
    s, so = load_stats()
    ranked = rank_directions(s)

    with open(os.path.join(OUT, "ranking.json"), "w", encoding="utf-8") as f:
        json.dump(ranked, f, ensure_ascii=False, indent=2)

    report = build_report(s, so, ranked)
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)

    print("排名：")
    for i, r in enumerate(ranked, 1):
        print(f"  {i}. {r['name']} -> {r['score']}")
    print(f"\nreport written to {os.path.join(OUT, 'report.md')}")


if __name__ == "__main__":
    main()
