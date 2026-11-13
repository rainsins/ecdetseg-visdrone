# ECDet-S / EdgeCrafter-S：VisDrone 模块筛选与实验基准

> 文档日期及调研截止：2026-11-10。
> 用途：保存上一轮研究结论，作为模块选择、对照实验与预算规划的基准。
> 状态：研究方案，尚未在当前 ECDet 上实现或验证所列新模块；文献结果不等于本项目结果。
> 本次落盘只核对了本地仓库与 SET 源文件，没有重新联网复核全部文献。外部结果、出版状态和搜索记录均归档自上一轮报告；存在缺口的条目继续标记为待核实。

## 使用规则

- 优先顺序：**SET-HBS → SO-DETR 风格输出蒸馏 → 真实 stride-8 细节注入**。
- 推进顺序：单模块 → 两两组合 → 三模块组合 → 多 seed → 独立数据集及部署验证。
- 第一篇以精度和小目标机制为主；量化、整数化与大规模结构压缩留给第二篇。
- 收益统一用 AP 百分点记录，例如 28.8→29.8 为 +1.0 AP；不要混用 0.288 和 28.8。
- 禁止把不同论文、数据划分、输入尺寸、硬件和推理后端的数字直接拼接为可比结论。
- “已公开代码”仅表示发现源码，不表示已复现；仓库存在不等于实现完整。
- 对已有方法做简化或改造时，名称使用“方法名-inspired / 适配”，不冒充原论文完整复现。

## 0. 本地基线与版本锚点

### 0.1 本次核对的仓库状态

| 项目 | 记录 |
|---|---|
| 项目目录 | `/home/rainsin/Projects/Career/Paper/EdgeCrafter` |
| 主仓库 HEAD | `f0ea9999f56ef191e0d088f3d9d4119d87ff1c05` |
| 主仓库工作区 | 存在用户未提交修改；HEAD 不能单独代表当前实验代码 |
| SET 本地目录 | `competition/SET` |
| SET origin | `https://github.com/HuixinSun/SET.git` |
| SET HEAD | `2ce6e65d0af61d4a8990cd6de0abe03ceecccad1` |
| SET 工作区 | 本次检查 `git status --short` 无输出 |
| 已读取核心实现 | `competition/SET/mmdet/models/detectors/fcos_set.py` |
| 已确认配置存在 | `competition/SET/configs/aitod/fcos_r50_set.py` |
| 已确认最佳实验快照存在 | `ecdetseg/outputs/p150u/e3_s42/snapshot/config_merged.yml` |

正式实验必须保存实际配置、代码差异、依赖版本和权重标识；不能仅以主仓库 HEAD 锁定含未提交修改的工作区。本次仅新增本文档，不更改已有实验代码或 SET 实现。

### 0.2 上轮已审查的基线事实

以下为上一轮源码与实验记录审查结论，不代表本次重跑结果。

| 维度 | 基线记录 |
|---|---|
| 模型 | ECDet-S / EdgeCrafter-S，ECViT-T，embed_dim=192 |
| 参数量 | 9,803,112，约 9.8M |
| FLOPs | 用户原始描述约 26G；未按统一工具重新核算，不当成已确认测量值 |
| 输入 | 640×640；任务为 VisDrone2019 DET，10 类 |
| ConvStem | 通道 24/48/96/192，对应 stride 2/4/8/16 |
| ViT 特征 | interaction_indexes=[10,11]，末两层同尺度 token 均值 |
| 金字塔 | token 经 bilinear 生成 P3/P4/P5；真实 stem stride-8 尚未直接进入 P3 |
| Neck | hidden_dim=192；P5 AIFI，nearest 上采样，sum 融合，双向 FPN/PAN |
| Decoder | 4 层，Q=300，points=[3,6,3]，reg_max=32 |
| 已有机制 | Gate、FDR/LQE、MAL/FGL/DDF，不能重新包装成新增能力 |
| 蒸馏状态 | 当前检测训练未发现外部 teacher 在线蒸馏 |
| 硬件目标 | Intel Arc A770 / NVIDIA T4 级别，B1 推理目标 <10ms |
| 期刊目标 | TGRS / EAAI / T-ITS / TMM；匹配度由最终贡献与实验决定 |

历史 best EMA 结果（AP 均为百分制）：

| 实验 | AP | APs | 用途 |
|---|---:|---:|---|
| `p150u/e1_s42` | 25.817 | 16.412 | 历史参考 |
| `p150u/e1_s43` | 24.896 | 15.675 | 与 e1_s42 配置仅 seed/output_dir 不同，提示波动 |
| `p150u/e2_s42` | 28.050 | 18.057 | 历史参考 |
| `p150u/e3_s42` | 28.819 | 18.671 | 正式 NVIDIA 基线重跑的配置来源 |

注意：e1/e2/e3 之间不能自动解释为单模块消融。正式基线 B 必须从 e3 完整 snapshot 重建，在同一 CUDA 环境与新方法比较；不能将 A770 历史结果和 NVIDIA 新实验直接归因比较。

### 0.3 接入前必须尊重的本地约束

- `ecdetseg/engine/edgecrafter/ecvit.py`：保留预训练 key 兼容；新参数放入 `.backbone` 命名空间可能被分到较低学习率，必须检查参数分组。
- `ecdetseg/engine/edgecrafter/hybrid_encoder.py`：已有全局建模及双向融合，新增模块要证明额外能力。
- `ecdetseg/engine/edgecrafter/decoder.py`：默认全尺度 max-class-logit topQ；选中 memory 和初始 bbox detach，硬 top-k 不会自动训练 selector。
- `ecdetseg/engine/edgecrafter/criterion.py`：已有 MAL/FGL/DDF；`forward()` 会调用 `_clear_cache()`，不要把“没有清缓存”当作现存缺陷。
- `ecdetseg/engine/edgecrafter/denoising.py`：DN cap100 仅限制 DN，不截断主 GT；两次 forward 的 DN 噪声及 query 对齐需要显式定义。
- `ecdetseg/engine/edgecrafter/postprocessor.py`：全局 query×class top300；其全类别上限与 COCO maxDets 的每类语义不同。
- P2 四尺度不是只改 `num_levels`：还需同步 adapter、encoder 索引、strides、points、anchors，本轮不作为低风险方案。
- 保留当前训练/选模/EMA 协议；val 已用于开发，最终不能把反复调参后的 val 当作独立泛化证据。

---

## Part 1：30 条候选与筛选记录

### 1.1 筛选口径

1. 时间窗口：2025-01-01 之后首次公开，截止 2026-11-10，优先 2026；正式出版年份不能替代首次公开年份。
2. 保留 DETR 范式，理论核心适配 <500 行；搬入整套代码不能规避此限制。
3. 成本按原要求为“新增参数 <10% **或**延迟增幅 <10%”；组合另需满足绝对 <10ms 和自身相对延迟目标。
4. 必须有 COCO/VisDrone/UAVDT/DOTA 等检测结果及小目标机制证据。
5. 排除 SAM/CLIP/LLM 挂载、依赖 TTA/多尺度推理/集成、完全无小目标分析或不可靠来源。
6. EdgeCrafter、DEIM、DEIMv2、D-FINE、RT-DETR、ViT-Adapter、DINOv3 不作为新推荐模块，表中仅在必要时作为文献基线。

**没有核得 30 篇同时满足全部条件的即插即用方法。** 以下是 30 条筛选记录，不是 30 个已过硬筛的方法：

- **优先验证**：证据相对完整，适合先做适配；并非已在 ECDet 通过成本门槛。
- **条件候选**：缺成本、实现或适配证据，补齐后再进入正式训练。
- **暂缓/淘汰**：保留检索记录，明确不进入首轮。
- 置信度表示来源和筛选判断可信度，不是涨点概率。
- 成本区分训练参数、部署参数、训练步时、FLOPs、实测延迟；不得相互替代。

### 1.2 优先验证与重点复现对象

| 排名 | 模块/方法 | 论文标题 | Venue/年份 | 链接 | 代码 | 核心机制 | 插入位置 | 收益证据与本地预期 | 额外成本 | 与 EdgeCrafter/DEIM 的差异 | 置信度/状态 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | SET：先 HBS，再 API | SET: Spectral Enhancement for Tiny Object Detection | CVPR 2025，正式 | [DOI:10.1109/CVPR52734.2025.00444](https://doi.org/10.1109/CVPR52734.2025.00444) | [官方](https://github.com/HuixinSun/SET)，本地 `competition/SET` | GT 前景保护、背景平滑、训练期对抗扰动 | neck/loss | FCOS AI-TOD AP 12.0→14.2；VisDrone 19.9→21.9；本地待测 | 推理路径不增加；完整 SET 原实验 0.17→0.42 s/batch；训练分支有参数 | 处理特征背景干扰，不是定位分布损失 | 高；优先验证 |
| 2 | SO-DETR-KD | SO-DETR: Leveraging Dual-Domain Features and Knowledge Distillation for Small Object Detection | arXiv 2025，正式 venue 待核 | [arXiv:2504.11470](https://arxiv.org/abs/2504.11470) | [官方](https://github.com/ValiantDiligent/SO_DETR) | 末层分类与定位蒸馏，配合权重调度 | loss/训练入口 | VisDrone EV2 AP 28.2→28.8；GIoU KD 配置 APs 19.7→20.0 | 学生 12.1M/33.3G 不变；增加 teacher 成本 | 外部监督，不等同于现有层间 DDF | 中高；优先验证 |
| 3 | TinyFormer-SSA | TinyFormer: Preserving Tiny Objects in YOLO-DETR Hybrid Real-time Detectors | arXiv 2026，正式接收待核 | [arXiv:2605.25046](https://arxiv.org/abs/2605.25046) | [官方](https://github.com/mmpmmpmmpjosh/TinyFormer) | 卷积浅层空间信息补充 ViT 特征 | backbone/adapter | 独立 SSA 大模型消融 AP 57.13→58.38，APs 38.50→39.33 | 该配置 48.8→49.8M、146.8→151.1G；10M 模型成本待测 | 补真实细节，而非粗 token 上采样 | 中高；优先验证轻量适配 |
| 4 | LoRA-DETR | Integrating Diverse Assignment Strategies into DETRs | AAAI 2026，正式；上轮已核官方论文 | [arXiv:2601.09247](https://arxiv.org/abs/2601.09247) | [官方](https://github.com/Z1zyw/LoRA-DETR) | 训练期低秩辅助分支与多样化 assignment | decoder/matcher/loss | COCO300Q 单辅助分支 AP 43.7→48.8，APs 26.4→30.4；完整训练方案收益 | 训练参数 47.3→48.1M，推理移除辅助分支 | 增加监督多样性，不仅增加正样本数量 | 中高；第二批验证 |
| 5 | TPRD | Enhanced Knowledge Distillation for Detection Transformers via Teacher Prediction Refinement | arXiv 2026，正式 venue 待核 | [arXiv:2609.19964](https://arxiv.org/abs/2609.19964) | [官方](https://github.com/xingyitong1/TPRD) | 跨层修正 teacher、保留非目标暗知识 | KD loss | RT-DETR+KD AP 46.2→46.9，APs 28.1→28.6 | 学生推理不变，依赖已有 KD 路径 | 修正外部监督，而非学生自身定位精炼 | 中；条件候选 |
| 6 | SFDNet-CPD | Adaptive Spectrum-Aware Feature Disentangled Network for Small Object Detection | ECCV 2026，正式接收；上轮核对官网 | [arXiv:2606.29029](https://arxiv.org/abs/2606.29029) | [官方](https://github.com/ManOfStory/SFDNet) | 类别原型约束小目标实例特征 | neck/loss | AI-TOD CPD 单项 AP 24.8→27.2，APvt 9.3→12.0 | 训练原型分支可不进入推理；DETR 成本待测 | 类别语义约束，与定位 DDF 不同 | 中；条件候选 |
| 7 | MDS-DETR | MDS-DETR: DETR with Masked Duplicate Suppressor | arXiv 2026，正式接收待核 | [arXiv:2605.23507](https://arxiv.org/abs/2605.23507) | [官方](https://github.com/DChoLee/MDS-DETR) | 前层 O2M、末层 O2O，置信度因果 mask 去重 | decoder/matcher | COCO300Q AP 47.0→49.8，APs 30.1→32.2 | 训练40→42 min/epoch，显存7534→8252MB；本机推理待测 | 改层职责，不是单一 attention mask | 中；高风险条件候选 |
| 8 | EagleConv | EagleConv: Bio-inspired Dual-Foveated Convolution for Robust Small Object Detection | UAI 2026，正式 | [PMLR337:3922–3932](https://proceedings.mlr.press/v337/liu26a.html)；DOI/arXiv 未检得 | 官方可运行代码待核；论文有公式 | 三高斯中心—周围滤波、部分通道卷积、残差门控 | 浅层卷积/neck | YOLO11s VisDrone960 AP 28.46→31.83(K7)/32.22(K17) | K7 9.43→8.74M，5.92→6.48ms(+9.5%)；K17 +8.3% 延迟 | 浅层频率选择，不等于 decoder Gate | 中；书目与复现条件候选 |
| 9 | CLoCKDistill | CLoCKDistill: Consistent Location-and-Context-aware Knowledge Distillation for DETRs | arXiv 2025，正式 venue 待核 | [arXiv:2502.10683](https://arxiv.org/abs/2502.10683) | [仓库](https://github.com/lanqz7766/CLoCKDistill)为占位，不能计为可用实现 | 位置感知 memory 与 target-aware query 蒸馏 | encoder memory/loss | DINO-R18 AP43.5→46.4，APs25.3→28.6 | 学生31M/204G不变；teacher 增训练成本 | 上下文/位置蒸馏，不是普通逐层L2 | 中；自行复现候选 |
| 10 | FIG | Feature Information Driven Position Gaussian Distribution Estimation for Tiny Object Detection | CVPR 2025，正式 | [DOI:10.1109/CVPR52734.2025.02828](https://doi.org/10.1109/CVPR52734.2025.02828) | 可运行官方仓库待核；补充材料有结构 | 信息量图与 GT 高斯位置图增强弱区域 | neck/辅助loss | 文内组件消融 AP26.3→28.3，不是 ECDet 结果 | 独立参数与延迟证据不足 | 显式空间监督，不等于框边界分布精炼 | 中低；成本未过筛 |

TinyFormer-S 整体对比不能全部归因于 SSA；复用现有 stem 的方案应称为 SSA-inspired 适配。EagleConv 正式论文可核实，但缺 DOI/arXiv，因此不包装成完全满足本轮书目要求的条目。

### 1.3 有价值但证据尚未闭环的候选

| 排名 | 模块/方法 | 论文标题 | Venue/年份 | 链接 | 代码 | 核心机制 | 插入位置 | 收益证据与本地预期 | 额外成本 | 与 EdgeCrafter/DEIM 的差异 | 置信度/状态 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 11 | QCAR/GCQA | Query-calibrated attention and visibility-aware anisotropic regression for small object detection | Neurocomputing 2026，正式 | [DOI:10.1016/j.neucom.2026.134465](https://doi.org/10.1016/j.neucom.2026.134465) | 待核 | GT椭圆高斯约束 query attention 与分阶段几何回归 | decoder/loss | 摘要确认 VisDrone、AI-TODv2；具体表格未核齐 | 作者称推理结构不变；训练成本待核 | attention 对齐有区别，回归与FGL/DDF可能重叠 | 中低；待全文 |
| 12 | FRFDet：IBS/SFRCF | FRFDet: Efficient UAV Small Object Detection with Symmetric Sampling and Scalable Fusion | arXiv 2026，正式 venue 待核 | [arXiv:2607.04125](https://arxiv.org/abs/2607.04125) | [论文所链仓库](https://github.com/HZAI-ZJNU/FRFDet)，未完整审查 | 对称重采样保细节、按规模选择融合 | neck/downsample | VisDrone消融 AP19.3→22.2；IBS-D单项20.7 | 完整2.6M不变、3.2→4.6G；IBS-D2.1M/3.6G | 改采样和融合，不新增全局 attention | 中；补单模块延迟 |
| 13 | PaQ-DETR | PaQ-DETR: Learning Pattern and Quality-Aware Dynamic Queries for Object Detection | arXiv 2026 | [arXiv:2603.06917](https://arxiv.org/abs/2603.06917) | 待核 | pattern动态 query 与质量感知 O2M | decoder/matcher | COCO DINO AP50.3→51.9，APs34.1→35.1 | 本地参数、延迟、行数未闭环 | 已有图像自适应top-k，需说明增量能力 | 中低；暂缓 |
| 14 | ReDHF | Reviving the Discarded High-Resolution Feature for Transformer-Based Tiny Object Detection | arXiv 2026 | [arXiv:2611.07994](https://arxiv.org/abs/2611.07994) | 待核 | 高分辨率融合、浅层监督、全尺度query交互 | adapter/neck/decoder | VisDrone AP35.9→37.5，APs28.3→30.3 | AI-TOD一组配置13.2→15.9M(+20.5%)，173.1→188.2G；延迟待核 | 同时改特征/监督/query，不是单支路 | 中；整套暂缓 |
| 15 | BUTR | Dynamic Query as Budgeting for Tiny Object Detection | arXiv 2026 | [arXiv:2610.33430](https://arxiv.org/abs/2610.33430) | [官方](https://github.com/zhangyi-magic/BUTR) | 无参数query预算器 | query selection | COCO D-FINE-S AP48.5→48.6 | 25.2→24.6G，但T4 3.63→4.24ms(+16.8%) | 调整query数量，非解决浅层信息损失 | 高；不进当前低延迟组合 |
| 16 | Dome-DETR | Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection | ACM MM 2025，正式 | [arXiv:2505.05741](https://arxiv.org/abs/2505.05741)；[DOI:10.1145/3746027.3754861](https://doi.org/10.1145/3746027.3754861) | [官方](https://github.com/RicePasteM/Dome-DETR) | 密度增强、窗口稀疏化、自适应query | neck/decoder | VisDrone800 AP31.2→33.5 | S配置11.6→13.2M，112.4→176.5G；目标硬件延迟待核 | 密度与query预算耦合，改动面大 | 中；整套暂缓 |
| 17 | EFSI-DETR：FFR | EFSI-DETR: Efficient Frequency-Semantic Integration for Real-Time Small Object Detection in UAV Imagery | arXiv 2026 | [arXiv:2601.18597](https://arxiv.org/abs/2601.18597) | 待核 | 浅层保留、频率/空间融合、语义聚合 | neck | FFR消融 AP26.9→31.3 | 25.6→27.7M(+8.2%)；单项延迟待核 | 与浅层注入接近，整体框架远复杂于单分支 | 中；仅保留FFR线索 |
| 18 | FSDC-DETR | FSDC-DETR: A Frequency-Spatial Domain Collaborative DETR for Small Object Detection | ECCV 2026，正式接收；上轮核官网 | [arXiv:2607.05176](https://arxiv.org/abs/2607.05176) | [仓库](https://github.com/nevereverinsomnia/FSDC-DETR)未发现可用Python实现 | 频率/空间双分支、融合与动态下采样 | backbone/neck | VisDrone test800 AP24.7→31.1；有子项消融 | 整体32→40.3M(+25.9%)；独立成本未知 | 整体提升不能归到一个模块 | 中；整套暂缓 |
| 19 | Sigmoid-τ KD | Protocol-consistent Sigmoid-τ logit distillation for lightweight object detection | Image and Vision Computing 2026，正式 | [DOI:10.1016/j.imavis.2026.106045](https://doi.org/10.1016/j.imavis.2026.106045) | 待核 | 类别独立蒸馏概率协议，避免softmax错配 | loss | 覆盖COCO、VisDrone、COCO-Small；完整数值待核 | 推理无需teacher；训练成本待核 | 对sigmoid头有启发，DETR仍需query对齐 | 中低；待全文 |
| 20 | SAFPN | SAFPN: spatial adaptive feature pyramid network for small object detection | Scientific Reports 2026，正式 | [DOI:10.1038/s41598-026-66640-9](https://doi.org/10.1038/s41598-026-66640-9) | 待核 | 侧向表征、空间纠偏、动态融合 | neck | 摘要确认COCO等实验；完整增益/成本未核 | 参数和延迟证据不足 | 空间纠偏不同于nearest+sum，完整替换必要性未知 | 低；不进实验 |

BUTR 无新增参数，可能满足“参数 OR 延迟”的参数条件；不推荐进本轮组合的原因是其已报告 T4 延迟增幅。ReDHF、Dome-DETR、FSDC-DETR 参数超限且尚无满足目标的延迟证据，因此当前不能判合格，不能仅凭 FLOPs 判延迟。

### 1.4 高效 ViT、训练策略与部署筛选记录

| 排名 | 模块/方法 | 论文标题 | Venue/年份 | 链接 | 代码 | 核心机制 | 插入位置 | 收益证据与本地预期 | 额外成本 | 与 EdgeCrafter/DEIM 的差异 | 置信度/状态 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 21 | MambaFPN | MambaFPN: A SSM-based feature pyramid network for object detection | Neural Networks 2026，正式 | [DOI:10.1016/j.neunet.2026.108544](https://doi.org/10.1016/j.neunet.2026.108544) | 待核 | SSM跨层多尺度上下文 | neck | 摘要确认COCO/LVIS提升及参数减少；APs/成本待核 | A770/T4算子效率及改动量待核 | 已有全局attention/双向融合，需证明新增能力 | 低；暂缓 |
| 22 | TAP | Putting Registers to Work: Task Registers for Token Pruning in Vision Transformers | arXiv 2026 | [arXiv:2608.10989](https://arxiv.org/abs/2608.10989) | 可运行实现待核 | task register指导token预算与稠密恢复 | backbone | COCO AP54.0→53.7；全模型吞吐约1.20× | 约+2.3K参数；目标硬件待测 | 需处理稀疏token到规则金字塔恢复 | 中；缺小目标专项证据 |
| 23 | DiD | DiD It in 87 Minutes: A Label-Free Softmax-to-Linear Adaptation of Vision Transformers for Object Detection | arXiv 2026 | [arXiv:2608.22368](https://arxiv.org/abs/2608.22368) | 待核 | 检测接口蒸馏下softmax到线性attention | backbone | DOTA RetinaNet AP33.5→31.1 | 骨干延迟−62%、显存−49%；不是全网等比例收益 | 改预训练骨干，首篇精度目标不占优 | 中；部署文再评估 |
| 24 | LiSA | LiSA: Linear Stochastic Attention for Long-Range Visual Modeling | arXiv 2026，正式venue待核 | [arXiv:2611.04216](https://arxiv.org/abs/2611.04216) | [论文所链](https://github.com/Sahardastani/lisa)，未完整核验 | 随机游走高阶线性attention | backbone | COCO PVT-T AP36.7→40.5；充分APs分析未核得 | 240→245G；并非ECViT-T替换成本 | 涉及骨干适配和预训练迁移 | 中低；证据未达标 |
| 25 | I-LW-DETR | Enabling Fully Integer-Only Inference for Lightweight Detection Transformers | arXiv 2026 | [arXiv:2607.24981](https://arxiv.org/abs/2607.24981) | [仓库](https://github.com/ltc286648/I-LW-DETR)未发现Python实现 | 量化友好结构和整数非线性 | 全网量化/部署 | VisDrone Tiny AP15.7→10.9，APs6.9→4.1 | 模型44.89→12.57MB；无可直接套用A770/T4时延 | 全整数化工程，不是简单QAT配置 | 高；首篇淘汰，第二篇对照 |
| 26 | QRT-DETR | QRT-DETR: Post-training quantization for real-time detection transformer | Neurocomputing 2026，正式 | [DOI:10.1016/j.neucom.2025.131957](https://doi.org/10.1016/j.neucom.2025.131957) | [仓库](https://github.com/ashleyhuo/QRT-DETR)有源码，入口待核 | EMA-MSE校准、渐进重建、量化优化 | PTQ | 摘要报告4-bit 22.6AP；FP→INT完整对照/APs待核 | 整网加速、算子回退待核 | 部署价值，不是精度增益模块 | 中；第二篇候选 |
| 27 | 场景理解Copy-Paste | A Realistic Instance-Level Data Augmentation Method for Small-Object Detection Based on Scene Understanding | Remote Sensing 2026，正式 | [DOI:10.3390/rs18040647](https://doi.org/10.3390/rs18040647) | 待核 | 匹配背景/尺度/光照/视角后粘贴 | aug | 论文报告VisDrone约+1.6AP | 数据生成需多模型流水线 | 使用SAM/CLIP，触发排除条件 | 高；淘汰 |
| 28 | OD³ | OD³: Optimization-free Dataset Distillation for Object Detection | 首版2025；ICLR2026正式 | [arXiv:2506.01942](https://arxiv.org/abs/2506.01942) | [官方](https://github.com/VILA-Lab/OD3) | 实例合成和observer筛选压缩数据集 | aug/dataset | COCO 1%压缩设置对既有方法>14AP50，不是全数据基线增益 | 离线合成成本，小目标保真不足 | 解决数据压缩，而非当前全量精度瓶颈 | 高；当前目标淘汰 |
| 29 | LVTP | Back to Fundamentals: Low-Level Visual Features Guided Progressive Token Pruning | arXiv 2025 | [arXiv:2504.17996](https://arxiv.org/abs/2504.17996) | 本轮未核实 | 低层视觉引导渐进token裁剪 | backbone | 未核得满足要求的检测基准证据 | 检测APs/延迟缺失 | 不能从分割收益推断小目标检测 | 高；按证据门槛淘汰 |
| 30 | LAF-YOLOv10：负结果参考 | Small Object Detection in Drone Aerial Imagery with LAF-YOLOv10 | arXiv 2026 | [arXiv:2609.14560](https://arxiv.org/abs/2609.14560) | [论文所链](https://github.com/Mudassiruddin7/Small-Object-Detection-in-UAV-Imagery) | 轻量卷积、融合、P2和回归改动组合 | 非移植对象 | 三seed24.0±0.4AP50，低于原YOLOv10n31.8AP50 | 2.14M；无符合当前目标的正增益 | 提醒检查预训练与模块负交互 | 高；仅参考实验设计 |

### 1.5 七方向结论

| 方向 | 结论 |
|---|---|
| 无人机小目标 | 有新工作；优先独立消融充分的训练监督或浅层细节模块 |
| 高效ViT/token压缩 | 有新工作，但未筛到小目标证据、低改动、目标硬件成本同时充分的首选 |
| 检测蒸馏 | 最值得投入之一；先证明teacher在tiny子集更强 |
| Neck/融合 | 优先补信息来源，而非重复堆双向融合块 |
| 分配/损失 | LoRA-DETR值得试；MDS改动深；QCAR需补全文 |
| 增强/训练策略 | SET最匹配；部分新Copy-Paste触发排除项 |
| 量化/端侧 | 留第二篇，不能当首篇精度路线 |

不能把“本次没有筛到合格候选”写成“整个方向已经饱和”。普通注意力叠加、常规上采样以及泛化Mosaic/Copy-Paste调参的贡献空间有限，但有限检索不足以证明领域饱和。

---

## Part 2：Top 10 深挖与最小验证

共同原则：固定预训练、数据、seed与选择规则；30 epoch短程只用于筛选，不能替代150 epoch结论。行数为核心适配估算，尚无实际补丁验证。每项1–2 GPU-day目标依赖目标卡步时、teacher是否现成；没有teacher时需另计训练成本。

### 2.1 SET：先HBS，后API

**机制：** HBS依据GT框区分前景与背景，对背景施加平滑并保护目标。API使用检测损失对特征的梯度构造扰动，再用辅助损失学习鲁棒预测。两者均为训练路径，完整SET训练成本明显增加。

- **小目标假设：** 弱小目标被道路、树叶、建筑高频边缘淹没，降低干扰可能优于再加全局建模。
- **位置：** `modeling.py`中的encoder→decoder边界；`criterion.py`组织原始/辅助loss；先仅P3。
- **工作量：** HBS约80–180行，完整HBS+API约200–400行，实际实现后复核。
- **实验：** B / B+HBS / B+HBS+API；目标48 GPU小时内完成短程对照，步时不满足则调整计划。
- **指标：** AP、APs、极小目标召回、背景误检、秒/step、显存和部署延迟。
- **继续条件：** HBS改善tiny recall且误检不显著升高；API必须提供额外收益才保留。
- **风险：** 小框mask下采样消失、漏标被当背景、两次前向DN/匹配差异。criterion已清缓存，不应再次把清缓存当成修复点。

### 2.2 SO-DETR-KD：最简外部teacher路线

**机制：** 原方法蒸馏末decoder层分类和框输出，分类用BCE、坐标用teacher置信度加权L1并配IoU损失。权重随训练调度。文献中GIoU版本也达到28.8AP，无需首步引入Expanded-SIoU。

- **假设：** 更强teacher为模糊小目标提供比单硬标签稳定的分类/定位监督。
- **位置：** 训练入口加载冻结teacher，`modeling.py`返回输出，`criterion.py`添加KD。
- **工作量：** 现成teacher时120–250行，teacher训练另计。
- **实验：** B / 分类KD / 分类+框KD；先GIoU，与现有FGL/DDF共存而非替换。
- **指标/目标：** APs、同GT正确率、分类/定位错误；学生推理图一致。
- **风险：** teacher整体AP高不代表tiny强；第i个query不天然对应，需要匹配或共同GT对齐；先不蒸馏所有层和reg_max分布。

### 2.3 TinyFormer-SSA：复用已有stride-8

**机制：** 原SSA增加卷积空间分支，与ViT表征融合，以补充粗token丢失的局部信息。其独立大模型消融不能直接推广到10M学生。本地建议复用已有96通道stride-8特征，属于SSA-inspired适配。

- **假设：** 上采样恢复空间尺寸，不能恢复已经丢失的小目标边界。
- **位置：** `ecvit.py`的ConvStem stride-8输出和adapter的P3构造，保持三尺度。
- **工作量：** 60–160行；新增参数设计预算≤0.15M，非实测。
- **实验：** B / 浅层简单相加 / 浅层投影+门控；比较APs、召回、两卡时延。
- **风险：** 浅层纹理增加误检；backbone低学习率影响新分支；返回值/命名变化破坏严格加载。
- **创新边界：** 浅层支路已有近似工作，不能单独作为新颖主张。

### 2.4 LoRA-DETR：一条辅助分支起步

**机制：** 训练decoder增加低秩分支，分支学习不同assignment，以增加监督策略多样性。推理仅保留主分支。文献的大幅提升来自完整训练机制，不能归给LoRA算子本身。

- **假设：** 弱目标在单一匹配策略中有效监督不足，多样分配缓解早期正样本稀缺。
- **位置：** `decoder.py`、matcher、`criterion.py`。
- **工作量：** 单分支250–450行，完整多分支是否<500行未知。
- **实验：** B / 单辅助assignment / 再加assignment-quality loss；固定主分支评测。
- **指标：** 收敛、GT有效正样本数、APs、主分支最终AP。
- **风险：** 与MAL、匹配并集、辅助层监督重叠，更多监督不必然更好。

### 2.5 TPRD：可靠KD之后再验证

**机制：** 跨层寻找更可靠teacher预测，修正过置信负样本并保留非目标暗知识。它主要是在既有KD上的增量，不是从零添加一个loss。

- **假设：** teacher对密集tiny目标也有漏检及层间退化，修正监督优于盲目服从。
- **位置：** teacher多层输出与KD criterion。
- **工作量：** 已有KD上150–300行，从零teacher管线不包含在内。
- **实验：** 固定teacher，对比B+KD与B+KD+TPRD；看修正质量、APs、错误类型。
- **风险：** 简化SO-DETR-KD不等同于论文基座，组合是新适配；teacher优势小可能抵不过优化噪声。

### 2.6 SFDNet-CPD：只研究原型监督

**机制：** 类别原型让实例特征紧凑且可分，CPD与频谱ASD有独立消融。原方法依赖GT/RoI、proposal与原型bank；换成DETR query不等于原样复现。

- **假设：** 小目标外观信息不足，类别级统计补充不稳定的单实例特征。
- **位置：** `criterion.py`匹配后的decoder embedding，或neck训练期GT区域特征。
- **工作量：** query适配180–350行；完整RoI路径是否<500未知。
- **实验：** B / 匹配query原型监督；看pedestrian/people、car/van及尺寸分层混淆。
- **风险：** 单原型过度压缩类内差异、少数类bank不稳定；query适配不可冒称原CPD复现。

### 2.7 MDS-DETR：decoder职责改造

**机制：** 前层O2M学习，末层O2O去重；末层置信度排序与严格因果attention，取消进一步cross-attention和定位细化，专注分类去重。完整收益不是一张mask产生的。

- **假设：** 仅当错误分析显示重复预测消耗名额时，训练内去重才有明确靶点。
- **位置：** `decoder.py`末层、matcher、`criterion.py`，检查FDR/FGL/DDF依赖。
- **工作量：** <500行不能保证，先接口设计和导出检查。
- **实验：** B / 分层O2M-O2O / 完整末层抑制；看重复、漏检、AP75/APs及时延。
- **风险：** 少一次定位细化伤害小框；密集真实邻居被误抑制；局部mask改法应明确是适配。

### 2.8 EagleConv：先一个位置测硬件

**机制：** 三高斯形成中心激励、周围抑制和外围上下文响应，部分通道深度滤波，再逐点混合与残差门控。核心是浅层频率选择，而非单纯扩大核。

- **假设：** 在不可逆下采样前保留弱边缘，减少tiny信息丢失。
- **位置：** 优先`hybrid_encoder.py`单个浅层卷积；直接换预训练stem更谨慎。
- **工作量：** 公式级100–220行，官方实现待核。
- **实验：** B / 单点K7；先A770/T4算子和整网时延，后训练。
- **风险：** 原参数下降来自替换而非额外添加；大核后端效率不确定；编号缺口仍存在。

### 2.9 CLoCKDistill：先memory-only

**机制：** 蒸馏包含上下文的encoder memory，位置权重突出目标区域；target-aware queries使teacher/student面向对应目标读取memory，缓解query错位。

- **假设：** 小目标本体信息不足时，上下文与位置联合监督有用。
- **位置：** `hybrid_encoder.py`输出或`decoder.py`输入投影后的memory，配criterion。
- **工作量：** memory-only120–250行；完整query管线是否<500待核。
- **实验：** B / 普通memory KD / 位置加权memory KD；看tiny召回、背景误检、梯度。
- **风险：** flatten位置/尺度需对齐，通道投影可能成为瓶颈；仓库占位提高复现风险。

### 2.10 FIG：先成本筛选

**机制：** 信息熵目标学习像素信息图，GT位置和大小定义高斯图，双图增强弱区域。原结构含卷积和转置卷积，不等价于零推理成本的高斯辅助loss。

- **假设：** P3弱小目标得到显式空间监督后可见性改善。
- **位置：** `hybrid_encoder.py`高分辨率输出与辅助criterion。
- **工作量：** 简化位置监督100–200行；完整FIG成本和行数未过筛。
- **实验：** 成本合格后B / 位置图监督 / 完整双图结构。
- **风险：** 简化不等于完整FIG；额外图增加开销；高斯过宽混淆密集相邻目标。

**启动建议：前四项优先做接口验证，后六项补齐条件再投入完整训练。**

---

## Part 3：三套组合与停止条件

所有ΔAP相对于同一NVIDIA环境的B，不直接相对历史A770的28.819。开发时间不包含排队训练。

| 方案 | 模块 | 研究目标（非保证） | 推理目标 | 工作量估算 | 失败风险 |
|---|---|---|---|---|---|
| 保守 | HBS + SO-DETR风格输出KD | +1.0AP，APs改善 | 原推理图，延迟不升且<10ms | 3–5人日，核心补丁争取250–450行 | teacher无tiny优势；误伤漏标；监督抵消 |
| 均衡 | HBS + 输出KD + stride-8细节注入 | +1.5～2.0AP | ≤+5%且<10ms | 5–8人日，合计是否<500行需单独审查 | 纹理误检，预训练/学习率不当 |
| 激进 | stride-8 + 输出KD + MDS式decoder | ≥+2.5AP | ≤+10%且<10ms | 8–12人日，总改动<500行未确认 | 与FDR/DDF冲突，定位退化，相邻目标误抑制 |

首选均衡型研究路线，但按单项、两两、三项递进。激进型只是条件方案，不能声称满足所有硬筛。绝对和相对时延必须同时满足：

```text
L_balanced < 10ms  且 L_balanced <= 1.05 * L_B
L_aggressive < 10ms 且 L_aggressive <= 1.10 * L_B
```

### 停止条件

- teacher在tiny子集没有稳定优势：暂停KD，先选teacher。
- 浅层注入增加时延而不改善APs/召回：移除，不继续堆attention补救。
- 完整SET相对HBS无稳定增量：放弃API训练开销。
- MDS必须重写大范围FDR/DDF或超过代码预算：不进入首篇主线。
- 组合只在一个seed涨点：不视为稳定贡献。

建议围绕可检验问题组织论文：**受限输入分辨率下，小目标的细节保留、背景可分性和监督可靠性如何共同影响检测？** 期刊竞争力来自机制、泛化、可重复性和实测效率，不来自模块数量。

---

## Part 4：最小实验矩阵、协议与预算

### 4.1 协议冻结

正式B来源：`ecdetseg/outputs/p150u/e3_s42/snapshot/config_merged.yml`完整配置。不能把e2的局部日程误当成e3，不能只抄部分参数。

- 固定640²、类别映射、预训练权重、有效batch。
- 固定Mosaic/Mixup、停止增强时机、LR、EMA、训练长度、checkpoint选择。
- 固定Q=300、全类别top300后处理、evaluator maxDets及ignore规则。
- AP/AP50/AP75/APs统一百分制；同时记录checkpoint epoch。
- 训练期参数、部署参数、FLOPs、显存、训练step time、纯模型/端到端时延分别记录。
- val是开发/选模集；方案锁定后增加独立评估，不能声称val调参收益是独立泛化。

补充诊断：按640缩放后`sqrt(w*h)`的`<8`、`8–16`、`16–32`、`>=32px`分召回；按每图目标数分密度；分解背景、分类、定位、重复和漏检；记录小框中心误差及真实邻居同时检出率。这些自定义分箱不是标准COCO APs。

历史e1两seed差约0.92AP，因此单次+0.3～0.5AP不构成稳定证据。

### 4.2 第一阶段：短程筛选

符号：H=HBS（不含API）；K=固定teacher的输出KD；S=复用真实stride-8的轻量注入。

| ID | 变体 | seed | 数据集/用途 | 长度 |
|---|---|---:|---|---:|
| P0 | B | 42 | VisDrone train→val，短程对照 | 30 epoch |
| P1 | B+H | 42 | 背景抑制假设 | 30 epoch |
| P2 | B+K | 42 | teacher监督假设 | 30 epoch |
| P3 | B+S | 42 | 浅层细节假设 | 30 epoch |
| P4 | B+H+K+S | 42 | 明显负交互检查 | 30 epoch |

短程必须统一缩放日程，不能一组已停止强增强、另一组仍强增强。提前结束只作为筛选，不能混入完整训练比较。HBS进入候选后，可加HBS+API短程对照；该额外实验不包含在上述5组预算内。

### 4.3 第二阶段：完整组合矩阵

| 变体 | seed42 | seed43 | seed44 | 目的 |
|---|---|---|---|---|
| B | 是 | 是 | 是 | 均值与波动 |
| B+H | 是 | — | — | 单模块 |
| B+K | 是 | — | — | 单模块 |
| B+S | 是 | — | — | 单模块 |
| B+H+K | 是 | — | — | 两两交互 |
| B+H+S | 是 | — | — | 两两交互 |
| B+K+S | 是 | — | — | 两两交互 |
| B+H+K+S | 是 | 是 | 是 | 最终组合重复性 |

共12次150 epoch训练，VisDrone train→val。最低版本支持最终组合重复性，但单项/交互统计仍弱；若交互是主结论，其他六变体补到三seed，总24次。

### 4.4 Teacher、泛化及部署项目

| 项目 | 最低要求 |
|---|---|
| Teacher | 确认可用权重；没有则另训练一次，成本不能隐藏 |
| Teacher诊断 | 同协议比较APs、极小召回、密集场景，不只看总体AP |
| 泛化 | 最终B/最终组合至少增加UAVDT或AI-TOD-v2对照；先锁定类别协议 |
| A770/T4 | 各自同后端、精度、B1、640²、warm-up和同步规则 |
| 延迟 | 分纯模型/端到端，至少中位数与P95；并记录具体包含哪些阶段 |
| 导出一致性 | 检查导出前后AP，训练分支确实移除 |

跨数据集训练时间依赖图数、分辨率和日程，未计入下面示例总预算。

### 4.5 GPU小时计算

规划假设：6471训练图，有效batch=10，每epoch约648 step（实际以sampler/drop_last为准），150 epoch。

```text
T_B = 150 * 648 * t_step / 3600 + T_val/save
若验证与存盘暂按15%预算：T_B ≈ 31.05 * t_step（GPU小时）
```

`t_step`必须在目标租赁GPU上实测。下面仅用0.25～0.60秒/step作预算示例，**不是A770、T4或任意租赁卡的性能预测**；对应一次B约7.8～18.6GPU小时。

| 变体 | 暂定训练倍率 |
|---|---:|
| B | 1.00× |
| H | 1.15× |
| K | 1.60× |
| S | 1.05× |
| HK | 1.75× |
| HS | 1.20× |
| KS | 1.65× |
| HKS | 1.80× |

这些倍率全部是占位假设，待实测替换；不能把HBS倍率用于完整HBS+API。

| 项目 | B等效量 | 示例GPU小时 |
|---|---:|---:|
| 5次短程筛选 | 1.32×T_B | 10～25 |
| 12次正式矩阵 | 16.8×T_B | 130～313 |
| 1次teacher训练 | 暂按2×T_B | 16～37 |
| teacher诊断、导出、评测 | 单独预留 | 2～4 |
| 小计 | — | 约158～379 |
| 加20%重跑预留 | — | 约190～455 |

有teacher时先做约10～25GPU小时短程决策，不必一次投入全预算。无teacher时先训练/诊断teacher，不承诺整体在1–2GPU-day内完成。多卡GPU小时=卡数×墙钟小时；不能不考虑多卡效率直接套单卡step时间。

### 4.6 运行记录模板

以下空位是待填写记录，不是未完成的实验结论；未测不要填0。

| run_id | 变体 | seed | 代码/差异标识 | 配置/权重标识 | GPU/后端 | epoch/选模规则 | AP | APs | tiny recall | 训练s/step | 峰值显存 | 部署参数 | p50/p95 ms | GPU小时 | 状态 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 待创建 | B | 42 | 待记录 | 待记录 | 待记录 | 待记录 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未运行 |

每次结果需同时保存：配置、日志、best/last选择规则、实际epoch、环境、teacher版本、GT分箱脚本版本、导出参数以及时延测试包含的处理阶段。

---

## Part 5：检索记录与可追溯来源

### 5.1 上轮报告登记的搜索式

检索日期记录为2026-11-10。下表归档上一轮报告登记的实际搜索式；**本次落盘没有重新执行搜索，也没有把新建议写成已执行查询**。

| 方向 | 查询 |
|---|---|
| UAV | `"small object detection" "UAV" "2026"` |
| ViT | `"token pruning" "detection" "2026"` |
| ViT定向 | `"token pruning" "detection" "2026" "TAP"` |
| KD | `"object detection" "distillation" "2026"` |
| Neck | `"feature pyramid" "small object" "2026"` |
| Assignment | `"DETR" "label assignment" "2026"` |
| Aug | `"copy-paste" "small object detection" "2026"` |
| Quant | `"quantization" "DETR" "2026"` |
| SET | `"SET: Spectral Enhancement for Tiny Object Detection"` |
| FIG | `"Feature Information Driven Position Gaussian"` |
| FIG官方 | `"Feature Information Driven Position Gaussian" site:openaccess.thecvf.com/content/CVPR2025/html` |
| 旧首版 | `"Uncertainty-Aware Gradient Stabilization" "arxiv"` |
| SAFPN | `"SAFPN: spatial adaptive feature pyramid network for small object detection"` |
| MambaFPN | `"MambaFPN: A SSM-based feature pyramid network for object detection"` |
| 数据蒸馏 | `"OD3" "Optimization-free" arxiv` |
| 蒸馏协议 | `"Protocol-consistent" "Sigmoid" "106045"` |
| QCAR数值 | `"Query-calibrated attention" "VisDrone" "AP"` |
| KD数值 | `"Protocol-consistent Sigmoid" "VisDrone" "mAP"` |
| Mamba数值 | `"MambaFPN" "COCO"` |
| SAFPN数值 | `"SAFPN" "66640" "AP"` |
| Eagle编号 | `"EagleConv" arxiv` |

### 5.2 来源与核验用途

- arXiv abstract/submission history：标题、编号、v1日期，区别修订与首版。
- arXiv HTML/PDF：实验表、独立消融、成本和机制。
- [CVF SET正式页面](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.html)。
- [SET官方PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.pdf)：重点Table2的VisDrone、Table5的训练成本。
- [CVF FIG正式页面](https://openaccess.thecvf.com/content/CVPR2025/html/Bian_Feature_Information_Driven_Position_Gaussian_Distribution_Estimation_for_Tiny_Object_CVPR_2025_paper.html)。
- [AAAI LoRA-DETR官方论文](https://ojs.aaai.org/index.php/AAAI/article/download/38293/42255)：正式出版状态。
- [ECCV2026 Accepted Papers](https://eccv.ecva.net/Conferences/2026/AcceptedPapers)：SFDNet/FSDC-DETR接收状态。
- [PMLR EagleConv](https://proceedings.mlr.press/v337/liu26a.html)与[PDF](https://raw.githubusercontent.com/mlresearch/v337/main/assets/liu26a/liu26a.pdf)：Table1精度、Table3时延，不能套到ECDet。
- 出版社页面/Crossref：题名、DOI、出版信息；全文不完整不编造数值。
- GitHub作者仓库/API树：区分源码、占位README与未核实实现。
- 本地源码/snapshot：接入位置、已有机制与评测口径。

上轮没有将Google Scholar或Semantic Scholar登记为实际使用渠道。部分CVF/Elsevier等请求出现403/其他访问限制，已知但未核全表的条目保留待核实状态。

### 5.3 日期与出版状态纠正

- **UGS排除**：[Uncertainty-Aware Gradient Stabilization for Small Object Detection，arXiv:2303.01803](https://arxiv.org/abs/2303.01803)，v1为2023-03-03；虽ICCV2025正式发表及2025修订，仍不符合首次公开窗口。DOI为`10.1109/ICCV51701.2025.00787`。
- **LoRA-DETR**：已查到AAAI2026正式论文，不能继续只标预印本。
- **Dome-DETR**：已核得ACM MM2025及DOI，不能继续写venue未知。
- **FSDC-DETR**：正式接收与源码完整性是两件事，接收不代表可运行实现齐备。
- **首次公开边界**：对部分正式期刊/CVF论文尚未穷尽所有更早作者版本；严格合规需锁定最终参考文献前复查。

---

## Part 6：不确定项与不能提前承诺的结论

### 6.1 文献与复现缺口

| 条目 | 缺口 | 决策影响 |
|---|---|---|
| EagleConv | DOI/arXiv未检得、官方代码待核、更早公开待排查 | 正式论文可信，但不满足所有书目/复现要求 |
| SET、FIG及部分期刊论文 | 已确认2025/2026出版，未穷尽排除更早预印本 | 首次公开严格合规仍需复查 |
| TinyFormer/TPRD/SO-DETR/CLoCKDistill/MDS等 | 未核正式接收 | 保持预印本标记，不按顶会接收加权 |
| CLoCKDistill | 仓库占位 | 提高复现工时，<500行更不确定 |
| FSDC-DETR/I-LW-DETR | 未发现Python实现 | 不写“完整开源可跑” |
| QCAR | 数值表、单项成本、源码未齐 | 不进入Top3或正式矩阵 |
| SAFPN/MambaFPN/Sigmoid-τ KD | 题名DOI明确，完整表格/实现证据不足 | 不填推测增益，不判硬筛通过 |
| FIG | 完整结构可能增开销，简化loss非原法 | 先过成本筛选 |
| TAP/LiSA/LVTP | 小目标检测证据不足 | 不能只凭总AP/FLOPs入选 |

### 6.2 工程与实验缺口

1. 所有候选均未在当前ECDet完成移植/训练；没有本地新模块实测收益。
2. <500行为规划估算，不是已验证补丁，MDS/完整CLoCKDistill/FIG/整数化尤其不确定。
3. A770/T4 <10ms尚无统一协议下的baseline实测；3090/4090或别的模型速度不能替代。
4. 训练专用分支必须实际从部署图移除，不能仅凭理论声称推理零成本；还应区分checkpoint中保留的训练参数和实际部署参数。
5. teacher需确认权重来源、tiny优势、训练成本与对齐方式。
6. val用于开发/选模，最终独立评估必须明确。
7. 新年份不等于创新；浅层信息已有TinyFormer/ReDHF，相似定位能力已有FGL/DDF。

---

## 附录 A：已clone的SET如何作为首轮基准

### A.1 本次直接读取的实现事实

文件：`competition/SET/mmdet/models/detectors/fcos_set.py`，commit见0.1。

- `DNResBlock`使用两个卷积、中间ReLU和残差；reduction=4，核大小依据feature stride选择。HBS在该实现中不是简单固定均值滤波。
- `FCOS_set.forward_train()`先执行原始FCOS head及loss。
- GT像素框先生成前景binary mask，再nearest插值到每层特征大小。
- 背景送入denoiser，输出再次乘背景mask；前景保留原feature。
- API分别求`loss_cls`、`loss_bbox`、`loss_centerness`对feature的梯度并归一化，按配置系数合成扰动。
- `x_set`再次送入head计算辅助loss，并按`scale`等权重加入原loss字典。
- 当前该文件的forward路径同时计算API梯度；HBS-only实验不能仅把扰动系数设为0后仍保留所有梯度计算，并据此声称测得HBS最低训练成本。
- 官方实现面向MMDetection FCOS，ECDet没有同名centerness语义，不能逐字粘贴loss接口。

### A.2 接入前检查表

- [ ] 固定SET commit及本地baseline snapshot，记录工作区diff。
- [ ] 选择P3-only HBS，保留P4/P5与原decoder输入契约。
- [ ] 明确GT坐标在增强后输入空间的表示；归一化cxcywh不能直接用于像素mask。
- [ ] 验证极小GT投影后仍覆盖至少合理的特征位置；如调整mask生成，明确标为适配并单独记录。
- [ ] 记录训练增强分支参数与部署参数，检查是否满足成本约束。
- [ ] HBS-only路径跳过API专用autograd计算，API开关独立。
- [ ] 定义两次decoder forward的DN、匹配、随机性和loss权重；避免辅助loss被重复加权。
- [ ] API若启用，使用ECDet实际分类/定位loss，不虚构centerness替代项；检查梯度非None、AMP与图保留。
- [ ] 确认criterion现有清缓存行为；仅针对实际问题改动。
- [ ] 检查新参数优化器分组、梯度、学习率和EMA。
- [ ] eval/deploy回到原预测路径，检查导出图及参数统计。
- [ ] B/H/完整SET同协议测训练步时、显存、AP/APs、tiny recall与时延。

### A.3 第一批SET专属对照

| ID | 内容 | 要回答的问题 |
|---|---|---|
| SET-B | 同环境原始B | 对照是否可复现 |
| SET-H | P3-only HBS适配 | 背景处理是否有效 |
| SET-HA | 同位置HBS+API适配 | API是否有超出HBS的收益 |

以上是未来执行计划，未创建配置、改模型、启动训练或生成实测结果。与Part4相比，SET-HA为额外短程实验，需另外计费/计时。

## 五句话决策摘要

1. 第一优先试SET-HBS，先检验背景抑制能否稳定改善极小目标召回，再决定是否支付API额外训练成本。
2. 第二优先试SO-DETR风格输出蒸馏，前提是teacher在tiny子集确实更强，并正确解决teacher/student query对齐。
3. 第三优先试复用真实stride-8的轻量细节注入，将其作为TinyFormer启发的适配对照，而非宣称浅层支路本身全新。
4. 第一篇最不值得先碰的是整套动态token/query重构，当前收益证据不足以抵消适配、稀疏恢复与实际延迟的不确定性。
5. 另一条不宜提前重投入的路线是整网低比特整数化，它更适合第二篇部署研究，当前应先扎实完成同环境基线、三seed精度和小目标错误分析。
