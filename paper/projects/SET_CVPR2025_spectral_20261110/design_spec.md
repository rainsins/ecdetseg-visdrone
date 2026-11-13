<!-- ppt-master-schema: design-spec/v1 -->
# SET Spectral Enhancement - Design Spec

## I. Project Information
| Item | Value |
| --- | --- |
| Project Name | SET_CVPR2025_spectral_20261110 |
| Canvas Format | ppt169, 1280 × 720 |
| Page Count | 18 |
| Primary Language | zh-CN |
| Target Audience | 具备目标检测基础的导师与课题组同学 |
| Communication Intent | 解释频域诊断、HBS/API机制、实验和训练成本，评估EdgeCrafter研究价值 |
| Desired Audience Outcome | 复述机制、评估证据、区分论文事实与待验证假设 |
| Core Message / Ask / Action | 压背景、抬前景的训练期增强；推理不加模块，训练成本明显增加 |
| Delivery Context | 中文组会，约20分钟，16:9投影 |
| Artifact Afterlife | 原论文证据图片，文字与公式可编辑，中文备注支持复习 |
| Reading Mode | balanced |
| Content Strategy | 用户授权依据PDF自主优化，不必严格遵循原Markdown |
| Design Style | 自由设计、白底证据型学术报告 |
| AI Image Acquisition Path | not applicable |
| Generation Mode | continuous |
| Spec Refinement | disabled |
| Speaker Notes | enabled — 用户讲稿基础上修订 |
| Custom Animations | disabled — delegated production decision |
| Narration Audio | disabled — delegated production decision |
| Created Date | 2026-11-11 |

用户最新授权取代旧14页等待确认状态；不伪造UI收据。比较白底证据型、暖白讲义型、暗底海报型，选择白底证据型，适配论文原图和会后复习。汇报人及汇报日期留作用户填写。

## II. Canvas Specification
| Property | Value |
| --- | --- |
| Format | PPT 16:9 |
| Dimensions | 1280 × 720 |
| viewBox | `0 0 1280 720` |
| Margins | 56 px |
| Content Area | 56,40 to 1224,650 |

## III. Visual Theme
### Theme Style
- **Mode**: custom
- **Mode Behavior**: 问题→诊断→设计→机制证据→实验与代价→研究设想，每页一个判断。
- **Visual style**: custom
- **Visual Style Behavior**: 白底深蓝标题，青绿HBS/增益，橙色API/成本；完整论文图为主，原生说明并置；封面和结尾深蓝大色面。
- **Theme**: 从频谱诊断到空间特征增强
- **Tone**: 准确、清晰、审慎
### Color Scheme
| Role | HEX | Purpose |
| --- | --- | --- |
| Background | #FFFFFF | 证据底色 |
| Secondary background | #F1F5F9 | 解释区域 |
| Primary | #142D4E | 标题 |
| Accent | #087E8B | HBS和收益 |
| Secondary accent | #BC5A24 | API和成本 |
| Body text | #243447 | 正文 |
| Secondary text | #58677A | 来源 |
| Divider | #D5DFE8 | 分隔 |

## IV. Typography System
### Font Plan
| Role | Character (Reference) | Primary | English if non-English | Fallback tail |
| --- | --- | --- | --- | --- |
| Title | 加粗清晰 | Microsoft YaHei | Arial | sans-serif |
| Body | 中性易读 | Microsoft YaHei | Arial | sans-serif |
- **Title stack**: Microsoft YaHei, Arial
- **Body stack**: Microsoft YaHei, Arial
### Font Size Hierarchy
| Purpose | Anchor Size (px) |
| --- | ---: |
| Body | 24 |
| Title | 40 |
| Subtitle | 30 |
| Annotation | 20 |
| Footnote | 16 |
| Display | 64 |
| Lead | 28 |

## V. Layout Principles
### Deck-wide Direction
- **Hierarchy direction**: 判断句标题→图/公式→解释→限定和来源
- **Composition tendency**: 证据图主导，非对称图文分区，横向大图配底部结论
- **Cross-page continuity**: HBS青绿、API橙色，短章节标签与一致页码
- **Spacing posture**: 方法页适中，证据页密集，联合目标与结尾留白
- **Spacing anchors**: page margin 56; block gap 24; column gutter 40; corner radius 12; body leading 38

## VI. Icon Usage Specification
- **Primary bundled library**: none
| Icon Path | Suitable Scenarios |
| --- | --- |

## VIII. Image Resource List
| Filename | Dimensions | Ratio | Purpose | Type | Image pattern | Crop Policy | Acquire Via | Status | Reference | text_policy | page_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fig1.png | 723 × 267 | 2.708 | 遮挡归因 | Paper figure | 完整大图配结论 | no-crop | user | Existing | PDF p1 Fig.1 | embedded | local |
| fig2.png | 720 × 372 | 1.935 | 频谱可视化 | Paper figure | 图像和动机相邻 | no-crop | user | Existing | PDF p2 Fig.2 | embedded | local |
| fig3.png | 1494 × 831 | 1.798 | 框架 | Paper figure | 横向主视觉 | no-crop | user | Existing | PDF p3 Fig.3 | embedded | local |
| hbs_detail.png | 753 × 480 | 1.569 | 解耦平滑 | Paper figure | 明示为框架局部 | no-crop | user | Existing | PDF p3 Fig.3 HBS局部 | embedded | local |
| fig4.png | 633 × 831 | 0.762 | PCA对照 | Paper figure | 纵向完整图 | no-crop | user | Existing | PDF p4 Fig.4 | embedded | local |
| fig5.png | 1494 × 531 | 2.814 | 显著性 | Paper figure | 完整横向对照 | no-crop | user | Existing | PDF p5 Fig.5 | embedded | local |
| fig6.png | 720 × 357 | 2.017 | 超参数 | Paper figure | 曲线大图与解释 | no-crop | user | Existing | PDF p8 Fig.6 | embedded | local |
| fig7.png | 714 × 336 | 2.125 | 频谱统计 | Paper figure | 统计与PCA并置 | no-crop | user | Existing | PDF p8 Fig.7 | embedded | local |
| table1.png | 1494 × 1014 | 1.473 | AI-TOD | Paper table | 原表配重点读数 | no-crop | user | Existing | PDF p6 Table 1 | embedded | local |
| table2.png | 720 × 351 | 2.051 | VisDrone | Paper table | 原表与摘要 | no-crop | user | Existing | PDF p7 Table 2 | embedded | local |
| table3.png | 720 × 315 | 2.286 | DOTA | Paper table | 与VisDrone并置 | no-crop | user | Existing | PDF p7 Table 3 | embedded | local |
| table4.png | 741 × 120 | 6.175 | COCO | Paper table | 宽表和限定 | no-crop | user | Existing | PDF p7 Table 4 | embedded | local |
| table5.png | 1497 × 384 | 3.898 | 消融代价 | Paper table | 大横表和成本解释 | no-crop | user | Existing | PDF p8 Table 5 | embedded | local |

## IX. Content Outline
### Part 1: 问题与诊断
#### Slide 01 - 从频谱诊断到特征增强
- **Audience move**: 不熟悉SET→记住压背景与抬前景
- **Relationships**: HBS与API分别针对背景和关键区域
- **Title**: SET：面向微小目标检测的频谱增强
- **Core message**: 压背景、抬前景，增强发生在训练期
- **Content**: Spectral Enhancement for Tiny Object Detection；CVPR 2025；Sun et al.；汇报人XXX，202X年X月X日；推理不增加模块。
- **Cover impact**: 压背景 × 抬前景；深蓝主题面双色重点
#### Slide 02 - 微小目标难在弱信号与强背景
- **Audience move**: 知道小目标难→了解两类瓶颈
- **Relationships**: 分辨率不足与背景杂波共同削弱判别性
- **Title**: 微小目标难在弱信号与强背景
- **Core message**: 下采样损失信息，背景又掩盖微弱目标
- **Content**: tiny≤16×16px；AI-TOD平均12.8px，COCO99.5px；DINO中等37.6、极微小9.9AP；FCOS极微小2.5AP。
- **Images**: fig2.png
#### Slide 03 - 先遮掉频段，再观察AP变化
- **Audience move**: 直觉归因→理解干预式分析
- **Relationships**: FPN→FFT→幅度遮挡→保留相位IFFT→评估
- **Title**: 先做诊断：遮掉频段，观察AP变化
- **Core message**: FFT是归因工具，不是SET推理新增模块
- **Content**: AI-TOD上1×FCOS；中频能量40%—60%，高频超过90%；APvt/APt/APs对应2—8/8—16/16—32px。
#### Slide 04 - 只去背景高频，三个尺度同时受益
- **Audience move**: 理解操作→读懂相对增益
- **Relationships**: 全特征与仅背景高频遮挡形成对照
- **Title**: 只去背景高频，三个尺度同时受益
- **Core message**: 前景和背景应异构处理
- **Content**: 全图高频1.14/1.08/0.96倍；背景高频1.15/1.08/1.04倍；中频遮挡APvt=0.54倍；相对AP，不是绝对AP点，结论限定此实验。
- **Images**: fig1.png
### Part 2: 方法与机制
#### Slide 05 - SET把诊断转化为训练期增强
- **Audience move**: 看见问题→理解两个训练模块
- **Relationships**: FPN经过HBS/API服务辅助损失，原检测分支保留
- **Title**: SET把诊断转化为训练期特征增强
- **Core message**: HBS压背景，API增强关键区域
- **Content**: GT掩码解耦；特征扰动；推理移除附加训练路径。
- **Images**: fig3.png
#### Slide 06 - HBS保护前景，只平滑背景
- **Audience move**: 模块名称→理解前景保护
- **Relationships**: 框掩码分成两路，前景与平滑背景相加
- **Title**: HBS：保护前景，只平滑背景
- **Core message**: GT框掩码是异构处理开关
- **Content**: M是框内二值掩码，非像素分割；前景原样保留，背景进入φ；各FPN层对应处理。
- **Mathematical content**: P_i^E=P_i\odot M+\phi_i(P_i\odot\neg M,r)
- **Images**: hbs_detail.png
#### Slide 07 - 压缩扩张瓶颈让背景平滑
- **Audience move**: 分路→理解压缩与核大小
- **Relationships**: 通道压缩、ReLU、扩张和残差组成平滑算子
- **Title**: 压缩—扩张瓶颈，让背景变得平滑
- **Core message**: r控制压缩，核匹配层级
- **Content**: r=4；残差保留背景上下文；低层小核，高层大核；非严格理想低通滤波器。
- **Mathematical content**: \phi_i(P_i^{bg},r)=\sigma(w_i^e\otimes\sigma(w_i^r\otimes P_i^{bg}))+P_i^{bg}
- **Mathematical content**: K_i=2\left\lfloor\frac{\log_2(S_i)}{2}\right\rfloor+1
#### Slide 08 - 可视化与频谱统计指向同一变化
- **Audience move**: 理解设计→检视机制证据
- **Relationships**: PCA对照和频谱统计共同支持平滑解释
- **Title**: 可视化与频谱统计指向同一个变化
- **Core message**: 低频增强，中高频衰减
- **Content**: Fig.4原图/FCOS/HBS；Fig.7在VisDrone训练集随机1000图；低频平均幅值0.3787→0.5205，中频0.5340→0.4249，高频0.6265→0.4518。
- **Images**: fig4.png, fig7.png
#### Slide 09 - API沿损失敏感方向注入扰动
- **Audience move**: 理解平滑→理解有约束特征扰动
- **Relationships**: 分支梯度归一化后加权融合
- **Title**: API：沿损失敏感方向注入受约束扰动
- **Core message**: 一阶近似，分支语义共同参与
- **Content**: L2约束ρ；分类/回归/center-ness分支加权；ρ=1；不是随机噪声，也不是像素攻击。
- **Mathematical content**: \epsilon_{i,cls}^{*}\approx\rho\frac{\nabla_{P_i}L_{cls}(P_i)}{\|\nabla_{P_i}L_{cls}(P_i)\|_2}
- **Mathematical content**: \epsilon_i^{adv}=\sum_{m=1}^{M}\lambda_m\epsilon_{i,m}^{*}
#### Slide 10 - API增强关键区域
- **Audience move**: 理解扰动→检视显著性及限定
- **Relationships**: 同场景FCOS和API显著性对照
- **Title**: API增强关键区域，tiny目标更显著
- **Core message**: 53.99→69.93仅为一个可视化示例
- **Content**: P3梯度L2范数显著性；示例约+29.5%；非全测试集统计，非所有对象保证提升。
- **Images**: fig5.png
#### Slide 11 - 辅助损失写入表征
- **Audience move**: 两模块→理解联合优化与推理边界
- **Relationships**: 原损失加各层增强特征辅助损失
- **Title**: 增强信号通过辅助损失写入表征
- **Core message**: 推理只保留检测器
- **Content**: λ=1；辅助特征P增强+ε；过强辅助约束不利检测；推理无需GT掩码或HBS/API路径。
- **Mathematical content**: L=L_{detection}+\lambda\sum_{i=1}^{N}L_i^{aux}(P_i^E+\epsilon_i^{adv})
### Part 3: 实验与判断
#### Slide 12 - 实验条件
- **Audience move**: 方法直觉→了解对比条件
- **Relationships**: 四数据集提供不同场景验证，CNN与DINO设置不同
- **Title**: 先看实验条件，再判断增益有多稳
- **Core message**: 多基准多检测器，限定论文设置
- **Content**: CNN ResNet-50，SGD12epochs，batch2，lr0.005；DINO5scale36epochs/Adam；RTX3090/MMDetection；AI-TOD800×800；AI-TOD/VisDrone/DOTA-v2/COCO；星号代表P2—P6。
#### Slide 13 - AI-TOD主结果
- **Audience move**: 协议→比较强弱基线收益
- **Relationships**: 每个检测器和自己的基线比较
- **Title**: SET不仅提升FCOS，也能叠加强基线
- **Core message**: RFLA+3.2，DINO+3.4 AP
- **Content**: FCOS12.0→14.2；FCOS*15.1→18.0；RFLA21.7→24.9；DINO23.2→26.6；RFLA+SR-TOD21.8，DINO+DNTR26.2。
- **Images**: table1.png
#### Slide 14 - 跨数据集验证
- **Audience move**: 单基准→识别泛化证据与疑点
- **Relationships**: 无人机、航拍和通用场景对照
- **Title**: 跨数据集有效，但结论要保留边界
- **Core message**: FCOS总体AP在三个数据集分别+2.0/+2.9/+1.0
- **Content**: VisDrone19.9→21.9，RFLA27.2→28.5；DOTA31.8→34.7；COCO36.4→37.4；COCO APt表值19.6→23.3与正文/上标+1.7矛盾，明确披露。
- **Images**: table2.png, table3.png, table4.png
#### Slide 15 - 消融与训练成本
- **Audience move**: 看到收益→判断贡献与代价
- **Relationships**: 各配置和FCOS基线比较；推理与训练代价区分
- **Title**: 联合效果最好，训练成本不能忽略
- **Core message**: SET14.2 AP；0.42/0.17≈2.47倍耗时
- **Content**: 基线12.0/全图12.7/HBS13.9/API13.2/SET14.2；1.9+1.2≠2.2，不可加；基线/HBS/API/SET耗时0.17/0.20/0.39/0.42 s/batch；全图平滑正文与表不一致，以表为准；参数/GFLOPs不等于峰值显存。
- **Images**: table5.png
#### Slide 16 - 超参数
- **Audience move**: 固定配置→理解搜索条件
- **Relationships**: 压缩、扰动、辅助权重分别影响性能
- **Title**: 适度增强优于过强约束
- **Core message**: r=4、ρ=1、λ=1是论文选择，不是普适最优
- **Content**: Fig.6 r/ρ搜索使用FCOS训练3epochs；r16低于基线；ρ5更差；ρ0仍有HBS；λ1三个框架较好。
- **Images**: fig6.png
#### Slide 17 - EdgeCrafter研究启发
- **Audience move**: 理解论文→形成可证伪实验
- **Relationships**: 固定基线→频域诊断→模块消融→误差与成本分析
- **Title**: 对EdgeCrafter：先验证诊断，再迁移模块
- **Core message**: 当前优先精度研究，部署轻量化后续独立进行
- **Content**: 未验证研究设想；VisDrone固定切分与尺度；频段遮挡；HBS/API/联合/全图平滑；分尺度AP、随机种子、训练时间、峰值显存；框掩码在拥挤背景的影响；可租更强GPU。
#### Slide 18 - 总结
- **Audience move**: 细节→记住方法论和边界
- **Relationships**: 诊断、设计、证据与成本共同形成判断
- **Title**: 最值得借鉴的是：先诊断，再设计
- **Core message**: 异构训练增强和可检验的机制链
- **Content**: 频域遮挡定位问题；HBS/API针对性设计；四基准验证；GT框依赖和训练增耗；EdgeCrafter迁移待验证。
- **Closing impact**: 先诊断，再设计；机制和成本一起看

## X. Speaker Notes Requirements
- **Generation**: enabled
- **Filename**: match each SVG filename under notes/
- **Content**: 依据最终页面和PDF中文讲述；来源标图表号与PDF页码；修正相对增益、成本、不可加和COCO矛盾，不把讲稿堆在页面。
- **Total duration**: 约20分钟
- **Notes style**: 自然中文组会讲述，逐图解释和转场
- **Presentation purpose**: 理解机制，评估证据，形成可验证的迁移思路
