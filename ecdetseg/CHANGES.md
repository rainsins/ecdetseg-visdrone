# CHANGES — P30u 协议重构

基准：旧配置 = `configs/ecdet/ecdet_s.yml`（合并 `dataset/custom.yml` + `ecdet.yml` 后的最终配置）。
新配置 = `configs/ecdet/ecdet_s_p30u.yml`（合并 `visdrone.yml` + `ecdet.yml` + `struct_s.yml` + `protocol/res640.yml` + `protocol/p30u.yml`）。
可复现命令：`python scripts/dump_config.py -c <config> -u ...`（旧 dump 见审阅时 `ecdet_s.yml`；新 dump 见 `configs_dump/E*.yml`）。

> 硬约束：正在跑的 O365 任务结束前**不改** `ecdet.yml` / `ecdet_s.yml` / `custom.yml` / `train_*.sh`。
> 本轮全部为**新增文件**。旧值列仅用于对照，不代表已改动这些文件。

## 1. 协议参数（protocol/p30u.yml 为准，旧值来自 ecdet_s.yml 合并结果）

| 参数 | 旧值 | 新值 (P30u) | 原因 |
|---|---|---|---|
| `epochs` | 30 | 30 | 不变 |
| `flat_epoch` | 28 | **15** | `flat_epoch=28` 时 `flat_iter == total_iter - no_aug_iter`，余弦区间为空 → 学习率台阶式直落 `min_lr`（audit Q2）。15 留出真实余弦退火。 |
| `no_aug_epoch` | 2 | 2 | **死键**（调度器用 `epochs-stop_epoch`，audit Q4）。保留仅为人类可读，与 `stop_epoch` 自洽。 |
| `stop_epoch` | 28 | 28 | 真正的 no-aug 边界；决定调度器 `no_aug_epochs=2`。 |
| `warmup_iter` | 2000 | 2000 | 不变 |
| `lr_gamma` | 0.5 | 0.5 | 不变 |
| **`Mosaic`（ops 首位）** | 存在 | **删除** | 默认 P30u **关闭 Mosaic**。 |
| `mosaic_prob` | 0.75 | **0** | 关闭。**必须显式置 0**（不能只删键）：冻结的 `ecdet.yml` 仍带 `0.5`，而 `container.py:83-96` 用它给 `RandomZoomOut/RandomIoUCrop` 做门控——见下“静默失效”。 |
| `mosaic_epoch` | 26 | **0** | 随 Mosaic 关闭置 0。 |
| `remove_ops` | `['Mosaic',RPD,RZO,RIoC]` | `['RPD','RZO','RIoC']` | 去掉已删除的 `'Mosaic'`（即使保留也不会报错，仅是死条目）。 |
| `RandomPhotometricDistort` / `RandomZoomOut` / `RandomIoUCrop` / `RandomHorizontalFlip` | 按原样 | **不变** | 任务要求保留。 |
| `mixup_prob` | 0 | 0 | 不变（ecdet_s 已是 0）。 |
| `mixup_epoch` | **36**（> epochs，永不生效） | **0**（合法） | 原值无意义；改为合法值 0。 |
| `train total_batch_size` | 8 | 8 | 不变 |
| `val total_batch_size` | 64 | **16** | 任务指定；降低 val 峰值显存。 |
| `num_workers` | 4（ecdet.yml 覆盖 custom 的 8） | 4 | 协议内**只出现一次**。 |
| `ema.decay` | 0.9999 | **0.9995** | 任务指定。 |
| `sync_bn` | True | **False** | 单卡 XPU 无 DDP，sync_bn 无意义；显式关闭。 |
| `use_amp` | True | True | 不变（XPU 上即 bf16，audit Q9）。 |
| `eval_maxDets` | （无） | **300** | 新增；配合上一轮 maxDets 修复（`num_queries=300`）。 |
| `output_dir` | `outputs/ecdet_s_coco_pre_weights`（写死） | **脚本注入** `outputs/p30u/<init>_s<seed>` | 从配置移除，避免实验间互相覆盖。 |
| `optimizer` | ecdet_s.yml 与 ecdet.yml **重复两处** | protocol 内**单处** | 消除重复；数值不变（lr 5e-4 / backbone 2.5e-5 / wd 1e-4）。 |
| 数据路径 | `custom.yml` 写死 `/home/rainsin/...` | `visdrone.yml` 占位 + `run_exp.sh` 用 `-u` 注入 `$VISDRONE_ROOT` | 加载器不支持 `${ENV}` 展开（audit 待确认 7）。 |
| `ViTAdapter.weights_path` | `ecvits/ecvitt.pth` | 同（`struct_s.yml`，可 `-u` 覆盖） | 结构字段；E2/E3 由 `-t` 提供 backbone。 |

## 2. 结构/文件变更（新增，不动旧文件）

| 动作 | 文件 | 说明 |
|---|---|---|
| 新增 | `configs/dataset/visdrone.yml` | data-only（路径占位、`num_classes:10`、evaluator）。 |
| 新增 | `configs/ecdet/struct_s.yml` | 仅 S 档结构差异（含 ViTAdapter S），不含 protocol/output_dir/optimizer。 |
| 新增 | `configs/ecdet/protocol/p30u.yml` | 训练协议唯一来源，**最后 include**；**Mosaic 关闭**（ops 无 Mosaic，`mosaic_prob: 0`）。 |
| 新增 | `configs/ecdet/protocol/aug_mosaic025.yml` | Mosaic@0.25 对照**覆盖文件**（ops 含 Mosaic，`mosaic_prob: 0.25`/`mosaic_epoch: 26`）。 |
| 新增 | `configs/ecdet/ecdet_s_p30u_mosaic025.yml` | 对照实验入口（在 p30u 之后再 include `aug_mosaic025.yml`）。 |
| 新增 | `configs/ecdet/protocol/res640.yml`、`res960.yml` | 分辨率；只设 `eval_spatial_size`（Mosaic/Resize 由 reset_cfg 派生，audit Q10）。 |
| 新增 | `configs/ecdet/ecdet_s_p30u.yml` | 实验入口（include 顺序：data<base<S<res<protocol）。 |
| 新增 | `scripts/dump_config.py` | 只读配置 dump（torch-free，规避 `load_config` 可变默认参数坑，audit Q1）。 |
| 新增 | `scripts/run_exp.sh`、`scripts/fake_train.sh`、`scripts/test_run_exp.sh` | 守护脚本 + 静态测试。 |
| 保留 | `ecdet.yml` / `ecdet_s.yml` / `custom.yml` / `train_o365.sh` / `train_ecdet.sh` | **未改动**（运行中任务依赖）。后续可折叠为 data/struct/protocol 三层（待确认 8）。 |

## 3. E1/E2/E3 唯一差异

三份 dump `configs_dump/E{1,2,3}.yml` 经 `diff` 验证**仅**两处不同：
`output_dir` 与 `tuning`（e1=None，e2=ecdet_s.pth，e3=ecdet_s_o3652coco.pth）。
运行时由 `run_exp.sh` 以 `-t`/`--output-dir` 传入（`-t` 还会置 `skip_load_backbone=True`，audit Q5）。

## 4. Mosaic 开关：为什么必须显式 `mosaic_prob: 0`

`container.py:79-101`（`stop_epoch_forward`）先独立计算
```python
if self.mosaic_prob > 0 and self.mosaic_epoch > self.cur_epoch:
    with_mosaic = random.random() <= self.mosaic_prob
else:
    with_mosaic = False
```
再遍历 `self.transforms`。关键点：**`with_mosaic` 与 `transforms` 里是否真有 Mosaic 无关**；
`container.py:96` 会在 `with_mosaic` 为真时跳过 `RandomZoomOut` / `RandomIoUCrop`（“Mosaic 与 ZoomOut/IoUCrop 不共存”）。

因此：从 `ops` 删除 Mosaic 后，若 `mosaic_prob` 仍为 `ecdet.yml` 里的 `0.5`（受硬约束不能改该文件），
`RandomZoomOut/RandomIoUCrop` 会有 ~50% 概率被**静默跳过**——不是报错，而是行为悄悄改变。
故默认协议显式 `mosaic_prob: 0`（`with_mosaic` 恒 False，ZoomOut/IoUCrop 始终生效）以消除该静默失效。

已核对的其它处理点（均安全）：
- `remove_ops` 里保留 `'Mosaic'` 只是死条目（`container.py:90` 只对**存在的** transform 比对，不会报错）；仍已清理。
- `remove_ops` 必须是非 None（否则 `container.py:90` 的 `in None` 会 `TypeError`）；`ecdet.yml` 与 `p30u.yml` 均提供，安全。
- `policy: stop_epoch` 名称未变，`get_forward`（`container.py:65-71`）照常解析。
- `reset_cfg`（`yaml_config.py:193-199`）对 `Mosaic` 项做 `output_size` 注入；无 Mosaic 时该分支不触发，`Resize` 分支照常，安全。

## 5. Mosaic 对照实验

默认入口 `ecdet_s_p30u.yml` = **Mosaic 关闭**。对照入口
`ecdet_s_p30u_mosaic025.yml`（末尾 include `protocol/aug_mosaic025.yml`）= **Mosaic@0.25**。
两者除增强块外完全一致（`configs_dump/E1.yml` vs `configs_dump/E1_mosaic025.yml`，`diff` 已验证仅增强块 + output_dir 不同）。
用法：`-c configs/ecdet/ecdet_s_p30u_mosaic025.yml`。

