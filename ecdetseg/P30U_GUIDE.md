# P30u 训练体系总说明（EdgeCrafter / ecdetseg）

> 目的：把「我帮你生成的配置 / 脚本 / dump / 文档」一次讲清，让你对每个文件干什么、
> 怎么用、有哪些坑心里有数。所有路径相对 `ecdetseg/`。
>
> 生成时间点：仓库 HEAD = `c0233ae init`（你已提交）。仍在跑的实验：`outputs/p30u/e2_s42`
> （pm2 启动）。**单张 Arc A770 16G：任何 XPU 操作前先 `pgrep -af train.py` 必须为空。**

---

## 1. 一句话总览

三个初始化实验 **E1/E2/E3** 共用同一套 **P30u 协议**，除「初始化权重」外完全一致；
配置分四层（数据 / 基础结构 / S 结构 / 协议），协议集中在 `protocol/p30u.yml`；
用 `scripts/run_exp.sh` 守护运行（快照、恢复、重试、信号、终评），可用 pm2 托管。

| 实验 | 初始化 |
|---|---|
| **E1** | 不加 `-t`（仅骨干 `ecvits/ecvitt.pth`） |
| **E2** | `-t ecdet_s.pth`（COCO 80 类） |
| **E3** | `-t ecdet_s_o3652coco.pth`（O365→COCO 80 类） |

---

## 2. 文件清单（我新增的）

### 2.1 配置（`configs/`）
| 文件 | 作用 |
|---|---|
| `configs/dataset/visdrone.yml` | **data-only**：VisDrone 路径（占位，脚本注入）、`num_classes: 10`、evaluator |
| `configs/ecdet/struct_s.yml` | **仅 S 档结构**（ViTAdapter/HybridEncoder/ECTransformer 的 S 参数），无协议、无 output_dir/optimizer |
| `configs/ecdet/protocol/p30u.yml` | **训练协议唯一来源**（最后 include，覆盖一切）；含 no-Mosaic 的 ops |
| `configs/ecdet/protocol/res640.yml` | 分辨率 640（只设 `eval_spatial_size`；Mosaic/Resize 自动派生） |
| `configs/ecdet/protocol/res960.yml` | 分辨率 960（Mosaic 自动 480） |
| `configs/ecdet/protocol/aug_mosaic025.yml` | **Mosaic 消融覆盖文件**（恢复 Mosaic@0.25） |
| `configs/ecdet/ecdet_s_p30u.yml` | **默认入口**（Mosaic 关闭） |
| `configs/ecdet/ecdet_s_p30u_mosaic025.yml` | **对照入口**（Mosaic@0.25） |
| `configs/ecdet/protocol/p74u.yml` | P74u 协议（官方 74-epoch，含 Mosaic op） |
| `configs/ecdet/protocol/p74u_mosaic0.yml` | P74u 消融（仅 4 个 aug 键为 0） |
| `configs/ecdet/ecdet_s_p74u.yml` / `ecdet_s_p74u_mosaic0.yml` | P74u 入口（640） |
| `configs/ecdet/ecdet_s_p74u_960.yml` / `ecdet_s_p74u_mosaic0_960.yml` | P74u 入口（960） |

include 顺序（后覆盖前）：`visdrone.yml < ecdet.yml(冻结) < struct_s.yml < res640.yml < p30u.yml`。

### 2.2 配置 dump（`configs_dump/`）
`E1.yml` / `E2.yml` / `E3.yml`（默认，Mosaic 关）、`E1_mosaic025.yml`（对照）。
三份 E* 经 `diff` 验证**仅** `output_dir` 与 `tuning` 不同。

### 2.3 脚本（`scripts/`）
| 文件 | 作用 | 用不用 XPU |
|---|---|---|
| `run_exp.sh` | **主守护脚本**：`run_exp.sh <e1\|e2\|e3> <seed> [key=val ...]`。快照/恢复/重试/信号/终评 | 是（调 train.py） |
| `run_all_p30u.sh` | 串行批跑 `seed × init`（pm2 用），信号级联 | 间接 |
| `ecosystem.config.js` | pm2 配置（`p30u-all`，`autorestart:false`，`kill_timeout:60s`） | — |
| `dump_config.py` | **只读**打印合并后配置（torch-free，规避 `load_config` 可变默认参数坑） | 否 |
| `eval_final.py` | 评估单个 ckpt：`--weights ema\|raw\|both`，输出含 `weights_type/epoch` | 是 |
| `verify_p30u_xpu.sh` | 训练结束后的 XPU 冒烟（内置 `pgrep` 守卫，64 图子集） | 是 |
| `make_subset.py` | 从 COCO json 取前 N 张做子集 | 否 |
| `polt.py` | 从某实验目录 `log.txt` 画曲线（cwd 无关，参数传实验目录） | 否 |
| `test_run_exp.sh` | 守护脚本静态测试（假 train/eval，**19 项**） | 否 |
| `fake_train.sh` / `fake_eval.sh` | 测试替身 | 否 |

> 你原有的 `train_o365.sh` / `train_ecdet.sh` 仍在（被移到了 `ecdetseg/scripts/script/`），**我没动过**。

### 2.4 文档
| 文件 | 内容 |
|---|---|
| `audit.md` | 阶段 0 只读审计（11 问，含「结论/待确认」） |
| `CHANGES.md` | 旧配置→P30u 的逐项差异与原因（含 Mosaic 静默失效分析） |
| `README_exp.md` | 如何跑、如何新增实验、注意项 |
| **`P30U_GUIDE.md`** | 本文件（总索引） |

---

## 3. 怎么跑（最短路径）

```bash
export VISDRONE_ROOT=/home/rainsin/Projects/Career/Paper/dataset/VisDrone   # 必填
cd /home/rainsin/Projects/Career/Paper/EdgeCrafter/ecdetseg
pgrep -af train.py          # 必须为空！

# 单个实验（前台）
bash scripts/run_exp.sh e1 42
# 全部 3 seed × 3 init（串行）
for s in 42 43 44; do for e in e1 e2 e3; do bash scripts/run_exp.sh "$e" "$s"; done; done

# pm2 批跑（推荐）
VISDRONE_ROOT=$VISDRONE_ROOT pm2 start ecosystem.config.js --only p30u-all
pm2 logs p30u-all ; pm2 stop p30u-all ; pm2 delete p30u-all

# Mosaic 对照
P30U_CONFIG=configs/ecdet/ecdet_s_p30u_mosaic025.yml \
P30U_OUT_ROOT=$PWD/outputs/p30u_mosaic VISDRONE_ROOT=$VISDRONE_ROOT \
  pm2 start ecosystem.config.js --only p30u-all
```

额外覆盖用 **裸 `key=val`**（脚本已自动兼容你是否带 `-u`）：
```bash
bash scripts/run_exp.sh e1 7 epochs=4 stop_epoch=2 flat_epoch=1     # 冒烟
P30U_RES=960 bash scripts/run_exp.sh e1 42                          # 换 960
```

守护脚本语义：
- **恢复**：`DONE` 存在 → 只补跑终评；否则 `last.pth`→最新 `checkpointXXXX.pth`→冷启动。
- **重试**：连续 3 次无新 checkpoint → 退出码 3（疑似 OOM）。
- **信号**：SIGINT/SIGTERM 级联到训练，不重试，退出码 130。
- **结束**：exit 0 → 写 `DONE` → 终评。

---

## 4. P30u 协议速查（`protocol/p30u.yml`）

| 键 | 值 | 备注 |
|---|---|---|
| `epochs` | 30 | |
| `stop_epoch` | 28 | **真正**的 no-aug 边界；调度器用 `epochs-stop_epoch`=2 |
| `flat_epoch` | 15 | 留出真实余弦尾（28 会退化成台阶，见 audit Q2） |
| `no_aug_epoch` | 2 | **死键**（从不被读），仅为人类可读 |
| `warmup_iter` / `lr_gamma` | 2000 / 0.5 | |
| `optimizer` | lr 5e-4 / backbone 2.5e-5 / wd 1e-4 | 单处定义 |
| `ema` | decay 0.9995, warmups 1000 | |
| `sync_bn` / `use_amp` | False / True | XPU 上即 **bf16** |
| `eval_maxDets` | 300 | 配合 maxDets 修复（`num_queries=300`） |
| `mosaic_prob` / `mosaic_epoch` | **0 / 0** | Mosaic 关闭（对照走 `aug_mosaic025.yml`） |
| `mixup_prob` / `mixup_epoch` | 0 / 0 | 关闭 |
| train bs / val bs | 8 / 16 | 单卡 |
| `num_workers` | 4 | 协议内只出现一次 |

---

## 4b. P74u / P74u-960 协议族（EdgeCrafter 官方 S 档 74-epoch 基线）

- `protocol/p74u.yml`：`epochs=74 stop_epoch=72 flat_epoch=36 lr_gamma=0.5`，增强 **Mosaic 0.75@36 + Mixup 0.75@36**，
  `ema.decay=0.9999`；**含 Mosaic op**（P30u 的 ops 已删 Mosaic，P74u 必须恢复，否则 `mosaic_prob` 只门控 ZoomOut/IoUCrop）。
- `protocol/p74u_mosaic0.yml`：与 p74u **仅**差 4 个键（`mosaic_prob/mosaic_epoch/mixup_prob/mixup_epoch=0`），ops 相同。

| 入口 | schedule | 增强 | res | 输出目录 |
|---|---|---|---|---|
| `configs/ecdet/ecdet_s_p74u.yml` | 74/72/36 | Mosaic+Mixup 0.75@36 | 640 | `outputs/p74u` |
| `configs/ecdet/ecdet_s_p74u_mosaic0.yml` | 74/72/36 | OFF | 640 | `outputs/p74u_mosaic0` |
| `configs/ecdet/ecdet_s_p74u_960.yml` | 74/72/36 | 同 p74u | 960 | `outputs/p74u_960` |
| `configs/ecdet/ecdet_s_p74u_mosaic0_960.yml` | 74/72/36 | OFF | 960 | `outputs/p74u_mosaic0_960` |

- 960 由 `protocol/res960.yml` 设 `eval_spatial_size=[960,960]` → `reset_cfg` 自动派生 Mosaic `output_size=480`、Resize `960`（单旋钮）。
- 运行（单实验）：`P74U_CONFIG=configs/ecdet/ecdet_s_p74u.yml P74U_OUT_ROOT=$PWD/outputs/p74u VISDRONE_ROOT=$VISDRONE_ROOT bash scripts/run_exp.sh e1 42`
- pm2 预设：`p30u-all`、`p74u-all`(640)、`p74u960-all`(960)。输出根默认 `ecdetseg/outputs/{p30u,p74u,p74u_960}`。
- 选择顺序：配置 `P30U_CONFIG > P74U_CONFIG`；输出根 `P30U_OUT_ROOT > P74U_OUT_ROOT`；分辨率 `P74U_RES > P30U_RES > 640`。
- 960 显存≈640×2.25；16G 上必要时追加 `total_batch_size=4`。

## 5. 输出与评估

每个实验目录 `outputs/p30u/<init>_s<seed>/`：

| 文件 | 含义 |
|---|---|
| `snapshot/{config.yml,git_*.txt,git_diff.patch,pixi.lock.sha256,command.txt,start_time.txt}` | 启动快照 |
| `train.log` | stdout+stderr 合并 |
| `checkpointXXXX.pth` / `last.pth` | 每轮（`checkpoint_freq=1`）；`last.pth` **每轮都写**（见 §6 修复） |
| `best.pth` + `best.pth.info` | **在 val 上选出**（旁注文件说明） |
| `final.pth` | **最后一轮权重**（与 val 排名无关） |
| `log.txt` | 每轮 train/test 指标的 JSONL（`polt.py` 读它） |
| `DONE` / `crash_<时间>.log` | 完成标记 / 崩溃尾部 200 行 |
| `eval.jsonl` + `eval_*.json` | 终评结果（**只在 val，不跑 test-dev**） |

`eval.jsonl` 每行字段：`label, weights_type(ema\|raw), epoch, ckpt, AP, AP50, AP75, APS, APM, APL, params, note`。
默认三行：`final_ema`、`final_raw`、`best_ema`（best 标注「selected on VAL → reference only」）。

每个 `.pth` 内**同时含** `model`(原始) 与 `ema`(EMA)，另有 optimizer/scaler/criterion/postprocessor。

画曲线：`python scripts/polt.py outputs/p30u/e1_s42` → 生成 `.../dfine_curve.png`。

## 5b. 冒烟与 `_smoke` 清理约定

冒烟（短跑验证写入/曲线）**必须输出到 `outputs/_smoke/`**，别落到正式根 `outputs/p30u|p74u`：
```bash
cd /home/rainsin/Projects/Career/Paper/EdgeCrafter/ecdetseg
export VISDRONE_ROOT=/home/rainsin/Projects/Career/Paper/dataset/VisDrone   # 注意结尾 e
P30U_OUT_ROOT=$PWD/outputs/_smoke \
  bash scripts/run_exp.sh e1 7 epochs=4 stop_epoch=2 flat_epoch=1 warmup_iter=200
# 产物: outputs/_smoke/e1_s7/  (epochs=4 -> ckpt0000..0003 + last.pth/final.pth/best.pth + eval.jsonl + DONE)
```
- 判读 `log.txt` 的 `train_lr`：**warmup(二次上升) → flat → cosine → min** 三段齐全、无台阶即通过。
  注意 `train_lr` 取的是 `param_groups[0]`（**backbone**，初值 2.5e-5），head 是 5e-4。
- 冒烟的 `epochs/stop_epoch/flat_epoch/warmup_iter` 是 `run_exp.sh` 末尾的**额外 `key=val` 覆盖**（脚本自动加 `-u`，勿自己再带 `-u`）。
- **命名约定**：`outputs/_smoke/` 只放冒烟；正式实验放 `outputs/<protocol>[_mosaic0]/<init>_s<seed>/`。
  失败的冒烟保留为 `e1_s7_failed_<时间>`，成功者用 `e1_s7`（便于对照，不删）。
- 误落到正式根的冒烟搬走：`mv outputs/p30u/e1_s7 outputs/_smoke/e1_s7`（先 `pgrep -af train.py` 确认为空）。
- 冒烟已验证通过（2026-10-12）：LR 三段无台阶；`last.pth` 每轮写、`final.pth` 落盘、`best.pth.info` 标注；
  `eval.jsonl` 出 `final_ema/final_raw/best_ema`（示例 `final_ema AP=0.1088 > final_raw AP=0.1006`）。

---

## 6. 关键行为 / 坑（务必知道）

1. **`stop_epoch` 处会 reload `best.pth`**：no-aug 阶段从「val 上选出的 best」重新出发
   （model+EMA+optimizer 全被覆盖）。→ val 既用于选 best 又用于评估，**不是留出集**；
   论文最终数字须报 test-dev（本流程不跑）。
2. **`last.pth` 曾在 `stop_epoch` 后冻结**（旧行为，会低估最终模型）——**已修复**：
   现在每轮写 `last.pth`，并在结束写 `final.pth`。参见 `ec_solver.py` 改动。
3. **`mosaic_prob` 必须显式置 0**：即使 ops 里删了 Mosaic，`container.py:83-96` 仍用它门控
   `RandomZoomOut/RandomIoUCrop`（否则静默跳过 ~50%）。参见 `CHANGES.md §4`。
4. **分辨率只有 `eval_spatial_size` 一个有效旋钮**：`reset_cfg` 自动派生 Mosaic `output_size=//2`
   与所有 `Resize`；yaml 里 640/320 字面量是死值。
5. **`no_aug_epoch` 是死键**；真正控制是 `stop_epoch`（且被 `reset_cfg` 校验在 `[epochs-5, epochs]`）。
6. **`num_classes=10`**（VisDrone id 0..9，无 ignored/crowd）→ 正确，无需 11 或重映射。
7. **E2/E3 分类头随机初始化**：10 类 vs 80 类形状不符，`map_class_weights` 回退，
   分类头不加载（backbone/encoder/decoder 正常加载）。
8. **`PYTORCH_SYCL_ALLOC_CONF` 当前 torch 2.9.1+xpu 不识别**（no-op）；
   `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` 被 UR LevelZero 适配器识别。

---

## 7. 我改动过的**核心代码**（非配置；供你审阅）

| 文件 | 改动 | 来源任务 |
|---|---|---|
| `engine/solver/ec_solver.py` | 每轮写 `last.pth`；结束写 `final.pth`+`best.pth.info`；best 保存打日志 | 本次(last.pth 修复) |
| `scripts/eval_final.py` | `--weights ema\|raw\|both`，输出 `weights_type/epoch` | 本次 |
| `engine/data/dataset/coco_eval.py` | 自定义 `coco_summarize`，`maxDets[-1]`；`build_max_dets`；`max_dets` 参数 | 更早(maxDets) |
| `engine/core/yaml_config.py` | 透传 `eval_maxDets` → evaluator | 更早 |
| `engine/optim/amp.py` | `get_amp_dtype`（cuda→fp16/xpu→bf16）；`GradScaler` 按设备自适应 `enabled` | 更早(AMP) |
| `engine/solver/ec_engine.py` | autocast dtype 改为 `get_amp_dtype` | 更早 |
| `train.py` | `--use-amp` 默认 `None`（yaml 的 `use_amp` 生效） | 更早 |
| `engine/edgecrafter/{decoder,criterion}.py` | dn 分支 GT 截断（`cap_dn_targets`）与 `dn_targets` | 更早(dn OOM) |

**未改动（冻结）**：`configs/ecdet/ecdet.yml`、`configs/ecdet/ecdet_s.yml`、`configs/dataset/custom.yml`、
`train_o365.sh`、`train_ecdet.sh`。新实验通过「新文件 + 最后 include 覆盖」实现，不碰它们。

---

## 8. 待你确认的决策

1. `flat_epoch=15`、`ema.decay=0.9995`、`mosaic_prob`（默认关）、`num_classes=10` 是否定稿？
2. E2/E3 分类头是否要做 VisDrone 专用类映射（会改 `engine/solver/_solver.py`）？
3. 终评口径：默认给出 `final_ema`/`final_raw`/`best_ema`，是否足够？
4. 训练结束后是否把冻结的三层（`custom/ecdet/ecdet_s.yml`）正式折叠进新结构？
5. 是否把 `polt.py` 改名为 `plot.py`、并清理根目录多余的 `dfine_curve.png` / `test.py` / `scripts/script/` 归位？

---

## 9. 快速排障

| 现象 | 处理 |
|---|---|
| `VISDRONE_ROOT` 未设 | `export VISDRONE_ROOT=...` 后重跑 |
| `label: unbound variable` | 旧版脚本 bug，已修；更新 `scripts/run_exp.sh` |
| 终评报 `-1.000` / 缺槽位 | 见 `coco_eval.py` maxDets 修复（已含） |
| 想重训某实验 | `rm -f outputs/p30u/<init>_s<seed>/DONE` |
| 想强制重算终评 | `FORCE_FINAL_EVAL=1 bash scripts/run_exp.sh e1 42` |
| 想跳过终评 | `SKIP_FINAL_EVAL=1 ...` |
| XPU 冒烟 | `VISDRONE_ROOT=... bash scripts/verify_p30u_xpu.sh`（先 `pgrep`） |
| 运行中改 `run_exp.sh` 后报“未预期的记号 (” | bash 边读边执行；**先 `pm2 stop`，改完再 `pm2 start`**（文件本身没问题，`bash -n` 通过） |
| `FileNotFoundError .../VisDron/...` | `VISDRONE_ROOT` 拼错（少 `e`）；脚本已**早失败**并提示。已训练目录会自动用其快照里的正确路径 |
