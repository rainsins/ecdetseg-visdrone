# HANDOFF — EdgeCrafter 仓库现状 & 当前工作

> 用途：换新对话框 / 新 agent 时，先读这份即可全面了解仓库与正在进行的工作。
> 生成时间：2026-10-12（仓库 HEAD `12abfe2 Update .gitignore`）。
> ⚠️ 生成时有训练在跑：`outputs/p74u/e1_s42`（P74u/e1/seed42）。**单卡，勿再启第二个 XPU 任务。**

---

## 0. 30 秒速读
- 仓库 = **EdgeCrafter (TMLR 2026)**：`ecdetseg/`（检测/分割）+ `ecpose/`（姿态）。
- 我（助手）最近的主线工作全在 **`ecdetseg/`**：为 **VisDrone** 建立 **P30u / P74u 训练协议体系**、
  守护脚本 `scripts/run_exp.sh`、评估工具、以及若干核心代码修复。
- 环境：**pixi**（仓库根 `pixi.toml`）+ **torch 2.9.1+xpu** + 单张 **Intel Arc A770 16G**，设备 `-d xpu`，AMP=**bf16**。
- 数据：`/home/rainsin/Projects/Career/Paper/dataset/VisDrone`（10 类，id 0..9）。
- 只读了解细节请按 §9 的顺序读文档；跑实验看 §4；踩坑看 §5。

---

## 1. 仓库与环境
- 路径：`/home/rainsin/Projects/Career/Paper/EdgeCrafter`
- 子项目：
  - `ecdetseg/`：检测（ECDet）/ 实例分割（ECSeg），configs/engine/tools/scripts。
  - `ecpose/`：姿态（ECPose），独立一套。
- 包管理：`pixi`（`pixi.toml` 在仓库根，env 在 `.pixi/envs/default`）。
  - deps：`python 3.10-3.11`、`numpy<2`、`scipy 1.11.4`、`opencv`、`matplotlib`；
    pypi：`torch==2.9.1+xpu`、`torchvision==0.24.1+xpu`、`pytorch-triton-xpu`、`pycocotools`、`pyyaml`、`tensorboard`、`calflops`、`transformers`、`tabulate`、`onnx`、`onnxruntime`。
- 运行训练/评估一律 `pixi run python ...`（自动用 XPU 环境）。CPU 静态脚本用系统 `python` 亦可（有 yaml/matplotlib）。
- 硬件：**单卡 Arc A770 16G**。任何 XPU 操作前必须 `pgrep -af train.py` 为空。

---

## 2. 数据与权重
- 数据根（**注意结尾 e**）：`/home/rainsin/Projects/Career/Paper/dataset/VisDrone`
  - `train/`、`val/` 图像目录；`annotations_VisDrone_train.json`、`annotations_VisDrone_val.json`。
  - 统计：train 6471 图 / 343205 标注；val 548 图 / 38759 标注；**10 类 id 0..9**（pedestrian…motor），`iscrowd` 全 0，无 ignored。→ `num_classes: 10` 正确。
- 权重（`ecdetseg/`）：`ecdet_s.pth`（COCO-80）、`ecdet_s_o3652coco.pth`、`ecdet_s_o365.pth`、`ecdet_l.pth`；骨干 `ecvits/ecvitt.pth`。

---

## 3. 当前工作主线：P30u / P74u 协议体系（`ecdetseg/`）

### 3.1 设计
配置分四层，**后 include 覆盖前 include**（`load_config` 递归合并，list 整体替换）：
`数据(visdrone.yml) < 基础结构(ecdet.yml，冻结) < S 结构(struct_s.yml) < 分辨率(res*.yml) < 协议(p30u|p74u，最后)`。
`YAMLConfig.reset_cfg` 会据 `eval_spatial_size` **自动派生** Mosaic `output_size=//2` 与所有 `Resize`（→ 分辨率只有这一个旋钮）。

### 3.2 协议与入口
| 协议 | schedule (epochs/stop/flat) | 增强 | 入口(640) | 输出根 |
|---|---|---|---|---|
| **P30u** | 30/28/15 | Mosaic **OFF** | `ecdet_s_p30u.yml` | `outputs/p30u` |
| P30u+ | 30/28/15 | Mosaic@0.25 | `ecdet_s_p30u_mosaic025.yml` | `outputs/p30u` |
| **P74u**（官方基线） | **74/72/36** | Mosaic+Mixup 0.75@36 | `ecdet_s_p74u.yml` | `outputs/p74u` |
| P74u− | 74/72/36 | OFF | `ecdet_s_p74u_mosaic0.yml` | `outputs/p74u_mosaic0` |
| P74u 960 | 74/72/36 | 同 P74u | `ecdet_s_p74u_960.yml` | `outputs/p74u_960` |
| P74u 960− | 74/72/36 | OFF | `ecdet_s_p74u_mosaic0_960.yml` | `outputs/p74u_mosaic0_960` |
协议文件在 `configs/ecdet/protocol/`：`p30u.yml`、`p74u.yml`、`p74u_mosaic0.yml`、`aug_mosaic025.yml`、`res640.yml`、`res960.yml`。
> 关键：**P74u 的 ops 含 Mosaic op**（P30u 把 Mosaic 从 ops 删了）；否则 `mosaic_prob>0` 只会门控 ZoomOut/IoUCrop（`container.py:83-96`）。

### 3.3 三个初始化实验
`run_exp.sh <e1|e2|e3> <seed>`：**E1** 无 `-t`（仅骨干 `ecvitt.pth`）；**E2** `-t ecdet_s.pth`；**E3** `-t ecdet_s_o3652coco.pth`。
（用 `-t`/`-r` 时 `train.py` 会置 `ViTAdapter.skip_load_backbone=True`；10 类 vs 80 类 → 分类头形状不符会随机初始化。）

### 3.4 守护脚本 `scripts/run_exp.sh`
- 快照（`snapshot/`：`config_merged.yml`、git hash/status/diff、pixi.lock sha256、cmd、time）。
- 恢复：`DONE` 存在→只补终评+画图；否则 `last.pth`→最新 `checkpointXXXX.pth`→冷启动。
- 重试：连续 3 次无新 checkpoint → exit 3（疑似 OOM）。
- 信号：SIGINT/SIGTERM 级联到训练，不重试，exit 130。
- **协议守卫**：resume 前比对快照 vs 当前的 `{epochs,flat_epoch,no_aug_epoch,stop_epoch,mosaic_prob,mosaic_epoch,mixup_prob,mixup_epoch}`，不一致 → `PROTOCOL_MISMATCH`，exit 4。
- **数据路径优先用本目录快照**（防 `VISDRONE_ROOT` 拼错/改动）；fresh 运行则用 `$VISDRONE_ROOT`，且**建模型前校验路径存在**（early-fail exit 2）。
- 环境变量：`VISDRONE_ROOT`（fresh 必填）、`WEIGHTS_DIR`、`P30U_CONFIG`/`P74U_CONFIG`、`P30U_OUT_ROOT`/`P74U_OUT_ROOT`、`P30U_RES`/`P74U_RES`、`SKIP_FINAL_EVAL`/`SKIP_PLOT`/`FORCE_FINAL_EVAL`、`PYTHON_BIN`/`EVAL_BIN`（测试钩子）。
- 环境优先级：config `P30U_CONFIG > P74U_CONFIG`；输出根 `P30U_OUT_ROOT > P74U_OUT_ROOT`；分辨率 `P74U_RES > P30U_RES > 640`。

### 3.5 训练产物（每个 `outputs/<root>/<init>_s<seed>/`）
`checkpointXXXX.pth`（`checkpoint_freq=1`，**每轮**）、`last.pth`（**每轮**，已修）、`final.pth`（末轮）、`best.pth`+`best.pth.info`（val 选出，标注）、`log.txt`（每轮 JSONL）、`train.log`、`DONE`、`crash_*.log`、`snapshot/`、`eval.jsonl`+`eval_*.json`、`dfine_curve.png`（收尾自动画）。
- 每个 `.pth` 同时含 `model`(原始) 与 `ema`；终评对 **final.pth 评 EMA 与 raw**，另评 `best.pth`(EMA)。
- `eval.jsonl` 字段：`label, weights_type(ema|raw), epoch, ckpt, AP, AP50, AP75, APS, APM, APL, params, note`。**只评 val，不跑 test-dev。**
- `pm2` 预设（`ecosystem.config.js`）：`p30u-all`、`p74u-all`、`p74u960-all`（单卡串行；`autorestart:false`）。

---

## 4. 运行命令速查

```bash
cd /home/rainsin/Projects/Career/Paper/EdgeCrafter/ecdetseg
export VISDRONE_ROOT=/home/rainsin/Projects/Career/Paper/dataset/VisDrone   # 注意结尾 e
export WEIGHTS_DIR=/home/rainsin/Projects/Career/Paper/EdgeCrafter/ecdetseg
pgrep -af train.py        # 必须为空！

# 主基线 P74u（单实验 / 全量 / pm2）
P74U_CONFIG=configs/ecdet/ecdet_s_p74u.yml P74U_OUT_ROOT=$PWD/outputs/p74u \
  bash scripts/run_exp.sh e1 42
for s in 42 43 44; do for e in e1 e2 e3; do
  P74U_CONFIG=configs/ecdet/ecdet_s_p74u.yml P74U_OUT_ROOT=$PWD/outputs/p74u bash scripts/run_exp.sh $e $s
done; done
P74U_CONFIG=configs/ecdet/ecdet_s_p74u.yml P74U_OUT_ROOT=$PWD/outputs/p74u VISDRONE_ROOT=$VISDRONE_ROOT \
  pm2 start ecosystem.config.js --only p74u-all

# P30u（把上面 P74U_CONFIG 换成 configs/ecdet/ecdet_s_p30u.yml，输出根 outputs/p30u）
# 960：P74U_CONFIG=configs/ecdet/ecdet_s_p74u_960.yml，输出根 outputs/p74u_960（OOM 时加 total_batch_size=4）
# 冒烟（短，必落 outputs/_smoke）：
P30U_OUT_ROOT=$PWD/outputs/_smoke bash scripts/run_exp.sh e1 7 epochs=4 stop_epoch=2 flat_epoch=1 warmup_iter=200

# 监控 / 停止 / 结果
pm2 logs p74u-all ; pm2 stop p74u-all
tail -f outputs/p74u/e1_s42/train.log
cat outputs/p74u/e1_s42/eval.jsonl
python scripts/polt.py outputs/p74u/e1_s42      # 手动画图（收尾已自动生成）
```

---

## 5. 铁律 & 常见坑
- **单卡**：同一时间只跑一个训练；启动前 `pgrep -af train.py` 必须为空。
- **别在训练进行时改 `scripts/run_exp.sh`**（bash 边读边执行会报“未预期的记号 (”，且会打断正在跑的守护进程）。要改先 `pm2 stop` / 等结束。
- **`VISDRONE_ROOT` 拼写**：常见 `VisDron`（缺 e）；脚本已 early-fail 提示。
- **`stop_epoch` 处会 `load_resume_state(best.pth)`**：no-aug 阶段从“val 选出的 best”重开；val 非留出集，最终数字须报 test-dev（本流程不跑）。
- **`mosaic_prob` 必须显式置值**：即便 ops 里没有 Mosaic，`mosaic_prob>0` 也会门控 `RandomZoomOut/RandomIoUCrop`（`container.py:83-96`）。
- **分辨率只有 `eval_spatial_size` 一个有效旋钮**（`reset_cfg` 派生 Mosaic/Resize）；yaml 里 640/320 字面量是死值。
- **`no_aug_epoch` 是死键**：真正控制是 `stop_epoch`（`no_aug_epochs = epochs - stop_epoch`）。
- **`last.pth` 曾冻结在 `stop_epoch-1`**（已修：现在每轮写，并写 `final.pth`）。
- 其它：XPU 上 AMP=**bf16**（`GradScaler` enabled=False）；`PYTORCH_SYCL_ALLOC_CONF` 当前 torch **不识别**（no-op），`SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` 被 UR LevelZero 适配器识别。

---

## 6. 我改动过的核心代码（非配置，注意不是上游原样）
| 文件 | 改动 | 目的 |
|---|---|---|
| `ecdetseg/engine/solver/ec_solver.py` | 每轮写 `last.pth`；结束写 `final.pth` + `best.pth.info`；best 保存打日志 | 修复“最终模型被低估” |
| `ecdetseg/engine/data/dataset/coco_eval.py` | `build_max_dets`、`coco_summarize`（用 `maxDets[-1]`）、`CocoEvaluator(max_dets=...)` | 评估 maxDets=300 生效、无 -1.000 |
| `ecdetseg/engine/core/yaml_config.py` | evaluator 透传 `eval_maxDets` | 同上 |
| `ecdetseg/engine/optim/amp.py` | `get_amp_dtype`（xpu→bf16 / cuda→fp16）；`GradScaler` 按设备自适应 `enabled` | XPU bf16 + CUDA 兼容 |
| `ecdetseg/engine/solver/ec_engine.py` | autocast dtype 改用 `get_amp_dtype` | 同上 |
| `ecdetseg/train.py` | `--use-amp` 默认 `None`（让 yaml 的 `use_amp` 生效） | 同上 |
| `ecdetseg/engine/edgecrafter/decoder.py` | 新增 `cap_dn_targets` / `MAX_GT_FOR_DN`（dn 分支 GT 截断） | 修 VisDrone 密集图 host OOM |
| `ecdetseg/engine/edgecrafter/criterion.py` | dn 分支改用 `dn_targets`（索引/归一化） | 同上 |

其余为**新增文件**（配置/脚本/文档）与 `ecpose/` 的同类 AMP 改动（更早）。

---

## 7. 冻结文件（除非明确要求，勿改）
`ecdetseg/configs/ecdet/ecdet.yml`、`configs/ecdet/ecdet_s.yml`、`configs/dataset/custom.yml`、
以及原 `train_o365.sh`/`train_ecdet.sh`（被移动到 `ecdetseg/scripts/script/`）。
新实验一律用「新文件 + 最后 include 覆盖」实现。

---

## 8. 已完成 / 进行中实验（`ecdetseg/outputs/`）
- `p30u/e1_s42`、`p30u/e2_s42`：P30u，已完成（有 `DONE`）。
- `p74u/e1_s42`：**正在训练**（P74u 主基线，seed42）。
- `_smoke/e1_s7`（成功冒烟，含 `dfine_curve.png`）、`_smoke/e1_s7_failed_2107`（拼错路径的失败冒烟，保留对照）。
- 旧实验：`ecdet_s/`、`ecdet_s_coco_pre_weights/`、`ecdet_s_o3652coco_visdrone/`。

---

## 9. 文档索引（建议阅读顺序）
1. **本文件 `HANDOFF.md`** — 全局地图。
2. `ecdetseg/P30U_GUIDE.md` — 文件清单、协议速查、P74u/960、冒烟与 `_smoke` 约定、坑与排障。
3. `ecdetseg/audit.md` — 只读审计（`__include__` 语义、flatcosine、EMA、`-t/-r`、checkpoint、seed、标注、AMP、分辨率、SYCL 变量）；含“待确认”。
4. `ecdetseg/README_exp.md` — 如何跑 E1/E2/E3、Mosaic 对照、P74u、静态/XPU 验证。
5. `ecdetseg/CHANGES.md` — 旧配置→P30u 的逐项差异与原因。
6. `ecdetseg/ecosystem.config.js` + `ecdetseg/scripts/*` — pm2 与守护/评估/画图脚本。

---

## 10. 待确认/未决
- `flat_epoch=15`、`ema.decay`（P30u=0.9995 / P74u=0.9999）、`mosaic_prob`、`num_classes=10` 是否定稿。
- E2/E3 是否要做 VisDrone 专用分类头类映射（会改 `engine/solver/_solver.py`）。
- 终评口径：默认给 `final_ema/final_raw/best_ema`，是否需要额外报告 test-dev。
- 训练结束后是否把冻结的三层（`custom/ecdet/ecdet_s.yml`）正式折叠进新结构。
- 是否把 `scripts/polt.py` 改名 `plot.py`、清理根目录杂项。

---

## 11. 新对话开场建议
先执行这几条只读命令建立认知，再动任何东西：
```bash
cd /home/rainsin/Projects/Career/Paper/EdgeCrafter
git log --oneline -3 ; git status --porcelain | head
pgrep -af train.py            # 是否有训练在跑（单卡！）
ls ecdetseg/configs/ecdet/protocol/ ecdetseg/scripts/ ecdetseg/outputs/
cat HANDOFF.md                # 本文
```
然后按 §9 阅读。要跑实验用 §4 命令；务必先 `pgrep`、别并行、别在跑的时候改脚本。
