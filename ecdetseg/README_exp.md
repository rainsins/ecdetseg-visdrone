# P30u — 运行 E1/E2/E3 × seeds

统一协议 **P30u**：三个初始化实验除初始化外**完全一致**（配置见 `configs/ecdet/protocol/p30u.yml`）。

## 0. 前置

```bash
# pixi 环境（仓库根 pixi.toml）
export VISDRONE_ROOT=/home/rainsin/Projects/Career/Paper/dataset/VisDrone   # 必填
# 可选：
export WEIGHTS_DIR=/home/rainsin/Projects/Career/Paper/EdgeCrafter/ecdetseg # ecdet_s.pth / ecdet_s_o3652coco.pth 所在
export P30U_RES=640        # 或 960（调用 reset_cfg 自动派生 Mosaic/Resize，见 audit Q10）
```

`-t` 权重：`e2 → $WEIGHTS_DIR/ecdet_s.pth`，`e3 → $WEIGHTS_DIR/ecdet_s_o3652coco.pth`（在 `run_exp.sh` 的 init 表里**只有一处**）。

## 1. 跑实验

```bash
cd ecdetseg
bash scripts/run_exp.sh e1 42
bash scripts/run_exp.sh e2 42
bash scripts/run_exp.sh e3 42
# 三个 seed
for s in 42 43 44; do for i in e1 e2 e3; do bash scripts/run_exp.sh $i $s; done; done
```
> 训练正在跑时**不要**在 XPU 上启动这些（单卡 16G 会 OOM）。守护脚本本身不占 XPU。

## 2. 输出布局 `outputs/p30u/<init>_s<seed>/`

| 文件 | 含义 |
|---|---|
| `snapshot/config.yml` | 启动时的最终合并配置（`dump_config.py` 产物） |
| `snapshot/git_commit.txt` / `git_status.txt` / `git_diff.patch` | 代码快照 |
| `snapshot/pixi.lock.sha256` | 环境锁定 |
| `snapshot/command.txt` / `start_time.txt` | 命令行/开始时间 |
| `train.log` | stdout+stderr 合并日志 |
| `last.pth` / `checkpointXXXX.pth` / `best.pth` | 训练产物（见“注意”） |
| `DONE` | 正常结束标记（存在即拒绝重复 resume） |
| `eval.jsonl` / `eval_best_ema.json` / `eval_last_ema.json` | 结束时 EMA 在 **val** 上的指标（不打 test-dev） |
| `crash_<时间>.log` | 崩溃时最后 200 行 |

## 3. 守护脚本语义（`scripts/run_exp.sh`）

- **恢复**：`DONE` 存在 → 直接退出；否则 `last.pth` 可 `torch.load` → `-r`；否则回退到最新可用
  `checkpointXXXX.pth`；都没有 → 冷启动（e1 无 `-t`，e2/e3 带 `-t`）。
- **重试**：以“是否产生新 `.pth`”判进展；连续 `NO_PROGRESS_LIMIT`(=3) 次无进展 → 退出码 3（疑似 OOM）。
- **信号**：SIGINT/SIGTERM（Ctrl-C / `pm2 stop`）转发到训练进程组，**不重试**，退出码 130。
- **结束**：exit 0 → 写 `DONE` → 自动在 val 上评估 `final.pth`(EMA+raw) 与 `best.pth`(EMA) 写 `eval.jsonl`
  → 自动调用 `scripts/polt.py` 生成 `<run>/dfine_curve.png`（`SKIP_FINAL_EVAL=1` / `SKIP_PLOT=1` 可分别跳过）。

### 注意
- `last.pth` 在 `epoch >= stop_epoch` 后**不再更新**（`ec_solver.py:111`），no-aug 阶段只更新 `best.pth`。
  故“最后一轮 EMA”通常取 `best.pth`；`eval.jsonl` 同时给出 `best_ema` 与 `last_ema`，**best 是在 val 上选的，仅供参考**。
- 训练结束时评估仅用 **val**，**不**自动跑 test-dev。
- `eval.jsonl` 每行含 `AP/AP50/AP75/APS/APM/APL/params/ckpt/label/note`。

## 4. 新增一个实验

在 `run_exp.sh` 的 `case "$INIT"` 里加一行（唯一改动点），例如：
```bash
e4) TUNING="$WEIGHTS_DIR/ecdet_s_myinit.pth" ;;
```
然后 `bash scripts/run_exp.sh e4 42`。想改结构/协议另建 include 分支的入口 config。

## 5. 改分辨率

- 首选：`P30U_RES=960 bash scripts/run_exp.sh e1 42`（脚本注入 `eval_spatial_size=[960,960]`）。
- 或把入口 config 的 `protocol/res640.yml` 换成 `protocol/res960.yml`。
- **只需这一处**：`reset_cfg` 会派生 `Mosaic.output_size=480`、`Resize=960`（audit Q10）。

## 5b. Mosaic 对照实验（默认关闭）

默认 P30u 入口 `ecdet_s_p30u.yml` 已**关闭 Mosaic**（`ops` 中删除 Mosaic，`mosaic_prob: 0`）。
对照实验用 Mosaic@0.25：

```bash
# 手工运行（注意：训练结束后再用）
cd ecdetseg
python train.py -c configs/ecdet/ecdet_s_p30u_mosaic025.yml -d xpu \
  -t ecdet_s.pth --output-dir outputs/p30u/e2_s42_mosaic025 \
  -u train_dataloader.dataset.img_folder=$VISDRONE_ROOT/train ... seed=42
```
除增强块外与默认入口完全一致（`configs_dump/E1.yml` vs `configs_dump/E1_mosaic025.yml`）。
> `mosaic_prob` 必须显式置 0/0.25：`container.py:83-96` 用它门控 `RandomZoomOut/RandomIoUCrop`，
> 与 Mosaic 是否存在无关（详见 CHANGES.md §4）。

## 5c. P74u（EdgeCrafter 官方 S 档 74-epoch 基线）

- 协议 `protocol/p74u.yml`：`epochs=74 stop_epoch=72 flat_epoch=36 lr_gamma=0.5`，
  增强 **Mosaic 0.75/ep36 + Mixup 0.75/ep36**，`ema.decay=0.9999`；其余键与 P30u 相同。
- 消融 `protocol/p74u_mosaic0.yml`：仅 `mosaic_prob/mosaic_epoch/mixup_prob/mixup_epoch` 置 0
  （ops 保持相同；Mosaic op 仍在列表但被 `mosaic_prob:0` 跳过）。
- 入口：`ecdet_s_p74u.yml`（基线，Mosaic 开）、`ecdet_s_p74u_mosaic0.yml`（Mosaic 关）。
- **960 臂**：`ecdet_s_p74u_960.yml`、`ecdet_s_p74u_mosaic0_960.yml`（include `protocol/res960.yml`，
  `output_dir` 为 `outputs/p74u_960` / `outputs/p74u_mosaic0_960`）。也可用 `P74U_RES=960` 覆盖（见下）。

```bash
# 单实验
P74U_CONFIG=configs/ecdet/ecdet_s_p74u.yml \
P74U_OUT_ROOT=$PWD/outputs/p74u VISDRONE_ROOT=$VISDRONE_ROOT \
  bash scripts/run_exp.sh e1 42

# pm2 批跑（串行 3 seed × 3 init）
P74U_CONFIG=configs/ecdet/ecdet_s_p74u.yml \
P74U_OUT_ROOT=$PWD/outputs/p74u VISDRONE_ROOT=$VISDRONE_ROOT \
  pm2 start ecosystem.config.js --only p74u-all
```
> 优先级：`P30U_CONFIG` > `P74U_CONFIG`；`P30U_OUT_ROOT` > `P74U_OUT_ROOT`；
> 分辨率 `P74U_RES` > `P30U_RES` > 640。
> resume 前会用 `snapshot/config_merged.yml` 校验协议/增强一致性，不一致则退出码 4。

**输出目录命名约定**：`outputs/<protocol>[_mosaic0]/<init>_s<seed>/`
（例：`outputs/p30u/e1_s42`、`outputs/p74u/e2_s42`、`outputs/p74u_mosaic0/e1_s42`）。

> ⚠️ P74u 基线**含 Mosaic op**（官方增强）。若逐字照抄 P30u（其 ops 已删 Mosaic）而只改
> `mosaic_prob:0.75`，Mosaic **不会生效**且会静默门控 `RandomZoomOut/RandomIoUCrop`
> （`container.py:83-96`，见 CHANGES.md §4）。故 P74u 恢复了 Mosaic op。

## 6. 静态验证（随时可跑，不碰 XPU）

```bash
cd ecdetseg
bash -n scripts/run_exp.sh scripts/fake_train.sh scripts/test_run_exp.sh
python scripts/dump_config.py -c configs/ecdet/ecdet_s_p30u.yml -o /tmp/p30u.yml
bash scripts/test_run_exp.sh          # 21 项：重试/进展/DONE/信号/终评/协议守卫
```

## 7. XPU 验证（**训练结束后**，脚本会先 `pgrep` 守卫）

```bash
cd ecdetseg
VISDRONE_ROOT=... bash scripts/verify_p30u_xpu.sh
```
用 64 张小子集、`epochs=1` 对 E1/E2/E3 各跑一次，检查：权重载入差异、学习率曲线、per-iter 耗时、
`-r` 可恢复。（峰值显存需要采样；见脚本内 TODO。）
