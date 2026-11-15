# 当前实验协议：P150u

后续基线、模块验证和消融统一使用 **P150u**。
主初始化固定为 **E2（ecdet_s.pth）**，种子为 **42/43/44**；E1/E3 仅作初始化对照。
主结果为 `final_ema`，`final_raw` 和 val 选择的 `best_ema` 单独保留。

| 设置 | 当前基线 |
|---|---|
| 输入 | 默认 640 × 640；960 是独立尺度实验 |
| 轮数 | 最多 150 |
| 训练 / 验证 batch | 10 / 32 |
| flat_epoch / stop_epoch | 72 / 148 |
| Mosaic / Mixup | 概率 0.75，截止参数 epoch=72 |
| 早停 | 从 epoch 100 起，patience=20，min_delta=0.001 |
| AMP / EMA | 保持现有设置 |

早停属于协议的一部分，因此不保证每个实验都跑满 150 轮。
配置权威来源：`configs/ecdet/protocol/p150u.yml`；此次整理未修改其数值。

## 新实验推荐入口

在 `ecdetseg/` 下配置 `VISDRONE_ROOT` 和 `WEIGHTS_DIR`，然后：

```bash
python scripts/benchmark.py plan B0 --init e2 --seed 42
python scripts/benchmark.py run B0 --init e2 --seed 42
python scripts/benchmark.py summarize -o benchmark_results.csv
```

默认协议已经是 P150u，输出为 `outputs/benchmark/B0-v1/p150u/r640/default/e2_s42/`。
模块接线状态和完整用法见 `BENCHMARK.md`。B1–B5 未实现前仍拒绝启动。
CUDA 环境需另行准备；设备参数不改变仓库默认 Pixi 的 XPU 依赖。
设备/AMP 修复、隔离环境、租卡命令及实测范围见 `docs/DEVICE_RUNTIME.md`。
使用 `--runtime-python /absolute/path/to/python --device cuda:0 --precision fp16`
可让训练、checkpoint 检查和终评共用独立环境；当前 CUDA 尚未实机验证。

## 原有运行与批量入口

```bash
# 原启动器：默认 P150u，输出 outputs/p150u/<init>_s<seed>/
bash scripts/run_exp.sh e2 42
# 批量运行，明确选择初始化和种子
INITS=e2 SEEDS="42 43 44" bash scripts/run_all.sh
# PM2 当前入口仅提供 P150u 训练队列
pm2 start ecosystem.config.js --only p150u-all
```

新环境变量为 `EXP_CONFIG`、`EXP_OUT_ROOT`、`EXP_RES`；其次为 `P150U_*`。
旧的 P30U/P74U 变量仅兼容历史调用，优先级较低；运行新基线前移除 shell 中残留的旧变量。
修改输入尺度时使用 benchmark 的 `--res`，以自动隔离目录。

## 历史协议归档

- 配置正文：`configs/archive/pre_p150u/`；旧配置路径仅保留兼容 include。
- 原说明与 PM2 配置：`archive/pre_p150u/`。
- P30u/P74u 历史结果与 checkpoint 保持原位，便于旧命令、快照和正在运行的进程继续定位。
- 历史 benchmark 操作须显式加入 `--allow-archived`；不与 P150u 汇总成同一比较组。
- 源代码快照校验仍严格：整理前已创建的受管理运行若因源指纹改变而拒绝恢复，
  应在原快照对应的代码环境恢复，不改写指纹绕过检查。

## 历史汇总与恢复

在 `ecdetseg/` 下刷新已有 P150u 结果：

```bash
python scripts/benchmark.py summarize --historical-root outputs/p150u \
  -o outputs/p150u/results.csv \
  --baseline-summary outputs/p150u/e2_baseline_summary.json
```

CSV 为逐运行/逐权重结果，指标单位为 0–1；JSON 给出 E2 完成种子、缺失种子、均值及样本标准差。
不足两次完成运行时标准差为空；三 seed 未齐时 `complete=false`，不能当作正式三 seed 结论。
历史配置按当前 P150u 展开核对，按配置、分辨率、初始化、设备和来源隔离。
历史源指纹留空，保留原始 snapshot；此次恢复的代码、输入校验和原因追加到 `resume_history/`，
当前代码快照不冒充历史代码。PM2 队列每完成一个运行后自动刷新上述汇总。

早停对象现在先于 checkpoint 恢复创建，每轮在验证和早停计数后保存 checkpoint。
旧 checkpoint 无早停状态时，从原日志重建；计入早停的历史缺失则拒绝猜测。
epoch 148 回载 best 时保留早停计数；已停止的 checkpoint 恢复后不再多训一轮。
`DONE` 只表示训练结束，三类终评及相关权重齐备才在汇总中标为 complete。

基线收尾后进入 SET：B1 无增强辅助监督 → B2 全图平滑 → B3 HBS → B4 查询式 API → B5 联合。
B1–B5 仍需实现和验证，当前不自动启动。B0 指当前项目基线，不表示上游论文逐项复现。
