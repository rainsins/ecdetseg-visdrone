# 当前实验协议：P150u

后续基线、模块验证和消融统一使用 **P150u**。

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

本次只整理文件和默认入口，不启动/停止训练或改动 PM2 中已运行的进程。
