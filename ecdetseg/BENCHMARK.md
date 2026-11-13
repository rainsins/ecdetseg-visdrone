# 可插拔模块与基准管理

实验登记入口：`configs/ecdet/experiments/registry.yml`。管理命令：
`scripts/benchmark.py`（只依赖标准库和 PyYAML，不导入 torch）。
真正训练仍由现有 `scripts/run_exp.sh` 负责恢复、重试、信号转发和终评。

## 唯一活动协议：P150u

后续 B0–B5 使用 P150u，默认 640 输入；150 为最大轮数，保留原早停规则。
P30u/P74u 已归档，只有显式 `--allow-archived --protocol p74u` 才可选择历史协议。
640 与 960 仍须分组比较；初始化和 batch 等设置也必须对齐。

## 当前状态

| 编号 | 实验 | 状态 |
|---|---|---|
| B0 | 原始 ECDet | 已接入管理入口；不代表本次已完成训练 |
| B1 | 不增强的辅助分支，控制额外监督收益 | planned |
| B2 | 全图平滑辅助分支 | planned |
| B3 | HBS 辅助分支 | planned |
| B4 | 适配查询式检测器的 API | planned |
| B5 | HBS + API | planned |

B1–B5 尚未接入模型和损失，`plan`、`run` 都会拒绝。登记表里的
`modules` 是设计记录，**不是已经生效的模型开关**。本次不产生方法精度结论。

## 使用

在 `ecdetseg/` 中运行；`python` 需要 PyYAML。

```bash
python scripts/benchmark.py list
export VISDRONE_ROOT=/path/to/VisDrone
export WEIGHTS_DIR=/path/to/initial_weights

# 只预览：展开真实协议、分辨率及覆盖项；不创建输出，不训练。
python scripts/benchmark.py plan B0 --protocol p150u --init e2 --seed 42 --res 640

# 显式训练/恢复：确认数据和权重齐全后再执行。
python scripts/benchmark.py run B0 --protocol p150u --init e2 --seed 42 --res 640

# 参数研究用独立 variant；租用 CUDA GPU 时指定实际设备。
python scripts/benchmark.py plan B0 --protocol p150u --variant lr-study --device cuda:0 \
  -u optimizer.lr=0.00025

# 导出逐次运行、逐权重类型的 CSV；指标沿用 eval.jsonl 的 0–1 单位。
python scripts/benchmark.py summarize -o benchmark_results.csv

# 无 torch / GPU 测试，全部数据位于独立临时目录。
python scripts/test_benchmark.py
```

`--data-root`、`--weights-dir` 可替代对应环境变量。已有运行未提供 data-root
时复用其清单路径；显式指定不同路径会触发一致性拒绝。`plan` 不检查数据或
权重是否实际存在；`run` 会检查，并校验标注与初始化权重 SHA256。E2/E3
缺失权重会拒绝运行，不会悄悄退回冷启动。E1 仍可能加载 backbone 预训练权重。

## 身份、快照与恢复

```text
outputs/benchmark/B0-v1/p150u/r640/default/e2_s42/
  benchmark.json                 # 方法、协议、配置、代码、输入指纹
  snapshot/benchmark_config.yml  # 最终合并且展开尺寸后的启动配置
  snapshot/source_files.json     # 源文件逐个 SHA256
  snapshot/source.tar.gz         # 实际源文件，包括未跟踪的新模块
  snapshot/config_merged.yml     # 现有启动器的实际配置快照
  DONE / train.log / eval.jsonl / checkpoints...
```

身份包含方法编号与版本、协议、分辨率、variant、初始化、seed；清单另外记录
device、weights_dir。完整配置指纹包含优化器、模块参数和数据路径等。相同身份
的配置、源代码/环境锁文件、输入权重或标注改变时拒绝恢复，`DONE` 后终评也
先检查。目录锁阻止同一个运行并发启动。扩展失败会报错，不跳过检查。

代码快照范围为 `engine/**/*.py`、所有配置、训练/管理/终评入口和 pixi 文件；
不归档图片、权重、结果目录或完整 Python 环境。图片内容和系统驱动没有逐项
指纹，因此实验时仍应固定数据集版本与硬件环境。当前源指纹保守地包含整个
配置树，修改其他实验配置也会要求新的 variant；它不自动判断语义等价。

请始终通过 `benchmark.py run` 恢复受管理实验，直接调用旧启动器会绕过这些
新增检查。已有 `outputs/p30u` 等历史目录不自动收编；汇总只扫描带
`benchmark.json` 的新目录。

## 后续模块如何插拔

1. 在模型侧实现实际模块及显式配置参数；关闭时保持原始前向及损失。建议
   接入点是 encoder 输出、decoder 输入之间。B0 不增加辅助分支。
2. B1 先实现无增强辅助分支；再逐步加入全图平滑、HBS 和 API。辅助训练路径
   可调用共享 decoder，但需明确 CDN、查询选择、匹配和损失计数，推理只走
   原路径。API 是查询式适配，不能直接把 FCOS 版本当作已复现。
3. 将真实构造器参数写入登记条目的 `overrides`，不要只改 `modules` 标签。
   固定模块实现版本 `revision`，同时记录论文/参考代码与适配差异。
4. 验证关闭等价性、输出形状、有限损失与梯度、训练/推理路径、保存恢复；
   验证后将条目标记 ready，并更新 `benchmark.py` 的 `IMPLEMENTED`。
5. 同一初始化、协议、输入尺度、batch 和 seed 下运行消融。改超参数用独立
   variant，改方法语义用新 revision；先短程筛选，再按相同正式协议多 seed
   对比。P30u/P74u/P150u 分开比较，P150u 的早停也属于协议的一部分。

汇总保留 `final_ema`、`final_raw`、`best_ema` 三类行、实际 checkpoint 路径、
协议及指纹，不跨组自动求均值。`best_ema` 在 val 上选取，只作明确标注的
参考；`final_*` 可能来自旧启动器的 last.pth 回退，应检查 ckpt 列。
