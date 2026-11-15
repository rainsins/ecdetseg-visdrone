# CPU / CUDA / XPU 运行与迁移

更新：2026-11-15。范围是设备、AMP、恢复和启动链修复；SET/HBS/API 仍未实现。

## 设备与精度

`engine/device.py` 集中解析并激活设备。`auto` 按 CUDA → XPU → CPU 选择；
显式 `cpu`、`cuda:0`、`xpu:0` 不可用时直接失败，不回退。模型、scaler、
日志、同步、内存探针与分布式使用同一设备。算法层继续继承输入 tensor 的设备。

| precision=auto，use_amp=true | autocast | GradScaler 默认 |
|---|---|---|
| CPU | 关闭，FP32 | 关闭 |
| CUDA | FP16 | 开启 |
| XPU | BF16 | 关闭，保留原基线策略 |

训练入口支持 `--precision auto/fp32/fp16/bf16` 或顶层配置 `precision`。
`use_amp=false` 只接受 auto/fp32；显式 fp16/bf16 与之冲突会报错。
`scaler.enabled=false` 只关闭梯度缩放，不关闭 autocast；FP32 禁止开启缩放。
显式 BF16 检查后端支持，预检执行实际矩阵运算；模型算子兼容性另由短测验证。
criterion 保留原来的 autocast-disabled 路径，不宣称其全部输入已转为 FP32。
验证与终评保持原 FP32 前向，加载 checkpoint 时不恢复训练 scaler。

CLI `train.py --device auto` 可自动选择；历史 launcher、benchmark、PM2 的默认仍为
`xpu`，避免旧命令在混合机器上悄悄换后端。租卡时必须显式指定 CUDA。
torchrun 使用无索引设备 `--device cuda`/`xpu`，按 LOCAL_RANK 激活；显式索引
冲突会报错。CPU/CUDA/XPU 分别选 gloo/nccl/xccl，初始化失败向外传播。
当前未进行多卡或多节点实测；对应后端必须由所装 PyTorch 提供。

## 环境隔离

现有 `pixi.toml` / `pixi.lock` 保持原 XPU 环境。切换 `--device` 不能把 XPU wheel
变成 CUDA wheel。使用独立 venv，运行时通过绝对可执行文件 `RUNTIME_PYTHON` 选择。
该解释器贯穿配置展开、运行预检、训练、checkpoint 检查、终评、绘图和执行记录。
默认不设置时仍使用原 Pixi 训练环境；配置工具仍使用系统 Python。

以下在仓库根目录执行，使用 Python 3.11。PyTorch 选 2.9.1、torchvision 选 0.24.1，
用于对齐本项目框架版本，而非宣称它们是最新版。官方安装表列出了这对版本的
CPU、cu126、cu128、cu130 wheel。此处 CUDA 12.6 仅为安装示例，先核对实际卡的
架构、驱动和 wheel 支持；不能只看 `nvidia-smi` 的 CUDA Version 选择 wheel。
NVIDIA 的 minor-version compatibility 存在功能限制，不能用最低驱动门槛
替代目标机器的算子短测。未在本机安装新的 CUDA/CPU 环境或系统驱动。

```bash
# CUDA 示例：确认目标卡/驱动支持该 wheel 后执行
python3.11 -m venv "$HOME/venvs/edgecrafter-cu126"
export RUNTIME_PYTHON="$HOME/venvs/edgecrafter-cu126/bin/python"
"$RUNTIME_PYTHON" -m pip install --upgrade pip
"$RUNTIME_PYTHON" -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu126
"$RUNTIME_PYTHON" -m pip install -r ecdetseg/requirements/runtime-common.txt \
  -c ecdetseg/requirements/runtime-torch-constraints.txt
"$RUNTIME_PYTHON" -m pip check

# CPU：独立环境，不覆盖 CUDA 或当前 Pixi 环境
python3.11 -m venv "$HOME/venvs/edgecrafter-cpu"
export RUNTIME_PYTHON="$HOME/venvs/edgecrafter-cpu/bin/python"
"$RUNTIME_PYTHON" -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cpu
"$RUNTIME_PYTHON" -m pip install -r ecdetseg/requirements/runtime-common.txt \
  -c ecdetseg/requirements/runtime-torch-constraints.txt
"$RUNTIME_PYTHON" -m pip check

# 现有 XPU 环境
pixi run --frozen python ecdetseg/scripts/runtime_preflight.py --device xpu
```

公共 requirements 不是完整 lock，尚未验证全新环境的依赖解析。安装后保留
`pip freeze`、预检结果并运行短测；torch constraints 用于阻止公共依赖升级框架。
本次官方资料核对日期：2026-11-15，来源：

```text
https://pytorch.org/get-started/previous-versions/#v291
https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html
```

## 租卡流程

1. 将当前修复后的代码（包括未跟踪的新文件）复制到租卡机器；仅 clone 当前 HEAD
   不会包含尚未提交的修改。配置数据和初始权重所在的绝对路径。
2. 在隔离环境中安装匹配的 wheel 和公共依赖，检查 `nvidia-smi` 与 `pip check`。
3. 执行预检和真实数据短测。短测用独立新目录，默认 320×320/batch2，
   只训练两步，中间保存/回载 model、EMA、optimizer、scaler，随后验证一批。
   这是诊断配置，不是 P150u 的精度或吞吐结果；`--res 640` 可进一步检查目标尺寸。

```bash
export RUNTIME_PYTHON="$HOME/venvs/edgecrafter-cu126/bin/python"
export VISDRONE_ROOT=/data/VisDrone
export WEIGHTS_DIR=/data/initial_weights
cd ecdetseg
"$RUNTIME_PYTHON" scripts/runtime_preflight.py --device cuda:0 --precision fp16
"$RUNTIME_PYTHON" scripts/smoke_runtime.py --device cuda:0 --precision fp16 \
  --data-root "$VISDRONE_ROOT" --weights "$WEIGHTS_DIR/ecdet_s.pth" \
  --out /data/smoke-cuda-first

# 预览正式 P150u，保持640、batch10/32、原早停
"$RUNTIME_PYTHON" scripts/benchmark.py plan B0 --device cuda:0 --precision fp16 \
  --runtime-python "$RUNTIME_PYTHON" --variant cuda-fp16 --seed 42
# 短测通过、资源确认后执行正式训练
"$RUNTIME_PYTHON" scripts/benchmark.py run B0 --device cuda:0 --precision fp16 \
  --runtime-python "$RUNTIME_PYTHON" --variant cuda-fp16 --seed 42
```

重复完全相同的 run 命令恢复该运行。variant 必须独立；更改设备、精度、源码、
环境或配置时，不通过覆盖指纹继续写旧目录。原 launcher 可用
`EXP_DEVICE=cuda:0 EXP_PRECISION=fp16 EXP_OUT_ROOT=/data/new-runs bash scripts/run_exp.sh e2 42`。
RUNTIME_PYTHON 必须同时 export；PM2 已转发这三个选择项。

退租前取回整个运行目录（checkpoint、eval、日志、snapshot、runtime.json、
resume_history），以及准确代码版本/未提交补丁和新文件、环境 freeze。
显存不足时不要静默缩小 batch、输入或 query；改变训练协议需要独立实验。
新方法与 B0 应在同硬件、软件、精度和协议下配对比较，不能直接将历史 XPU
基线与 CUDA 新方法合并为严格受控对照。

## 恢复与来源保护

- 新 checkpoint 保存 runtime_policy：backend、实际 precision、scaler_enabled。
  训练恢复遇到不一致时拒绝；模型权重评估/新实验 tuning 可以跨设备。
- launcher 先核对配置，再预检实际解释器/设备，写 runtime.json：实际设备名、
  显存、torch/torchvision/Python、CUDA runtime、主要包版本和时间。
  再次使用同目录时核对这些关键环境字段。该文件不声称覆盖所有系统依赖。
- legacy checkpoint 缺少策略时，launcher 要求旧 train.log/resume_history 的
  设备证据和旧快照精度策略一致；没有证据时拒绝。新记录仅证明当前运行，
  不补造历史源码/软件一致性。直接 train.py 恢复 legacy checkpoint 也要求
  launcher 已验证的 runtime.json；否则以 tuning 在新目录运行。
- `-t /path/to/checkpoint` 是新实验权重初始化，不恢复原优化器轨迹。跨后端
  迁移不承诺逐位一致；本轮不提供绕过保护的迁移恢复开关。

## 实际验证记录

2026-11-15，现有 Pixi：torch 2.9.1+xpu / torchvision 0.24.1+xpu。

| 验证 | 实际结果 |
|---|---|
| CPU FP32，真实 VisDrone、ECDet-S/E2、320/batch2 | 两次优化、checkpoint 回载、一批 EMA 验证通过 |
| Intel Arc A770，XPU BF16，相同诊断规模 | 同上；scaler 关闭，有限损失通过 |
| CUDA | 无可用实机；仅设备/AMP/DDP 分支模拟和启动链参数测试 |
| 分布式 | 后端选择、本地索引和初始化失败传播测试；未实测多卡 |
| 独立 CPU/CUDA wheel 环境 | 安装说明已核对官方资料，尚未新建环境验收 |

短测记录位于 `/tmp/opencode/edgecrafter-{cpu,xpu}-runtime-20261115/smoke.json`；
CPU loss 为 43.2147/42.6332，XPU 为 43.0199/46.0307，仅检查有限值及参数更新，
不把两步 loss 变化或子集 AP 当成训练收益。CPU 是在现有 XPU wheel 中运行 CPU
设备，不是独立 CPU wheel 的安装验证。历史 P150u 训练输出未作为短测目录。

回归命令（仓库根目录）：
本次结果：40 项 Python 测试通过，23 项 shell 启动器检查通过；
65 个 Python 文件语法解析、启动脚本 bash 语法与 git diff --check 均通过。
设备共存、CUDA 分支与启动链中的 GPU 调用使用模拟，不能替代 CUDA 实机验证。

```bash
pixi run --frozen python -m unittest discover -s ecdetseg/scripts -p 'test_*.py'
bash ecdetseg/scripts/test_run_exp.sh
bash -n ecdetseg/scripts/run_exp.sh ecdetseg/scripts/run_all.sh
git diff --check
```

## 改动位置

- `engine/device.py`、`engine/optim/amp.py`：共享设备和混合精度策略。
- `engine/core/{_config,yaml_config,yaml_utils}.py`：绑定实际配置；修复默认可变
  dict 导致连续创建配置时残留上次 device/precision 的问题。
- `engine/misc/dist_utils.py`、`train.py`：实际设备、DDP、本地 rank、同步和日志。
- `engine/solver/{_solver,ec_engine}.py`：运行策略、恢复检查、FP32 autocast 禁用。
- `scripts/{run_exp.sh,run_all.sh,benchmark.py,eval_final.py,record_execution.py}`、
  `ecosystem.config.js`：解释器与设备贯穿启动、恢复、终评和元数据。
- `scripts/{runtime_preflight,smoke_runtime,test_device_runtime}.py`、
  `scripts/test_benchmark.py`、`requirements/`：预检、诊断短测、回归和安装约束。
