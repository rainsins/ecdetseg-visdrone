# ECDet-S → ONNX → RKNN INT8 → 仿真评测

> 板端/接口契约的单一事实来源见 [`MODEL_CONTRACT.md`](MODEL_CONTRACT.md)。
> 2026-11-07：原 `.rknn` + `--grid-sample-cpu` 已通过 RK3588 真机 548 图验收，AP **28.7692**，相对 FP32 下降 **0.0516**；报告为 `artifacts/board_eval_summary.json`。633.577 ms/图为整次评测平均值。本轮单图 profiling 中，CPU 回调累计约占 infer 的 **6.6%**；独立逐层报告中最大耗时为 NPU `exSDPAttention`（**43.24%**）。显式 GPU 子图已成功编译路由，但当前板端运行库报不支持 GPU GridSample，未成功执行。8 个有界子图编译实验均通过，尚未复现 Gather 整网挂起。修复、计时及诊断边界见 [`RK3588_REPAIR.md`](RK3588_REPAIR.md)。

本目录是论文实验的部署流水线，目标平台 **RK3588 / INT8**，模型为
`ECDet-S`（ViTAdapter + HybridEncoder + ECTransformer，VisDrone 10 类，
300 queries，640×640 静态输入）。

> 本目录不修改训练代码。ONNX 导出直接复用仓库已有的
> `cfg.model.deploy()`（`engine/edgecrafter/modeling.py`），没有 `deploy()`
> 的分支才需要写包装类；这里仓库已有，因此只加了一个把 dict 输出变成
> 具名 tuple 的 `DetectorExport` 包装。

---

## 1. 流水线顺序

```
export_onnx.py  →  make_calib.py  →  preprocess.py（契约自检）
      →  convert.py  →  sim_eval.py
```

| 顺序 | 文件 | 作用 | 运行环境 |
|---|---|---|---|
| — | `eval_pytorch.py` | PyTorch 参考评测：同一份 stretch 预处理 + numpy 后处理；用于和 INT8 对齐（不在 `run_all.sh` 里） | `TRAIN_ENV` |
| 1 | `export_onnx.py` | 探测 EMA/raw ckpt，构建 deploy 模式模型，静态导出 ONNX（opset 17），`onnx.checker` + ORT 数值对齐 | `TRAIN_ENV` |
| 2 | `make_calib.py` | 按 10 类实例数分层抽 300 张（seed=42），直接 stretch 到 640×640 后保存 PNG；自动剔除会让 query-selection gather 出现 inf 的图并同层补样 | `CALIB_ENV` |
| 3 | `preprocess.py` | 唯一预处理实现：direct stretch → BGR→RGB → CHW，不做 mean/std；同时是板端对拍 oracle | `CALIB_ENV` |
| — | `postprocess.py` | numpy sigmoid + topk300 + cxcywh→xyxy + stretch 坐标反算 | 任意 |
| 4 | `convert.py` | `rknn.config(mean/std/rk3588)` → `load_onnx` → `build(int8)` → `.rknn` | `RKNN_ENV` |
| 5 | `sim_eval.py` | 仿真器逐张跑 val 548 张，COCOeval 与参照基线对比，单边门限结论文本化 | `RKNN_ENV` |
| — | `accuracy_analysis.py` | 单张代表图的逐层量化误差 snapshot | `RKNN_ENV` |
| — | `run_all.sh` | 顺序执行 1→5，失败即停，最后打印 `RESULT` / `GATE` | 宿主 shell |
| — | `pixi_feature.toml` | 可合并进 `pixi.toml` 的 rknn feature 片段 | — |

`deploy/__init__.py` 只放“项目事实”和路径/日志工具，使整个目录可 `import`。
`deploy/cpp/` 是板端 C++ 实现，用 CMake 构建，不参与这条 Python 链路。

---

## 2. 环境准备

### 2.1 合并 rknn feature

`pixi.toml` 没有 include 语法，需要手工把 `pixi_feature.toml` 里的
`[feature.rknn]` 和 `[environments].rknn` 两段复制进去，然后：

```bash
pixi install -e rknn
pixi run -e rknn python -c "from rknn.api import RKNN; print(RKNN)"
```

`rknn-toolkit2` 的 x86 wheel 不一定在默认 PyPI 镜像里。若 pixi 报
`No matching distribution`，从 Rockchip 的
`airockchip/rknn-toolkit2/packages` 下载与本机 Python 匹配的
`cp310/cp311 manylinux` wheel，然后把 `pixi_feature.toml` 中的 pypi 条目换成
本地路径：

```toml
rknn-toolkit2 = { path = "packages/rknn_toolkit2-2.3.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" }
```

当前工作区实际只有 `default` 环境时，导出/标定可以用：

```bash
TRAIN_ENV=default RKNN_ENV=rknn ./deploy/run_all.sh
```

### 2.2 路径

所有脚本都可 `--help`。默认路径从 `deploy/` 反推工作区，不写死绝对路径：

- ECDet 代码：自动识别 `<workspace>/ecdetseg` 或扁平布局
- 配置：`ecdetseg/configs/ecdet/ecdet_s_p150u.yml`
- 权重：`ecdetseg/outputs/p150u/e3_s42/best_ema.pth`；若该文件名不存在则自动回退到同目录 `best.pth`（同样是选中的 EMA 权重，日志会告警）
- 数据：依次尝试 `<workspace>/dataset/VisDrone`、
  `<workspace>/../dataset/VisDrone`、`<cwd>/dataset/VisDrone`
- 产物：`deploy/artifacts/`

数据路径不确定时显式传：

```bash
--data-root /path/to/VisDrone
```

---

### 2.3 PyTorch stretch 基线

```bash
pixi run -e default python deploy/eval_pytorch.py \
  --config ecdet_s_p150u.yml \
  --weights best.pth \
  --preprocess stretch \
  --ann annotations_VisDrone_val.json
```

文件名会自动从 `ecdetseg/configs/ecdet/`、`ecdetseg/outputs/p150u/e3_s42/`、
`<data-root>/` 回退解析。`--preprocess stretch` 与 RKNN 流水线使用同一
PIL bilinear resize；全量 548 张实测 AP=28.82。`--keep-lqe` 用于保留原始
LQE 分支（当前 `run_all.sh` 默认开启并配合 auto_hybrid）。

## 3. 一键执行

```bash
./deploy/run_all.sh
```

或指定环境/数据：

```bash
TRAIN_ENV=default RKNN_ENV=rknn \
./deploy/run_all.sh --data-root /path/to/VisDrone --num-calib 300
```

三个环境变量各管一段，默认值 `TRAIN_ENV=default`、`CALIB_ENV=$RKNN_ENV`、
`RKNN_ENV=rknn`：

| 变量 | 用途 | 需要的包 |
|---|---|---|
| `TRAIN_ENV` | ONNX 导出 | torch、onnx |
| `CALIB_ENV` | 校准 PNG + 预处理参考张量 | cv2、numpy、Pillow |
| `RKNN_ENV` | 量化、仿真评测 | rknn-toolkit2 |

`CALIB_ENV` 默认等于 `RKNN_ENV` 而不是 `TRAIN_ENV`：校准图、板端参考张量和
仿真评测的输入都由 `deploy/preprocess.py` 的 PIL 双线性重采样生成，三者必须是同一
份 Pillow 构建，否则校准分布和推理分布会被不同版本重采样器"错位"改写。Pillow 12 与
13 的 pass-order 规则不同（`200x900`、`1000x3000` 的输出逐字节不同），所以
`run_all.sh` 在启动时会读取两个环境的 `PIL.__version__`，不一致直接 FATAL 退出，
而不是让它静默影响 mAP。VisDrone 的三种真实尺寸（1360×765、960×540、1920×1080）
在这两个版本下恰好一致，正因如此，这种错位更难被发现。

逐步执行：

```bash
# 2. ONNX
pixi run -e default python deploy/export_onnx.py \
  --config ecdetseg/configs/ecdet/ecdet_s_p150u.yml \
  --weights ecdetseg/outputs/p150u/e3_s42/best_ema.pth

# 3. 校准集
pixi run -e rknn python deploy/make_calib.py --num 300 --seed 42

# 4. 预处理自检：生成参考张量，或与板端 dump 对拍
#    必须用原始分辨率的图，640x640 的校准 PNG 不会触发 resize 路径。
pixi run -e rknn python deploy/preprocess.py \
  --image /path/to/VisDrone/val/0000001_02999_d_0000005.jpg \
  --output-npy /tmp/ecdet_input.npy \
  --output-meta /tmp/ecdet_meta.json \
  --require-resize

# 用板端产出的张量对拍（C++ 侧 --dump-npy 写出）
pixi run -e rknn python deploy/preprocess.py \
  --image /path/to/VisDrone/val/0000001_02999_d_0000005.jpg \
  --compare-candidate /tmp/board_input.npy \
  --require-resize

# 6. RKNN
pixi run -e rknn python deploy/convert.py

# 7. 全量仿真（548 张；调试可加 --limit 10）
pixi run -e rknn python deploy/sim_eval.py --fp32-baseline --hybrid-cos-thresh 0.9999

# 8. 掉点诊断
pixi run -e rknn python deploy/accuracy_analysis.py
```

`sim_eval.py` 的门限是**单边**判定：只有 `INT8 AP` 比参照基线**低得超过**
`--fail-drop`（默认 1.0）时才失败。INT8 反超基线不算失败——双边绝对差会把它判成失败，
而这显然是错误结论。结论写在 summary 的 `gate` 块里（`passed` / `drop` /
`fail_drop` / `reason`），`run_all.sh` 结束时会打印一行 `GATE : PASSED|FAILED`。

参照基线由 `--fp32-baseline` 决定：开启时用同一份 stretch 预处理的 FP32 ONNX，
关闭时退回 `PYTORCH_REFERENCE`。前者隔离量化损失，后者混入了导出/图优化差异。

---

## 4. 关键设计说明（论文里可直接引用）

### 4.1 为什么后处理留在 ONNX 图外

导出的图只有：

```
images [1,3,640,640] float32
  ├─ pred_logits [1,300,10] float32   # sigmoid 前
  └─ pred_boxes  [1,300,4]  float32   # 归一化 cxcywh
```

最终 `sigmoid → flatten → topk300 → box_convert → gather` 在
`postprocess.py` 中用 numpy 完成，原因：

1. TopK/Sort/Gather 在 RKNN INT8 中支持差、动态输出难被板端消费；
2. 板端通常还要按业务阈值筛选，放在图外可调而不必重新转换；
3. 图保持“图像 → 原始 decoder 输出”的纯函数，逐层量化误差才可解释。

注意：D-FINE 解码器内部还有一次 encoder query selection TopK，它是模型语义的
一部分，**无法去掉**；这里排除的是最终检测后处理 TopK。

### 4.2 为什么归一化走 RKNN config

`preprocess.py` 输出 **uint8 RGB 0..255**，不做 `/255`，不减 mean、不除 std。
归一化由：

```python
rknn.config(
    mean_values=[[123.675, 116.28, 103.53]],   # ImageNet mean × 255, RGB
    std_values=[[58.395, 57.12, 57.375]],      # ImageNet std  × 255, RGB
    quant_img_RGB2BGR=False,
)
```

统一完成。如果 Python 再做一次，就会二次归一化，INT8 校准统计全错；同时板端
可以直接喂相机 uint8 buffer，少一次浮点转换。

### 4.3 为什么校准集要分层

VisDrone train 实例数极不均衡（car 144,867 vs awning-tricycle 3,246）。
纯随机 300 张可能完全抽不到稀有类，导致这些类别的 head 量化范围几乎没数据。
`make_calib.py`：

- 以每张图的“主导类别”分层（一张图只属于一个层，避免重复）；
- 按各类实例数用 D'Hondt 分配 300 个配额，每类至少 1 张；
- 层内用 `random.Random(42)` 抽样；
- 校准图先做与推理完全相同的 direct resize（stretch 640×640，无 pad），再存无损 PNG；
- 用文件名集合断言 train/val 无交集；
- 打印每类配额、被抽中图片数、实例数，并写 `calib_manifest.json`。

### 4.4 为什么预处理仍保留 scale/pad 字段

stretch 没有 padding，也没有 per-axis letterbox scale；`pad_x/pad_y=0`、
`scale_x/scale_y=1.0`。`postprocess.py` 不再使用这些字段，而是按 PyTorch
参考实现把归一化坐标直接乘以 `orig_w/orig_h`。保留字段只是为了不破坏
`preprocess_reference.json` / 板端旧调用方的 JSON 契约。

### 4.5 与 PyTorch 参考值的口径

参考值来自 `outputs/p150u/e3_s42/eval_best_ema.json`：

```
AP=28.82  AP50=48.54  AP75=28.63
```

`sim_eval.py` 使用与仓库 `coco_summarize` 一致的 12 项统计，`maxDets` 末位取
300；原生 pycocotools 的 AP 行硬编码 100，会得到不同数值，不能直接比较。

PyTorch val 和部署流水线现在都使用直接 stretch 到 640×640，因此可以直接和
28.82 比较。`sim_eval.py` 仍加 `--fp32-baseline`：用同一份 stretch 预处理跑
FP32 ONNX，既验证 ONNX 与 PyTorch 对齐，也把量化损失和导出/运行时差异分开；
门限是单边的 `INT8 AP` 比参照基线低超过 `--fail-drop`（默认 1.0）才判失败。
`summary JSON` 的历史字段名 `fp32_letterbox_baseline` / `drop_vs_fp32_letterbox`
保持不变，但值是 stretch 基线，不是 letterbox。

本机实测（548 张 val，seed=42，300 张校准）：

```
PyTorch(stretch)   AP = 28.82
FP32 ONNX(stretch) AP = 28.82
RKNN INT8(stretch) AP = 28.71
INT8 - FP32           = -0.11
```

即真实量化损失仅 0.11 点。归档的原始构建测得 28.69（drop 0.13），与本次重建差
0.02——同一份校准集、同一个 seed，属于量化过程的批次噪声。`.rknn` 不保证字节
可复现，两次构建的 sha256 不同是预期行为，不要拿哈希当回归判据。

### 4.6 为什么最终保留 LQE

早期 letterbox 实验里，LQE 的 33-bin TopK/Softmax 被 RKNN 量化得很差，分数
全部压到 0.01，因此曾临时关闭 LQE。stretch 对齐 + `auto_hybrid` 0.9999 后，
LQE 可以正常量化；关闭 LQE 的 FP32 上限只有 28.29，无法达到 28.6 的目标。
最终 `run_all.sh` 导出时显式传 `--keep-lqe`，并用
`--hybrid-cos-thresh 0.9999` 让敏感层回退 FP16。

### 4.7 为什么 clamp encoder score head

部分 VisDrone train 帧的 FP32 encoder 分数会低到 -310；如果 RKNN 用这个范围
标定 score head，正样本峰值（约 +2.6）会被压成 0，300 个 query 全部选错。
`export_onnx.py` 默认对 score head 的输入 clamp 到 ±10、输出 clamp 到 ±30，
并在最终 logits 上再 clamp ±30。这个操作改变 topk 选择，FP32 AP 从 28.19 提升到 28.82，同时把
量化范围限制在可用区间。

### 4.8 为什么校准集要剔除 non-finite 图

即使加了 clamp，仍有个别 train 图在 `GatherElements_output_0`（query selection
后的 anchor gather）产生 +inf。只要校准集里有 1 张这样的图，该层 scale 就会
变成 inf，整个 INT8 模型输出接近 0。`calib.py` 在生成 300 张后，用 FP32 ONNX
逐张检查这个张量；发现 inf/NaN 就按“同 dominant class”补一张新的 train 图，
保持 300 张和分层配额不变。stretch + PIL 下当前 seed=42 会替换 1 张
（`0000790...` -> `0001635...`，日志会打印）。
关闭该行为用 `--no-nonfinite-filter`。该过滤在训练环境里需要 `onnx` 和
`onnxruntime`（本项目默认环境已包含）；缺依赖时脚本会告警并跳过，此时务必确认
`calib.log` 里没有 `non-finite query-selection` 记录。

### 4.9 为什么开启 auto_hybrid

D-FINE 的 logits 极稀疏，纯 w8a8 会把少数正峰裁掉。`convert.py` / `sim_eval.py` /
`run_all.sh` 使用 `auto_hybrid=True, auto_hybrid_cos_thresh=0.9999`：这个参数是
**cosine 距离**阈值（RKNN 默认 0.98），量化前后输出向量的距离超过它时，该敏感算子
回退 FP16（RK3588 支持 FP16）。取值越大越激进，0.9999 几乎等于"只要有一点偏差就回退"，
代价是模型从约 37 MiB 增至约 49 MiB；反过来把它当成相似度阈值来理解会得到相反结论。
`--no-auto-hybrid` 可关闭，仅用于对比实验。

---

## 5. 产物

```
deploy/artifacts/
├── ecdet_s_p150u_640.onnx
├── ecdet_s_p150u_640_meta.json          # 输入/输出名、shape、mean/std、参考值
├── calib.txt                            # 一行一个相对路径
├── calib_imgs/                          # 300 张 stretch 640 PNG
├── calib_manifest.json                  # 分层抽样审计
├── preprocess_reference.npy / .json     # 板端自检参考张量（全分辨率 val 图）
├── ecdet_s_p150u_640_rk3588_int8.rknn
├── ecdet_s_p150u_640_rk3588_int8.rknn.json  # 转换配置 + sha256
├── detections_sim.json                  # COCO detection 列表
├── sim_eval_summary.json                # 指标、掉点、耗时
└── logs/*.log                           # run_all 每步完整日志

deploy/snapshot/
├── input.npy / input.png
└── error_analysis.txt                   # 逐层误差（版本不同文件名略有差异）

deploy/cpp/                              # 板端实现，CMake 构建，不产出到 artifacts/
├── pil_resize.h/.cpp                    # 与 PIL 逐比特一致的双线性重采样
├── rknn_engine.h/.cpp                   # RKNPU2 封装（rknn_inputs_set 路径）
├── postprocess.h/.cpp                   # 与 postprocess.py 逐条对齐
├── main.cpp                             # CLI：--timing / --dump-npy / --dump-dets
├── CMakeLists.txt
└── tests/                               # 宿主机对拍 CLI（-DEC_BUILD_TESTS=ON）
```

`sim_eval.py` 预期控制台输出（`--fp32-baseline`，2026-10-27 全量 548 张实测）：

```
metric    PyTorch(stretch)  FP32(stretch)   RKNN INT8      INT8-FP32
AP                   28.82            28.82       28.71          -0.11
AP50                 48.54            48.54       48.51          -0.03
AP75                 28.63            28.63       28.39          -0.24
```

`run_all.sh` 结尾打印：

```
RESULT: PyTorch(stretch) mAP=28.82 | RKNN INT8(stretch) mAP=28.71 | INT8-FP32=-0.11
GATE  : PASSED -- INT8 AP 28.71 is 0.11 points below FP32 stretch baseline 28.82 (within limit 1.00)
NOTE  : INT8 mAP came from an in-process rebuild, not from load_rknn of the shipped .rknn.
```

`INT8-FP32` 是 `INT8 - FP32`，负值表示 INT8 略低；它和 `GATE` 用的是同一个参照
（`gate.reference_AP`），所以两行不会互相矛盾。`GATE` 是按 `--fail-drop` 单边判定的
结论，不是 `RESULT` 数字的隐含推论——反超基线时差值为正但依然通过。`NOTE` 只在仿真走了
进程内 rebuild 时出现（RKNN 2.5 的模拟器不接受 `load_rknn`），提醒这段 mAP 不是直接对
交付的 `.rknn` 测出来的。

---

## 6. 常见报错对照表

| 现象 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: rknn` | 在 training env 里跑了 convert/sim | 用 `pixi run -e rknn ...`；确认 rknn feature 已合并并安装 |
| `ModuleNotFoundError: pkg_resources` | setuptools>=81 不再带 pkg_resources | 已固定 `setuptools<80.1`；重新 `pixi install -e rknn` |
| `load_rknn ... not support inference on the simulator` | RKNN 2.5 模拟器不接受 load_rknn | `sim_eval.py` 已自动改为进程内 `load_onnx + build`；不要加 `--no-rebuild` |
| 仿真跑很久、看起来卡住 | INT8 分支要走进程内 rebuild（load_onnx + 量化 + build），实测约 11 分钟，之后才是逐张推理 | 不要套 `timeout 600` 之类的上限；真被杀了会没有 `sim_eval_summary.json`，前四步产物仍然可用，重跑第 5 步即可 |
| `No matching distribution found for rknn-toolkit2` | Rockchip wheel 不在默认 PyPI | 下载匹配 cp310/cp311 的 manylinux wheel，用 `path =` 安装 |
| `numpy.core.multiarray failed to import` / ABI 错误 | rknn env 混入 numpy>=2 | 固定 `numpy>=1.26,<2`，重建 rknn 环境 |
| `Unsupported op` / `Not support op` | ONNX 含 RKNN 2.3 未实现算子 | 保持 `--opset 17`；检查导出图算子；简化或替换不支持的算子 |
| `dataset ... not exist` / calib 报错 | `calib.txt` 路径基准错误 | 必须写相对路径；`convert.py` 会 chdir 到 `calib.txt` 所在目录 |
| `The input size is not match` | 静态 shape / 输入名不一致 | ONNX 必须是 `[1,3,640,640]`，输入名 `images`；不要开 dynamic axes |
| 仿真输入 shape 报错 | `data_format` 与 tensor layout 不一致 | 本流水线固定 `data_format="nchw"`，输入 `(1,3,640,640)` uint8 |
| 仿真结果全是低分/乱框 | RGB/BGR 或 mean/std 顺序错 | preprocess 输出 RGB；config mean/std 也是 RGB；`quant_img_RGB2BGR=False` |
| 只掉几个点但 AP50 正常 | 正常的 INT8 量化误差 | 先看 `sim_eval_summary.json` 的 APs/APm/APl，再跑 accuracy_analysis |
| AP 大幅下降（明显超过 `--fail-drop` 的 1.0，且 FP32 stretch 也正常） | 校准集含 non-finite 图 / 某层量化溢出 | 看 `calib.log` 是否替换过图；跑 `accuracy_analysis.py`；确认 auto_hybrid 开启 |
| 全部输出分数 ~0.01 | query selection 被量化破坏 | 确认 ONNX 默认关闭 LQE、score head clamp、auto_hybrid=True |
| 校准日志出现 `non-finite query-selection` | 该图在 GatherElements 产生 inf | `make_calib.py` 已自动同层替换；不要用 `--no-nonfinite-filter` |
| `accuracy_analysis` 不认 `.npy` | 旧版 toolkit 只吃图片路径 | 脚本会自动回退 `input.png`；仍失败时升级到 2.3.x+ |
| 仿真很慢 | x86 模拟器逐层模拟 INT8，不是板端速度 | 用 `--limit 10` 先打通；性能数据必须在 RK3588 板端测 |
| `best_ema.pth` load 报 missing/unexpected | 选错容器或结构与配置不符 | 先看日志里的 top-level keys；用 `--weights-key ema|model|raw`；本仓库若无 `best_ema.pth` 会自动用 `best.pth` |
| ckpt 加载时联网下载 backbone | ViTAdapter 初始化触发了预训练下载 | 导出脚本已设置 `skip_load_backbone=True`；确认配置里有 `ViTAdapter` 段 |

---

## 7. 板端实现

板端代码已经交付在 [`deploy/cpp/`](cpp/)，不是待移植的伪代码：

| 文件 | 对应 Python | 一致性保证 |
|---|---|---|
| `pil_resize.h/.cpp` | `preprocess.py` 的 `Image.resize(BILINEAR)` | 逐比特一致（560 用例 `max_abs_diff == 0`：548 张真实 val 帧 + 12 个合成尺寸，含 vertical-first 分支） |
| `rknn_engine.h/.cpp` | 无（直接对 `rknn_api.h`） | 只调用头文件声明的符号，全量返回码检查 |
| `postprocess.h/.cpp` | `postprocess.py::postprocess` | 逐条一致（`maxdiff ≈ 5e-7`，标签零错配） |
| `main.cpp` | `sim_eval.py` 的单帧路径 | 输出可被 `preprocess.py --compare-candidate` 验收 |

构建（板端本机）：

```bash
cd deploy/cpp
cmake -S . -B build -DEC_ENABLE_GRID_SAMPLE_CPU=ON && cmake --build build -j
./build/ec_infer --model ecdet_s_p150u_640_rk3588_int8.rknn \
                  --image frame.jpg --grid-sample-cpu --timing
```

板端不做 `mean/std`、不做 `/255`——这些已经在 `.rknn` 的 `mean_values` /
`std_values` 里，C++ 侧再算一次就会二次归一化，INT8 校准统计全错。

CMake 需要 `librknnrt.so` + `rknn_api.h`。板端镜像通常已经把两者装在
`/usr/lib` 和 `/usr/include`，此时无需任何参数；交叉编译或非标准布局时按优先级传：

```bash
# 1) 前缀目录，期望 <root>/include/rknn_api.h + <root>/lib/librknnrt.so
cmake -S . -B build -DEC_RKNN_ROOT=/path/to/rknpu2/runtime/Linux/librknn_api
# 2) 或直接指定两个路径（优先于 EC_RKNN_ROOT）
cmake -S . -B build -DEC_RKNN_INCLUDE_DIR=... -DEC_RKNN_LIBRARY=...
```

找不到时 CMake 直接 FATAL 而不是降级——一个"能编译成功但没链上 NPU"的二进制
比编译失败更难查。宿主机只想跑重采样/后处理对拍时：

```bash
cmake -S . -B build-host -DEC_BUILD_BOARD=OFF -DEC_BUILD_TESTS=ON
cmake --build build-host -j
```

这两个 target 完全不碰 librknnrt 和 OpenCV，只要求 C++17 编译器；
`EC_BUILD_BOARD` 默认 ON，所以板端本机构建不需要额外参数。

### 7.1 验收方法

板端与 Python 的一致性不靠肉眼，靠 `preprocess.py` 当 oracle：

```bash
# 板端写出 uint8 CHW 张量
./build/ec_infer --model m.rknn --image frame.jpg --dump-npy /tmp/board.npy

# Python 用同一张图重算并逐元素比较；不一致则退出码非 0
pixi run -e rknn python deploy/preprocess.py \
  --image frame.jpg --compare-candidate /tmp/board.npy --require-resize
```

`--require-resize` 是必要的：640×640 的输入图会让 PIL 返回字节相同的副本，resize
路径根本没被执行，检查会无条件通过。这一点在旧版 smoke test 里真实发生过。

`--dump-dets` 的输出（label / xyxy / score）可以用 `postprocess.py` 在同一份
float32 输出上复算后逐条比对；多图模式下会写成 `PATH.0`、`PATH.1`…，避免静默覆盖。
