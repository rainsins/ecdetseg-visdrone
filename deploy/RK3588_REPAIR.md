# RK3588 GridSample / FP16 输入修复记录

更新：2026-11-07。原模型 + CPU 自定义回调已通过真机 548 图精度验收；本轮 SSH profiling 确认回调累计约占 infer 的 6.6%。显式 GPU 子图编译路由成功，但当前板端运行库报不支持 GPU GridSample，未成功执行。

## 1. 根因与修复边界

- 报错表明当前 `.rknn` 的 GridSample 被路由到 CPU，但所加载的运行库没有对应实现。x86 模拟器能执行该算子，不能证明板端运行库也能执行。
- `RKNN_QUERY_INPUT_ATTR.type` 描述模型张量；`rknn_input.type/fmt` 描述调用方 buffer。原 C++ 把两者混为一谈。已允许 FP16/FP32/INT8/UINT8 模型输入，调用仍固定 UINT8 RGB NCHW、`pass_through=0`，归一化仍在 RKNN 中。不能仅凭输入类型断定是 auto_hybrid 导致。
- 本地原 ONNX 实测 **12 个 GridSample，均在 decoder cross-attention**，不是 36 个 ViTAdapter 节点。先在主机和板端对齐模型 SHA256。
- 配置为 `w8a8 + auto_hybrid=True`，应称为 INT8/FP16 混合精度；不用修改 mean/std 或关闭 auto_hybrid 来强行制造 UINT8 内部输入。

原 ONNX SHA256：`3e2a20507eee52d0aafb3a5ca261d2333a5f352bb3a6696c247848554d551be8`。
原 RKNN SHA256：`ff61f31c8ef1dd565c613c1bb59a7c40fb17a5e955e44503db19f928ede35130`。

## 2. 方案 A：ONNX 等价改写

`rewrite_grid_sample.py` 把每个双线性 GridSample 改为四邻点 Gather、逐邻点越界 mask 和双线性加权。`align_corners=0` 的横坐标是 `((gx+1)*W-1)/2`，不能丢掉半像素偏移；只做 clamp 而没有 mask 会错误变成 border padding。

支持 opset 16–19、静态 NCHW feature、FLOAT feature/grid、bilinear、zeros、两种 align_corners；要求有限输入。不支持的模式会拒绝，不静默近似。不改训练权重，不裁剪模块。

以下为历史流程记录，不要再次无界执行整网转换。当前使用 §8 的有界子图诊断。输出必须使用新路径；已有 `_gather` 产物时跳过改写。

```bash
.pixi/envs/rknn/bin/python -B deploy/rewrite_grid_sample.py \
  --input deploy/artifacts/ecdet_s_p150u_640.onnx \
  --output deploy/artifacts/ecdet_s_p150u_640_gather.onnx \
  --meta-in deploy/artifacts/ecdet_s_p150u_640_meta.json \
  --meta-out deploy/artifacts/ecdet_s_p150u_640_gather_meta.json

.pixi/envs/rknn/bin/python -B deploy/convert.py \
  --onnx deploy/artifacts/ecdet_s_p150u_640_gather.onnx \
  --meta deploy/artifacts/ecdet_s_p150u_640_gather_meta.json \
  --output deploy/artifacts/ecdet_s_p150u_640_gather.rknn

.pixi/envs/rknn/bin/python -B deploy/sim_eval.py \
  --rknn deploy/artifacts/ecdet_s_p150u_640_gather.rknn \
  --onnx deploy/artifacts/ecdet_s_p150u_640_gather.onnx \
  --meta deploy/artifacts/ecdet_s_p150u_640_gather_meta.json \
  --summary deploy/artifacts/sim_eval_gather_summary.json \
  --out-detections deploy/artifacts/detections_gather_sim.json \
  --fp32-baseline --fp32-onnx deploy/artifacts/ecdet_s_p150u_640.onnx
```

2026-11-07 收到后台任务终止通知：SIGKILL。已检查 `artifacts/logs/convert_gather.log`，最后记录为构建阶段的 `OpFusing 0: 99%`（日志最后修改时间 2026-11-06 20:37:26；这不是已确认的终止时间）。没有导出 `_gather.rknn` 或转换报告；`sim_eval_gather.log`、评测 summary 和 detections 均不存在，故评测未启动。这里的 99% 是该优化阶段进度，不是整条转换流程完成度。默认 gate 允许相对 FP32 下降 1.0 AP 点；通过该 gate 不等于“零掉点”。后续应同时比较新模型与原混合精度重建的 28.7075 AP，并报告 AP/AP50/AP75/小目标 AP 的实际变化。

**等价 ONNX 不等于等价量化，也不保证全 NPU 执行。** 真实尺寸采样子图在 RKNN 2.5.0 下 build/export 返回 0，但仍报告 `Unkown op target: 0`。因此本路线尚不能称为板端修复成功。完整新模型 AP 在取得新 summary 前不得继承旧模型成绩。模拟器会从 ONNX 在进程内重建，不直接执行已导出的 `.rknn`。

## 3. 方案 B：原模型 + 可选 CPU 回调

文件：`cpp/grid_sample_cpu.h/.cpp`，由 `rknn_engine.cpp` 在 `rknn_init` 后注册 `GridSample`。编译开关与运行开关均需启用。

这是已通过真机验收的严格参考 fallback：
- 仅接受 packed、未量化 FLOAT32 回调张量；feature/output 支持 NCHW/NHWC。支持 NHWC `[N,OH,OW,2]`、已转置 NCHW `[N,2,OH,OW]`，以及真机标为 NCHW 但实际 channel-last 的 grid。
- 使用 `mem.virt_addr + mem.offset`；验证尺寸、stride、可用内存及 dtype，不把 FP16 字节解释成 float。
- 字符串属性接受裸值和真机带双引号的 `"bilinear"` / `"zeros"`；`align_corners` 真机为 INT64。缺失或未知编码仍拒绝。
- 真机 feature `[8,24,H,W]`，H/W=80/40/20；grid `[8,300,3|6,2]`；output `[8,24,300,3|6]`。先结合 output 几何选择 grid 解释，再做 stride 校验；标准 NCHW 的 OW=2 不再误判。两种解释均满足 `[N,2,2,2]` 时尊重显式 fmt，不能仅凭 shape 判断实际存储。
- 真机两处修复已保留；本轮新增 56 组带引号、真机交付与 OW=2 回归，加上原 80 组，共 136 组 ORT 数值对比。新增布局泛化与计时修改须与原板端二进制对拍。

板端安装与 librknnrt 2.5.0 配套的 `rknn_api.h`、`rknn_custom_op.h`，按实际路径设置：

```bash
cmake -S deploy/cpp -B deploy/cpp/build-board \
  -DEC_ENABLE_GRID_SAMPLE_CPU=ON \
  -DEC_RKNN_INCLUDE_DIR=/path/to/rknn/include \
  -DEC_RKNN_LIBRARY=/path/to/librknnrt.so
cmake --build deploy/cpp/build-board -j4

ldd deploy/cpp/build-board/ec_infer
sha256sum deploy/artifacts/ecdet_s_p150u_640_rk3588_int8.rknn

deploy/cpp/build-board/ec_infer \
  --model deploy/artifacts/ecdet_s_p150u_640_rk3588_int8.rknn \
  --image /path/to/non640-frame.jpg --grid-sample-cpu \
  --dump-npy /tmp/board.npy --dump-raw /tmp/board \
  --dump-dets /tmp/board.dets.txt
```

检查 `ldd` 实际加载路径以及程序打印的 SDK/driver 版本，不只检查安装包版本。日志中的 `model_dtype=1` 可以合法；caller 必须仍是 UINT8 NCHW、pass_through=0。注册成功只说明 API 接受注册，不证明回调已正确执行。

方案 A 上板时使用 `_gather.rknn`，**不加** `--grid-sample-cpu`；两条路径分别验收。

## 4. 已完成的本地检查

| 检查 | 结果与范围 |
|---|---|
| ONNX 子图 | 2 个 unittest，通过；覆盖 8 组 shape/align、越界、矩形、单像素维度及拒绝 nearest |
| ONNX 整网 | 随机输入 + 3 张校准图片；两输出 max_abs_diff 均为 0，见 `artifacts/ecdet_s_p150u_640_gather.parity.json` |
| CPU 回调 | 与 ORT 对比 80 组 shape/align/layout，通过；验证非零未对齐 offset，并拒绝缺失属性、nearest、FP16、padding stride、内存不足、NC1HWC2、量化张量、NaN grid |
| C++ 输入契约 | 4 种模型 dtype × 2 种 layout 共 8 组通过；验证 UINT8 caller、pass_through=0、native stride 不误拒绝、反序输出识别及非法契约拒绝 |
| C++ 编译检查 | engine（回调开/关）、callback 和 main 均通过 `-Wall -Wextra -Werror -fsyntax-only`；不是 aarch64 链接/运行验收 |
| 新 RKNN 完整模型 | 2026-11-07 确认转换任务被 SIGKILL；无新 RKNN，548 图模拟器评测未启动 |
| RK3588 真机 | 用户完成 CPU 路径 548 图验收，AP 28.7691625；本轮已通过 ssh topeet 连接并开始独立目录 profiling |

复跑本地检查（SDK 路径按环境修改）：

```bash
EC_RKNN_INCLUDE_DIR=/tmp/opencode/ec-rknn-sdk \
  .pixi/envs/rknn/bin/python -B -m unittest discover -s deploy/tests -p 'test_grid_sample*.py' -v
g++ -std=c++17 -Wall -Wextra -Werror -Ideploy/cpp \
  -I/tmp/opencode/ec-rknn-sdk deploy/cpp/rknn_engine.cpp \
  deploy/cpp/tests/rknn_engine_contract.cpp -o /tmp/opencode/ec-engine-contract
/tmp/opencode/ec-engine-contract /tmp/opencode/ec-engine-mock.rknn
```

## 5. 真机精度与性能验收

1. 将同一非 640×640 原图与板端 `board.npy` 拷回主机，运行：

   ```bash
   .pixi/envs/rknn/bin/python -B deploy/preprocess.py \
     --image /path/to/non640-frame.jpg \
     --compare-candidate /path/to/board.npy --require-resize
   ```

2. `--dump-raw PREFIX` 输出第一帧的 `PREFIX.logits.f32`（3000 个 float32）与 `PREFIX.boxes.f32`（1200 个 float32）；RK3588 为小端。用 `numpy.fromfile(path,dtype='<f4').reshape(1,300,10)` / `(1,300,4)` 读取，检查有限值、范围和与同图参考输出的 max/mean absolute difference。FP32 ONNX 参考必须使用 `(uint8.astype(float32)-mean)/std`，RKNN caller 则绝不能再归一化。量化输出不要求逐位等于 FP32。
3. 原图与预处理完全对齐后，用 `--image-list` 跑同一验证集、`--score-thresh 0 --max-det 300 --dump-dets PREFIX`。多图文件分别为 `PREFIX.0`、`PREFIX.1` 等；每行是 `label x1 y1 x2 y2 score`，不是原始 logits。将对应图的标注 image_id 与结果对齐，转 COCO xywh 后复用 `deploy.sim_eval.evaluate_coco`，maxDets 为 `[1,10,100,300]`。先 50–100 图排障，正式报告使用完整 548 图。**小子集只能与同子集的 FP32/模拟器结果比较，不能与全量 28.82 直接比。**
4. 单图验证无误后运行：

   ```bash
   deploy/cpp/build-board/ec_infer \
     --model deploy/artifacts/ecdet_s_p150u_640_rk3588_int8.rknn \
     --image /path/to/non640-frame.jpg --grid-sample-cpu \
     --timing --warmup 20 --iters 100
   ```

   日志标签已从 `npu` 改为 `infer`：包括 inputs_set/run/outputs_get 和 fallback，不是纯 NPU kernel 时间。记录 pre/infer/post/total、核掩码、温度；持续负载再检查热降频。

   加 `--grid-sample-profile` 汇总 callback/sampling 时间与调用次数；计时循环会在 warmup 后清零，关闭时不调用回调内部时钟。`callback_sum` 包含校验与采样，`sampling_sum` 是采样循环（含坐标检查/访存），不是纯算术周期。多线程回调若重叠，累加值不是墙钟时间。`infer-minus-callback-sum` 还包括 NPU、调度、转换和输出获取，不能称为搬运时间。

   `--perf-detail PATH` 单独采集逐层报告；正式测速去掉性能收集，并对比关闭回调计时的基线。

## 6. 两条路径都不能达到目标时

先对含 GridSample 的真实 decoder 子图做复现，附 shape、属性、SDK/驱动版本、完整日志和模型哈希交给厂商。不能从一个回退报错推导整个 ViTAdapter/DETR 不可行。

可比较连续 NPU backbone/encoder + CPU decoder 分区，避免逐算子频繁跨设备搬运；这是后续工程，不是本次已经实现的功能。更换普通 attention、离散采样或裁剪 decoder 会改变模型语义，需要微调/蒸馏与全量 AP 评估，不能视作无损部署优化。现有 `cross_attn_method='discrete'` 分支也不能直接切换后声称等价。只有真机 profiling 确认代价后才值得改结构。

## 官方接口依据

仓库：`airockchip/rknn-toolkit2`。本地核对的文件：
- `rknpu2/runtime/Linux/librknn_api/include/rknn_api.h`：rknn_input.type/fmt/pass_through。
- 同目录 `rknn_custom_op.h`：在 rknn_init 后注册，compute 回调签名，compute_native 暂不支持。
- `rknpu2/examples/rknn_custom_op_demo/rknn_custom_cpu_op_demo/src/rknn_api_test_custom_cpu_op.cpp`：按 virt_addr + offset 访问数据。

本机检查使用下载的官方头文件；板端应以安装的 2.5.0 SDK 配套头文件和真实运行库重新构建。

## 7. 实测段错误后的追加处理：常量折叠与执行后端

用户真机探测：init/inputs_set 返回 0，init 报 12 次不支持 CPU GridSample，run 退出 139。这确认了当前执行路径存在明确阻塞，但只有修复该路径并完成运行后，才能排除其他执行问题；没有 backtrace 时，不能断言段错误内部的具体位置。

### 7.1 已实现的静态分支折叠

新增 `fold_static_ops.py`：只计算白名单允许且不依赖运行输入的完整常量分支，保留动态 mask 和可被调用方覆盖的 initializer。不是用一张图片的结果冻结张量。

已生成并验证：

| ONNX 产物 | 节点数 | GridSample | Sin/Cos | Equal/Where |
|---|---:|---:|---:|---|
| `ecdet_s_p150u_640_static_v2.onnx` | 2621 | 12 | 0 / 0 | 原有两者均移除 |
| `ecdet_s_p150u_640_gather_static_v2.onnx` | 3881 | 0 | 0 / 0 | 原有两者均移除，保留采样所需 48 个动态 Where |

两个版本相对原始 ONNX 的随机输入 + 3 张校准图片整网对比，两输出 max_abs_diff 均为 0。报告为同名 `.static.json`、`.parity.json`；对应 `_meta.json` 已生成。这不等于新的量化 AP 或板端性能通过。较早的 `_static.onnx` 文件只折叠了 Equal/Where；使用这里列出的 `_static_v2.onnx`。

复现折叠：

```bash
# OUTPUT 必须是新路径；不覆盖已有的 _static_v2 产物。
.pixi/envs/rknn/bin/python -B deploy/fold_static_ops.py --input INPUT.onnx --output OUTPUT.onnx
```

历史整网转换命令（暂不直接重跑；先执行 §8）：

```bash
.pixi/envs/rknn/bin/python -B deploy/convert.py \
  --onnx deploy/artifacts/ecdet_s_p150u_640_gather_static_v2.onnx \
  --meta deploy/artifacts/ecdet_s_p150u_640_gather_static_v2_meta.json \
  --output deploy/artifacts/ecdet_s_p150u_640_gather_static_v2.rknn
```

之前的完整转换/评测任务已被 SIGKILL，未自动重启。完整模型编译、量化 AP 和板端验收仍需独立完成。节点数从 2656 增到 3881，包含动态 Gather、Floor、整数索引与 mask；因此“消除 GridSample”不等于“全 NPU 或更快”。

终止原因现已由用户确认：整网转换持续 22h46m、CPU 99.9%，停在 OpFusing；用户手动 `kill -9`，并非 OOM 或自行崩溃。此前“SIGKILL 来源未知”的判断已被此交接取代。后续仅执行带超时和资源监控的实验。

### 7.2 另一个低改动探测：官方 GPU-priority fallback

官方 `airockchip/rknn-toolkit2` README 的 v2.5.0 更新列出 GridSample (GPU)。头文件提供 `RKNN_FLAG_EXECUTE_FALLBACK_PRIOR_DEVICE_GPU`（0x400）。新增 CLI `--fallback-gpu`，在 init 时传该标志；不加该选项时沿用原行为。它是 GPU 优先策略，不是保证指定节点必定使用 GPU；需要板端配套 GPU/OpenCL 用户态与运行库支持。

```bash
deploy/cpp/build-board/ec_infer \
  --model deploy/artifacts/ecdet_s_p150u_640_rk3588_int8.rknn \
  --image /path/to/non640-frame.jpg --fallback-gpu \
  --dump-raw /tmp/board-gpu --perf-detail /tmp/board-gpu.perf.txt
```

上述原模型命令已由用户测试失败：修复 Mali ICD 后 OpenCL 平台可见，0x400 仍未将 12 个 GridSample 路由到 GPU，run exit 139。与 `--grid-sample-cpu` 互斥。不要将 OpenCL 可见或 runtime flag 当成 GPU 执行证据。

新增 `--perf-detail PATH`：启用 `RKNN_FLAG_COLLECT_PERF_MASK`，第一帧成功后查询 `RKNN_QUERY_PERF_DETAIL`，原样保存。空报告或不支持时明确报错。性能收集会增加开销，正式测速去掉该选项。GPU/perf 标志及报告导出已通过 C API mock 回归；本轮已取得真机逐层报告与独立回调计时，见 §9.1。显式 GPU 子图的板端运行库报不支持 GridSample，未成功执行，见 §9.2。

### 7.3 其他候选算子的处理顺序

**19 条 `Unkown op target: 0` 不能直接解读成 19 个 CPU 算子。** 列出的原图候选数量相加为 20；编译器还可能折叠、融合、拆分节点，日志计数与 ONNX 节点数不存在已证明的一一映射。先解决崩溃，再结合逐层报告、编译 verbose 日志判断实际落点。

| 算子 | 本地图中的用途 | 处理建议 |
|---|---|---|
| Sin / Cos 各 1 | RoPE 常量位置表，各输出 `[1600,64]` | 已折叠；没有运行输入依赖。编译器也可能原本已折叠，不能承诺加速幅度 |
| 原 Equal / Where 各 1 | 常量 shape 构造，输出仅 `[3]` | 已折叠；优先级低于运行正确性 |
| Erf ×12 | backbone 12 个 MLP 的 exact GELU 表达式 | 先确认是否已被 RKNN 融合。若确实造成频繁回退，再做 GELU pattern 最小子图实验 |
| TopK ×2 | 内部 query 选择；最后一层 LQE 分支 | 动态依赖特征，不能常量折叠，也不是已移到模型外的最终检测 TopK。先保留，按时间占比决定是否做连续子图拆分 |
| GatherElements ×2 | 按动态 query 索引取 anchors/features | anchors 虽是常量，索引不是。不能把输出固化；先保留，与 query 选择一起分析 |

Erf 链本来就是 `0.5*x*(1+erf(x/sqrt(2)))`。标准 ONNX `Gelu` 从 opset 20 才引入，当前是 opset 17；不能只把节点改名为 Gelu，或直接把 opset 数字改成 20。也不要将 ORT 的 contrib-domain 融合图默认当成 RKNN 可导入图。若后续尝试 tanh GELU，是近似替换而不是等价融合，必须重新检查 FP32 输出、量化 AP 和真实延迟。本轮没有替换这 12 个激活。

依据：官方仓库 README v2.5.0、`rknn_api.h` 中 GPU/performance 标志及查询结构；ONNX 官方 Gelu-20 定义（含 exact 与 tanh 两种公式）。

本轮实测更新：逐层报告已有 12 个 `ConvExGelu`，均在 NPU；无需为这 12 个 Erf 链改近似激活。TopK 两次合计 4.479ms（0.63%），GatherElements 两次合计 46.686ms（6.60%），应按这个实测优先级分析。

## 8. 2026-11-07 有界编译诊断与 GPU 配置

`diagnose_rknn.py` 从真实第一层 decoder 提取单采样、cross-attention、单层 decoder；常量祖先保留，动态边界取自真实图片的整网 ORT 激活。原版/改写版输出均与整网中间值 max_abs_diff=0。单张激活只用于编译排障，不是有效量化精度评估。

```bash
# --out 必须不存在；现有 grid_v1 不要覆盖
.pixi/envs/rknn/bin/python -B deploy/diagnose_rknn.py prepare \
  --source deploy/artifacts/ecdet_s_p150u_640.onnx \
  --image deploy/artifacts/calib_imgs/0000008_0000338_01569_d_0000146.png \
  --out deploy/artifacts/diagnostics/grid_v2
.pixi/envs/rknn/bin/python -B deploy/diagnose_rknn.py run \
  --prepared deploy/artifacts/diagnostics/grid_v2 --scope decoder --variant gather \
  --out deploy/artifacts/diagnostics/decoder_v2 --timeout 30 --rss-mib 4096 --quantize
```

每例新进程组，200ms 采样进程树 RSS/CPU；超时或 RSS 超限强制终止整个进程组，记录 `result.json`、`resources.jsonl`、`build.log`。RSS 是采样值且共享页可能重复计算，不是精确峰值 rusage。

| 子图/配置 | 耗时秒 | 结果 |
|---|---:|---|
| 单采样 Gather，FP16，O3 | 1.63 | 导出成功 |
| cross-attention Gather，FP16，O3 | 3.87 | 导出成功 |
| 单层 decoder Gather，FP16，O3 | 4.68 | 导出成功 |
| 单采样 Gather，量化+hybrid，O3 | 2.04 | 导出成功 |
| cross-attention Gather，量化+hybrid，O3 | 6.52 | 导出成功 |
| 单层 decoder Gather，量化+hybrid，O3 | 7.54 | 导出成功 |
| 单层 decoder Gather，FP16，O0 | 4.70 | 导出成功 |
| 原单采样，FP16，显式 GPU，O3 | 1.02 | 日志 GridSample FLOAT16 GPU；导出成功 |

证据目录：`artifacts/diagnostics/grid_v1/`。**尚未定位整网挂起触发点**；单层通过不能推出多层、前后处理连接或完整量化上下文通过。下一步在同样有界条件下逐步扩大连接边界，而不是再次无界整网转换。

官方 `doc/EN/03_RKNN2_Toolkit_Python_API_Reference_V2.5.0.pdf` 第10页确认：`op_target` 接受 op_type 或输出名，值可为 cpu/npu/gpu/auto。`doc/EN/06_RKNN2_Supported_Operators_and_Constraints_Reference_V2.5.0.pdf` 第377页：GPU GridSample 只支持 FP16、bilinear、zeros；未在该页给出完整 shape/align 限制。文件从 `airockchip/rknn-toolkit2` 下载，经本地代理 127.0.0.1:7890 取得。

已接入 `convert.py --grid-sample-target gpu`，写入 `op_target={"GridSample":"gpu"}` 并记录于转换报告，要求显式新输出路径；若 API 不支持则拒绝静默丢弃。`sim_eval.py` 默认拒绝将无路由配置的 rebuild 冒充这个实验的同配置模拟。尚未转换 GPU 整网，不能对原产物加 flag 就声称应用了此配置。

## 9. 真机交接结论

- TOPEET iTOP-RK3588、Ubuntu 24.04.4、kernel 6.1.141、librknnrt 2.5.0、NPU driver 0.9.8。CPU 回调 AP **28.76916252162901** / AP50 **48.37683725216193** / AP75 **28.638130657815363**，548 图 / 164400 detections，gate 通过。
- 347.2 秒、633.577 ms/图、1.578 FPS 是整次评测均值；原报告的“CPU-bound”是待定位归因，不是逐层证据。与模拟器相差 0.061698 AP 的原因也不能仅凭 rebuild 差异确定。
- 通过 SSH 查询 `/root/ecdet_cpp/model.rknn` 的 SHA256 与本文原产物一致。原始 548 图报告没有哈希，本次是后续文件核对，不能追溯证明当时执行文件未变。
- 保留用户原部署 `/root/ecdet_cpp`，本轮构建位于独立 `/root/ecdet-diag-e94P1v`。精度报告原始测量值保持不动；新增实验和解释记录独立归档。

### 9.1 SSH 性能实测

证据：`artifacts/diagnostics/board_v1/summary.json`、`profile.log`、`baseline.log`、`runtime.perf.txt`。使用 `/root/ecdet_cpp/test.png`（640×640），warmup=5、iters=20；因此这不是 resize 验收或完整验证集延迟分布。

- 不启用运行库逐层采集时：infer **626.982ms**，CPU callback 累计 **41.392ms**，采样循环 **41.347ms**；20帧共240次回调，0失败，回调累计/infer **6.60%**。关闭回调计时的对照 infer **627.606ms**，总流程 **630.726ms**，约 **1.59 FPS**。两次均值差不能当作精确的计时开销测量。
- 新旧二进制同图 logits/boxes 各自逐位一致（max_abs_diff=0）；本轮没有重新跑548图，历史全量AP仍归属于用户验收版本。
- 独立的逐层采集首帧：`exSDPAttention` 306.054ms / **43.24%**；GridSample 99.691ms / 14.08%；Transpose 63.778ms / 9.01%；GatherElements 46.686ms / 6.60%；ConvExGelu 19.528ms / 2.76%。这些是运行库报告口径，不能与前一轮41.392ms直接相减来计算搬运时间。
- 原报告中将整网慢直接归为 CPU 回调瓶颈，证据不足；当前应优先分析 attention 的 NPU 执行与数据变换/索引路径，保持 exact GELU。尚未做 NEON、线程优化或模型结构修改。

### 9.2 GPU 子图上板结果

`grid_sample_board_probe.cpp` 使用真实捕获的 FLOAT32 中间激活，经 `pass_through=0` 转换为子图内部FP16；该诊断入口不是整网 UINT8 caller。编译日志明确 GridSample FLOAT16 GPU。板端 `rknn_init=0`、输入检查通过、`inputs_set=0`，但加载时报 **Unsupport GPU op: GridSample in this librknnrt.so**，未得到 run 返回日志或合法输出，SSH命令退出255；不能把255写成已证明的段错误信号。证据为 `gpu_probe_v2.log`。

当前实际加载 `/lib/librknnrt.so`（系统库 `/usr/lib/librknnrt.so` 的哈希为 `4b6cc575bedf8952b13db4b1190fa3b3d2ac213f00be774ff6ccd19d651970c0`）。因此问题已分成两层：转换期显式GPU路由已成立，当前运行库未提供可用的对应实现。下一步向Rockchip核对匹配的GPU实现/运行库构建；本轮没有替换系统库或更改驱动。

### 9.3 核配置对照与慢层定位（2026-11-07）

证据：`artifacts/diagnostics/core_masks_v1/summary.json`，同目录保存各轮日志、原始输出和三核逐层报告。使用同一模型、二进制与640×640图片，顺序为 AUTO → 三核 → core0 → AUTO；每轮3次warmup、8次计时，单次远程推理命令限时40秒。

| 配置 | core-mask | infer均值 ms |
|---|---:|---:|
| AUTO（前） | 0 | 631.078 |
| 三核 | 7 | 687.333 |
| core0 | 1 | 625.605 |
| AUTO（后） | 0 | 626.072 |

所有配置首帧 logits/boxes 与上一轮参考输出逐位一致。三核设置调用成功，但本轮未获提速；保留默认 AUTO。这是短时顺序筛查，不能据此断言三核始终更慢，或 core0 显著优于 AUTO。轮次边界温度约41.6–45.3°C、NPU频率读数均为1GHz；未连续采样，不能排除瞬态频率变化。

逐层报告进一步定位：
- 先前 AUTO 的17个 attention 中，backbone 12层合计 **292.418ms**，encoder 1层 **4.000ms**，decoder 4层 **9.636ms**。优先目标是 backbone 的1601-token attention，而不是 decoder 的300-query self-attention。
- 三核独立首帧报告中，backbone attention 合计 **291.426ms**，仍报告 `100.0%/0.0%/0.0%` 的核负载；说明本次 runtime mask 未使这些算子在报告中分摊到三核。该字段不是独立硬件追踪，也不能推广为平台不支持多核。
- CPU `GatherElements:/model/decoder/GatherElements_1` 为 **45.603ms**，输入 `(1,8400,1,192)`、输出 `(1,300,1,192)`。
- 最大 Transpose 是 CPU `/model/backbone/Concat_output_0_tp`，**26.951ms**，从 `(1,192,2,1600)` 到 `(1,2,1600,192)`。

以上逐层数据来自独立首帧采集，不与计时循环均值直接相减。下一实验为提取 backbone 第一层 attention 的真实激活子图，在既有超时/RSS边界内对比编译配置及板端输出；之后分别隔离上述 feature GatherElements 与 backbone Transpose。尚未实施这些子图优化或重跑全量AP。
