# EdgeCrafter / ecdetseg — VisDrone 训练配置只读审计

> 阶段 0 产物。**纯静态分析**：不加载模型、不读数据、不初始化 XPU。
> 所有结论均落到 `文件:行号`。未经代码证实的事项列入“待我确认”。
>
> 审计时的运行态：`pgrep -af train.py` 显示 PID 431205（O365 初始化，60 轮）在跑；
> 因此本次**只新增文件，未改动** `ecdet.yml` / `ecdet_s.yml` / `custom.yml` / `train_*.sh`。

---

## 结论

### Q1. `__include__` 合并顺序与语义

- 语义：`load_config`（`engine/core/yaml_utils.py:25-49`）对 `__include__` 列表**按顺序**逐个
  递归加载，用 `merge_dict` 合并进同一个 `cfg`；最后再 `merge_dict(cfg, 本文件 body)`。
  `merge_dict`（`yaml_utils.py:52-67`）是**递归深合并**：两边都是 dict 就下钻，否则**后者整体覆盖前者**
  （list 不逐元素合并，是整体替换）。
- 结论：**列表里靠后的 base 覆盖靠前的 base；包含文件自身 body 覆盖所有 base。**
  即 `__include__: [A, B]` ⟹ 优先级 `A < B < 本文件`。
- ⚠️ **隐患（重要）**：`def load_config(file_path, cfg=dict())`（`yaml_utils.py:25`）用了可变默认参数。
  同一个 Python 进程里**多次顶层调用** `load_config(path)`（不显式传 `cfg`）会共用同一个 dict，
  导致配置互相污染。单个训练进程只加载一次配置，所以训练不受影响；但**任何一次加载多个配置的脚本
  必须显式传 `cfg={}`**（本审计的 `scripts/dump_config.py` 已如此处理）。
- 另：`YAMLConfig.__init__`（`engine/core/yaml_config.py:23-34`）在合并后会调用 `reset_cfg()`，
  再只把 `super().__dict__` 里**已存在**的非下划线 key 拷进 `self`（因此 yaml 里出现新 key 不会自动成为
  cfg 属性，除非 `BaseConfig` 已声明）。

**合并后最终配置**：见 `scripts/dump_config.py`（只读替身）与 `configs_dump/`。命令：
```bash
cd ecdetseg && python scripts/dump_config.py -c configs/ecdet/ecdet_s.yml
```

### Q2. flatcosine 调度器语义与学习率曲线

源码：`engine/optim/lr_scheduler.py:10-35`（`flat_cosine_schedule`）、`38-83`（`FlatCosineLRScheduler`）。

按迭代 `current_iter` 分三段（`total_iter = iter_per_epoch*epochs`，`flat_iter = iter_per_epoch*flat_epochs`，
`no_aug_iter = iter_per_epoch*no_aug_epochs`）：
1. `iter <= warmup_iter`：`lr = init_lr * (iter/warmup_iter)^2`（二次 warmup）。
2. `warmup_iter < iter <= flat_iter`：`lr = init_lr`（恒定）。
3. `iter >= total_iter - no_aug_iter`：`lr = min_lr = init_lr * lr_gamma`（恒定，判在 cosine 之前）。
4. 其余（`flat_iter < iter < total_iter - no_aug_iter`）：余弦 `min_lr + (init_lr-min_lr)*0.5*(1+cos(pi*(iter-flat_iter)/(total_iter-flat_iter-no_aug_iter)))`。

**关键**：调度器的 `no_aug_epochs` 不是取 cfg 的 `no_aug_epoch`，而是
`no_aug_epochs = epochs - stop_epoch`（`engine/solver/ec_solver.py:31-34`）。`flat_epochs` 取
`args.flat_epoch`，若为 `None` 则回退到 `mosaic_epoch`（`ec_solver.py:34`）。

因此当 `epochs=30, flat_epoch=28, no_aug_epoch=2, stop_epoch=28`：
`flat_iter == total_iter - no_aug_iter`（28e == 28e），**余弦区间为空**，学习率在 epoch 28 由 `init_lr`
**直接台阶式掉到** `min_lr`，没有退火。`flat_epoch=28` 实际上没有余弦尾。

> 曲线（`iter_per_epoch = 6471//8 = 808`，`warmup_iter = min(2000, 3*808) = 2000`，
> `stop_epoch=28` ⟹ `no_aug_epochs=2`）。下为每个 epoch 起始迭代的学习率（head 组 `init_lr=5e-4`）：

**flat_epoch=10**
```
e00:0.00e+00 e01:8.16e-05 e02:3.26e-04 e03:5.00e-04 e04:5.00e-04 e05:5.00e-04 e06:5.00e-04 e07:5.00e-04 e08:5.00e-04 e09:5.00e-04
e10:5.00e-04 e11:4.98e-04 e12:4.92e-04 e13:4.83e-04 e14:4.71e-04 e15:4.55e-04 e16:4.38e-04 e17:4.18e-04 e18:3.97e-04 e19:3.75e-04
e20:3.53e-04 e21:3.32e-04 e22:3.13e-04 e23:2.95e-04 e24:2.79e-04 e25:2.67e-04 e26:2.58e-04 e27:2.52e-04 e28:2.50e-04 e29:2.50e-04
```
**flat_epoch=15**（P30u 默认）
```
e00:0.00e+00 e01:8.16e-05 e02:3.26e-04 e03:5.00e-04 e04:5.00e-04 e05:5.00e-04 e06:5.00e-04 e07:5.00e-04 e08:5.00e-04 e09:5.00e-04
e10:5.00e-04 e11:5.00e-04 e12:5.00e-04 e13:5.00e-04 e14:5.00e-04 e15:5.00e-04 e16:4.96e-04 e17:4.86e-04 e18:4.69e-04 e19:4.46e-04
e20:4.19e-04 e21:3.90e-04 e22:3.60e-04 e23:3.31e-04 e24:3.04e-04 e25:2.81e-04 e26:2.64e-04 e27:2.54e-04 e28:2.50e-04 e29:2.50e-04
```
**flat_epoch=20**
```
e00:0.00e+00 e01:8.16e-05 e02:3.26e-04 e03:5.00e-04 e04:5.00e-04 e05:5.00e-04 e06:5.00e-04 e07:5.00e-04 e08:5.00e-04 e09:5.00e-04
e10:5.00e-04 e11:5.00e-04 e12:5.00e-04 e13:5.00e-04 e14:5.00e-04 e15:5.00e-04 e16:5.00e-04 e17:5.00e-04 e18:5.00e-04 e19:5.00e-04
e20:5.00e-04 e21:4.90e-04 e22:4.63e-04 e23:4.23e-04 e24:3.75e-04 e25:3.27e-04 e26:2.87e-04 e27:2.60e-04 e28:2.50e-04 e29:2.50e-04
```
**flat_epoch=28**
```
e00:0.00e+00 e01:8.16e-05 e02:3.26e-04 e03:5.00e-04 e04:5.00e-04 e05:5.00e-04 e06:5.00e-04 e07:5.00e-04 e08:5.00e-04 e09:5.00e-04
e10:5.00e-04 e11:5.00e-04 e12:5.00e-04 e13:5.00e-04 e14:5.00e-04 e15:5.00e-04 e16:5.00e-04 e17:5.00e-04 e18:5.00e-04 e19:5.00e-04
e20:5.00e-04 e21:5.00e-04 e22:5.00e-04 e23:5.00e-04 e24:5.00e-04 e25:5.00e-04 e26:5.00e-04 e27:5.00e-04 e28:5.00e-04 e29:2.50e-04
```
> backbone 组同理，初值 `2.5e-5`，min `1.25e-5`。生成脚本见本次会话 `/tmp/opencode/lr_table.py`。

### Q3. EMA（`engine/optim/ema.py:22-84`）

- 有效衰减：`d(u) = decay * (1 - exp(-u / warmups))`（`ema.py:48`），`u` 为已更新次数（`ema.py:60-61`）。
  `warmups=1000` 时 `u=1000` 处 `d≈0.632*decay`；`u→∞` 时 `d→decay`。`decay` 是**渐近上界**。
- `start=0` 表示从第一次 update 就生效（`ema.py:55-57` 的 `before_start<start` 不触发）。
- 更新式：`v = d*v + (1-d)*model`（仅浮点 buffer/param，`ema.py:63-66`）。
- **评估用 EMA**：`ec_solver.py:119` `module = self.ema.module if self.ema else self.model` 后再 `evaluate`。
  训练中途的重复评估（resume 前）同样用 EMA（`ec_solver.py:57`）。
- **保存**：`state_dict()`（`_solver.py:125-138`）遍历 `self.__dict__`，凡有 `state_dict()` 的对象都存，
  因此 checkpoint 同时含 `model` 与 `ema`（还有 `optimizer`、`scaler` 等）。

### Q4. mosaic_epoch / stop_epoch / no_aug_epoch / mixup_epoch 的真实作用

- `mosaic_epoch`（`engine/data/transforms/container.py:83`）：Mosaic **仅在** `mosaic_prob>0 且 cur_epoch < mosaic_epoch`
  时可能被应用。
- `stop_epoch`（`container.py:90`）：`cur_epoch >= stop_epoch` 后禁用 `remove_ops` 里的强增强
  （ecdet.yml 的 `remove_ops` 含 `Mosaic`）。**同时**驱动：调度器 `no_aug_epochs = epochs - stop_epoch`
  （`ec_solver.py:33`）、checkpoint 保存窗口（`ec_solver.py:111`）、以及在 `epoch==stop_epoch` 时
  重新载入 `best.pth` 进入 no-aug 阶段（`ec_solver.py:82-86`）。
- `mixup_epoch`（`engine/data/dataloader.py:113`）：Mixup **仅在** `rand<mixup_prob 且 epoch < mixup_epoch` 时应用。
- `no_aug_epoch`（cfg 属性，`engine/core/_config.py:64`）：**全仓库唯一出现处就是声明**，
  **从未被读取**（调度器用的是 `epochs-stop_epoch`）。它是**死键**。
- 冲突/耦合：
  - `stop_epoch` 是真正的“no-aug 边界”，`no_aug_epoch` 只是它的镜像，二者必须自洽
    （`reset_cfg` 会校验 `epochs-stop_epoch∈[0,5]`，否则改 `stop_epoch=epochs-2`，`yaml_config.py:201-211`）。
  - 若 `mosaic_epoch > stop_epoch`，`stop_epoch` 规则先命中（Mosaic 被 `remove_ops` 移除），`mosaic_epoch` 失效；
    若 `mosaic_epoch < stop_epoch`，Mosaic 在 `mosaic_epoch` 停（如 P30u：26<28）。
  - `mixup_prob=0` 时 `mixup_epoch` 无意义。

### Q5. `-t`（tuning）与 `-r`（resume）

- 参数：`train.py:77-78`；二者互斥（`train.py:45-46`）。
- **两者都会**把 `ViTAdapter.skip_load_backbone=True`（`train.py:55-57`），即**跳过 `ecvits/ecvitt.pth`**。
- 加载先后（`_solver.py:53-58`）：先 `self.model = cfg.model`（触发 `ViTAdapter.__init__`，
  若未 skip 则 `_load_weights` **strict=True** 载入 `ecvitt.pth`），**再** `load_tuning_state`。所以
  `-t`/`-r` 时顺序是“构建（跳过 ecvitt）→ 叠加 -t 权重”。
- **`-t`**（`_solver.py:173-196`）：`strict=False`；先 `_adjust_head_parameters` 做 COCO↔O365 类映射
  （`_solver.py:214-259`，基于 `obj365_ids`），失败则回退直接匹配。打印
  `Load model.state_dict, {'missed':[...], 'unmatched':[...]}`（`_solver.py:196`）——
  即**打印的是 missed/unmatched 的 key 清单**（而非计数；计数可 `len()` 得出）。missed=当前模型有、ckpt 无；
  unmatched=形状不一致。
  - VisDrone 10 类 vs COCO/O365 80 类：`denoising_class_embed.weight` 尺寸不符会先被删（`_solver.py:217-219`）；
    head 映射 `map_class_weights`（`_solver.py:244-259`）在 `pretrain>cur` 时按 `obj365_ids` 逐行赋值，
    对 10 类会索引越界→抛异常→被 `try/except` 捕获回退（`_solver.py:189-193`），最终**分类头随机初始化**、
    其余层尽量载入。**待我确认**：E2/E3 分类头是否应改成真正的 VisDrone 映射（见末节）。
- **`-r`**（`_solver.py:163-171` + `140-161`）：`torch.load(..., weights_only=True)`；
  对 `model`/`optimizer`/`ema`/`scaler` 等逐个 `v.load_state_dict(state[k])`，**model 是 strict=True**
  （未传 strict → 默认 True），**不打印 matched/missing 计数**，只打印 `Load {k}.state_dict`。
- **`-t` + `weights_path` 同时存在**：因为 `skip_load_backbone=True`，`weights_path` 被跳过，
  由 `-t` ckpt 提供 backbone。

### Q6. checkpoint 保存（`ec_solver.py:111-117, 144-145`）

- 命名：`last.pth`（每轮，**仅当 `epoch < stop_aug_epoch`**）、`checkpoint{epoch:04d}.pth`
  （当 `(epoch+1) % checkpoint_freq == 0`，默认 `checkpoint_freq=1`）、`best.pth`（val 指标刷新时）。
- **非原子写**：`dist_utils.save_on_master(self.state_dict(), path)` → `torch.save(path)` 直接写，
  无 temp+rename（`dist_utils.py:140-142`）。
- 内容：见 Q3，含 `model` 与 `ema`、`optimizer`、`scaler` 等，及 `last_epoch`、`date`。
- ⚠️ **行为坑**：`last.pth` 在 `epoch >= stop_aug_epoch` 后**不再更新**；no-aug 阶段从
  `best.pth` 重启（`ec_solver.py:82-86`），其间只可能更新 `best.pth`。故“最后一轮权重”通常要取
  **训练结束后的 `best.pth`（EMA）**，`last.pth` 可能停在 `stop_aug_epoch-1`。脚本恢复逻辑需据此处理。

### Q7. seed / deterministic

- `train.py:80` `--seed` 默认 0；`setup_distributed(..., seed=args.seed)`（`train.py:43`）→
  `setup_seed(seed)`（`dist_utils.py:78-79`），设 `random/np/torch` 及 `torch.cuda.manual_seed_all`。
- **未启用确定性**：`setup_seed(seed, deterministic=False)` 默认 False，且仅设置
  `cudnn.deterministic`（XPU 不适用）。XPU 上没有 deterministic 设置。

### Q8. 标注 json 统计（只读，CPU）

源文件：`/home/rainsin/Projects/Career/Paper/dataset/VisDrone/annotations_VisDrone_{train,val}.json`

| split | images | anns | categories | names |
|---|---|---|---|---|
| train | 6471 | 343205 | id 0..9 (10) | pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor |
| val | 548 | 38759 | id 0..9 (10) | 同上 |

- `iscrowd` 全为 0；`category_id==0` 是**真实类别 pedestrian**（79337/8844 个），**不是 ignored region**。
- 每类标注分布：train `{0:79337,1:27059,2:10480,3:144867,4:24956,5:12875,6:4812,7:3246,8:5926,9:29647}`。
- 类别 id 已经是 **0..9 连续**，且 `remap_mscoco_category: False`（`custom.yml:9`）→ 标签直接用
  `category_id`（`coco_dataset.py:134-140`）。**结论：`num_classes: 10` 正确**，不需要 11，也不需要重映射。

### Q9. AMP dtype 与 GradScaler（XPU）

- `engine/optim/amp.py:22-28` `get_amp_dtype`：`xpu → torch.bfloat16`，`cuda → float16`，其余 `float32`。
- `engine/solver/ec_engine.py:49` `torch.autocast(device_type=device.type, dtype=get_amp_dtype(device.type), ...)`；
  在 Arc/A770 上即 **bf16 autocast**。
- `GradScaler.__init__`（`amp.py:32-37`）：`enabled` 未显式给定时按 `device=='cuda'` 决定；
  XPU 上 **`enabled=False`**（bf16 不需要 loss scaling）。`use_amp: True` 只是让 `cfg.scaler` 被创建
  （`yaml_config.py:96-99`），从而 `ec_engine.py:47` 走 autocast 分支。

### Q10. 分辨率相关参数出现位置

- **唯一有效来源：`eval_spatial_size`**（`ecdet_s.yml:62`、`ecdet.yml:88` 等）。
  `reset_cfg`（`yaml_config.py:183-199`）会把 **Mosaic 的 `output_size` 改成 `eval_spatial_size//2`**、
  把 **所有 Resize 的 `size` 改成 `(eval_spatial_size[0],)*2`**。
- 因此以下字面量是**死值**（会被 `reset_cfg` 覆盖）：`ecdet.yml:94 Mosaic output_size: 320`、
  `ecdet.yml:101 train Resize [640,640]`、`ecdet.yml:132 val Resize [640,640]`。
- 其它引用：`decoder.py:459,488,509,651,679` 用 `eval_spatial_size` 生成 anchors / 分割头尺寸。
- 结论：**改分辨率只需改 `eval_spatial_size` 一处**（640→960 时 Mosaic 自动 480）。P30u 用
  `protocol/res640.yml` / `res960.yml` 两个片段表达，实际只设 `eval_spatial_size`。

### Q11. 两个 XPU 环境变量是否被识别（只核实）

- `PYTORCH_SYCL_ALLOC_CONF`：在 `.pixi/envs/default` 内**无任何字符串引用**（`grep -rl` 为空）。
  torch 2.9.1+xpu 只含 `PYTORCH_CUDA_ALLOC_CONF`（`libc10.so`）。⟹ **当前版本下不被识别，等同 no-op**。
- `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS`：被 `libur_adapter_level_zero*.so` 引用
  ⟹ **由 Unified Runtime 的 Level Zero 适配器识别**，设置有效。
- torch 版本：`2.9.1+xpu`，`torch.xpu.is_available() == True`。

---

## 待我确认（不要猜的项）

1. **`flat_epoch` 默认值**：我按“留出真实余弦尾”取了 **15**（epochs=30, stop_epoch=28）。
   若你更想要短退火（更接近 28）或长平段（更小），请定夺。
2. **`EMA decay`**：任务要求 0.9995；但现网在跑的实验用 0.9999、warmups 1000。需确认 P30u 归一为 0.9995。
3. **`mosaic_prob`**：任务给 0.25；原 `ecdet_s.yml` 为 0.75（被 ecdet.yml 的 0.5 覆盖前的历史值）。
   是否 E1/E2/E3 一律 0.25？
4. **`num_classes`**：审计结论为 10（id 0..9，无 ignored）。确认沿用。
5. **E2/E3 分类头初始化**：10 类 vs 80 类形状不符，`map_class_weights` 会越界回退 →
   分类头随机初始化。是否需要实现 VisDrone 专用类映射？（会改到 `_solver.py`，超出“只新增”范围。）
6. **no-aug 阶段权重**：因 `last.pth` 在 `stop_aug_epoch` 后冻结，最终评估应取 `best.pth`（EMA）
   还是“训练结束时的 EMA”？run_exp 计划评估**最终 EMA（best.pth，EMA 模块）**并另记录 best 数字（见脚本注释）。
7. **数据根**：加载器**不支持** `${ENV}` 展开（`ConvertCocoPolysToMask` 只读字面路径）。
   故 P30u 数据路径由 `run_exp.sh` 用 `-u` 注入；`visdrone.yml` 内为占位符。确认可否。
8. **是否允许在训练结束后**把 `custom.yml/ecdet.yml/ecdet_s.yml` 正式折叠成 data-only / struct-only /
   protocol 三层（本轮受硬约束只新增文件）。
