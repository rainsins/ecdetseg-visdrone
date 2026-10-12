#!/bin/bash
# ============================================================
# EdgeCrafter ECDet-S 训练守护脚本
# 功能：自动增量恢复 + 崩溃重试 + 防死循环 + 崩溃日志留档
# 用法：配合 pm2 使用，也可直接 bash 运行
# ============================================================

# ---------- 配置区 ----------
REPO_ROOT="/home/rainsin/Projects/Career/Paper/EdgeCrafter"          # pixi.toml 所在
PROJECT_DIR="${REPO_ROOT}/ecdetseg"                                   # train.py 所在
CONFIG="configs/ecdet/ecdet_s.yml"                                    # 相对 ecdetseg
INIT_WEIGHT="${PROJECT_DIR}/ecdet_s.pth"                              # COCO 预训练权重
CKPT_DIR="${PROJECT_DIR}/outputs/ecdet_s_coco_pre_weights"            # ← 启动前 ls 确认实际目录名
LAST_CKPT="${CKPT_DIR}/last.pth"
MAX_RETRY=20                                                          # 连续崩溃重试上限
RETRY_WAIT=15                                                         # 崩溃后等待秒数
# ---------------------------

# ---------- 路径检查 ----------
cd "$REPO_ROOT" || { echo "[FATAL] repo root not found: $REPO_ROOT"; exit 1; }
[ -d "$PROJECT_DIR" ] || { echo "[FATAL] project dir not found: $PROJECT_DIR"; exit 1; }
[ -f "$REPO_ROOT/pixi.toml" ] || { echo "[FATAL] pixi.toml not found"; exit 1; }

# ---------- XPU 环境 ----------
export PYTORCH_SYCL_ALLOC_CONF=expandable_segments:True
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0

mkdir -p "$CKPT_DIR"

retry_count=0
while true; do
    # ---------- 智能选择恢复模式 ----------
    if [ -f "$LAST_CKPT" ]; then
        MODE="-r $LAST_CKPT"
        TAG="resume"
        echo "[$(date '+%F %T')] Resuming from checkpoint: $LAST_CKPT"
    elif [ -f "$INIT_WEIGHT" ]; then
        MODE="-t $INIT_WEIGHT"
        TAG="cold"
        echo "[$(date '+%F %T')] Cold start from: $INIT_WEIGHT"
    else
        echo "[FATAL] neither checkpoint nor init weight found"
        exit 1
    fi

    # ---------- 训练（崩溃 traceback 留档） ----------
    cd "$PROJECT_DIR"
    pixi run --manifest-path "$REPO_ROOT/pixi.toml" \
        python train.py -c "$CONFIG" -d xpu $MODE \
        2> >(tee -a "${CKPT_DIR}/crash_${TAG}_$(date '+%m%d_%H%M').log" >&2)
    EXIT_CODE=$?
    cd "$REPO_ROOT"

    # ---------- 正常结束 ----------
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date '+%F %T')] Training finished normally."
        break
    fi

    # ---------- 防死循环 ----------
    retry_count=$((retry_count + 1))
    if [ $retry_count -ge $MAX_RETRY ]; then
        echo "[FATAL] crashed $MAX_RETRY times in a row, giving up."
        exit 2
    fi

    echo "[$(date '+%F %T')] Crash #$retry_count (exit=$EXIT_CODE), restarting in ${RETRY_WAIT}s..."
    sleep $RETRY_WAIT
done
