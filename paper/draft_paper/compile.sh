#!/bin/bash

# 自动化学报 LaTeX 编译脚本

MAIN_FILE="main.tex"
PDF_FILE="main.pdf"

echo "📄 开始编译 $MAIN_FILE..."

# 第一次编译
echo "➤ 第 1 次编译（生成辅助文件）..."
xelatex -interaction=nonstopmode $MAIN_FILE > /dev/null 2>&1

# 第二次编译（解决交叉引用和目录）
echo "➤ 第 2 次编译（解决交叉引用）..."
xelatex -interaction=nonstopmode $MAIN_FILE > /dev/null 2>&1

# 检查编译结果
if [ -f "$PDF_FILE" ]; then
    SIZE=$(du -h "$PDF_FILE" | cut -f1)
    echo "✅ 编译成功！"
    echo "📊 PDF 文件: $PDF_FILE ($SIZE)"
    echo "📍 路径: $(pwd)/$PDF_FILE"
else
    echo "❌ 编译失败，请检查 main.tex"
    exit 1
fi

# 清理临时文件（可选）
# rm -f *.aux *.log *.out *.toc
