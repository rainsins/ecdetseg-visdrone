#!/usr/bin/env python3
"""Generate the ECDet-S RK3588 group-meeting deck.

Single source of truth: MODEL_CONTRACT.md next to this script. Every number,
threshold and command below is copied from that file; nothing is recalled.

Usage:
    python make_ppt.py

Requires python-pptx only.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

try:
    import pptx
except ImportError:  # pragma: no cover - one-shot bootstrap
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-pptx"])
    import pptx

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Length, Pt
from pptx.oxml.ns import qn

# --------------------------------------------------------------------------
# Fonts
# --------------------------------------------------------------------------
# 微软雅黑 (Microsoft YaHei) is the requested face and is present on Windows.
# On Linux it is usually absent; the closest metric-compatible CJK face is
# Noto Sans CJK SC.  python-pptx only writes the latin typeface, so the East
# Asian slot must be set through a:ea as well -- see set_font().
CN_FONT = "微软雅黑"
CN_FONT_FALLBACK = "Noto Sans CJK SC"
MONO_FONT = "Consolas"

# --------------------------------------------------------------------------
# Palette (solid fills only; no images anywhere)
# --------------------------------------------------------------------------
INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x5A, 0x5A, 0x5A)
ACCENT = RGBColor(0x0B, 0x5C, 0xAB)
GOOD = RGBColor(0x1E, 0x7D, 0x3A)
WARN = RGBColor(0xB3, 0x2D, 0x12)
RULE = RGBColor(0xD0, 0xD7, 0xDE)
BAND = RGBColor(0xEE, 0xF3, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

# --------------------------------------------------------------------------
# Layout constants (13.333 x 7.5 in, 16:9)
# --------------------------------------------------------------------------
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
MARGIN_X = Inches(0.62)
CONTENT_TOP = Inches(1.52)
CONTENT_W = SLIDE_W - 2 * MARGIN_X

TITLE_PT = 30
COVER_TITLE_PT = 40
BODY_PT = 18
TABLE_PT = 14
CODE_PT = 14
MIN_PT = 14


# --------------------------------------------------------------------------
# Run formatting helpers
# --------------------------------------------------------------------------
def _apply_ea(run, font_name: str) -> None:
    """Write a:ea so the CJK glyphs actually use `font_name`."""
    rpr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rpr.find(qn(tag))
        if el is None:
            el = rpr.makeelement(qn(tag), {})
            rpr.append(el)
        el.set("typeface", font_name)


def set_font(run, size: int = BODY_PT, bold: bool = False,
             color: RGBColor = INK, mono: bool = False) -> None:
    """Single entry point for every run, including table cells."""
    assert size >= MIN_PT, f"font {size}pt is below the {MIN_PT}pt floor"
    face = MONO_FONT if mono else CN_FONT
    run.font.name = face              # latin slot
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    _apply_ea(run, CN_FONT if not mono else MONO_FONT)


def set_cell(cell, text: str, size: int = TABLE_PT, bold: bool = False,
             color: RGBColor = INK, align=PP_ALIGN.LEFT, mono: bool = False):
    cell.text = ""
    tf = cell.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    set_font(run, size=size, bold=bold, color=color, mono=mono)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.margin_left = Inches(0.07)
    cell.margin_right = Inches(0.07)
    cell.margin_top = Inches(0.03)
    cell.margin_bottom = Inches(0.03)


def add_textbox(slide, left, top, width, height):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = None               # never rely on autofit
    return box, tf


def para(tf, text: str, size: int = BODY_PT, bold: bool = False,
         color: RGBColor = INK, space_before=0,
         level: int = 0, mono: bool = False, first: bool = False):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    # callers pass either a point count or an already-built Emu/Length
    p.space_before = (space_before if isinstance(space_before, Length)
                      else Pt(space_before))
    p.space_after = Pt(0)
    p.level = level
    run = p.add_run()
    run.text = text
    set_font(run, size=size, bold=bold, color=color, mono=mono)
    return p


def rule_line(slide, top, left=None, width=None, color=RULE):
    left = MARGIN_X if left is None else left
    width = CONTENT_W if width is None else width
    shape = slide.shapes.add_shape(1, left, top, width, Pt(1.1))  # rectangle
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def band(slide, left, top, width, height, color=BAND):
    shape = slide.shapes.add_shape(1, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


# --------------------------------------------------------------------------
# Slide chrome
# --------------------------------------------------------------------------
def add_slide_with_title(prs, title: str, kicker: str | None = None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = WHITE

    box, tf = add_textbox(slide, MARGIN_X, Inches(0.52), CONTENT_W, Inches(0.72))
    p = tf.paragraphs[0]
    p.space_before = Pt(0)
    run = p.add_run()
    run.text = title
    set_font(run, size=TITLE_PT, bold=True, color=INK)
    rule_line(slide, Inches(1.30), color=ACCENT)

    if kicker:
        kbox, ktf = add_textbox(
            slide, MARGIN_X, Inches(1.34), CONTENT_W, Inches(0.30))
        para(ktf, kicker, size=MIN_PT, color=MUTED, first=True)
    return slide


def add_notes(slide, lines):
    tf = slide.notes_slide.notes_text_frame
    tf.text = ""
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        run = p.add_run()
        run.text = line
        set_font(run, size=MIN_PT, color=RGBColor(0x40, 0x40, 0x40))


def bullets(slide, items, top=None, width=None, size=BODY_PT,
            left=None, mono=False, shade_first=False):
    """At most 7 bullets; each is a plain string or (text, bold) tuple."""
    assert len(items) <= 7, f"{len(items)} bullets exceed the 7-item cap"
    left = MARGIN_X if left is None else left
    top = CONTENT_TOP if top is None else top
    width = CONTENT_W if width is None else width

    box, tf = add_textbox(slide, left, top, width, SLIDE_H - top - Inches(0.45))
    for i, item in enumerate(items):
        text, bold = (item, False) if isinstance(item, str) else item
        para(tf, text, size=size, bold=bold, mono=mono,
             space_before=0 if i == 0 else Pt(9), first=(i == 0))
    return box


# --------------------------------------------------------------------------
# P1 cover
# --------------------------------------------------------------------------
def slide_cover(prs, report_date: str):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = WHITE

    band(slide, Emu(0), Emu(0), SLIDE_W, Inches(0.22), ACCENT)

    box, tf = add_textbox(
        slide, MARGIN_X, Inches(2.05), CONTENT_W, Inches(1.55))
    para(tf, "ECDET-S × RK3588 端侧部署：模型契约与验收结果",
         size=COVER_TITLE_PT, bold=True, color=INK, first=True)

    rule_line(slide, Inches(3.72), width=Inches(4.2), color=ACCENT)

    sub, stf = add_textbox(
        slide, MARGIN_X, Inches(3.95), CONTENT_W, Inches(1.0))
    para(stf, report_date, size=BODY_PT, color=MUTED, first=True)
    para(stf, "汇报人：（占位）", size=BODY_PT, color=MUTED,
         space_before=Pt(6))

    foot, ftf = add_textbox(
        slide, MARGIN_X, Inches(6.35), CONTENT_W, Inches(0.5))
    para(ftf, "事实来源：deploy/MODEL_CONTRACT.md（冻结版）",
         size=MIN_PT, color=MUTED, first=True)

    add_notes(slide, [
        "开场一句话：模型已经冻结，今天汇报契约内容和仿真验收结果。",
        "先声明所有数字都来自 MODEL_CONTRACT.md，没有现场估算。",
        "真板数据还没有，第 11、12 页会明确标出未验证项。",
    ])


# --------------------------------------------------------------------------
# P2 one-page summary
# --------------------------------------------------------------------------
def slide_summary(prs):
    slide = add_slide_with_title(prs, "一页摘要", "§0 合同摘要 / §9 精度契约")

    items = [
        ("输入：静态 1×3×640×640，uint8 RGB NCHW，0..255", False),
        ("量化：RKNN INT8 + FP16 hybrid，auto_hybrid_cos_thresh=0.9999", False),
        ("预处理：PIL bilinear 直接拉伸，无 letterbox、无 pad", False),
        ("归一化烧进 .rknn；板端不再做 mean/std，不喂 float [0,1]", False),
        ("后处理：sigmoid → top-300 → 拉伸逆变换；无 NMS", False),
    ]
    bullets(slide, items, top=Inches(1.78), size=BODY_PT)

    # headline number block
    top = Inches(4.52)
    band(slide, MARGIN_X, top, CONTENT_W, Inches(2.05))

    lbox, ltf = add_textbox(
        slide, MARGIN_X + Inches(0.28), top + Inches(0.12),
        Inches(3.5), Inches(1.80))
    para(ltf, "28.71", size=64, bold=True, color=ACCENT, first=True)
    para(ltf, "RKNN INT8 AP", size=MIN_PT, color=MUTED, space_before=Pt(2))

    mbox, mtf = add_textbox(
        slide, MARGIN_X + Inches(4.05), top + Inches(0.12),
        Inches(3.5), Inches(1.80))
    para(mtf, "28.82", size=64, bold=True, color=INK, first=True)
    para(mtf, "FP32 ONNX AP（参照）", size=MIN_PT, color=MUTED,
         space_before=Pt(2))

    rbox, rtf = add_textbox(
        slide, MARGIN_X + Inches(7.85), top + Inches(0.12),
        Inches(4.2), Inches(1.80))
    para(rtf, "−0.11 分", size=64, bold=True, color=GOOD, first=True)
    para(rtf, "drop 0.1133 ≤ 1.0 → PASSED",
         size=MIN_PT, color=GOOD, space_before=Pt(2))

    add_notes(slide, [
        "先给结论：INT8 相比 FP32 只掉 0.11 分，单侧门限通过。",
        "五个约束一次性讲完：输入、量化、预处理、归一化、后处理。",
        "强调板端不许再做归一化，这是最常见的翻车点。",
    ])


# --------------------------------------------------------------------------
# P3 pipeline and frozen commands
# --------------------------------------------------------------------------
def slide_pipeline(prs):
    slide = add_slide_with_title(prs, "流水线与冻结命令", "§2 Pipeline / §1 制品哈希")

    stages = ["PyTorch", "ONNX\nopset 17", "RKNN\nINT8+FP16", "RK3588"]
    width = Inches(2.55)
    gap = Inches(0.42)
    left = MARGIN_X
    top = Inches(1.80)
    for i, name in enumerate(stages):
        shape = band(slide, left, top, width, Inches(0.92))
        shape.fill.fore_color.rgb = ACCENT if i in (1, 2) else BAND
        tf = shape.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        for j, part in enumerate(name.split("\n")):
            p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER
            run = p.add_run()
            run.text = part
            set_font(run, size=BODY_PT if j == 0 else MIN_PT,
                     bold=(j == 0),
                     color=WHITE if i in (1, 2) else INK)
        if i < len(stages) - 1:
            abox, atf = add_textbox(
                slide, left + width, top + Inches(0.18), gap, Inches(0.6))
            p = atf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            run = p.add_run()
            run.text = "→"
            set_font(run, size=BODY_PT, bold=True, color=MUTED)
        left += width + gap

    box, tf = add_textbox(
        slide, MARGIN_X, Inches(2.98), CONTENT_W, Inches(1.9))
    para(tf, "run_all.sh 五步（冻结调用）", size=BODY_PT, bold=True,
         color=INK, first=True)
    steps = [
        "1  export_onnx.py --keep-lqe       → opset-17 ONNX",
        "2  calib.py                        → 300 张 PNG + 非有限值过滤",
        "3  preprocess.py                   → 契约冒烟测试",
        "4  convert.py --hybrid-cos-thresh 0.9999   → .rknn",
        "5  sim_eval.py --fp32-baseline     → mAP",
    ]
    for s in steps:
        para(tf, s, size=CODE_PT, mono=True, color=INK, space_before=Pt(3))

    cbox, ctf = add_textbox(
        slide, MARGIN_X, Inches(4.98), CONTENT_W, Inches(0.5))
    para(ctf, "TRAIN_ENV=default RKNN_ENV=rknn ./deploy/run_all.sh",
         size=CODE_PT, mono=True, color=ACCENT, first=True)

    wbox, wtf = add_textbox(
        slide, MARGIN_X, Inches(5.62), CONTENT_W, Inches(1.35))
    para(wtf, "两个 .rknn 构建的哈希仅作记录，不是回归门：",
         size=BODY_PT, bold=True, color=WARN, first=True)
    para(wtf, "ff61f31c… 49,837,410 B · AP 28.7075（当前）",
         size=MIN_PT, mono=True, color=MUTED, space_before=Pt(3))
    para(wtf, "1d44dc17… 49,772,463 B · AP 28.6887（归档）",
         size=MIN_PT, mono=True, color=MUTED, space_before=Pt(2))
    para(wtf, "重建不可字节复现 → 用 AP 对比 gate.reference_AP",
         size=MIN_PT, color=MUTED, space_before=Pt(3))

    add_notes(slide, [
        "四段流水线，中间两步是工具链，最后一步才是验收。",
        "冻结调用必须带三个环境变量；CALIB_ENV 不设时等于 RKNN_ENV。",
        "两张哈希不是回归判据，重建出来的字节一定不同，比 AP。",
    ])


# --------------------------------------------------------------------------
# P4 model identity
# --------------------------------------------------------------------------
def slide_identity(prs):
    slide = add_slide_with_title(prs, "模型身份", "§1 Model identity")

    box, tf = add_textbox(
        slide, MARGIN_X, Inches(1.78), Inches(5.55), Inches(4.9))
    facts = [
        ("架构：ECDet-S（D-FINE 变体）", False),
        ("ViTAdapter + HybridEncoder + ECTransformer", False),
        ("输入：1×3×640×640，静态", False),
        ("Queries：300", False),
        ("制品：49,772,463 B（≈48 MiB，归档构建）", False),
        ("校准：300 张 VisDrone train，seed=42，分层", False),
        ("RKNN target rk3588 · toolkit 2.5.0", False),
    ]
    bullets(slide, facts, top=Inches(1.78), width=Inches(5.55), size=BODY_PT)

    # class table on the right
    left = Inches(6.72)
    width = Inches(5.99)
    rows, cols = 11, 2
    table_shape = slide.shapes.add_table(
        rows, cols, left, Inches(1.78), width, Inches(4.55))
    table = table_shape.table
    table.columns[0].width = Inches(0.95)
    table.columns[1].width = width - Inches(0.95)

    set_cell(table.cell(0, 0), "ID", size=TABLE_PT, bold=True,
             color=WHITE, align=PP_ALIGN.CENTER)
    set_cell(table.cell(0, 1), "VisDrone 类名（顺序不可变）", size=TABLE_PT,
             bold=True, color=WHITE)
    classes = ["pedestrian", "people", "bicycle", "car", "van",
               "truck", "tricycle", "awning-tricycle", "bus", "motor"]
    for i, name in enumerate(classes, start=1):
        set_cell(table.cell(i, 0), str(i - 1), size=TABLE_PT,
                 align=PP_ALIGN.CENTER)
        set_cell(table.cell(i, 1), name, size=TABLE_PT, mono=True)

    for r in range(rows):
        for c in range(cols):
            cell = table.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = (
                ACCENT if r == 0 else (BAND if r % 2 == 0 else WHITE))

    add_notes(slide, [
        "架构是 D-FINE 变体，骨干换成 ViTAdapter，编码器是 HybridEncoder。",
        "类序 0..9 是合同里写死的，category_id 不是 1..10。",
        "48 MiB 是归档构建的字节数，当前重建是 49,837,410 B。",
    ])


# --------------------------------------------------------------------------
# P5 preprocessing contract
# --------------------------------------------------------------------------
def slide_preprocess(prs):
    slide = add_slide_with_title(prs, "预处理契约（§3）", "§3.2 normative / §3.4 metadata")

    box, tf = add_textbox(
        slide, MARGIN_X, Inches(1.78), Inches(6.0), Inches(1.5))
    para(tf, "BGR uint8 → RGB → PIL bilinear 拉伸 640×640 → CHW",
         size=BODY_PT, bold=True, color=ACCENT, first=True)
    para(tf, "→ uint8 0..255，shape (1,3,640,640)", size=BODY_PT,
         color=INK, space_before=Pt(4))

    forbidden = [
        ("禁止 letterbox、禁止 pad", True),
        ("禁止 /255，禁止在 Python 里做 mean/std", True),
        ("禁止用 cv2 静默替换 PIL bilinear", True),
    ]
    bullets(slide, forbidden, top=Inches(3.02), width=Inches(6.0),
            size=BODY_PT)

    # measured penalty box
    top = Inches(4.72)
    band(slide, MARGIN_X, top, Inches(6.0), Inches(1.62))
    pbox, ptf = add_textbox(
        slide, MARGIN_X + Inches(0.22), top + Inches(0.14),
        Inches(5.6), Inches(1.35))
    para(ptf, "cv2 INTER_AREA 的实测代价", size=MIN_PT, bold=True,
         color=WARN, first=True)
    para(ptf, "FP32  28.82 → 28.65", size=CODE_PT, mono=True,
         color=INK, space_before=Pt(3))
    para(ptf, "INT8  28.69 → 28.44（归档构建）", size=CODE_PT, mono=True,
         color=INK, space_before=Pt(2))

    # metadata neutral panel
    mleft = Inches(6.90)
    mtop = Inches(1.78)
    band(slide, mleft, mtop, Inches(5.81), Inches(2.35), RGBColor(0xF6, 0xF7, 0xF9))
    mbox, mtf = add_textbox(
        slide, mleft + Inches(0.22), mtop + Inches(0.14),
        Inches(5.4), Inches(2.1))
    para(mtf, "pad/scale 是中性值，后处理不用", size=BODY_PT, bold=True,
         color=INK, first=True)
    for line in ("pad_x = 0     pad_y = 0",
                 "scale_x = 1.0     scale_y = 1.0",
                 "后处理直接用 orig_w / orig_h"):
        para(mtf, line, size=CODE_PT, mono=True, color=MUTED,
             space_before=Pt(4))

    nbox, ntf = add_textbox(
        slide, mleft, Inches(4.35), Inches(5.81), Inches(2.0))
    para(ntf, "冒烟检查要求 orig_w > 640 或 orig_h > 640",
         size=BODY_PT, bold=True, color=WARN, first=True)
    para(ntf, "640×640 源图会让 PIL 原样返回，resize 根本没跑，",
         size=MIN_PT, color=MUTED, space_before=Pt(5))
    para(ntf, "比对必然通过。--require-resize 强制这一点。",
         size=MIN_PT, color=MUTED, space_before=Pt(2))

    add_notes(slide, [
        "预处理只有一句话：BGR 转 RGB，PIL 双线性直接拉伸到 640。",
        "四条禁令对应的都是真实踩过的坑，尤其 cv2 替换会掉 0.17 分。",
        "pad 和 scale 留着是为了元数据完整，后处理完全不读它们。",
    ])


# --------------------------------------------------------------------------
# P6 RKNN normalization
# --------------------------------------------------------------------------
def slide_norm(prs):
    slide = add_slide_with_title(prs, "RKNN 归一化（§4）", "归一化在 .rknn 内部，运行时输入仍是 uint8")

    box, tf = add_textbox(
        slide, MARGIN_X, Inches(1.78), Inches(7.0), Inches(1.3))
    para(tf, "mean / std 烧进 .rknn（RGB，0..255 域）", size=BODY_PT,
         bold=True, color=INK, first=True)
    para(tf, "板端输入保持 uint8，不做任何浮点变换", size=MIN_PT,
         color=MUTED, space_before=Pt(4))

    code_top = Inches(3.05)
    band(slide, MARGIN_X, code_top, Inches(7.0), Inches(1.85),
         RGBColor(0xF6, 0xF7, 0xF9))
    cbox, ctf = add_textbox(
        slide, MARGIN_X + Inches(0.22), code_top + Inches(0.14),
        Inches(6.6), Inches(1.6))
    for i, line in enumerate((
        "mean_values=[[123.675, 116.28, 103.53]]",
        "std_values =[[ 58.395,  57.12,  57.375]]",
        "quant_img_RGB2BGR=False",
    )):
        para(ctf, line, size=CODE_PT, mono=True, color=INK,
             first=(i == 0), space_before=0 if i == 0 else Pt(4))

    rules = [
        ("禁止板端二次归一化（会变成两次减均值两次除方差）", True),
        ("禁止喂 float32 [0,1]，除非显式配置 inputs_pass_through 并验证", True),
        ("quant_img_RGB2BGR=False 因为预处理已经输出 RGB", False),
    ]
    bullets(slide, rules, top=Inches(5.05), width=Inches(7.0), size=BODY_PT)

    # right: numbers
    rleft = Inches(8.10)
    for i, (val, label, color) in enumerate((
        ("123.675 / 116.28 / 103.53", "mean（RGB）", ACCENT),
        ("58.395 / 57.12 / 57.375", "std（RGB）", INK),
    )):
        top = Inches(1.78) + Inches(i * 1.55)
        band(slide, rleft, top, Inches(4.61), Inches(1.35))
        vbox, vtf = add_textbox(
            slide, rleft + Inches(0.2), top + Inches(0.12),
            Inches(4.2), Inches(1.1))
        para(vtf, val, size=BODY_PT, bold=True, color=color,
             mono=True, first=True)
        para(vtf, label, size=MIN_PT, color=MUTED, space_before=Pt(3))

    add_notes(slide, [
        "这三个数是 ImageNet ×255 的 RGB 版本，顺序不能错。",
        "归一化是 config 参数，编译进模型，运行时不用再管。",
        "板端唯一要做的是别多此一举，二次归一化会把分数打成噪声。",
    ])


# --------------------------------------------------------------------------
# P7 outputs and postprocess
# --------------------------------------------------------------------------
def _code_panel(slide, left, top, width, height, lines, size=CODE_PT):
    band(slide, left, top, width, height, RGBColor(0xF6, 0xF7, 0xF9))
    box, tf = add_textbox(
        slide, left + Inches(0.2), top + Inches(0.12),
        width - Inches(0.4), height - Inches(0.2))
    for i, line in enumerate(lines):
        para(tf, line, size=size, mono=True, color=INK,
             first=(i == 0), space_before=0 if i == 0 else Pt(3))
    return box


def slide_postprocess(prs):
    slide = add_slide_with_title(prs, "输出与后处理（§5–6）", "decoder 原始输出 → COCO 检测框")

    # output table
    rows, cols = 3, 4
    table_shape = slide.shapes.add_table(
        rows, cols, MARGIN_X, Inches(1.76), CONTENT_W, Inches(1.22))
    table = table_shape.table
    widths = [Inches(2.55), Inches(2.3), Inches(2.6), Inches(4.64)]
    for i, w in enumerate(widths):
        table.columns[i].width = w
    head = ["张量", "形状", "dtype", "语义"]
    for c, h in enumerate(head):
        set_cell(table.cell(0, c), h, size=TABLE_PT, bold=True, color=WHITE)
    set_cell(table.cell(1, 0), "pred_logits", size=TABLE_PT, mono=True)
    set_cell(table.cell(1, 1), "[1, 300, 10]", size=TABLE_PT, mono=True)
    set_cell(table.cell(1, 2), "float32", size=TABLE_PT, mono=True)
    set_cell(table.cell(1, 3), "pre-sigmoid 类别 logits", size=TABLE_PT)
    set_cell(table.cell(2, 0), "pred_boxes", size=TABLE_PT, mono=True)
    set_cell(table.cell(2, 1), "[1, 300, 4]", size=TABLE_PT, mono=True)
    set_cell(table.cell(2, 2), "float32", size=TABLE_PT, mono=True)
    set_cell(table.cell(2, 3), "归一化 cxcywh（640 拉伸系）", size=TABLE_PT)
    for r in range(rows):
        for c in range(cols):
            cell = table.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = (
                ACCENT if r == 0 else (BAND if r == 1 else WHITE))

    # algorithm
    abox, atf = add_textbox(
        slide, MARGIN_X, Inches(3.18), Inches(6.4), Inches(0.4))
    para(atf, "解码顺序", size=BODY_PT, bold=True, color=INK, first=True)
    _code_panel(slide, MARGIN_X, Inches(3.62), Inches(6.4), Inches(2.55), [
        "sigmoid(logits) → flatten 3000",
        "top-300 → label=idx%10, query=idx//10",
        "cxcywh → xyxy",
        "x *= orig_w   y *= orig_h",
        "clip → COCO xywh；无 NMS",
    ])

    # identify-by-count rule
    rleft = Inches(7.30)
    band(slide, rleft, Inches(3.18), Inches(5.41), Inches(1.32),
         RGBColor(0xFD, 0xF2, 0xEC))
    ibox, itf = add_textbox(
        slide, rleft + Inches(0.2), Inches(3.30), Inches(5.05), Inches(1.1))
    para(itf, "按元素数识别，不按输出顺序", size=BODY_PT, bold=True,
         color=WARN, first=True)
    para(itf, "3000 → logits    1200 → boxes", size=CODE_PT,
         mono=True, color=INK, space_before=Pt(4))

    nbox, ntf = add_textbox(
        slide, rleft, Inches(4.68), Inches(5.41), Inches(1.9))
    para(ntf, "split_outputs 拒绝猜测：", size=MIN_PT, bold=True,
         color=INK, first=True)
    para(ntf, "num_classes == 4 直接抛错（两个张量都以 4 结尾）",
         size=MIN_PT, color=MUTED, space_before=Pt(4))
    para(ntf, "同角色多个 4 维候选也抛错，不静默误配",
         size=MIN_PT, color=MUTED, space_before=Pt(3))
    para(ntf, "category_id 是 0..9，不是 1..10", size=MIN_PT,
         color=MUTED, space_before=Pt(3))

    add_notes(slide, [
        "图里没有 TopK，这步故意留在 float32，便于和 PyTorch 对齐和调阈值。",
        "一个 query 可以同时出现在多个类下，这是 D-FINE 的官方行为，不是 bug。",
        "输出顺序 RKNN 不保证，必须按 3000/1200 认，交换了照样能出框但全错。",
    ])


# --------------------------------------------------------------------------
# P8 calibration contract and the incident
# --------------------------------------------------------------------------
def slide_calibration(prs):
    slide = add_slide_with_title(
        prs, "校准契约与教训（§7）", "300 张、seed=42、分层抽样 + 非有限值过滤")

    facts = [
        ("数据集 VisDrone train，300 张，seed=42", False),
        ("分层：按图内主类 + D'Hondt 配额，每类至少 1 张", False),
        ("变换：与推理同一条 PIL bilinear 拉伸路径", False),
        ("重叠检查：assert train/val 文件名不相交", False),
    ]
    bullets(slide, facts, top=Inches(1.78), width=Inches(6.3), size=BODY_PT)

    # the two gathers
    gtop = Inches(4.30)
    band(slide, MARGIN_X, gtop, Inches(6.3), Inches(2.05),
         RGBColor(0xFD, 0xF2, 0xEC))
    gbox, gtf = add_textbox(
        slide, MARGIN_X + Inches(0.22), gtop + Inches(0.13),
        Inches(5.9), Inches(1.8))
    para(gtf, "图里有 2 个 decoder GatherElements，必须全部探测",
         size=MIN_PT, bold=True, color=WARN, first=True)
    para(gtf, "GatherElements_output_0      1/300 出 +inf",
         size=MIN_PT, mono=True, color=WARN, space_before=Pt(5))
    para(gtf, "GatherElements_1_output_0    全部干净",
         size=MIN_PT, mono=True, color=GOOD, space_before=Pt(3))

    # incident timeline
    rleft = Inches(7.15)
    band(slide, rleft, Inches(1.78), Inches(5.56), Inches(4.57),
         RGBColor(0xF6, 0xF7, 0xF9))
    tbox, ttf = add_textbox(
        slide, rleft + Inches(0.24), Inches(1.92), Inches(5.1), Inches(4.3))
    para(ttf, "事故复盘：INT8 AP = 0.00", size=BODY_PT, bold=True,
         color=WARN, first=True)
    timeline = [
        "① 旧过滤器取 matches[-1]",
        "    → 只盯住干净的第二个节点",
        "② 日志报 all 300 images clean",
        "③ 那张 inf 图留在校准集里",
        "④ RKNN 把某层 scale 标成 inf",
        "⑤ 整个 INT8 模型分数趋近 0",
        "⑥ FP32 同批帧仍有 35.08 AP",
    ]
    for line in timeline:
        para(ttf, line, size=MIN_PT, color=INK, space_before=Pt(6),
             mono=line.startswith(" "))

    fix, ftf = add_textbox(
        slide, rleft + Inches(0.24), Inches(5.72), Inches(5.1), Inches(0.7))
    para(ftf, "修复：探测全部节点，任一非有限即换图",
         size=MIN_PT, bold=True, color=GOOD, first=True)

    add_notes(slide, [
        "这是本次最值得讲的一页：0.00 不是模型坏了，是一张校准图。",
        "关键点在于单节点探测不可靠，两个 gather 只有一个会溢出。",
        "修复后 AP 回到 28.71，替换记录写进了合同 §7。",
    ])


# --------------------------------------------------------------------------
# P9 quantization and export
# --------------------------------------------------------------------------
def slide_quant(prs):
    slide = add_slide_with_title(prs, "量化与导出（§8）",
                                 "w8a8 + auto_hybrid 0.9999")

    # keep-lqe
    band(slide, MARGIN_X, Inches(1.76), Inches(6.1), Inches(1.5))
    lbox, ltf = add_textbox(
        slide, MARGIN_X + Inches(0.22), Inches(1.88), Inches(5.7), Inches(1.3))
    para(ltf, "必须 --keep-lqe", size=BODY_PT, bold=True, color=WARN,
         first=True)
    para(ltf, "no-LQE FP32 上限仅 28.29；LQE 是 28.82", size=MIN_PT,
         color=INK, space_before=Pt(4))
    para(ltf, "无 LQE 的 ONNX 不是这个模型", size=MIN_PT,
         color=MUTED, space_before=Pt(2))

    # clamp
    band(slide, Inches(7.02), Inches(1.76), Inches(5.69), Inches(1.5))
    cbox, ctf = add_textbox(
        slide, Inches(7.24), Inches(1.88), Inches(5.3), Inches(1.3))
    para(ctf, "score-head clamp 是必需的", size=BODY_PT, bold=True,
         color=WARN, first=True)
    para(ctf, "输入 ±10   输出 ±30   最终 logits ±30", size=CODE_PT,
         mono=True, color=INK, space_before=Pt(4))

    # why clamp
    wbox, wtf = add_textbox(
        slide, MARGIN_X, Inches(3.44), Inches(6.1), Inches(1.5))
    para(wtf, "为什么必须 clamp", size=BODY_PT, bold=True, color=INK,
         first=True)
    para(wtf, "部分帧 FP32 encoder logits 低至约 −310", size=MIN_PT,
         color=MUTED, space_before=Pt(5))
    para(wtf, "不 clamp 时 RKNN 按这条长尾标定 → 裁掉约 +2.6 的少数正峰",
         size=MIN_PT, color=MUTED, space_before=Pt(3))
    para(wtf, "→ query 选择被毁，分数全塌", size=MIN_PT, color=WARN,
         space_before=Pt(3))

    # cosine threshold
    ttop = Inches(5.02)
    band(slide, MARGIN_X, ttop, CONTENT_W, Inches(1.42),
         RGBColor(0xEE, 0xF6, 0xEF))
    tbox, ttf_ = add_textbox(
        slide, MARGIN_X + Inches(0.22), ttop + Inches(0.12),
        CONTENT_W - Inches(0.44), Inches(1.2))
    para(ttf_, "auto_hybrid_cos_thresh = 0.9999 是 cosine 距离界，"
               "不是相似度阈值", size=BODY_PT, bold=True, color=GOOD,
         first=True)
    para(ttf_, "RKNN 默认 0.98；越大 → 偏差一点点就回退 FP16 → 更激进",
         size=MIN_PT, color=MUTED, space_before=Pt(4))
    para(ttf_, "代价：模型从约 37 MiB 增至约 49 MiB",
         size=MIN_PT, color=MUTED, space_before=Pt(2))

    add_notes(slide, [
        "LQE 有两个数：28.29 和 28.82，差了 0.5 分，这是保留它的理由。",
        "clamp 不是运行时后处理，它是导出图的一部分。",
        "0.9999 容易讲反，它是距离界，越大越激进。",
    ])


# --------------------------------------------------------------------------
# P10 accuracy acceptance
# --------------------------------------------------------------------------
def slide_accuracy(prs):
    slide = add_slide_with_title(prs, "精度验收（§9）",
                                 "VisDrone val 548 张，COCOeval bbox")

    rows, cols = 7, 5
    table_shape = slide.shapes.add_table(
        rows, cols, MARGIN_X, Inches(1.74), Inches(7.55), Inches(2.55))
    table = table_shape.table
    widths = [Inches(1.35), Inches(1.55), Inches(1.75), Inches(1.6),
              Inches(1.3)]
    for i, w in enumerate(widths):
        table.columns[i].width = w

    for c, h in enumerate(["指标", "PyTorch", "FP32 ONNX", "RKNN INT8", "drop"]):
        set_cell(table.cell(0, c), h, size=TABLE_PT, bold=True,
                 color=WHITE,
                 align=PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT)

    data = [
        ("AP", "28.82", "28.8208", "28.7075", "0.1133", True),
        ("AP50", "48.54", "48.5410", "48.5089", "0.0321", False),
        ("AP75", "28.63", "28.6290", "28.3918", "0.2372", False),
        ("APs", "18.67", "18.6712", "18.5602", "0.1111", False),
        ("APm", "41.47", "41.4749", "41.3606", "0.1143", False),
        ("APl", "61.67", "61.6736", "61.4027", "0.2708", False),
    ]
    for r, (name, a, b, c_, d, strong) in enumerate(data, start=1):
        set_cell(table.cell(r, 0), name, size=TABLE_PT, bold=strong)
        set_cell(table.cell(r, 1), a, size=TABLE_PT, mono=True,
                 align=PP_ALIGN.RIGHT)
        set_cell(table.cell(r, 2), b, size=TABLE_PT, mono=True,
                 align=PP_ALIGN.RIGHT)
        set_cell(table.cell(r, 3), c_, size=TABLE_PT, mono=True, bold=strong,
                 align=PP_ALIGN.RIGHT)
        set_cell(table.cell(r, 4), d, size=TABLE_PT, mono=True,
                 align=PP_ALIGN.RIGHT)

    for r in range(rows):
        for c in range(cols):
            cell = table.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = (
                ACCENT if r == 0 else (BAND if r % 2 == 0 else WHITE))

    # gate panel
    rleft = Inches(8.45)
    band(slide, rleft, Inches(1.74), Inches(4.26), Inches(2.55),
         RGBColor(0xEE, 0xF6, 0xEF))
    gbox, gtf = add_textbox(
        slide, rleft + Inches(0.22), Inches(1.86), Inches(3.85), Inches(2.3))
    para(gtf, "单侧门", size=BODY_PT, bold=True, color=GOOD, first=True)
    para(gtf, "INT8_AP ≥ reference_AP − 1.0", size=MIN_PT, mono=True,
         color=INK, space_before=Pt(5))
    para(gtf, "28.7075 ≥ 28.8208 − 1.0", size=MIN_PT, mono=True,
         color=INK, space_before=Pt(3))
    para(gtf, "→ PASSED", size=BODY_PT, bold=True, color=GOOD,
         space_before=Pt(6))
    para(gtf, "INT8 反超基线也算通过", size=MIN_PT, color=MUTED,
         space_before=Pt(4))

    # gate json
    jbox, jtf = add_textbox(
        slide, MARGIN_X, Inches(4.48), Inches(7.55), Inches(1.5))
    para(jtf, "sim_eval_summary.json → gate 字段", size=BODY_PT,
         bold=True, color=INK, first=True)
    _code_panel(slide, MARGIN_X, Inches(4.90), Inches(7.55), Inches(1.42), [
        '"passed": true, "one_sided": true, "fail_drop": 1.0',
        '"reference_AP": 28.820758052964102',
        '"drop": 0.11329371972247415',
    ], size=MIN_PT)

    wbox, wtf = add_textbox(
        slide, Inches(8.45), Inches(4.48), Inches(4.26), Inches(1.6))
    para(wtf, "注意", size=BODY_PT, bold=True, color=WARN, first=True)
    para(wtf, "模拟器 ≈1.21 s/img，不是板端时延", size=MIN_PT,
         color=MUTED, space_before=Pt(4))
    para(wtf, "旧文档的 AP≥28.6 / drop≤0.3 已退役", size=MIN_PT,
         color=MUTED, space_before=Pt(3))
    para(wtf, "归档 28.6887 只是记录，不是阈值", size=MIN_PT,
         color=MUTED, space_before=Pt(3))

    add_notes(slide, [
        "六行指标，只有 AP 行加粗，其他都是分解项。",
        "门限是单侧的，因为 INT8 偶然超过 FP32 不算回归。",
        "最后强调 1.21 s/img 是仿真速度，别当成板端数字。",
    ])


# --------------------------------------------------------------------------
# P11 board validation protocol
# --------------------------------------------------------------------------
def slide_board(prs):
    slide = add_slide_with_title(
        prs, "板端验证协议（§10–11）", "顺序不可换：张量 → 输出 → mAP → 时延")

    steps = [
        ("①", "张量字节级一致", "max_abs_diff = 0", "用非 640 源图"),
        ("②", "输出对比", "logits / boxes / 解码框", "先张量后输出"),
        ("③", "COCO mAP", "≥ 28.8208 − 1.0", "50–100 张 val"),
        ("④", "时延 / 温度 / 功耗", "warmup 20 + 100 次", "30–60 min 持续测"),
    ]
    top = Inches(1.74)
    height = Inches(0.86)
    for i, (num, name, detail, note) in enumerate(steps):
        y = top + Inches(i * (0.98))
        band(slide, MARGIN_X, y, Inches(7.3), height,
             BAND if i % 2 == 0 else RGBColor(0xF6, 0xF7, 0xF9))
        nbox, ntf = add_textbox(
            slide, MARGIN_X + Inches(0.14), y + Inches(0.10),
            Inches(0.5), Inches(0.6))
        para(ntf, num, size=BODY_PT, bold=True, color=ACCENT, first=True)
        tbox, ttf = add_textbox(
            slide, MARGIN_X + Inches(0.62), y + Inches(0.06),
            Inches(2.6), Inches(0.75))
        para(ttf, name, size=BODY_PT, bold=True, color=INK, first=True)
        dbox, dtf = add_textbox(
            slide, MARGIN_X + Inches(3.30), y + Inches(0.06),
            Inches(2.25), Inches(0.75))
        para(dtf, detail, size=MIN_PT, mono=True, color=INK, first=True)
        rbox, rtf = add_textbox(
            slide, MARGIN_X + Inches(5.62), y + Inches(0.06),
            Inches(1.6), Inches(0.75))
        para(rtf, note, size=MIN_PT, color=MUTED, first=True)

    # require-resize warning
    wtop = Inches(5.72)
    band(slide, MARGIN_X, wtop, Inches(7.3), Inches(0.95),
         RGBColor(0xFD, 0xF2, 0xEC))
    wbox, wtf = add_textbox(
        slide, MARGIN_X + Inches(0.16), wtop + Inches(0.08),
        Inches(7.0), Inches(0.8))
    para(wtf, "--require-resize 不能省", size=MIN_PT, bold=True,
         color=WARN, first=True)
    para(wtf, "640×640 源图 PIL 原样返回，resize 跳过 → 白测",
         size=MIN_PT, color=MUTED, space_before=Pt(2))

    # C++ panel
    rleft = Inches(8.28)
    band(slide, rleft, Inches(1.74), Inches(4.43), Inches(3.55),
         RGBColor(0xF6, 0xF7, 0xF9))
    cbox, ctf = add_textbox(
        slide, rleft + Inches(0.22), Inches(1.86), Inches(4.0), Inches(3.3))
    para(ctf, "C++ 路径要点（§10.4）", size=BODY_PT, bold=True,
         color=ACCENT, first=True)
    for line in ("want_float = 1",
                 "pass_through = 0",
                 "按元素数 3000 / 1200 识别输出",
                 "rknn_inputs_set + NCHW + UINT8",
                 "不用零拷贝 NC1HWC2"):
        para(ctf, "· " + line, size=MIN_PT, color=INK, space_before=Pt(6))

    nbox, ntf = add_textbox(
        slide, rleft, Inches(5.42), Inches(4.43), Inches(1.35))
    para(ntf, "桩测试证明的是调用序列与 decoder 一致性，",
         size=MIN_PT, color=MUTED, first=True)
    para(ntf, "不证明 NPU 数值与真实时延。",
         size=MIN_PT, color=MUTED, space_before=Pt(2))
    para(ntf, "§11.3 与温升/功耗必须在真板跑。", size=MIN_PT,
         color=WARN, space_before=Pt(4))

    add_notes(slide, [
        "四步有严格顺序，第 1 步不过就别查量化，那是白费时间。",
        "第 3 步的参照物是 gate.reference_AP，不是硬编码的旧值 28.6887。",
        "C++ 那栏三个点是最容易写错的：want_float、pass_through、元素数。",
    ])


# --------------------------------------------------------------------------
# P12 failure matrix + invariants + next steps
# --------------------------------------------------------------------------
def slide_failure(prs):
    slide = add_slide_with_title(
        prs, "失败矩阵 + 不变量 + 下一步", "§12 / §13")

    # failure table
    rows = 7
    table_shape = slide.shapes.add_table(
        rows, 2, MARGIN_X, Inches(1.72), Inches(7.05), Inches(3.35))
    table = table_shape.table
    table.columns[0].width = Inches(2.75)
    table.columns[1].width = Inches(4.30)
    set_cell(table.cell(0, 0), "症状", size=TABLE_PT, bold=True,
             color=WHITE)
    set_cell(table.cell(0, 1), "正确做法", size=TABLE_PT, bold=True,
             color=WHITE)
    fails = [
        ("分数全 ~0.01", "确认 --keep-lqe + clamp + 0.9999"),
        ("AP 卡在 28.4", "cv2 换回 PIL 双线性"),
        ("框偏移 / 缩放", "用 x*=orig_w, y*=orig_h，不减 pad"),
        ("双归一化", "删掉板端 mean/std"),
        ("哈希当回归门", "比 AP，不比字节"),
        ("NHWC / NCHW 混淆", "对齐 data_format 并转置"),
    ]
    for r, (sym, fix) in enumerate(fails, start=1):
        set_cell(table.cell(r, 0), sym, size=TABLE_PT)
        set_cell(table.cell(r, 1), fix, size=TABLE_PT, color=INK)
    for r in range(rows):
        for c in range(2):
            cell = table.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = (
                ACCENT if r == 0 else (BAND if r % 2 == 0 else WHITE))

    # invariants
    rleft = Inches(7.92)
    band(slide, rleft, Inches(1.72), Inches(4.79), Inches(3.35),
         RGBColor(0xEE, 0xF6, 0xEF))
    ibox, itf = add_textbox(
        slide, rleft + Inches(0.22), Inches(1.84), Inches(4.35), Inches(3.1))
    para(itf, "6 条关键不变量（§13）", size=BODY_PT, bold=True,
         color=GOOD, first=True)
    inv = [
        "640×640 静态 uint8 RGB NCHW",
        "PIL 双线性直接拉伸，无 pad/letterbox",
        "mean/std 在 .rknn 内，Python 不做",
        "输出 3000 logits / 1200 boxes",
        "x*=orig_w, y*=orig_h，无 pad 减法",
        "无 NMS；category_id 0..9",
    ]
    for line in inv:
        para(itf, "· " + line, size=MIN_PT, color=INK, space_before=Pt(5))

    # next steps
    ntop = Inches(5.28)
    band(slide, MARGIN_X, ntop, CONTENT_W, Inches(1.42),
         RGBColor(0xF6, 0xF7, 0xF9))
    nbox, ntf = add_textbox(
        slide, MARGIN_X + Inches(0.22), ntop + Inches(0.12),
        CONTENT_W - Inches(0.44), Inches(1.2))
    para(ntf, "下一步 / 风险", size=BODY_PT, bold=True, color=INK,
         first=True)
    para(ntf, "① RK3588 真板跑 §11.1–11.4，补齐板端 AP 与时延   "
              "② 量化批次噪声 0.02 分，需多次重建取均值",
         size=MIN_PT, color=MUTED, space_before=Pt(4))
    para(ntf, "③ 产品侧 score_thresh 0.25–0.4 待定   "
              "TODO：板端温度/功耗实测数据",
         size=MIN_PT, color=WARN, space_before=Pt(3))

    add_notes(slide, [
        "六个症状是最常被问到的，对应六条不变量。",
        "最后一行是明天要讨论的：真板数据还没拿到，这里如实标 TODO。",
        "量化噪声 0.02 分不建议追，属批次波动。",
    ])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
REPORT_DATE = "2026-10-28"
OUTPUT_NAME = "ECDet_RK3588_组会汇报.pptx"
TITLES = [
    "ECDet-S × RK3588 端侧部署：模型契约与验收结果",
    "一页摘要",
    "流水线与冻结命令",
    "模型身份",
    "预处理契约（§3）",
    "RKNN 归一化（§4）",
    "输出与后处理（§5–6）",
    "校准契约与教训（§7）",
    "量化与导出（§8）",
    "精度验收（§9）",
    "板端验证协议（§10–11）",
    "失败矩阵 + 不变量 + 下一步",
]


def build(path: Path) -> Path:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    slide_cover(prs, REPORT_DATE)
    slide_summary(prs)
    slide_pipeline(prs)
    slide_identity(prs)
    slide_preprocess(prs)
    slide_norm(prs)
    slide_postprocess(prs)
    slide_calibration(prs)
    slide_quant(prs)
    slide_accuracy(prs)
    slide_board(prs)
    slide_failure(prs)

    prs.save(str(path))
    return path


def self_check(path: Path) -> None:
    prs = Presentation(str(path))
    assert len(prs.slides) == 12, f"expected 12 slides, got {len(prs.slides)}"
    print(f"\nself-check: {len(prs.slides)} slides OK")
    for i, slide in enumerate(prs.slides, start=1):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                texts.append(shape.text_frame.text.strip().split("\n")[0])
        headline = texts[0] if texts else "(no text)"
        assert texts, f"slide {i} has no text"
        print(f"  P{i:02d}  shapes={len(slide.shapes):2d}  {headline}")


def main() -> int:
    here = Path(__file__).resolve().parent
    contract = here / "MODEL_CONTRACT.md"
    if not contract.exists():
        print(f"WARNING: {contract} not found", file=sys.stderr)
    else:
        print(f"source: {contract.name}")

    out = here / OUTPUT_NAME
    try:
        build(out)
        self_check(out)
    except Exception:
        traceback.print_exc()
        return 1

    size = out.stat().st_size
    print(f"\noutput: {out}")
    print(f"size  : {size:,} bytes ({size / 1024 / 1024:.2f} MiB)")
    print("\ntitles:")
    for i, t in enumerate(TITLES, start=1):
        print(f"  {i:2d}. {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
