<!-- ppt-master-schema: spec-lock/v1 -->
# Execution Lock

## canvas
- viewBox: 0 0 1280 720
- format: PPT 16:9

## communication
- primary_language: zh-CN
- audience: 具备目标检测基础的导师与课题组同学
- objective: 解释频域诊断与HBS/API，评估实验成本并区分论文事实和迁移假设
- core_message: 压背景、抬前景的训练期增强；推理不加模块，训练成本明显增加
- consumption_mode: balanced

## mode
- mode: custom
- mode_behavior: 问题、诊断、设计、证据、实验与代价、研究设想，每页一个判断

## visual_style
- visual_style: custom
- visual_style_behavior: 白底深蓝标题，青绿HBS及收益，橙色API及成本；完整论文图主导，原生说明并置，深蓝封面结尾

## colors
- background: #FFFFFF
- secondary_background: #F1F5F9
- primary: #142D4E
- accent: #087E8B
- secondary_accent: #BC5A24
- body_text: #243447
- secondary_text: #58677A
- divider: #D5DFE8

## typography
- font_family: Microsoft YaHei, Arial
- title_family: Microsoft YaHei, Arial
- body_family: Microsoft YaHei, Arial
- body: 24
- title: 40
- subtitle: 30
- annotation: 20
- footnote: 16
- display: 64
- lead: 28

## icons
- library: none
- inventory: none

## page_rhythm
- P01: anchor
- P02: dense
- P03: dense
- P04: dense
- P05: dense
- P06: dense
- P07: dense
- P08: dense
- P09: dense
- P10: dense
- P11: breathing
- P12: dense
- P13: dense
- P14: dense
- P15: dense
- P16: dense
- P17: dense
- P18: anchor

## pptx_structure
- mode: flat

## images
- fig1: images/fig1.png | source=user | crop=no-crop
- fig2: images/fig2.png | source=user | crop=no-crop
- fig3: images/fig3.png | source=user | crop=no-crop
- hbs_detail: images/hbs_detail.png | source=user | crop=no-crop
- fig4: images/fig4.png | source=user | crop=no-crop
- fig5: images/fig5.png | source=user | crop=no-crop
- fig6: images/fig6.png | source=user | crop=no-crop
- fig7: images/fig7.png | source=user | crop=no-crop
- table1: images/table1.png | source=user | crop=no-crop
- table2: images/table2.png | source=user | crop=no-crop
- table3: images/table3.png | source=user | crop=no-crop
- table4: images/table4.png | source=user | crop=no-crop
- table5: images/table5.png | source=user | crop=no-crop

## forbidden
- `mask`, `<style>`, `class`, external CSS, `<foreignObject>`, `textPath`, `@font-face`, `<animate*>`, `<set>`, `<script>` / event attributes, `<iframe>`
- HTML named entities in text; write typography as raw Unicode and escape XML reserved characters
- 不要用那个模板了 (user)
