对我们来说有三点：第一，VisDrone 的 tiny 目标同样受高频影响，值得先做频域诊断；第二，SET 只改训练，适合 EdgeCrafter 这种端侧模型；第三，它不加 FLOPs，可以做 INT8 量化后再验证增益是否保持。
