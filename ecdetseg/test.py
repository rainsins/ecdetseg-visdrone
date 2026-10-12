import torch
ck = torch.load("outputs/p30u/e1_s42/last.pth", map_location="cpu", weights_only=False)
print(list(ck.keys()))
print(ck.get("last_epoch"))   # 记录 epoch 的字段，名字可能不同
