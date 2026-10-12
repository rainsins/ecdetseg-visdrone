#!/bin/bash
cd "$(dirname "$0")"

exec pixi run python ../train.py \
  -c ../configs/ecdet/ecdet_s.yml \
  -d xpu \
  -r ../outputs/ecdet_s_o3652coco_visdrone/last.pth \
  --output-dir ../outputs/ecdet_s_o3652coco_visdrone \
  -u epochs=60 flat_epoch=56 no_aug_epoch=4 mosaic_prob=0.25 mosaic_epoch=52
