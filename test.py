from pycocotools.coco import COCO

coco = COCO("/home/rainsin/Projects/Career/EdgeCrafter/VisDrone/annotations_VisDrone_train.json")
cats = coco.loadCats(coco.getCatIds())
print("类别：", [(c['id'], c['name']) for c in cats])

# 检查所有 annotation 的 category_id
ann_ids = coco.getAnnIds()
anns = coco.loadAnns(ann_ids)
cat_ids = set(a['category_id'] for a in anns)
print("实际出现的 category_id：", sorted(cat_ids))
print("最大 category_id：", max(cat_ids))
