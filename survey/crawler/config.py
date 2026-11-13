"""Central configuration for the TOD survey crawler + analysis.

Contains:
- arXiv search queries (full-spectrum small/tiny object detection coverage)
- Venue aliases (for parsing arXiv comment/journal_ref fields)
- Module/direction keyword taxonomy (for the "modules & trends" analysis)
- Detector/baseline keywords (for the "baselines" analysis)
- Domain keywords (for the application-area analysis)

NOTE on keyword matching: single-token keywords are matched with word boundaries
(see analysis/features.py) to avoid false hits like "ct" in "object", "rs" in
"papers", "sea" in "search", "frame" in "framework", "vit" in "activity".
Multi-word phrases are matched as substrings.
"""

# ---------------------------------------------------------------------------
# Date range
# ---------------------------------------------------------------------------
YEAR_START = 2022
YEAR_END = 2026
ARXIV_DATE_FILTER = (
    f"submittedDate:[{YEAR_START}01010000 TO {YEAR_END}12312359]"
)

# ---------------------------------------------------------------------------
# arXiv search queries
# ---------------------------------------------------------------------------
QUERIES = [
    ("small_object_detection", 'all:"small object detection"', "core"),
    ("tiny_object_detection", 'all:"tiny object detection"', "core"),
    ("small_target_detection", 'all:"small target detection"', "core"),
    ("small_object_detector", 'all:"small object detector"', "core"),
    ("detecting_small_objects", 'all:"detecting small objects"', "core"),
    ("small_instance", 'all:"small instance"', "core"),
    ("tiny_object", 'all:"tiny object"', "variant"),
    ("small_objects", 'ti:"small objects"', "variant"),
    ("tiny_instances", 'all:"tiny instance"', "variant"),
    ("infrared_small_target", 'all:"infrared small target"', "domain"),
    ("dim_small_target", 'all:"dim small target"', "domain"),
    ("remote_sensing_small", 'all:"remote sensing" AND all:"small object"', "domain"),
    ("aerial_small", 'all:"aerial" AND all:"small object"', "domain"),
    ("uav_small", 'all:"UAV" AND all:"small object"', "domain"),
    ("drone_small", 'all:"drone" AND all:"small object"', "domain"),
    ("maritime_small", 'all:"maritime" AND all:"small object"', "domain"),
    ("super_resolution_small", 'all:"super-resolution" AND all:"small object"', "technique"),
    ("feature_pyramid_small", 'all:"feature pyramid" AND all:"small object"', "technique"),
    ("frequency_small", 'all:"frequency" AND all:"small object"', "technique"),
    ("attention_small", 'all:"attention" AND all:"small object"', "technique"),
    ("multiscale_small", 'all:"multi-scale" AND all:"small object"', "technique"),
    ("context_small", 'all:"context" AND all:"small object"', "technique"),
    ("detr_small", 'all:"DETR" AND all:"small object"', "technique"),
    ("yolo_small", 'all:"YOLO" AND all:"small object"', "technique"),
    ("fpn_small", 'all:"FPN" AND all:"small object"', "technique"),
    ("query_small", 'all:"query" AND all:"small object"', "technique"),
]

# ---------------------------------------------------------------------------
# Module / direction taxonomy
# ---------------------------------------------------------------------------
MODULE_KEYWORDS = {
    "frequency/spectral": [
        "frequency", "spectral", "fourier", "fft", "wavelet", "dct", "dwt",
        "frequency domain", "high-frequency", "low-frequency",
        "amplitude", "phase spectrum", "frequency enhancement",
        "spectral enhancement", "fourier transform", "band-pass",
    ],
    "super-resolution": [
        "super-resolution", "super resolution", "srgan", "generative",
        "upsampling", "upscale", "super-resolv", "sr branch",
        "high-resolution branch", "generator",
    ],
    "attention/transformer": [
        "attention", "self-attention", "transformer", "channel attention",
        "spatial attention", "non-local", "vision transformer", "vit",
        "cbam", "squeeze-and-excitation", "se block",
    ],
    "multi-scale/feature-fusion": [
        "multi-scale", "multiscale", "feature pyramid", "fpn", "pafpn", "bifpn",
        "feature fusion", "feature aggregation", "cross-scale", "scale-aware",
        "path aggregation",
    ],
    "context/receptive-field": [
        "context", "contextual", "receptive field", "dilated", "atrous",
        "global context", "context enhancement", "large kernel",
    ],
    "query/DETR": [
        "detr", "query", "deformable detr", "dino", "rt-detr", "query-based",
        "object query", "query selection", "denoising query", "set prediction",
    ],
    "loss/label-assignment": [
        "focal loss", "label assignment", "nwd", "wasserstein", "rfla",
        "gaussian", "simota", "task-aligned", "dynamic label",
        "giou", "ciou", "diou", "quality focal loss", "assignment strategy",
    ],
    "data-augmentation": [
        "augmentation", "copy-paste", "copy paste", "mosaic", "tiling",
        "slicing", "mixup", "cutmix", "oversampling", "data synthesis",
    ],
    "knowledge-distillation": [
        "distillation", "teacher-student", "teacher", "student",
        "knowledge transfer", "kd", "distill",
    ],
    "lightweight/efficient": [
        "lightweight", "efficient", "real-time", "mobile", "embedded",
        "low-latency", "low latency", "pruning", "quantization", "flops",
        "tiny model", "compact", "edge device", "edge computing", "on-device",
    ],
    "anchor-free/one-stage": [
        "anchor-free", "anchor free", "fcos", "centernet", "one-stage",
        "single-stage", "yolo", "retinanet", "point-based",
    ],
    "two-stage/region-proposal": [
        "faster r-cnn", "faster rcnn", "faster rcnn", "r-cnn", "rcnn",
        "two-stage", "region proposal", "roi", "mask r-cnn", "cascade",
    ],
    "temporal/video": [
        "video", "temporal", "tracking", "track", "sequence", "motion",
        "optical flow",
    ],
    "uncertainty/robustness": [
        "uncertainty", "robust", "noise", "degradation", "blur", "occlusion",
        "low-quality", "low resolution", "domain adaptation", "generalization",
    ],
}

# ---------------------------------------------------------------------------
# Detector / baseline keywords
# ---------------------------------------------------------------------------
DETECTORS = {
    "YOLO (v5-v11)": ["yolov5", "yolov6", "yolov7", "yolov8", "yolov9",
                       "yolov10", "yolov11", "yolo11", "yolo", "yolox"],
    "Faster R-CNN": ["faster r-cnn", "faster rcnn", "faster-rcnn"],
    "Cascade R-CNN": ["cascade r-cnn", "cascade rcnn"],
    "Mask R-CNN": ["mask r-cnn", "mask rcnn"],
    "FCOS": ["fcos"],
    "RetinaNet": ["retinanet", "retina net"],
    "CenterNet": ["centernet"],
    "SSD": ["ssd"],
    "DETR": ["detr"],
    "Deformable DETR": ["deformable detr", "deformable"],
    "DINO": ["dino"],
    "RT-DETR": ["rt-detr", "rt detr"],
    "D-FINE": ["d-fine", "dfine"],
    "DEIM": ["deim"],
    "EfficientDet": ["efficientdet"],
    "QueryDet": ["querydet"],
    "RFLA": ["rfla"],
    "Swin Transformer": ["swin transformer", "swin"],
    "ViT": ["vision transformer", "vit"],
}

# ---------------------------------------------------------------------------
# Application-domain keywords
# ---------------------------------------------------------------------------
DOMAIN_KEYWORDS = {
    "aerial/UAV": ["aerial", "uav", "drone", "unmanned", "visdrone", "dota",
                   "airborne"],
    "remote-sensing/satellite": ["remote sensing", "satellite", "geospatial",
                                  "earth observation", "optical remote"],
    "infrared": ["infrared", "thermal", "dim small target"],
    "maritime/underwater": ["maritime", "underwater", "marine", "ocean",
                            "vessel", "ship detection"],
    "autonomous-driving/traffic": ["autonomous driving", "traffic", "vehicle",
                                    "pedestrian", "kitti", "bdd", "cityscapes",
                                    "road"],
    "medical": ["medical", "lesion", "tumor", "pathology", "x-ray", "mri",
                "histopathology", "clinical"],
    "surveillance/security": ["surveillance", "security", "crowd", "person",
                               "monitoring"],
    "industrial/defect": ["defect", "industrial", "pcb", "manufacturing",
                           "anomaly"],
}

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
DATASET_KEYWORDS = {
    "VisDrone": ["visdrone"],
    "AI-TOD": ["ai-tod", "ai tod", "aitod"],
    "DOTA": ["dota"],
    "COCO": ["coco", "ms coco", "mscoco"],
    "TinyPerson": ["tinyperson", "tiny person"],
    "SODA": ["soda"],
    "SODA-D": ["soda-d", "sodad"],
    "xView": ["xview"],
    "VEDAI": ["vedai"],
    "SeaPerson": ["seaperson", "sea person"],
}
