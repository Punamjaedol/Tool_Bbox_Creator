from pathlib import Path

TOOL_DIR = Path(r"D:/KCH\workspace/tool_bbox_creator")
WORK_DIR = TOOL_DIR / "workspace"

DRAFT_LABEL_DIR = WORK_DIR / "draft_labels"
DRAFT_VIS_DIR = WORK_DIR / "draft_visualized"
DATASET_DIR = WORK_DIR / "dataset"
RUNS_DIR = WORK_DIR / "runs"

CLASS_NAME = "None"
MODEL_NAME = "hanging_product_yolo"

RESET_WORK_DIR = False
PREPARE_DATASET = False
AUGMENT_DATASET = True
TRAIN_MODEL = False

EPOCHS = 80
IMGSZ = 640
BATCH = 4

