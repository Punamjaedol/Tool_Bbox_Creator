"""
main.py

Build a YOLO detection model for hanging conveyor products.
"""
import shutil
# from pathlib import Path
from auto_bbox_gui import open_gui
from auto_bbox_utils import *
from config import *

def main_fn():
    """Prepare data and train a YOLO model"""

    debug_info("AUTO BBOX MODE")
    debug_item(f"work_dir={WORK_DIR}")
    debug_item(f"class_name={CLASS_NAME}")
    
    # image_files = load_imgs(TOOL_DIR / "input")
    # if not image_files: return
    # print(image_files)
    # print(RESET_WORK_DIR)
    
    if RESET_WORK_DIR:
        if WORK_DIR.exists():
            shutil.rmtree(WORK_DIR)
            debug_info(f"Removed directory: {WORK_DIR}")
        else:
            WORK_DIR.mkdir(parents=True, exist_ok=True)
            debug_info(f"Created directory: {WORK_DIR}")
    else:
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        debug_info(f"Using work directory: {WORK_DIR}")    
    
    open_gui(label_dir=DRAFT_LABEL_DIR,)

    """
    if PREPARE_DATASET:
        split_dataset(
            image_files=image_files,
            draft_label_dir=DRAFT_LABEL_DIR,
            dataset_dir=DATASET_DIR,
            val_ratio=0.2,
        )

        if AUGMENT_DATASET:
            augment_train_dataset(DATASET_DIR, copies_per_image=3)

        data_yaml = write_data_yaml(DATASET_DIR, class_name=CLASS_NAME)
    else:
        data_yaml = DATASET_DIR / "data.yaml"
        debug_warn("PREPARE_DATASET=False. Dataset split/augmentation was not started.")

    if TRAIN_MODEL:
        train_detection_model(
            data_yaml=data_yaml,
            pretrained_model=PRETRAINED_MODEL,
            project_dir=RUNS_DIR,
            model_name=MODEL_NAME,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
        )
    else:
        debug_warn("TRAIN_MODEL=False. Training was not started.")
        debug_warn("After reviewing labels, set PREPARE_DATASET=True and TRAIN_MODEL=True if you want to train.")
"""

if __name__ == "__main__":
    main_fn()
