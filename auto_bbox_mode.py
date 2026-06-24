"""
auto_bbox_mode.py

Build a YOLO detection model for hanging conveyor products.

Target object:
- Use one class only: hanging_product
"""

import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from auto_bbox_gui import open_gui
from auto_bbox_utils import (
    debug_info,
    debug_item,
    debug_warn,
    read_yolo_labels,
    save_yolo_label,
    xyxy_to_yolo,
)
from config import *


def load_model(model_path=None):
    """Load a YOLO detection model."""
    debug_info(f"Initializing YOLO model: {model_path}")

    try:
        if not model_path:
            model_path = "yolov8n.pt"
        model = YOLO(model_path)
        debug_info(f"Loaded YOLO model: {model_path}")
        return model
    except Exception as e:
        debug_warn(f"Error loading {model_path}: {e}")
        return None


def load_imgs(input_dir, ):
    """Load source images that match the target conveyor screenshot pattern."""
    input_path = Path(input_dir)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    image_files = [f for f in input_path.rglob("*") if f.suffix.lower() in valid_exts]
    
    if not image_files:
        debug_warn(f"No images found in input directory: {input_path}")
        return []
    else: debug_info(f"Found {len(image_files)} target images.")
    print()
    return image_files


def create_yolo_dirs(dataset_dir):
    """Create YOLO train/val folder structure."""
    dataset_dir = Path(dataset_dir)
    dirs = {
        "train_images": dataset_dir / "images" / "train",
        "val_images": dataset_dir / "images" / "val",
        "train_labels": dataset_dir / "labels" / "train",
        "val_labels": dataset_dir / "labels" / "val",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
        debug_info(f"Prepared directory: {path}")

    return dirs


def auto_create_labels(model, image_files, label_dir, visualized_dir, conf=0.05, imgsz=640):
    """
    Create first-pass labels with a pretrained model.

    Important:
    - These labels are only a draft.
    - Manually delete wrong boxes and fix missing product boxes before training.
    """
    if model is None:
        debug_warn("Model is not loaded. Auto-label step skipped.")
        return

    label_dir = Path(label_dir)
    visualized_dir = Path(visualized_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    visualized_dir.mkdir(parents=True, exist_ok=True)

    debug_info("Starting first-pass auto labeling.")

    for img_path in image_files:
        debug_info(f"Processing image: {img_path}")
        img = cv2.imread(str(img_path))
        if img is None:
            debug_warn(f"Failed to read image: {img_path}")
            continue

        img_h, img_w = img.shape[:2]
        boxed_img = img.copy()
        yolo_boxes = []

        results = model.predict(
            source=img_path,
            conf=conf,
            imgsz=imgsz,
            agnostic_nms=True,
            verbose=False,
        )
        debug_info("Model inference completed.")

        for r_idx, result in enumerate(results):
            if not hasattr(result, "boxes") or result.boxes is None:
                debug_warn(f"No boxes found in result index {r_idx}")
                continue

            debug_info(f"Processing result index {r_idx}, boxes: {len(result.boxes)}")
            for b_idx, box in enumerate(result.boxes):
                score = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                x1 = max(0, min(img_w - 1, int(round(x1))))
                y1 = max(0, min(img_h - 1, int(round(y1))))
                x2 = max(0, min(img_w - 1, int(round(x2))))
                y2 = max(0, min(img_h - 1, int(round(y2))))

                debug_item(f"Box {b_idx}: ({x1}, {y1}, {x2}, {y2}) conf={score:.4f}")
                yolo_boxes.append(xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h))
                cv2.rectangle(boxed_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        save_yolo_label(label_dir / f"{img_path.stem}.txt", yolo_boxes)
        cv2.imwrite(str(visualized_dir / f"{img_path.stem}_draft_boxed{img_path.suffix}"), boxed_img)
        debug_info(f"Total boxes detected: {len(yolo_boxes)}")


def split_dataset(image_files, draft_label_dir, dataset_dir, val_ratio=0.2, seed=42):
    """Copy images and reviewed labels into YOLO train/val structure."""
    dirs = create_yolo_dirs(dataset_dir)
    image_files = list(image_files)
    random.Random(seed).shuffle(image_files)

    val_count = max(1, int(len(image_files) * val_ratio)) if len(image_files) > 1 else 0
    val_set = set(image_files[:val_count])

    debug_info(f"Splitting dataset. train={len(image_files) - val_count}, val={val_count}")

    for img_path in image_files:
        split = "val" if img_path in val_set else "train"
        dst_img_dir = dirs[f"{split}_images"]
        dst_label_dir = dirs[f"{split}_labels"]

        label_path = Path(draft_label_dir) / f"{img_path.stem}.txt"
        if not label_path.exists():
            debug_warn(f"Missing label. Empty label will be created: {label_path}")
            label_path.write_text("", encoding="utf-8")

        shutil.copy2(img_path, dst_img_dir / img_path.name)
        shutil.copy2(label_path, dst_label_dir / f"{img_path.stem}.txt")
        debug_item(f"{split}: {img_path.name}")


def augment_train_dataset(dataset_dir, copies_per_image=3):
    """
    Add simple augmentations that preserve bounding boxes.

    Used augmentations:
    - horizontal flip
    - brightness/contrast change
    - light blur/noise
    """
    train_img_dir = Path(dataset_dir) / "images" / "train"
    train_label_dir = Path(dataset_dir) / "labels" / "train"
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [f for f in train_img_dir.iterdir() if f.suffix.lower() in valid_exts]

    debug_info(f"Starting augmentation. images={len(image_files)}, copies_per_image={copies_per_image}")

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            debug_warn(f"Failed to read image for augmentation: {img_path}")
            continue

        label_lines = read_yolo_labels(train_label_dir / f"{img_path.stem}.txt")

        for idx in range(copies_per_image):
            aug_img = img.copy()
            aug_labels = list(label_lines)
            aug_name = f"{img_path.stem}_aug{idx}{img_path.suffix}"

            if idx % 3 == 0:
                aug_img = cv2.flip(aug_img, 1)
                aug_labels = flip_yolo_labels(label_lines)
                debug_item(f"Horizontal flip: {aug_name}")
            elif idx % 3 == 1:
                alpha = random.uniform(0.85, 1.20)
                beta = random.randint(-20, 20)
                aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)
                debug_item(f"Brightness alpha={alpha:.2f}, beta={beta}: {aug_name}")
            else:
                aug_img = cv2.GaussianBlur(aug_img, (3, 3), 0)
                noise = np.random.normal(0, 4, aug_img.shape).astype(np.int16)
                aug_img = np.clip(aug_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                debug_item(f"Blur/noise: {aug_name}")

            cv2.imwrite(str(train_img_dir / aug_name), aug_img)
            (train_label_dir / f"{Path(aug_name).stem}.txt").write_text(
                "\n".join(aug_labels) + ("\n" if aug_labels else ""),
                encoding="utf-8",
            )


def flip_yolo_labels(label_lines):
    """Flip YOLO labels horizontally."""
    flipped = []
    for line in label_lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id, x, y, w, h = parts
        flipped_x = 1.0 - float(x)
        flipped.append(f"{class_id} {flipped_x:.6f} {float(y):.6f} {float(w):.6f} {float(h):.6f}")
    return flipped


def write_data_yaml(dataset_dir, class_name="hanging_product"):
    """Write Ultralytics data.yaml."""
    dataset_dir = Path(dataset_dir).resolve()
    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset_dir.as_posix()}",
                "train: images/train",
                "val: images/val",
                "nc: 1",
                f"names: ['{class_name}']",
                "",
            ]
        ),
        encoding="utf-8",
    )
    debug_info(f"Saved dataset yaml: {data_yaml}")
    return data_yaml


def train_detection_model(data_yaml, pretrained_model, project_dir, model_name, epochs=80, imgsz=640, batch=4):
    """Train a one-class YOLO detection model."""
    model = load_model(pretrained_model)
    if model is None:
        debug_warn("Training skipped because model failed to load.")
        return None

    debug_info("Starting YOLO training.")
    debug_item(f"data={data_yaml}")
    debug_item(f"pretrained_model={pretrained_model}")
    debug_item(f"epochs={epochs}, imgsz={imgsz}, batch={batch}")

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(project_dir),
        name=model_name,
        task="detect",
        single_cls=True,
        patience=20,
    )
    debug_info("Training completed.")
    return results


def build_hanging_product_model():
    """Prepare data and train a YOLO model"""

    debug_info("AUTO BBOX MODE")
    debug_item(f"work_dir={WORK_DIR}")
    debug_item(f"class_name={CLASS_NAME}")

    image_files = load_imgs(TOOL_DIR / "input")
    if not image_files: return

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

    if AUTO_CREATE_DRAFT_LABELS:
        auto_label_model = load_model(AUTO_LABEL_MODEL)
        auto_create_labels(
            model=auto_label_model,
            image_files=image_files,
            label_dir=DRAFT_LABEL_DIR,
            visualized_dir=DRAFT_VIS_DIR,
            conf=0.05,
            imgsz=IMGSZ,
        )
        debug_warn("Review draft labels before training.")
        debug_warn(f"Edit labels in: {DRAFT_LABEL_DIR}")
        debug_warn(f"Check draft boxes in: {DRAFT_VIS_DIR}")

    if REVIEW_LABELS: 
        open_gui(image_files=image_files, label_dir=DRAFT_LABEL_DIR,)

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
    build_hanging_product_model()
