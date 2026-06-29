import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from auto_bbox_utils import *
import auto_bbox_db as db
from config import *


# --- connect DB ---
def connect_db(host, user, password, port, dbname):
    """주어진 정보로 DB 연결을 시도하고, 성공/실패와 무관하게 정보를 저장한다."""
    return db.try_connect(host, user, password, port, dbname, save=True)

def disconnect_db():
    db.disconnect()

def is_db_connected():
    return db.is_connected()

def get_db_config():
    return db.get_db_config()

def get_table_name(table_name):
    table_dic = {"class_table": db.CLASS_TABLE,
                "bbox_table": db.BBOX_TABLE}
    return table_dic[table_name]

def set_table_name(table_name, updated_name):
    """DB 모듈의 전역 변수 값을 직접 업데이트합니다."""
    if table_name == "class_table":
        db.CLASS_TABLE = updated_name
    elif table_name == "bbox_table":
        db.BBOX_TABLE = updated_name

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

# --- classes.txt local cache ---
def load_class_names(classes_txt_path):
    classes_txt_path = Path(classes_txt_path)
    if not classes_txt_path.exists():
        debug_warn(f"Classes file not found: {classes_txt_path}")
        debug_info("Trying to populate it from the active class source (DB or local_classes.json).")
        return sync_classes_from_db(classes_txt_path) or []
    return [
        line.strip()
        for line in classes_txt_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
def init_local_classes(classes_txt_path):
    """
    Returns:
      ("db_connected", None)   -> DB 연결 상태에서는 사용 불가
      ("ok", class_names)      -> 초기화 완료 (빈 리스트)
    """
    if db.is_connected():
        return "db_connected", None
    db.reset_local_classes()
    return "ok", sync_classes_from_db(classes_txt_path)

def save_class_names(classes_txt_path, class_names):
    classes_txt_path = Path(classes_txt_path)
    classes_txt_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(class_names)
    if text:
        text += "\n"
    classes_txt_path.write_text(text, encoding="utf-8")

def sync_classes_from_db(classes_txt_path):
    """Synchronize active classes from the database to classes.txt and return the class name list. Returns None if synchronization fails."""
    rows = db.get_active_classes()
    if rows is None:
        debug_warn("Failed to load classes from db_class_info.")
        return None
    class_names = [name for _, name in rows]
    save_class_names(classes_txt_path, class_names)
    return class_names


# --- Class Add/Rename/Hide ---
def add_class(class_name, classes_txt_path):
    """Returns: ("exists", None) | ("ok", class_names)"""
    existing = db.find_class(class_name)
    if existing:
        class_id, _, use_yn = existing
        if use_yn == "Y":
            return "exists", None
        db.set_class_use_yn(class_id, "Y")
    else:
        db.insert_class(class_name, use_yn="Y")
    return "ok", sync_classes_from_db(classes_txt_path)

def rename_class(old_name, new_name, classes_txt_path):
    """Returns: ("same", None) | ("name_taken", None) | ("not_found", None) | ("ok", class_names)"""
    if old_name == new_name:
        return "same", None
    if db.find_class(new_name):
        return "name_taken", None
    row = db.find_class(old_name, "Y")
    if not row:
        return "not_found", None
    db.rename_class(row[0], new_name)
    return "ok", sync_classes_from_db(classes_txt_path)

def hide_class(class_name, classes_txt_path):
    """Returns: ("not_found", None) | ("ok", class_names)"""
    row = db.find_class(class_name, "Y")
    if not row:
        return "not_found", None
    db.set_class_use_yn(row[0], "N")
    return "ok", sync_classes_from_db(classes_txt_path)

def get_db_class_id(class_name):
    """Return the CLASS_ID of an active class, or None if not found."""
    row = db.find_class(class_name, "Y")
    return row[0] if row else None

def sync_model_classes_to_db(model_class_names, classes_txt_path):
    """Add missing auto-label model classes to the database and synchronize the class list."""
    existing = db.get_active_classes()
    existing_names = {name for _, name in existing} if existing else set()
    for class_name in model_class_names:
        if class_name not in existing_names:
            db.insert_class(class_name, use_yn="Y", remark="Added by Auto Label Model")
    return sync_classes_from_db(classes_txt_path)


# --- Save bbox  ---
def save_bboxes_to_db(image_path, class_id, boxes):
    """해당 이미지의 bbox를 전부 지우고 다시 저장."""
    image_id = Path(image_path).name
    if not db.clear_bboxes(image_id):
        debug_warn(f"Failed to clear bbox rows: image_id={image_id}")
        return False

    bbox_seq = 1
    for (x1, y1, x2, y2) in boxes:
        if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
            continue
        db.insert_bbox(image_id, bbox_seq, class_id, x1, y1, x2, y2)
        bbox_seq += 1

    debug_info(f"Saved bbox rows to DB: image_id={image_id} boxes={bbox_seq - 1}")
    return True


# --- Auto Label Model  ---
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

def get_model_class_names(model):
    """Extract the class name list from a YOLO model object."""
    return list(model.names.values())

def setup_auto_label_model(model_path, classes_txt_path):
    """
    Load a YOLO model and synchronize its classes with the database.

    Returns:
        ("invalid_model", None, None)              -> Failed to load the model
        ("db_error", None, None)                   -> Failed to synchronize with the database
        ("ok", model_class_names, class_names)     -> Success
    """
    model = load_model(model_path)
    if model is None:
        return "invalid_model", None, None

    model_class_names = get_model_class_names(model)
    class_names = sync_model_classes_to_db(model_class_names, classes_txt_path)
    if class_names is None:
        return "db_error", None, None

    return "ok", model_class_names, class_names

def auto_create_labels(model, image_files, label_dir, visualized_dir, conf_thres=0.5, imgsz=640):
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
    label_dir.mkdir(parents=True, exist_ok=True)

    debug_info("Starting first-pass auto labeling.")
    all_results = []

    for img_path in image_files:
        debug_info(f"Processing image: {img_path}")
        img = cv2.imread(str(img_path))
        if img is None:
            debug_warn(f"Failed to read image: {img_path}")
            continue

        img_h, img_w = img.shape[:2]
        yolo_boxes = []
        cls_id = -1

        results = model.predict(
            source=img_path,
            conf=conf_thres,
            imgsz=imgsz,
            agnostic_nms=True,
            verbose=False,
        )
        debug_info("Model inference completed.")
        
        img_boxes = []
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
                if score > conf_thres:
                    cls_id = int(box.cls[0])
                    img_boxes.append((x1, y1, x2, y2))
                    yolo_boxes.append(xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h))
                else: continue

        save_boxed_image(str(visualized_dir / f"{img_path.stem}_draft_boxed{img_path.suffix}"), img_path, img_boxes)
        debug_info(f"Total boxes detected: {len(yolo_boxes)}")

        all_results.append({
            "image_path": img_path,
            "class_id": cls_id,
            "yolo_boxes": yolo_boxes, 
            "boxes": img_boxes
        })
    return all_results

def save_boxed_image(save_path, image_path, boxes):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(image_path))
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.imwrite(str(save_path), image)

def auto_label_images(image_dir, auto_label_model_path):
    try:
        image_files = load_imgs(image_dir)
        auto_label_model = load_model(auto_label_model_path)
        
        results = auto_create_labels(
            model=auto_label_model,
            image_files=image_files,
            label_dir=DRAFT_LABEL_DIR,
            visualized_dir=DRAFT_VIS_DIR,
            conf_thres=0.05,
            imgsz=IMGSZ,
        )

        debug_warn("Review draft labels before training.")
        debug_warn(f"Edit labels in: {DRAFT_LABEL_DIR}")
        debug_warn(f"Check draft boxes in: {DRAFT_VIS_DIR}")
        return results
    except Exception as e:
        debug_warn(f"Auto Label Error: {e}")
        return False
    
