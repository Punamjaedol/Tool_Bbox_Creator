from pathlib import Path

def debug_info(message):
    print(f"[*] {message}")

def debug_warn(message):
    print(f"[!] {message}")

def debug_item(message):
    print(f"    [-] {message}")

def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """Convert pixel xyxy box to normalized YOLO xywh label."""
    box_w = max(0, x2 - x1)
    box_h = max(0, y2 - y1)
    x_center = x1 + box_w / 2
    y_center = y1 + box_h / 2

    return (
        x_center / img_w,
        y_center / img_h,
        box_w / img_w,
        box_h / img_h,
    )

def yolo_to_xyxy(line, img_w, img_h):
    """Convert one YOLO label line to pixel xyxy."""
    parts = line.split()
    if len(parts) != 5:
        return None

    _, x, y, w, h = parts
    x = float(x) * img_w
    y = float(y) * img_h
    box_w = float(w) * img_w
    box_h = float(h) * img_h

    x1 = int(round(x - box_w / 2))
    y1 = int(round(y - box_h / 2))
    x2 = int(round(x + box_w / 2))
    y2 = int(round(y + box_h / 2))

    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))
    return [x1, y1, x2, y2]

def read_yolo_labels(label_path):
    """Read YOLO label lines."""
    label_path = Path(label_path)
    if not label_path.exists():
        return []
    return [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]

def save_yolo_label(label_path, boxes, class_id=0):
    """Save boxes in YOLO label format."""
    label_path = Path(label_path)
    with open(label_path, "w", encoding="utf-8") as f:
        for box in boxes:
            x, y, w, h = box
            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    debug_info(f"Saved label: {label_path} boxes={len(boxes)}")

def save_xyxy_labels(label_path, boxes, img_w, img_h, class_id=0):
    """Save pixel xyxy boxes as YOLO labels."""
    yolo_boxes = []
    for x1, y1, x2, y2 in boxes:
        if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
            continue
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        yolo_boxes.append(xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h))

    save_yolo_label(label_path, yolo_boxes, class_id=class_id)
