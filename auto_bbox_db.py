import json
from pathlib import Path

import DBConnection_mariadb as dbconn
from DBConnection_mariadb import select, insert, update, delete
from config import WORK_DIR

CLASS_TABLE = "db_class_info"
BBOX_TABLE = "db_bbox_data"
LOCAL_CLASSES_PATH = WORK_DIR / "local_classes.json"

_connected = False
# ---------- 연결 상태 ----------
def is_connected():
    return _connected

def get_db_config():
    return dbconn.get_db_config()

def try_connect(host, user, password, port, dbname, save=True):
    global _connected
    dbconn.set_db_config(host, user, password, port, dbname, save=save)
    _connected = dbconn.test_connection()
    return _connected

def disconnect():
    global _connected
    _connected = False


# ---------- 로컬 폴백 저장소 (DB 연결 안 됐을 때) ----------
def _load_local_classes():
    if not LOCAL_CLASSES_PATH.exists():
        LOCAL_CLASSES_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOCAL_CLASSES_PATH.write_text(json.dumps([], ensure_ascii=False, indent=2), encoding="utf-8")
        return []
    try:
        return json.loads(LOCAL_CLASSES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []

def _save_local_classes(rows):
    LOCAL_CLASSES_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_CLASSES_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

def _next_local_id(rows):
    return max((r["id"] for r in rows), default=0) + 1

# --- class_info 테이블 ---
def get_active_classes():
    """Retrieve all active classes (USE_YN='Y') ordered by CLASS_ID. Returns a list of (CLASS_ID, CLASS_NAME) tuples, or None if the query fails."""
    if is_connected():
        return select(
            CLASS_TABLE,
            columns="CLASS_ID, CLASS_NAME",
            where="USE_YN = %s",
            params=("Y",),
            extra="ORDER BY CLASS_ID",
        )
    rows = sorted((r for r in _load_local_classes() if r["use_yn"] == "Y"), key=lambda r: r["id"])
    return [(r["id"], r["name"]) for r in rows]

def find_class(class_name, use_yn=None):
    """Retrieve a class by its name. Returns a tuple (CLASS_ID, CLASS_NAME, USE_YN), or None if not found."""
    if is_connected():
        where = "CLASS_NAME = %s"
        params = [class_name]
        if use_yn is not None:
            where += " AND USE_YN = %s"
            params.append(use_yn)
        rows = select(
            CLASS_TABLE,
            columns="CLASS_ID, CLASS_NAME, USE_YN",
            where=where,
            params=tuple(params),
            extra="ORDER BY CLASS_ID LIMIT 1",
        )
        return rows[0] if rows else None
    for r in sorted(_load_local_classes(), key=lambda r: r["id"]):
        if r["name"] == class_name and (use_yn is None or r["use_yn"] == use_yn):
            return (r["id"], r["name"], r["use_yn"])
    return None

def insert_class(class_name, use_yn="Y", remark=None):
    if is_connected():
        return insert(CLASS_TABLE, {"CLASS_NAME": class_name, "USE_YN": use_yn, "REMARK": remark})
    rows = _load_local_classes()
    rows.append({"id": _next_local_id(rows), "name": class_name, "use_yn": use_yn, "remark": remark})
    _save_local_classes(rows)


def set_class_use_yn(class_id, use_yn):
    if is_connected():
        return update(CLASS_TABLE, {"USE_YN": use_yn}, "CLASS_ID = %s", (class_id,))
    rows = _load_local_classes()
    for r in rows:
        if r["id"] == class_id:
            r["use_yn"] = use_yn
    _save_local_classes(rows)

def rename_class(class_id, new_name):
    if is_connected():
        return update(CLASS_TABLE, {"CLASS_NAME": new_name}, "CLASS_ID = %s", (class_id,))
    rows = _load_local_classes()
    for r in rows:
        if r["id"] == class_id:
            r["name"] = new_name
    _save_local_classes(rows)

def reset_local_classes():
    """로컬 클래스 저장소를 완전히 초기화."""
    _save_local_classes([])
    
# --- bbox_data 테이블 ---
def clear_bboxes(image_id):
    if is_connected():
        return delete(BBOX_TABLE, "IMAGE_ID = %s", (image_id,))
    return True

def insert_bbox(image_id, bbox_seq, class_id, x1, y1, x2, y2):
    if is_connected():
        return insert(
            BBOX_TABLE,
            {
                "IMAGE_ID": image_id,
                "BBOX_SEQ": bbox_seq,
                "CLASS_ID": class_id,
                "X_MIN": x1, "Y_MIN": y1, "X_MAX": x2, "Y_MAX": y2,
            },
        )
    return None  # 미연결 시 아무것도 안 함