from DBConnection_mariadb import select, insert, update, delete

CLASS_TABLE = "db_class_info"
BBOX_TABLE = "db_bbox_data"

# --- class_info 테이블 ---
def get_active_classes():
    """Retrieve all active classes (USE_YN='Y') ordered by CLASS_ID. Returns a list of (CLASS_ID, CLASS_NAME) tuples, or None if the query fails."""
    return select(
        CLASS_TABLE,
        columns="CLASS_ID, CLASS_NAME",
        where="USE_YN = %s",
        params=("Y",),
        extra="ORDER BY CLASS_ID",
    )

def find_class(class_name, use_yn=None):
    """Retrieve a class by its name. Returns a tuple (CLASS_ID, CLASS_NAME, USE_YN), or None if not found."""
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

def insert_class(class_name, use_yn="Y", remark=None):
    return insert(CLASS_TABLE, {"CLASS_NAME": class_name, "USE_YN": use_yn, "REMARK": remark})

def set_class_use_yn(class_id, use_yn):
    return update(CLASS_TABLE, {"USE_YN": use_yn}, "CLASS_ID = %s", (class_id,))

def rename_class(class_id, new_name):
    return update(CLASS_TABLE, {"CLASS_NAME": new_name}, "CLASS_ID = %s", (class_id,))


# --- bbox_data 테이블 ---
def clear_bboxes(image_id):
    return delete(BBOX_TABLE, "IMAGE_ID = %s", (image_id,))

def insert_bbox(image_id, bbox_seq, class_id, x1, y1, x2, y2):
    return insert(
        BBOX_TABLE,
        {
            "IMAGE_ID": image_id,
            "BBOX_SEQ": bbox_seq,
            "CLASS_ID": class_id,
            "X_MIN": x1, "Y_MIN": y1, "X_MAX": x2, "Y_MAX": y2,
        },
    )