import os, sys, ssl, json
from pathlib import Path


# Windows 환경에서 파이썬 ssl 모듈이 Windows Root Certificate Store(인증서 저장소)에서
# 깨진 인증서를 로드하려고 할 때 ssl.SSLError(NOT_ENOUGH_DATA)가 발생하는 버그를 우회합니다.
try:
    _original_load_default_certs = ssl.SSLContext.load_default_certs
    def _patched_load_default_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
        try:
            _original_load_default_certs(self, purpose)
        except Exception:
            pass
    ssl.SSLContext.load_default_certs = _patched_load_default_certs
except AttributeError:
    pass

# Windows 환경에서 mariadb 라이브러리의 C DLL 관련 예외처리 버그(RuntimeError: Failed to create exception)를
# 피하기 위해 순수 파이썬 라이브러리인 PyMySQL을 우선 사용합니다.
# PyMySQL이 없는 경우에만 기존 mariadb 모듈을 사용하도록 폴백 처리합니다.
try:
    import pymysql
    USE_PYMYSQL = True
    DBError = pymysql.Error
except ImportError:
    USE_PYMYSQL = False
    if os.name == 'nt':
        conda_dll_path = r"D:\conda\conda_envs\myungsung_env\Library\bin"
        if os.path.exists(conda_dll_path):
            os.add_dll_directory(conda_dll_path)
    # pyrefly: ignore [missing-import]
    import mariadb
    DBError = mariadb.Error

# localhost 사용 시 로컬 소켓/네임드파이프로 접근하려다 생기는 연결 실패를 막기 위해
# 명시적으로 127.0.0.1을 지정하여 TCP/IP 연결을 강제합니다.
_SETTINGS_PATH = Path(__file__).resolve().parent / "db_settings.json"

DB_HOST = "127.0.0.1"
DB_USER = "root"
DB_PASSWORD = "root"
DB_PORT = 3306
DB_NAME = "yolobasedDB"

def _load_saved_settings():
    """이전에 GUI에서 저장해둔 접속정보가 있으면 기본값 위에 덮어씌운다."""
    global DB_HOST, DB_USER, DB_PASSWORD, DB_PORT, DB_NAME
    if not _SETTINGS_PATH.exists():
        return
    try:
        data = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
        DB_HOST = data.get("host", DB_HOST)
        DB_USER = data.get("user", DB_USER)
        DB_PASSWORD = data.get("password", DB_PASSWORD)
        DB_PORT = int(data.get("port", DB_PORT))
        DB_NAME = data.get("dbname", DB_NAME)
    except Exception as e:
        print(f"[DB][SETTINGS] Failed to load saved settings: {e}")

_load_saved_settings()

def get_db_config():
    return {"host": DB_HOST, "user": DB_USER, "password": DB_PASSWORD, "port": DB_PORT, "dbname": DB_NAME}

def set_db_config(host, user, password, port, dbname, save=True):
    """GUI의 DB Connect 창에서 입력한 값으로 접속정보를 교체."""
    global DB_HOST, DB_USER, DB_PASSWORD, DB_PORT, DB_NAME
    DB_HOST, DB_USER, DB_PASSWORD, DB_PORT, DB_NAME = host, user, password, int(port), dbname
    if save:
        try:
            _SETTINGS_PATH.write_text(json.dumps(get_db_config(), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[DB][SETTINGS] Failed to save settings: {e}")

def get_connection():
    """
    DB 연결 객체를 반환하는 함수
    """
    if USE_PYMYSQL:
        return pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            port=int(DB_PORT),
            database=DB_NAME,
            charset='utf8mb4',
            connect_timeout=5,

        )
    else:
        return mariadb.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            port=int(DB_PORT),
            database=DB_NAME,
            connect_timeout=5,

        )
def test_connection():
    """연결 시도 후 바로 닫고 성공 여부만 반환."""
    conn = connect()
    if conn:
        close_connection(conn)
        return True
    return False

def connect():
    """
    DB 연결 및 Connection 객체 반환 (실패 시 None)
    """
    try:
        conn = get_connection()
        print(f"[DB][CONNECT] {DB_HOST}:{DB_PORT} -> SUCCESS")
        return conn

    except DBError as e:
        print(f"[DB][DISCONNECT] Error connecting to MariaDB: {e}")
        # sys.exit(1)을 하면 GUI 자체가 다운되므로 경고를 띄울 수 있도록 None을 반환하도록 수정하는 것이 안전합니다.
        return None
        
def close_connection(conn):
    if conn:
        conn.close()

# =====================
#       SQL Query
# =====================
def select(table, columns="*", where=None, params=None, extra=""):
    conn = None
    try:
        conn = connect()
        if not conn:
            return None
        with conn.cursor() as cur:
            sql = f"SELECT {columns} FROM {table}"
            if where:
                sql += f" WHERE {where}"
            if extra:
                sql += f" {extra}"

            cur.execute(sql, params)
            result = cur.fetchall()

            print(f"[DB][SELECT] {table} -> {len(result)} rows")
            return result

    except DBError as e:
        print(f"[DB][SELECT][ERROR] {table} | {e}")
        return None
    finally:
        close_connection(conn)

def is_exist(table, where, params=None):
    try:
        conn = connect()
        if not conn:
            return False
        with conn.cursor() as cur:
            sql = f"SELECT 1 FROM {table} WHERE {where} LIMIT 1"
            cur.execute(sql, params)
            result = cur.fetchone() is not None

            print(f"[DB][EXISTS] {table} -> {result}")    
            return result
    except DBError as e:
        print(f"[DB][SELECT][ERROR] {table} | {e}")
        return False
    finally:
        close_connection(conn)
    
def insert(table, data: dict):
    try:
        conn = connect()
        if not conn:
            return
        cols = ",".join(data.keys())
        vals = ",".join(["%s"] * len(data))
        sql = f"INSERT INTO {table} ({cols}) VALUES ({vals})"
        
        with conn.cursor() as cur:
            cur.execute(sql, tuple(data.values()))
        conn.commit()
        print(f"[DB][INSERT] {table} -> SUCCESS")

    except DBError as e:
        print(f"[DB][INSERT][ERROR] {table} | {e}")
    finally:
        close_connection(conn)

def update(table, data: dict, where, params=None):
    try:
        conn = connect()
        if not conn:
            return
        set_clause = ",".join([f"{k}=%s" for k in data.keys()])
        sql = f"UPDATE {table} SET {set_clause} WHERE {where}"
        values = tuple(data.values())

        if params:
            values = values + tuple(params)
        with conn.cursor() as cur:
            cur.execute(sql, values)
            affected_rows = cur.rowcount
        conn.commit()
        print(f"[DB][UPDATE] {table} -> {affected_rows} rows")
        
    except DBError as e:
        print(f"[DB][UPDATE][ERROR] {table} | {e}")
    finally:
        close_connection(conn)

def delete(table, where, params=None):
    try:
        conn = connect()
        if not conn:
            return False
        sql = f"DELETE FROM {table} WHERE {where}"

        with conn.cursor() as cur:
            cur.execute(sql, params)
            affected_rows = cur.rowcount
        conn.commit()
        print(f"[DB][DELETE] {table} -> {affected_rows} rows")
        return True

    except DBError as e:
        print(f"[DB][DELETE][ERROR] {table} | {e}")
        return False
    finally:
        close_connection(conn)

