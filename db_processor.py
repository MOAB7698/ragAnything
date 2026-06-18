"""
╔══════════════════════════════════════════════════════════════════════╗
║  db_processor.py  —  پردازش پایگاه‌های داده برای RAG-Anything       ║
╠══════════════════════════════════════════════════════════════════════╣
║  پشتیبانی:                                                           ║
║    • SQLite   (.sqlite, .db, .sqlite3)  — بدون نیاز به نصب          ║
║    • MySQL    (.sql dump یا اتصال زنده)  — نیاز: pymysql            ║
║    • PostgreSQL                          — نیاز: psycopg2            ║
║    • SQL dump (.sql)                     — parse متنی                ║
║                                                                      ║
║  خروجی: content_list سازگار با RAGAnything.insert_content_list       ║
╚══════════════════════════════════════════════════════════════════════╝

رویکرد:
  هر جدول → یک آیتم "table" در content_list (برای ساختار/schema)
  هر ردیف  → یک آیتم "text"  در content_list (برای محتوای قابل جستجو)

  این ساختار باعث می‌شود:
    - بتوان روی ساختار جدول جستجو کرد ("جدول users چه ستون‌هایی دارد؟")
    - بتوان روی مقادیر جستجو کرد ("ایمیل کاربر علی چیست؟")
    - روابط بین جداول در knowledge graph ذخیره شود
"""

import re
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════════
# تنظیمات پردازش دیتابیس
# ══════════════════════════════════════════════════════════════════════

@dataclass
class DBProcessorConfig:
    # حداکثر تعداد ردیف از هر جدول (None = بدون محدودیت)
    max_rows_per_table: Optional[int] = 1000

    # حداکثر تعداد ستون برای نمایش در هر ردیف متنی
    max_columns_in_text: int = 20

    # آیا ردیف‌های NULL-heavy نادیده گرفته شوند؟
    skip_empty_rows: bool = True

    # آیا schema جداول سیستمی (sqlite_*) نادیده گرفته شود؟
    skip_system_tables: bool = True

    # پیشوند جداولی که باید نادیده گرفته شوند
    excluded_table_prefixes: List[str] = field(
        default_factory=lambda: ["sqlite_", "pg_", "information_schema"]
    )

    # جداولی که صریحاً نادیده گرفته می‌شوند
    excluded_tables: List[str] = field(default_factory=list)

    # حداکثر طول مقدار رشته‌ای در هر سلول (کوتاه‌سازی)
    max_cell_length: int = 500


# ══════════════════════════════════════════════════════════════════════
# ابزارهای مشترک
# ══════════════════════════════════════════════════════════════════════

def _truncate(value: Any, max_len: int) -> str:
    """تبدیل مقدار به رشته و کوتاه‌کردن در صورت نیاز"""
    if value is None:
        return "NULL"
    s = str(value)
    if len(s) > max_len:
        return s[:max_len] + f"…(+{len(s)-max_len})"
    return s


def _row_to_text(
    table_name: str,
    columns: List[str],
    row: tuple,
    max_cell: int,
    max_cols: int,
) -> str:
    """
    یک ردیف را به متن توصیفی تبدیل می‌کند.
    مثال:
      [جدول: users] id=1 | name=علی | email=ali@example.com | age=30
    """
    cols = columns[:max_cols]
    vals = [_truncate(v, max_cell) for v in row[:max_cols]]
    pairs = " | ".join(f"{c}={v}" for c, v in zip(cols, vals))

    if len(columns) > max_cols:
        pairs += f" | …(+{len(columns)-max_cols} ستون دیگر)"

    return f"[جدول: {table_name}] {pairs}"


def _table_to_markdown(
    table_name: str,
    columns: List[str],
    rows: List[tuple],
    max_cell: int,
    max_rows_preview: int = 5,
) -> str:
    """
    ساخت Markdown جدول برای نمایش schema + نمونه داده
    """
    header = "| " + " | ".join(columns) + " |"
    sep    = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines  = [header, sep]

    preview = rows[:max_rows_preview]
    for row in preview:
        cells = [_truncate(v, max_cell) for v in row]
        lines.append("| " + " | ".join(cells) + " |")

    if len(rows) > max_rows_preview:
        lines.append(f"| *... {len(rows)-max_rows_preview} ردیف دیگر ...* |" + " |" * (len(columns)-1))

    return "\n".join(lines)


def _is_excluded(table_name: str, cfg: DBProcessorConfig) -> bool:
    if table_name in cfg.excluded_tables:
        return True
    for prefix in cfg.excluded_table_prefixes:
        if table_name.lower().startswith(prefix.lower()):
            return True
    return False


def _build_content_list(
    table_name: str,
    columns: List[str],
    rows: List[tuple],
    cfg: DBProcessorConfig,
    page_offset: int = 0,
) -> List[dict]:
    """
    ساخت content_list از یک جدول:
      صفحه N+0: آیتم table (schema + پیش‌نمایش)
      صفحه N+1: آیتم‌های text (یک آیتم به ازای هر ردیف)
    """
    items: List[dict] = []

    # ── آیتم جدول (schema + نمونه) ───────────────────────────────
    md = _table_to_markdown(table_name, columns, rows, cfg.max_cell_length)
    items.append({
        "type": "table",
        "table_body": md,
        "table_caption": [f"جدول دیتابیس: {table_name}"],
        "table_footnote": [
            f"تعداد ستون: {len(columns)} | "
            f"تعداد ردیف: {len(rows)}"
            + (f" (نمایش {cfg.max_rows_per_table} ردیف اول)" if cfg.max_rows_per_table and len(rows) >= cfg.max_rows_per_table else "")
        ],
        "page_idx": page_offset,
    })

    # ── آیتم‌های متنی (یک به ازای هر ردیف) ─────────────────────
    for i, row in enumerate(rows):
        if cfg.skip_empty_rows:
            non_null = sum(1 for v in row if v is not None and str(v).strip())
            if non_null == 0:
                continue

        text = _row_to_text(
            table_name, columns, row,
            cfg.max_cell_length, cfg.max_columns_in_text,
        )
        items.append({
            "type": "text",
            "text": text,
            "page_idx": page_offset + 1 + (i // 100),  # هر ۱۰۰ ردیف = یک صفحه
        })

    return items


# ══════════════════════════════════════════════════════════════════════
# پردازشگر SQLite
# ══════════════════════════════════════════════════════════════════════

class SQLiteProcessor:
    """
    پردازش فایل‌های SQLite (.sqlite, .db, .sqlite3)
    بدون هیچ dependency خارجی — از sqlite3 استاندارد پایتون استفاده می‌کند.
    """

    EXTENSIONS = {".sqlite", ".db", ".sqlite3", ".db3"}

    def __init__(self, cfg: DBProcessorConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.EXTENSIONS

    def process(self, file_path: Path) -> Tuple[List[dict], Dict[str, Any]]:
        """
        Returns:
            content_list: لیست آیتم‌های آماده برای insert_content_list
            stats: آمار پردازش
        """
        self.logger.info(f"SQLite پردازش: {file_path.name}")
        content_list: List[dict] = []
        stats: Dict[str, Any] = {
            "file": file_path.name,
            "type": "sqlite",
            "tables": [],
            "total_rows": 0,
            "total_items": 0,
            "skipped_tables": [],
        }

        try:
            conn = sqlite3.connect(str(file_path))
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            # لیست جداول
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            table_names = [row[0] for row in cur.fetchall()]

            page_offset = 0
            for tbl in table_names:
                if _is_excluded(tbl, self.cfg):
                    stats["skipped_tables"].append(tbl)
                    self.logger.debug(f"  جدول رد شد: {tbl}")
                    continue

                try:
                    # ستون‌ها
                    cur.execute(f'PRAGMA table_info("{tbl}")')
                    pragma = cur.fetchall()
                    columns = [row[1] for row in pragma]
                    col_types = {row[1]: row[2] for row in pragma}

                    if not columns:
                        continue

                    # داده‌ها
                    limit_clause = f"LIMIT {self.cfg.max_rows_per_table}" if self.cfg.max_rows_per_table else ""
                    cur.execute(f'SELECT * FROM "{tbl}" {limit_clause}')
                    rows = [tuple(row) for row in cur.fetchall()]

                    items = _build_content_list(tbl, columns, rows, self.cfg, page_offset)
                    content_list.extend(items)

                    tbl_stat = {
                        "name": tbl,
                        "columns": len(columns),
                        "rows_processed": len(rows),
                        "items_generated": len(items),
                        "column_types": col_types,
                    }
                    stats["tables"].append(tbl_stat)
                    stats["total_rows"] += len(rows)
                    stats["total_items"] += len(items)

                    page_offset += 2 + (len(rows) // 100)
                    self.logger.info(
                        f"  ✅ {tbl}: {len(columns)} ستون, {len(rows)} ردیف → {len(items)} آیتم"
                    )

                except Exception as e:
                    self.logger.warning(f"  ⚠️ خطا در جدول {tbl}: {e}")

            conn.close()

        except Exception as e:
            self.logger.error(f"خطا در باز کردن SQLite: {e}")
            raise

        return content_list, stats


# ══════════════════════════════════════════════════════════════════════
# پردازشگر فایل SQL Dump
# ══════════════════════════════════════════════════════════════════════

class SQLDumpProcessor:
    """
    پردازش فایل‌های SQL dump متنی (.sql)
    بدون اجرا — فقط parse ساختار CREATE TABLE و INSERT INTO
    """

    EXTENSIONS = {".sql"}

    # regex های پارس
    _RE_CREATE = re.compile(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"']?(\w+)[`\"']?\s*\(([^;]+?)\)\s*;",
        re.IGNORECASE | re.DOTALL,
    )
    _RE_INSERT = re.compile(
        r"INSERT\s+INTO\s+[`\"']?(\w+)[`\"']?\s*(?:\(([^)]+)\))?\s*VALUES\s*(.*?);",
        re.IGNORECASE | re.DOTALL,
    )
    _RE_VALUES_ROW = re.compile(r"\(([^()]*(?:\([^()]*\)[^()]*)*)\)")
    _RE_COL_DEF = re.compile(
        r"[`\"']?(\w+)[`\"']?\s+\w+",
        re.IGNORECASE,
    )

    def __init__(self, cfg: DBProcessorConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.EXTENSIONS

    def _parse_value(self, s: str) -> Any:
        s = s.strip()
        if s.upper() == "NULL":
            return None
        if (s.startswith("'") and s.endswith("'")) or \
           (s.startswith('"') and s.endswith('"')):
            return s[1:-1].replace("\\'", "'").replace('\\"', '"')
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            pass
        return s

    def _parse_values_list(self, values_str: str) -> List[List[Any]]:
        """چندین ردیف VALUES(...), (...) را parse می‌کند"""
        rows = []
        for match in self._RE_VALUES_ROW.finditer(values_str):
            raw = match.group(1)
            # split با احتیاط (کاما داخل رشته را نشکند)
            cells = []
            depth = 0
            current = []
            in_str = False
            str_char = None
            for ch in raw:
                if in_str:
                    current.append(ch)
                    if ch == str_char:
                        in_str = False
                elif ch in ("'", '"'):
                    in_str = True
                    str_char = ch
                    current.append(ch)
                elif ch == "," and depth == 0:
                    cells.append("".join(current).strip())
                    current = []
                else:
                    current.append(ch)
            if current:
                cells.append("".join(current).strip())
            rows.append([self._parse_value(c) for c in cells])
        return rows

    def process(self, file_path: Path) -> Tuple[List[dict], Dict[str, Any]]:
        self.logger.info(f"SQL Dump پردازش: {file_path.name}")

        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            raise RuntimeError(f"خطا در خواندن فایل SQL: {e}")

        # ── پارس CREATE TABLE ها ──────────────────────────────────
        schemas: Dict[str, List[str]] = {}
        for m in self._RE_CREATE.finditer(text):
            tbl_name = m.group(1)
            body = m.group(2)
            cols = []
            for line in body.split("\n"):
                line = line.strip().rstrip(",")
                # خطوط constraint را رد کن
                up = line.upper()
                if any(up.startswith(k) for k in
                       ("PRIMARY", "UNIQUE", "KEY", "INDEX", "CONSTRAINT",
                        "FOREIGN", "CHECK", "--", "/*")):
                    continue
                cm = self._RE_COL_DEF.match(line)
                if cm:
                    cols.append(cm.group(1))
            if cols:
                schemas[tbl_name] = cols

        # ── پارس INSERT INTO ها ──────────────────────────────────
        table_rows: Dict[str, Tuple[List[str], List[List[Any]]]] = {}
        for m in self._RE_INSERT.finditer(text):
            tbl_name = m.group(1)
            if _is_excluded(tbl_name, self.cfg):
                continue
            cols_str = m.group(2)
            values_str = m.group(3)

            if cols_str:
                cols = [c.strip().strip("`\"'") for c in cols_str.split(",")]
            elif tbl_name in schemas:
                cols = schemas[tbl_name]
            else:
                continue

            rows = self._parse_values_list(values_str)
            if not rows:
                continue

            if tbl_name not in table_rows:
                table_rows[tbl_name] = (cols, [])

            existing_cols, existing_rows = table_rows[tbl_name]
            limit = self.cfg.max_rows_per_table
            if limit is None or len(existing_rows) < limit:
                remaining = (limit - len(existing_rows)) if limit else len(rows)
                existing_rows.extend(rows[:remaining])

        # ── ساخت content_list ────────────────────────────────────
        content_list: List[dict] = []
        stats: Dict[str, Any] = {
            "file": file_path.name,
            "type": "sql_dump",
            "tables": [],
            "total_rows": 0,
            "total_items": 0,
            "skipped_tables": [],
        }

        page_offset = 0
        for tbl_name, (cols, rows) in table_rows.items():
            items = _build_content_list(tbl_name, cols, rows, self.cfg, page_offset)
            content_list.extend(items)
            stats["tables"].append({
                "name": tbl_name,
                "columns": len(cols),
                "rows_processed": len(rows),
                "items_generated": len(items),
            })
            stats["total_rows"] += len(rows)
            stats["total_items"] += len(items)
            page_offset += 2 + (len(rows) // 100)
            self.logger.info(f"  ✅ {tbl_name}: {len(cols)} ستون, {len(rows)} ردیف → {len(items)} آیتم")

        # جداول فقط schema (بدون INSERT)
        for tbl_name, cols in schemas.items():
            if tbl_name not in table_rows and not _is_excluded(tbl_name, self.cfg):
                items = _build_content_list(tbl_name, cols, [], self.cfg, page_offset)
                content_list.extend(items)
                stats["tables"].append({
                    "name": tbl_name,
                    "columns": len(cols),
                    "rows_processed": 0,
                    "items_generated": len(items),
                    "note": "فقط schema — بدون داده INSERT",
                })
                stats["total_items"] += len(items)
                page_offset += 1

        return content_list, stats


# ══════════════════════════════════════════════════════════════════════
# پردازشگر MySQL زنده
# ══════════════════════════════════════════════════════════════════════

class MySQLProcessor:
    """
    اتصال زنده به MySQL/MariaDB
    نیاز: pip install pymysql
    """

    def __init__(self, cfg: DBProcessorConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger

    def _connect(self, conn_params: dict):
        try:
            import pymysql
            return pymysql.connect(
                host=conn_params.get("host", "localhost"),
                port=conn_params.get("port", 3306),
                user=conn_params["user"],
                password=conn_params["password"],
                database=conn_params["database"],
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor,
            )
        except ImportError:
            raise ImportError("برای MySQL نیاز است: pip install pymysql")

    def process_live(
        self,
        conn_params: dict,
        db_label: Optional[str] = None,
    ) -> Tuple[List[dict], Dict[str, Any]]:
        """
        conn_params: {"host": ..., "port": ..., "user": ..., "password": ..., "database": ...}
        """
        db_name = conn_params.get("database", "mysql_db")
        label = db_label or db_name
        self.logger.info(f"MySQL اتصال زنده: {label}")

        conn = self._connect(conn_params)
        content_list: List[dict] = []
        stats: Dict[str, Any] = {
            "file": label,
            "type": "mysql_live",
            "tables": [],
            "total_rows": 0,
            "total_items": 0,
            "skipped_tables": [],
        }

        try:
            with conn.cursor() as cur:
                cur.execute("SHOW TABLES")
                table_names = [list(row.values())[0] for row in cur.fetchall()]

                page_offset = 0
                for tbl in table_names:
                    if _is_excluded(tbl, self.cfg):
                        stats["skipped_tables"].append(tbl)
                        continue

                    try:
                        cur.execute(f"DESCRIBE `{tbl}`")
                        columns = [row["Field"] for row in cur.fetchall()]

                        limit = self.cfg.max_rows_per_table or 1000
                        cur.execute(f"SELECT * FROM `{tbl}` LIMIT {limit}")
                        rows_dicts = cur.fetchall()
                        rows = [tuple(r.values()) for r in rows_dicts]

                        items = _build_content_list(tbl, columns, rows, self.cfg, page_offset)
                        content_list.extend(items)
                        stats["tables"].append({
                            "name": tbl, "columns": len(columns),
                            "rows_processed": len(rows), "items_generated": len(items),
                        })
                        stats["total_rows"] += len(rows)
                        stats["total_items"] += len(items)
                        page_offset += 2 + (len(rows) // 100)
                        self.logger.info(f"  ✅ {tbl}: {len(columns)} ستون, {len(rows)} ردیف")

                    except Exception as e:
                        self.logger.warning(f"  ⚠️ {tbl}: {e}")
        finally:
            conn.close()

        return content_list, stats


# ══════════════════════════════════════════════════════════════════════
# پردازشگر PostgreSQL زنده
# ══════════════════════════════════════════════════════════════════════

class PostgreSQLProcessor:
    """
    اتصال زنده به PostgreSQL
    نیاز: pip install psycopg2-binary
    """

    def __init__(self, cfg: DBProcessorConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger

    def _connect(self, conn_params: dict):
        try:
            import psycopg2
            import psycopg2.extras
            return psycopg2.connect(
                host=conn_params.get("host", "localhost"),
                port=conn_params.get("port", 5432),
                user=conn_params["user"],
                password=conn_params["password"],
                dbname=conn_params["database"],
            )
        except ImportError:
            raise ImportError("برای PostgreSQL نیاز است: pip install psycopg2-binary")

    def process_live(
        self,
        conn_params: dict,
        schema: str = "public",
        db_label: Optional[str] = None,
    ) -> Tuple[List[dict], Dict[str, Any]]:
        db_name = conn_params.get("database", "postgres_db")
        label = db_label or db_name
        self.logger.info(f"PostgreSQL اتصال زنده: {label} (schema: {schema})")

        conn = self._connect(conn_params)
        content_list: List[dict] = []
        stats: Dict[str, Any] = {
            "file": label,
            "type": "postgresql_live",
            "tables": [],
            "total_rows": 0,
            "total_items": 0,
            "skipped_tables": [],
        }

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = %s AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                    """,
                    (schema,),
                )
                table_names = [row[0] for row in cur.fetchall()]

                page_offset = 0
                for tbl in table_names:
                    if _is_excluded(tbl, self.cfg):
                        stats["skipped_tables"].append(tbl)
                        continue

                    try:
                        cur.execute(
                            """
                            SELECT column_name FROM information_schema.columns
                            WHERE table_schema = %s AND table_name = %s
                            ORDER BY ordinal_position
                            """,
                            (schema, tbl),
                        )
                        columns = [row[0] for row in cur.fetchall()]

                        limit = self.cfg.max_rows_per_table or 1000
                        cur.execute(f'SELECT * FROM "{schema}"."{tbl}" LIMIT {limit}')
                        rows = cur.fetchall()

                        items = _build_content_list(tbl, columns, rows, self.cfg, page_offset)
                        content_list.extend(items)
                        stats["tables"].append({
                            "name": tbl, "columns": len(columns),
                            "rows_processed": len(rows), "items_generated": len(items),
                        })
                        stats["total_rows"] += len(rows)
                        stats["total_items"] += len(items)
                        page_offset += 2 + (len(rows) // 100)
                        self.logger.info(f"  ✅ {tbl}: {len(columns)} ستون, {len(rows)} ردیف")

                    except Exception as e:
                        self.logger.warning(f"  ⚠️ {tbl}: {e}")
        finally:
            conn.close()

        return content_list, stats


# ══════════════════════════════════════════════════════════════════════
# مدیریت مرکزی — DatabaseManager
# ══════════════════════════════════════════════════════════════════════

class DatabaseManager:
    """
    نقطه ورودی واحد برای پردازش همه نوع دیتابیس.

    استفاده:
        mgr = DatabaseManager(cfg, logger)

        # SQLite فایل
        cl, stats = mgr.process_file(Path("mydb.sqlite"))

        # SQL Dump فایل
        cl, stats = mgr.process_file(Path("backup.sql"))

        # MySQL زنده
        cl, stats = mgr.process_mysql({"host":..., "user":..., ...})

        # PostgreSQL زنده
        cl, stats = mgr.process_postgresql({"host":..., "user":..., ...})

        # درج در RAG
        await rag.insert_content_list(cl, file_path="mydb.sqlite", display_stats=True)
    """

    # پسوندهایی که به عنوان دیتابیس شناخته می‌شوند
    DB_EXTENSIONS = (
        SQLiteProcessor.EXTENSIONS |
        SQLDumpProcessor.EXTENSIONS
    )

    def __init__(self, cfg: Optional[DBProcessorConfig] = None, logger: Optional[logging.Logger] = None):
        self.cfg = cfg or DBProcessorConfig()
        self.logger = logger or logging.getLogger("RAG.DB")
        self._sqlite = SQLiteProcessor(self.cfg, self.logger)
        self._sqldump = SQLDumpProcessor(self.cfg, self.logger)
        self._mysql = MySQLProcessor(self.cfg, self.logger)
        self._postgresql = PostgreSQLProcessor(self.cfg, self.logger)

    def is_database_file(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.DB_EXTENSIONS

    def process_file(self, file_path: Path) -> Tuple[List[dict], Dict[str, Any]]:
        """پردازش خودکار فایل دیتابیسی بر اساس پسوند"""
        ext = file_path.suffix.lower()

        if ext in SQLiteProcessor.EXTENSIONS:
            return self._sqlite.process(file_path)
        elif ext in SQLDumpProcessor.EXTENSIONS:
            return self._sqldump.process(file_path)
        else:
            raise ValueError(f"فرمت دیتابیسی ناشناخته: {ext}")

    def process_mysql(
        self,
        conn_params: dict,
        db_label: Optional[str] = None,
    ) -> Tuple[List[dict], Dict[str, Any]]:
        return self._mysql.process_live(conn_params, db_label)

    def process_postgresql(
        self,
        conn_params: dict,
        schema: str = "public",
        db_label: Optional[str] = None,
    ) -> Tuple[List[dict], Dict[str, Any]]:
        return self._postgresql.process_live(conn_params, schema, db_label)

    def print_stats(self, stats: Dict[str, Any]):
        print(f"\n📊 آمار پردازش دیتابیس: {stats['file']} ({stats['type']})")
        print(f"  📋 جداول پردازش‌شده: {len(stats['tables'])}")
        print(f"  📝 کل ردیف‌ها: {stats['total_rows']}")
        print(f"  📦 کل آیتم‌های RAG: {stats['total_items']}")
        if stats.get("skipped_tables"):
            print(f"  ⏭️  رد‌شده: {', '.join(stats['skipped_tables'])}")
        print()
        for t in stats["tables"]:
            note = f"  ← {t.get('note', '')}" if t.get("note") else ""
            print(f"    • {t['name']}: {t['columns']} ستون | {t['rows_processed']} ردیف | {t['items_generated']} آیتم{note}")
        print()
