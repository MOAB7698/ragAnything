"""
╔══════════════════════════════════════════════════════════════════════╗
║          RAG-Anything Production System  —  v3.0                    ║
║          پشتیبانی کامل از تمام قابلیت‌های RAG-Anything              ║
╚══════════════════════════════════════════════════════════════════════╝

قابلیت‌ها:
  ✅ پردازش چندوجهی: متن، تصویر، جدول، معادله
  ✅ VLM Enhanced Query (تحلیل تصاویر در زمان جستجو)
  ✅ Multimodal Query با محتوای ورودی (aquery_with_multimodal)
  ✅ سه حالت query: text / vlm-enhanced / multimodal
  ✅ Direct Content List Insertion (بدون پارسر)
  ✅ بارگذاری LightRAG موجود (بدون پردازش مجدد)
  ✅ پشتیبانی از سه پارسر: mineru / docling / paddleocr
  ✅ Batch folder processing
  ✅ Custom Modal Processors
  ✅ پارامترهای پیشرفته MinerU (صفحه، زبان، GPU، backend)
  ✅ پشتیبانی Vision LLM برای تصاویر (LM Studio)
  ✅ Embedding چندزبانه (multilingual-e5-large)
  ✅ کش هوشمند فایل‌های پردازش‌شده (جلوگیری از پردازش مجدد)
  ✅ لاگ‌گیری کامل
  ✅ پردازش دیتابیس: SQLite (فایل)، SQL Dump، MySQL زنده، PostgreSQL زنده

وابستگی‌های اختیاری دیتابیس:
  pip install pymysql          # برای MySQL
  pip install psycopg2-binary  # برای PostgreSQL
  (SQLite و SQL Dump نیاز به نصب اضافه ندارند)
"""

import os
import sys
import asyncio
import logging
import json
import re
import base64
import hashlib
import traceback
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import time
from datetime import datetime

import numpy as np
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from sentence_transformers import SentenceTransformer

from raganything import RAGAnything, RAGAnythingConfig
from raganything.modalprocessors import (
    ImageModalProcessor,
    TableModalProcessor,
    EquationModalProcessor,
    GenericModalProcessor,
)
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc

# ماژول دیتابیس (در همان پوشه)
try:
    from db_processor import DatabaseManager, DBProcessorConfig
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False

# ماژول جستجوی تصویر (در همان پوشه)
try:
    from image_search import ImageSearchEngine, ImageSearchConfig
    _IMG_SEARCH_AVAILABLE = True
except ImportError:
    _IMG_SEARCH_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════
# تنظیمات اصلی
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # مسیرها
    data_dir: Path = Path("./data")
    working_dir: Path = Path("./rag_storage")
    output_dir: Path = Path("./output")
    parser_output_dir: Path = Path("./parser_output")
    cache_dir: Path = Path("./cache")
    log_dir: Path = Path("./logs")
    processed_files_db: Path = Path("./cache/processed_files.json")

    # LLM (LM Studio)
    llm_api_url: str = "http://192.168.190.18:1234/v1/chat/completions"
    llm_model_name: str = "google/gemma-4-31b"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 4096
    llm_timeout: int = 300
    # اندازه chunk متن برای استخراج entity — کوچکتر = کمتر از context رد میشه
    llm_chunk_token_size: int = 512
    llm_chunk_overlap_token_size: int = 50

    # Embedding
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_dim: int = 1024
    embedding_max_token_size: int = 512
    embedding_batch_size: int = 32

    # پارسر (mineru | docling | paddleocr)
    parser: str = "mineru"
    parse_method: str = "auto"   # auto | ocr | txt

    # پارامترهای پیشرفته MinerU
    mineru_lang: Optional[str] = None        # مثال: "fa", "en", "zh"
    mineru_device: str = "cpu"               # cpu | cuda | cuda:0 | npu | mps
    mineru_start_page: Optional[int] = None  # صفحه شروع (0-based)
    mineru_end_page: Optional[int] = None    # صفحه پایان (0-based)
    mineru_formula: bool = True              # پردازش فرمول‌ها
    mineru_table: bool = True                # پردازش جداول
    mineru_backend: str = "pipeline"         # pipeline | hybrid-auto-engine | vlm-auto-engine

    # پردازش محتوا
    enable_image_processing: bool = True
    enable_table_processing: bool = True
    enable_equation_processing: bool = True
    display_content_stats: bool = True
    max_concurrent_files: int = 1
    recursive_folder_processing: bool = True

    # Context
    context_window: int = 2
    context_mode: str = "page"
    max_context_tokens: int = 2000
    include_headers: bool = True
    include_captions: bool = True
    use_full_path: bool = True
    context_filter_content_types: List[str] = field(
        default_factory=lambda: ["text", "table", "image", "equation"]
    )
    content_format: str = "minerU"

    # جستجو
    top_k: int = 40
    chunk_top_k: int = 15

    # ── Query Enhancement ──────────────────────────────────────────
    # Multi-Query: بازنویسی سوال برای پوشش بیشتر embedding space
    enable_multi_query: bool = True
    multi_query_count: int = 3          # تعداد بازنویسی

    # HyDE: ساخت متن فرضی و embed کردن آن برای جستجو
    enable_hyde: bool = True

    # Cross-Encoder Reranker: رتبه‌بندی دقیق نتایج
    enable_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"   # multilingual
    reranker_top_k: int = 5             # بعد از rerank چند نتیجه نگه داریم
    reranker_score_threshold: float = -1.5   # زیر این → fallback به نتایج dense

    # Intent Router: تشخیص نوع سوال (search | chat)
    enable_intent_router: bool = True

    # ── "نمی‌دانم" — IDK Detection ────────────────────────────────
    # اگر پاسخ RAG کافی نبود، به جای ساختن جواب الکی، منابع نزدیک را نشان می‌دهد
    enable_idk: bool = True
    # عبارت‌هایی که نشان می‌دهند RAG جواب نداشته
    idk_phrases: List[str] = field(default_factory=lambda: [
        "i don't know", "i do not know", "no information",
        "not mentioned", "not found", "cannot find",
        "no relevant", "insufficient", "not available",
        "اطلاعاتی ندارم", "نمی‌دانم", "یافت نشد",
        "اطلاعاتی در منابع", "در منابع موجود نیست",
        "پاسخی پیدا نشد", "منبعی پیدا نشد",
    ])
    # حداقل طول پاسخ برای اینکه «یافت نشد» نباشد
    idk_min_length: int = 40

    # ── نمایش منبع پاسخ — Source Citation ────────────────────────
    enable_source_citation: bool = True
    # حداکثر تعداد منابع نمایش داده‌شده
    citation_max_sources: int = 5

    # فرمت‌های پشتیبانی‌شده
    supported_extensions: List[str] = field(default_factory=lambda: [
        ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp",
        ".txt", ".md", ".csv",
    ])

    # پسوندهای دیتابیسی (جدا از RAG parser — با db_processor پردازش می‌شوند)
    db_extensions: List[str] = field(default_factory=lambda: [
        ".sqlite", ".db", ".sqlite3", ".db3", ".sql",
    ])

    # تنظیمات پردازش دیتابیس
    db_max_rows_per_table: Optional[int] = 1000
    db_max_columns_in_text: int = 20
    db_skip_empty_rows: bool = True
    db_max_cell_length: int = 500

    # اتصال MySQL زنده (اختیاری — در صورت نیاز پر کنید)
    mysql_conn: Optional[Dict[str, Any]] = None
    # مثال: {"host": "localhost", "port": 3306, "user": "root", "password": "...", "database": "mydb"}

    # اتصال PostgreSQL زنده (اختیاری)
    postgresql_conn: Optional[Dict[str, Any]] = None
    postgresql_schema: str = "public"

    # ── Image Search (DINOv2 + FAISS) ─────────────────────────────
    enable_image_search: bool = True

    # مدل DINOv2:
    #   facebook/dinov2-small  (dim=384, سریع)
    #   facebook/dinov2-base   (dim=768, متوازن)   ← پیش‌فرض
    #   facebook/dinov2-large  (dim=1024, دقیق‌تر)
    dino_model_id: str = "facebook/dinov2-base"
    dino_dim: int = 768

    image_index_dir: Path = Path("./image_index")
    image_search_top_k: int = 10
    image_search_min_score: float = 0.0
    image_index_batch_size: int = 16

    # آیا تصاویر داخل اسناد (parser_output_dir) هم index شوند؟
    index_document_images: bool = True

    def ensure_directories(self):
        for d in [self.data_dir, self.working_dir, self.output_dir,
                  self.parser_output_dir, self.cache_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# لاگ‌گیری
# ══════════════════════════════════════════════════════════════════════

def setup_logging(config: Config) -> logging.Logger:
    config.ensure_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.log_dir / f"rag_{timestamp}.log"

    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt, datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("RAG")
    logger.info(f"Log → {log_file}")
    return logger


def setup_cache(cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_dir / "tiktoken_cache")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")


# ══════════════════════════════════════════════════════════════════════
# کش فایل‌های پردازش‌شده
# ══════════════════════════════════════════════════════════════════════

class ProcessedFilesCache:
    """ردیابی فایل‌های پردازش‌شده تا پردازش مجدد نشن"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self):
        if self.db_path.exists():
            try:
                self._data = json.loads(self.db_path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    def _save(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _file_hash(file_path: Path) -> str:
        """محاسبه MD5 فایل برای تشخیص تغییر محتوا (صرف‌نظر از mtime)"""
        h = hashlib.md5()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def is_processed(self, file_path: Path) -> bool:
        key = str(file_path.resolve())
        stored = self._data.get(key)
        if stored is None:
            return False
        current_mtime = str(file_path.stat().st_mtime)
        if isinstance(stored, dict):
            # فرمت جدید: اول mtime بررسی میشه (سریع)، اگر فرق داشت hash چک میشه
            if stored.get("mtime") == current_mtime:
                return True
            return stored.get("hash") == self._file_hash(file_path)
        # فرمت قدیمی (فقط mtime)
        return stored == current_mtime

    def mark_processed(self, file_path: Path):
        key = str(file_path.resolve())
        self._data[key] = {
            "mtime": str(file_path.stat().st_mtime),
            "hash": self._file_hash(file_path),
        }
        self._save()

    def remove(self, file_path: Path):
        key = str(file_path.resolve())
        self._data.pop(key, None)
        self._save()

    def clear(self):
        self._data = {}
        self._save()

    def count(self) -> int:
        return len(self._data)


# ══════════════════════════════════════════════════════════════════════
# LLM Functions
# ══════════════════════════════════════════════════════════════════════

@retry(
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
)
def _call_llm_sync(messages: List[dict], config: Config) -> str:
    resp = requests.post(
        config.llm_api_url,
        json={
            "model": config.llm_model_name,
            "messages": messages,
            "temperature": config.llm_temperature,
            "max_tokens": config.llm_max_tokens,
        },
        timeout=config.llm_timeout,
    )
    if not resp.ok:
        # بدنه خطای واقعی LM Studio را لاگ می‌کنیم — برای دیباگ 400/422/...
        try:
            err_body = resp.json()
        except Exception:
            err_body = resp.text[:1000]
        logging.getLogger("RAG").error(
            f"LLM API خطا [{resp.status_code}] برای {config.llm_api_url}: {err_body}"
        )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def llm_model_func(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[dict]] = None,
    config: Config = None,
    **kwargs,
) -> str:
    if config is None:
        config = Config()
    if history_messages is None:
        history_messages = []
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    return await asyncio.to_thread(_call_llm_sync, messages, config)


_MIME_TYPES: Dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}


async def vision_model_func(
    prompt: str = "Describe this image in detail",
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[dict]] = None,
    image_data: Optional[str] = None,       # base64 string
    image_path: Optional[str] = None,       # مسیر فایل — برای تشخیص MIME type دقیق
    messages: Optional[List[dict]] = None,  # فرمت VLM Enhanced
    config: Config = None,
    **kwargs,
) -> str:
    """
    Vision LLM — سه حالت:
      1. messages مستقیم (VLM Enhanced Query)
      2. image_data به صورت base64 (پردازش تصویر)
      3. فقط متن (fallback)

    MIME type از image_path استخراج می‌شود؛ در غیاب آن از kwargs["img_path"] یا
    پیش‌فرض image/jpeg استفاده می‌شود.
    """
    if config is None:
        config = Config()
    if history_messages is None:
        history_messages = []

    # ── تشخیص MIME type ─────────────────────────────────────────────
    def _resolve_mime(path_hint: Optional[str]) -> str:
        if path_hint:
            ext = Path(path_hint).suffix.lower()
            return _MIME_TYPES.get(ext, "image/jpeg")
        return "image/jpeg"

    # image_path می‌تواند از پارامتر مستقیم یا kwargs["img_path"] بیاید
    _img_path = image_path or kwargs.get("img_path")
    mime_type = _resolve_mime(_img_path)

    try:
        if messages:
            # حالت ۱: VLM Enhanced Query — پیام‌ها آماده‌اند
            return await asyncio.to_thread(_call_llm_sync, messages, config)

        elif image_data:
            # حالت ۲: تحلیل تصویر با base64 و MIME type دقیق
            msgs: List[dict] = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.extend(history_messages)
            msgs.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{mime_type};base64,{image_data}"
                    }},
                ],
            })
            return await asyncio.to_thread(_call_llm_sync, msgs, config)

        else:
            # حالت ۳: فقط متن
            return await llm_model_func(prompt, system_prompt, history_messages, config=config)

    except Exception as e:
        logging.getLogger("RAG").warning(f"Vision LLM خطا: {e} — از متن استفاده می‌شود")
        return await llm_model_func(prompt, system_prompt, history_messages, config=config)


# ══════════════════════════════════════════════════════════════════════
# Embedding Service
# ══════════════════════════════════════════════════════════════════════

class EmbeddingService:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._model = None

    def get_model(self) -> SentenceTransformer:
        if self._model is None:
            self.logger.info(f"بارگذاری مدل Embedding: {self.config.embedding_model}")
            self._model = SentenceTransformer(
                self.config.embedding_model,
                cache_folder=str(self.config.cache_dir / "sentence_transformers"),
            )
            dim = self._model.get_embedding_dimension()
            self.logger.info(f"✓ Embedding بارگذاری شد — بعد: {dim}")
        return self._model

    async def embed_texts(self, texts: List[str]) -> np.ndarray:
        model = self.get_model()
        result = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=self.config.embedding_batch_size,
        )
        return np.asarray(result, dtype=np.float32)

    def get_embedding_func(self) -> EmbeddingFunc:
        return EmbeddingFunc(
            embedding_dim=self.config.embedding_dim,
            max_token_size=self.config.embedding_max_token_size,
            func=self.embed_texts,
        )


# ══════════════════════════════════════════════════════════════════════
# ساخت RAGAnything
# ══════════════════════════════════════════════════════════════════════

def build_rag_config(config: Config) -> RAGAnythingConfig:
    """ساخت RAGAnythingConfig از Config اصلی"""
    return RAGAnythingConfig(
        working_dir=str(config.working_dir),
        parser=config.parser,
        parse_method=config.parse_method,
        parser_output_dir=str(config.parser_output_dir),
        display_content_stats=config.display_content_stats,
        enable_image_processing=config.enable_image_processing,
        enable_table_processing=config.enable_table_processing,
        enable_equation_processing=config.enable_equation_processing,
        max_concurrent_files=config.max_concurrent_files,
        recursive_folder_processing=config.recursive_folder_processing,
        context_window=config.context_window,
        context_mode=config.context_mode,
        max_context_tokens=config.max_context_tokens,
        include_headers=config.include_headers,
        include_captions=config.include_captions,
        context_filter_content_types=config.context_filter_content_types,
        content_format=config.content_format,
        use_full_path=config.use_full_path,
        supported_file_extensions=config.supported_extensions,
    )


def build_image_search_engine(
    config: Config,
    logger: logging.Logger,
) -> Optional["ImageSearchEngine"]:
    """
    ساخت ImageSearchEngine بر اساس Config.
    اگر image search غیرفعال یا وابستگی‌ها نصب نباشند، None برمی‌گرداند.
    """
    if not config.enable_image_search:
        return None

    if not _IMG_SEARCH_AVAILABLE:
        logger.warning(
            "image_search.py یا وابستگی‌های آن پیدا نشد — image search غیرفعال\n"
            "  pip install faiss-cpu torch transformers"
        )
        return None

    img_cfg = ImageSearchConfig(
        dino_model_id       = config.dino_model_id,
        dino_dim            = config.dino_dim,
        index_dir           = config.image_index_dir,
        image_extensions    = [e for e in config.supported_extensions
                                if e in {".jpg", ".jpeg", ".png", ".bmp",
                                         ".tiff", ".tif", ".gif", ".webp"}],
        batch_size          = config.image_index_batch_size,
        default_top_k       = config.image_search_top_k,
        min_score           = config.image_search_min_score,
        index_parser_output = config.index_document_images,
        model_cache_dir     = str(config.cache_dir / "transformers"),
    )

    return ImageSearchEngine(img_cfg, logger)


async def _preflight_multimodal_test(rag: RAGAnything, logger: logging.Logger) -> None:
    """
    یک تست کوچک برای تأیید اینکه pipeline کامل (از جمله entity extraction) کار می‌کند.
    اگر role_llm_funcs مشکل داشته باشه اینجا خطا میده — نه وسط پردازش ۲۶۹ آیتم.
    """
    import inspect

    # ---- ۱. بررسی ساختار داخلی LightRAG ----
    lg = rag.lightrag
    if lg is None:
        raise RuntimeError("lightrag instance is None inside RAGAnything")

    role_states = getattr(lg, "_role_llm_states", None)
    logger.info(f"  _role_llm_states present: {role_states is not None}")
    if role_states:
        for role, state in role_states.items():
            wrapped = getattr(state, "wrapped", None)
            logger.info(f"    role '{role}': wrapped={wrapped is not None}")

    # ---- ۲. بررسی global_config (اگر متد وجود داشت) ----
    if hasattr(lg, "_build_global_config"):
        try:
            gc = lg._build_global_config()
            rlf = gc.get("role_llm_funcs", "KEY_MISSING")
            if rlf == "KEY_MISSING":
                logger.error("  ❌ global_config['role_llm_funcs'] وجود ندارد!")
            else:
                none_roles = [k for k, v in rlf.items() if v is None]
                logger.info(f"  global_config['role_llm_funcs'] roles: {list(rlf.keys())}")
                if none_roles:
                    logger.warning(f"  ⚠️  این role ها None هستند: {none_roles}")
        except Exception as e:
            logger.error(f"  ❌ خطا در _build_global_config: {e}\n{traceback.format_exc()}")

    # ---- ۳. بررسی lightrag_kwargs ----
    lkw = getattr(rag, "lightrag_kwargs", {})
    rlf_in_kw = lkw.get("role_llm_funcs", "MISSING")
    logger.info(f"  rag.lightrag_kwargs['role_llm_funcs']: {'present' if rlf_in_kw != 'MISSING' else 'MISSING'}")

    # ---- ۴. تست واقعی: insert یک متن کوچک ----
    # اگر این موفق شد، entity extraction با role_llm_funcs کار می‌کنه
    test_text = "تست سیستم: این یک متن آزمایشی کوتاه است."
    logger.info("  درحال تست ainsert با متن کوچک...")
    try:
        await lg.ainsert(test_text)
        logger.info("  ✅ ainsert موفق — entity extraction کار می‌کند")
    except KeyError as e:
        logger.error(f"  ❌ KeyError در ainsert: {e}\n{traceback.format_exc()}")
        raise
    except Exception as e:
        # خطاهای دیگه (مثل LLM timeout) رو فقط log می‌کنیم، متوقف نمی‌کنیم
        logger.warning(f"  ⚠️  خطای غیر KeyError در ainsert (احتمالاً LLM): {type(e).__name__}: {e}")


async def create_rag(config: Config, embedding_service: EmbeddingService) -> RAGAnything:
    """ساخت یا بارگذاری نمونه RAGAnything"""
    rag_cfg = build_rag_config(config)

    async def _llm(prompt, system_prompt=None, history_messages=None, **kw):
        return await llm_model_func(prompt, system_prompt, history_messages, config=config, **kw)

    async def _vision(prompt="", system_prompt=None, history_messages=None,
                      image_data=None, image_path=None, messages=None, **kw):
        return await vision_model_func(
            prompt, system_prompt, history_messages,
            image_data=image_data, image_path=image_path,
            messages=messages, config=config, **kw
        )

    # ── ساخت role_llm_funcs برای LightRAG جدید ───────────────────────
    # LightRAG جدید (lightrag-hku) داخل lightrag_kwargs انتظار دارد
    # یک dict به نام role_llm_funcs با چهار نقش بگیرد.
    # اگر این dict نباشد، KeyError: 'role_llm_funcs' می‌دهد.
    role_llm_funcs = {
        "extract": _llm,   # استخراج موجودیت و رابطه از اسناد
        "keyword": _llm,   # استخراج کلیدواژه
        "query":   _llm,   # پاسخ به سوال کاربر
        "vlm":     _vision, # تحلیل تصویر
    }

    # lightrag_kwargs به RAGAnything پاس داده می‌شود و RAGAnything آن را به LightRAG constructor می‌دهد.
    # llm_model_func باید اینجا باشد تا LightRAG بتواند LLM را initialize کند.
    # role_llm_funcs را اینجا نمی‌گذاریم چون نسخه نصب‌شده LightRAG آن را در constructor نمی‌پذیرد.
    lightrag_kwargs: Dict[str, Any] = {
        "llm_model_func": _llm,
        "chunk_token_size": config.llm_chunk_token_size,
        "chunk_overlap_token_size": config.llm_chunk_overlap_token_size,
    }

    # اگر storage موجود باشد، LightRAG را مستقیم بارگذاری می‌کنیم
    existing = config.working_dir.exists() and any(config.working_dir.iterdir())

    _logger = logging.getLogger(__name__)

    if existing:
        # بارگذاری LightRAG موجود — به تدریج پارامترها را امتحان می‌کنیم
        lightrag_instance = None
        used_path = None
        for path_name, lg_kwargs in [
            ("full (role_llm_funcs + chunk_size)", dict(
                working_dir=str(config.working_dir),
                role_llm_funcs=role_llm_funcs,
                embedding_func=embedding_service.get_embedding_func(),
                chunk_token_size=config.llm_chunk_token_size,
                chunk_overlap_token_size=config.llm_chunk_overlap_token_size,
            )),
            ("role_llm_funcs only", dict(
                working_dir=str(config.working_dir),
                role_llm_funcs=role_llm_funcs,
                embedding_func=embedding_service.get_embedding_func(),
            )),
            ("legacy llm_model_func", dict(
                working_dir=str(config.working_dir),
                llm_model_func=_llm,
                embedding_func=embedding_service.get_embedding_func(),
            )),
        ]:
            try:
                lightrag_instance = LightRAG(**lg_kwargs)
                used_path = path_name
                _logger.info(f"✅ LightRAG init path: {path_name}")
                break
            except TypeError as e:
                _logger.debug(f"  ↳ {path_name} → TypeError: {e}")
                continue

        if lightrag_instance is None:
            raise RuntimeError("ساخت LightRAG با هیچ‌یک از پیکربندی‌ها ممکن نشد")

        # ── FIX: تزریق role_llm_funcs به __dict__ ──────────────────────────
        # raganything از self.lightrag.__dict__ (نه _build_global_config) به عنوان
        # global_config به extract_entities و merge_nodes_and_edges پاس میده.
        # اما role_llm_funcs فقط داخل _build_global_config ساخته میشه — نه در __dict__.
        # راه‌حل: مقدار نهایی role_llm_funcs رو مستقیم به instance اضافه کنیم.
        if hasattr(lightrag_instance, "_build_global_config"):
            _gc = lightrag_instance._build_global_config()
            _computed_rlf = _gc.get("role_llm_funcs")
        else:
            _computed_rlf = None

        if _computed_rlf:
            # نمی‌توانیم مستقیم assign کنیم چون property است (no setter)
            # اما راه‌حل: مستقیم در __dict__ می‌نویسیم تا raganything پیداش کنه
            lightrag_instance.__dict__["role_llm_funcs"] = _computed_rlf
            _logger.info(f"✅ role_llm_funcs injected into lightrag.__dict__: {list(_computed_rlf.keys())}")
        else:
            lightrag_instance.__dict__["role_llm_funcs"] = role_llm_funcs
            _logger.warning("⚠️  role_llm_funcs از _build_global_config نامعتبر بود — fallback مستقیم")

        # تشخیص وضعیت role_llm_funcs در instance
        _role_states = getattr(lightrag_instance, "_role_llm_states", None)
        _logger.info(f"🔍 LightRAG._role_llm_states: {list(_role_states.keys()) if _role_states else 'MISSING/EMPTY'}")
        _logger.info(f"🔍 lightrag_kwargs keys: {list(lightrag_kwargs.keys())}")

        await lightrag_instance.initialize_storages()
        await initialize_pipeline_status()

        try:
            rag = RAGAnything(
                lightrag=lightrag_instance,
                config=rag_cfg,
                llm_model_func=_llm,
                vision_model_func=_vision,
                lightrag_kwargs=lightrag_kwargs,
            )
        except TypeError:
            rag = RAGAnything(
                lightrag=lightrag_instance,
                config=rag_cfg,
                vision_model_func=_vision,
                lightrag_kwargs=lightrag_kwargs,
            )

        # ── FIX 2: patch global_config روی هر modal processor ──────────────
        # BaseModalProcessor.__init__ از asdict(lightrag) برای global_config استفاده می‌کند.
        # asdict فقط dataclass fields رو می‌گیره، نه @property ها.
        # پس role_llm_funcs که property است داخل global_config نیست → KeyError.
        # راه‌حل: مستقیم به dict هر processor تزریق می‌کنیم.
        _rlf_to_inject = lightrag_instance.__dict__.get("role_llm_funcs") or role_llm_funcs
        _patched = 0
        modal_processors = getattr(rag, "modal_processors", None) or {}
        for _pname, _proc in modal_processors.items():
            _gc = getattr(_proc, "global_config", None)
            if isinstance(_gc, dict) and "role_llm_funcs" not in _gc:
                _gc["role_llm_funcs"] = _rlf_to_inject
                _patched += 1
        if _patched:
            _logger.info(f"✅ role_llm_funcs injected into {_patched} modal processor global_configs")

        # تست سریع قبل از پردازش اصلی
        _logger.info("🧪 تست سریع pipeline قبل از پردازش...")
        try:
            await _preflight_multimodal_test(rag, _logger)
            _logger.info("✅ تست pre-flight موفق بود")
        except Exception as e:
            _logger.error(f"❌ تست pre-flight ناموفق: {e}\n{traceback.format_exc()}")
    else:
        # ساخت RAGAnything جدید — RAGAnything خودش LightRAG را درون خودش می‌سازد
        try:
            rag = RAGAnything(
                config=rag_cfg,
                llm_model_func=_llm,
                vision_model_func=_vision,
                embedding_func=embedding_service.get_embedding_func(),
                lightrag_kwargs=lightrag_kwargs,
            )
        except TypeError:
            rag = RAGAnything(
                config=rag_cfg,
                llm_model_func=_llm,
                vision_model_func=_vision,
                embedding_func=embedding_service.get_embedding_func(),
            )

        # ── FIX: تزریق role_llm_funcs بعد از init (existing=False) ──────────
        # RAGAnything از _ensure_lightrag_initialized برای ساخت LightRAG استفاده می‌کند.
        # چون role_llm_funcs را در constructor نمی‌توان پاس داد، بعد از init تزریق می‌کنیم.
        _orig_ensure = getattr(rag, "_ensure_lightrag_initialized", None)
        if _orig_ensure is not None and callable(_orig_ensure):
            async def _patched_ensure(_orig=_orig_ensure):
                # CRITICAL: نتیجه تابع اصلی را برمی‌گردانیم.
                # raganything انتظار dict {"success": True} دارد؛ اگر None برگردانیم
                # → "LightRAG initialization failed: unknown error".
                _result = await _orig()
                lg = getattr(rag, "lightrag", None)
                if lg is not None:
                    # محاسبه role_llm_funcs از _build_global_config یا fallback
                    if hasattr(lg, "_build_global_config"):
                        try:
                            _rlf = lg._build_global_config().get("role_llm_funcs") or role_llm_funcs
                        except Exception:
                            _rlf = role_llm_funcs
                    else:
                        _rlf = role_llm_funcs
                    # تزریق به __dict__ (نه property — چون setter ندارد)
                    lg.__dict__["role_llm_funcs"] = _rlf
                    _logger.info(f"✅ [new] role_llm_funcs injected into lightrag.__dict__: {list(_rlf.keys())}")
                    # تزریق به global_config هر modal processor
                    _patched = 0
                    for _pname, _proc in (getattr(rag, "modal_processors", None) or {}).items():
                        _gc = getattr(_proc, "global_config", None)
                        if isinstance(_gc, dict):
                            _gc["role_llm_funcs"] = _rlf
                            _patched += 1
                    if _patched:
                        _logger.info(f"✅ [new] role_llm_funcs injected into {_patched} modal processor global_configs")
                return _result

            rag._ensure_lightrag_initialized = _patched_ensure
            _logger.info("🔧 _ensure_lightrag_initialized monkey-patched برای تزریق role_llm_funcs")
        else:
            _logger.warning("⚠️  _ensure_lightrag_initialized پیدا نشد — ممکن است role_llm_funcs تزریق نشود")

    return rag


# ══════════════════════════════════════════════════════════════════════
# پردازش اسناد
# ══════════════════════════════════════════════════════════════════════

def _mineru_kwargs(config: Config) -> dict:
    """پارامترهای پیشرفته MinerU"""
    kw = {}
    if config.mineru_lang:
        kw["lang"] = config.mineru_lang
    if config.mineru_device != "cpu":
        kw["device"] = config.mineru_device
    if config.mineru_start_page is not None:
        kw["start_page"] = config.mineru_start_page
    if config.mineru_end_page is not None:
        kw["end_page"] = config.mineru_end_page
    if config.parser == "mineru":
        kw["formula"] = config.mineru_formula
        kw["table"] = config.mineru_table
        kw["backend"] = config.mineru_backend
    return kw


async def process_documents(
    rag: RAGAnything,
    files: List[Path],
    config: Config,
    logger: logging.Logger,
    file_cache: ProcessedFilesCache,
    force_reprocess: bool = False,
) -> Dict[str, Any]:
    """پردازش دسته‌ای با کش هوشمند"""

    to_process = []
    skipped = 0
    for f in files:
        if not force_reprocess and file_cache.is_processed(f):
            skipped += 1
        else:
            to_process.append(f)

    logger.info(f"📊 کل: {len(files)} | پردازش جدید: {len(to_process)} | رد‌شده (کش): {skipped}")

    if not to_process:
        print("\n✅ همه فایل‌ها قبلاً پردازش شده‌اند — مستقیم به جستجو بروید.\n")
        return {"total": len(files), "success": 0, "skipped": skipped, "failed": 0, "failed_files": []}

    success, failed_files = 0, []
    mineru_kw = _mineru_kwargs(config)

    for i, fp in enumerate(to_process, 1):
        try:
            logger.info(f"[{i}/{len(to_process)}] پردازش: {fp.name}")
            await rag.process_document_complete(
                file_path=str(fp),
                output_dir=str(config.output_dir),
                parse_method=config.parse_method,
                display_stats=False,
                **mineru_kw,
            )
            file_cache.mark_processed(fp)
            success += 1
            logger.info(f"  ✅ موفق ({success}/{i})")
        except Exception as e:
            logger.error(f"  ❌ خطا: {fp.name} — {e}\n{traceback.format_exc()}")
            failed_files.append(fp.name)

        if i % 10 == 0:
            print(f"📊 پیشرفت: {i}/{len(to_process)} | ✅ {success} | ❌ {len(failed_files)}")

    return {
        "total": len(files), "success": success,
        "skipped": skipped, "failed": len(failed_files), "failed_files": failed_files,
    }


# ══════════════════════════════════════════════════════════════════════
# قابلیت Direct Content List Insertion
# ══════════════════════════════════════════════════════════════════════

async def process_databases(
    rag: RAGAnything,
    config: Config,
    logger: logging.Logger,
    file_cache: ProcessedFilesCache,
    force_reprocess: bool = False,
) -> Dict[str, Any]:
    """
    پردازش همه فایل‌های دیتابیسی در data_dir و اتصال‌های زنده تعریف‌شده.

    جریان کار:
      1. فایل‌های .sqlite/.db/.sql را در data_dir پیدا می‌کند
      2. هر فایل را با DatabaseManager پردازش می‌کند → content_list
      3. content_list را با insert_content_list وارد RAG می‌کند
      4. در صورت تنظیم mysql_conn یا postgresql_conn، اتصال زنده هم پردازش می‌شود
    """
    if not _DB_AVAILABLE:
        logger.error("db_processor.py پیدا نشد — پردازش دیتابیس غیرفعال است")
        return {"error": "db_processor not found"}

    db_cfg = DBProcessorConfig(
        max_rows_per_table=config.db_max_rows_per_table,
        max_columns_in_text=config.db_max_columns_in_text,
        skip_empty_rows=config.db_skip_empty_rows,
        max_cell_length=config.db_max_cell_length,
    )
    mgr = DatabaseManager(db_cfg, logger)

    results = {
        "files_processed": 0,
        "files_skipped": 0,
        "files_failed": 0,
        "live_connections": 0,
        "total_tables": 0,
        "total_items": 0,
        "details": [],
    }

    # ── فایل‌های دیتابیسی ────────────────────────────────────────
    db_files: List[Path] = []
    for ext in config.db_extensions:
        db_files.extend(config.data_dir.rglob(f"*{ext}"))
        db_files.extend(config.data_dir.rglob(f"*{ext.upper()}"))
    db_files = sorted(set(db_files))

    if db_files:
        logger.info(f"📂 {len(db_files)} فایل دیتابیسی پیدا شد")

    for fp in db_files:
        if not force_reprocess and file_cache.is_processed(fp):
            logger.info(f"  ⏭️  رد‌شده (کش): {fp.name}")
            results["files_skipped"] += 1
            continue

        try:
            content_list, stats = mgr.process_file(fp)

            if not content_list:
                logger.warning(f"  ⚠️ {fp.name}: هیچ محتوایی استخراج نشد")
                results["files_failed"] += 1
                continue

            await rag.insert_content_list(
                content_list=content_list,
                file_path=str(fp),
                display_stats=False,
            )

            file_cache.mark_processed(fp)
            mgr.print_stats(stats)

            results["files_processed"] += 1
            results["total_tables"] += len(stats["tables"])
            results["total_items"] += stats["total_items"]
            results["details"].append(stats)
            logger.info(f"  ✅ {fp.name}: {stats['total_items']} آیتم وارد شد")

        except Exception as e:
            logger.error(f"  ❌ {fp.name}: {e}")
            results["files_failed"] += 1

    # ── اتصال زنده MySQL ─────────────────────────────────────────
    if config.mysql_conn:
        try:
            logger.info("MySQL اتصال زنده در حال پردازش...")
            content_list, stats = mgr.process_mysql(
                config.mysql_conn,
                db_label=config.mysql_conn.get("database", "mysql"),
            )
            if content_list:
                await rag.insert_content_list(
                    content_list=content_list,
                    file_path=f"mysql://{config.mysql_conn.get('database', 'db')}",
                    display_stats=False,
                )
                mgr.print_stats(stats)
                results["live_connections"] += 1
                results["total_tables"] += len(stats["tables"])
                results["total_items"] += stats["total_items"]
                results["details"].append(stats)
                logger.info(f"  ✅ MySQL: {stats['total_items']} آیتم وارد شد")
        except Exception as e:
            logger.error(f"  ❌ MySQL خطا: {e}")

    # ── اتصال زنده PostgreSQL ────────────────────────────────────
    if config.postgresql_conn:
        try:
            logger.info("PostgreSQL اتصال زنده در حال پردازش...")
            content_list, stats = mgr.process_postgresql(
                config.postgresql_conn,
                schema=config.postgresql_schema,
                db_label=config.postgresql_conn.get("database", "postgres"),
            )
            if content_list:
                await rag.insert_content_list(
                    content_list=content_list,
                    file_path=f"postgresql://{config.postgresql_conn.get('database', 'db')}",
                    display_stats=False,
                )
                mgr.print_stats(stats)
                results["live_connections"] += 1
                results["total_tables"] += len(stats["tables"])
                results["total_items"] += stats["total_items"]
                results["details"].append(stats)
                logger.info(f"  ✅ PostgreSQL: {stats['total_items']} آیتم وارد شد")
        except Exception as e:
            logger.error(f"  ❌ PostgreSQL خطا: {e}")

    return results


# ══════════════════════════════════════════════════════════════════════
# Direct Content List Insertion — مثال
# ══════════════════════════════════════════════════════════════════════

async def insert_content_list_example(rag: RAGAnything, logger: logging.Logger):
    """
    مثال: درج مستقیم محتوای آماده بدون پارسر
    (برای محتوایی که از منابع دیگر آماده شده)
    """
    logger.info("درج content list مستقیم...")

    content_list = [
        {
            "type": "text",
            "text": "این یک متن نمونه است که مستقیماً درج می‌شود.",
            "page_idx": 0,
        },
        {
            "type": "table",
            "table_body": "| روش | دقت | سرعت |\n|-----|-----|------|\n| RAGAnything | 95.2% | 120ms |\n| Baseline | 87.3% | 180ms |",
            "table_caption": ["جدول ۱: مقایسه عملکرد"],
            "table_footnote": ["نتایج روی مجموعه تست"],
            "page_idx": 1,
        },
        {
            "type": "equation",
            "latex": r"P(d|q) = \frac{P(q|d) \cdot P(d)}{P(q)}",
            "text": "فرمول احتمال بازیابی سند",
            "page_idx": 2,
        },
    ]

    await rag.insert_content_list(
        content_list=content_list,
        file_path="manual_content.pdf",
        display_stats=True,
    )
    logger.info("✅ content list با موفقیت درج شد")


# ══════════════════════════════════════════════════════════════════════
# Query Enhancement — Multi-Query، HyDE، Intent Router، Reranker
# ══════════════════════════════════════════════════════════════════════

def _strip_thinking_tags(text: str) -> str:
    """
    برخی مدل‌ها chain-of-thought را داخل تگ می‌گذارند.
    مثال Gemma: <|channel>thought...</channel|>
    مثال Qwen:  <think>...</think>
    """
    text = re.sub(r"<\|channel\>thought.*?<channel\|>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*?</think>",                "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<\|.*?\|>",                         "", text)
    return text.strip()


class QueryEnhancer:
    """
    لایه‌ای بین سوال کاربر و RAG:
      1. Intent Router  — تشخیص جستجو / چت
      2. Multi-Query    — بازنویسی سوال به چند نسخه
      3. HyDE           — ساخت متن فرضی برای embed
      4. Reranker       — رتبه‌بندی دقیق نتایج نهایی
    """

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._reranker = None
        self._reranker_loaded = False

    # ── Reranker (lazy load) ─────────────────────────────────────
    def _get_reranker(self):
        if self._reranker_loaded:
            return self._reranker
        self._reranker_loaded = True
        if not self.config.enable_reranker:
            return None
        try:
            from sentence_transformers import CrossEncoder
            self.logger.info(f"بارگذاری Reranker: {self.config.reranker_model}")
            self._reranker = CrossEncoder(self.config.reranker_model)
            self.logger.info("✓ Reranker آماده است")
        except ImportError:
            self.logger.warning("sentence-transformers نصب نیست — Reranker غیرفعال")
        except Exception as e:
            self.logger.warning(f"Reranker بارگذاری نشد: {e}")
        return self._reranker

    # ── LLM call (sync داخل thread) ─────────────────────────────
    async def _llm(self, prompt: str, temperature: float = 0.3) -> str:
        msgs = [{"role": "user", "content": prompt}]
        raw = await asyncio.to_thread(_call_llm_sync, msgs, self.config)
        return _strip_thinking_tags(raw)

    # ── 1. Intent Router ─────────────────────────────────────────
    async def detect_intent(self, query: str) -> str:
        """
        خروجی: "search" یا "chat"
          search — جستجو در اسناد
          chat   — سوال عمومی / خارج از حوزه اسناد
        """
        if not self.config.enable_intent_router:
            return "search"

        # fast-path برای greetings
        greetings = ["سلام", "خوبی", "hello", "hi", "چطوری", "درود", "ممنون", "مرسی"]
        q_lower = query.strip().lower()
        for g in greetings:
            if q_lower.startswith(g):
                return "chat"

        prompt = f"""Classify this query into exactly one category.

Categories:
- search: the user wants to find or ask about information in documents
- chat: greetings, general questions unrelated to documents, or small talk

Query: "{query}"

Reply with ONLY one word: search or chat"""

        try:
            result = await self._llm(prompt, temperature=0.0)
            word = result.strip().lower().split()[0]
            if word in ("search", "chat"):
                return word
        except Exception as e:
            self.logger.debug(f"Intent router خطا: {e}")

        return "search"

    # ── 2. Multi-Query ───────────────────────────────────────────
    async def multi_query_rewrite(self, query: str) -> List[str]:
        """
        سوال را به N نسخه متنوع بازنویسی می‌کند.
        خروجی: [query_اصلی, rewrite1, rewrite2, ...]
        """
        if not self.config.enable_multi_query:
            return [query]

        n = self.config.multi_query_count
        prompt = f"""Rewrite the following search query in {n} different ways.
Each rewrite should have the same meaning but use different words or phrasing.
Mix Persian and English when helpful.
Output ONLY the {n} queries, one per line, no numbering, no explanation.

Original query: {query}

Rewrites:"""

        try:
            result = await self._llm(prompt, temperature=0.7)
            rewrites = [
                ln.strip() for ln in result.splitlines()
                if ln.strip() and not ln.strip().startswith(("#", "-", "*"))
            ][:n]
            all_queries = [query] + rewrites
            self.logger.info(f"Multi-Query: {len(all_queries)} نسخه ساخته شد")
            return all_queries
        except Exception as e:
            self.logger.debug(f"Multi-Query خطا: {e}")
            return [query]

    # ── 3. HyDE ──────────────────────────────────────────────────
    async def hyde_generate(self, query: str) -> Optional[str]:
        """
        HyDE: متن فرضی می‌سازد که گویی پاسخ سوال در آن هست.
        این متن embed می‌شود و با اسناد مقایسه می‌گردد.
        """
        if not self.config.enable_hyde:
            return None

        prompt = f"""Write a short hypothetical passage (2-3 sentences) that would directly answer this question.
Write as if this passage exists in a real document.
Use the same language as the question (Persian or English).
Output ONLY the passage, no explanation.

Question: {query}

Hypothetical passage:"""

        try:
            result = await self._llm(prompt, temperature=0.5)
            if result and len(result) > 10:
                self.logger.info(f"HyDE: متن فرضی ساخته شد ({len(result)} کاراکتر)")
                return result
        except Exception as e:
            self.logger.debug(f"HyDE خطا: {e}")
        return None

    # ── 4. Query Rewrite با تاریخچه ─────────────────────────────
    async def rewrite_with_history(
        self, query: str, chat_history: List[dict]
    ) -> str:
        """
        اگر سوال به تاریخچه مکالمه اشاره دارد (مثل «اون اولی» یا «بیشتر توضیح بده»)
        آن را به یک query مستقل و کامل تبدیل می‌کند.
        """
        if not chat_history:
            return query

        recent = chat_history[-4:]
        history_text = "\n".join(
            f"{'کاربر' if m['role'] == 'user' else 'دستیار'}: {m['content'][:200]}"
            for m in recent
        )

        prompt = f"""Given this conversation history:
{history_text}

New user message: "{query}"

If the new message is a follow-up or refers to something from history (uses pronouns, incomplete phrases, or context-dependent references), rewrite it as a complete standalone query.
If it's already a complete independent question, return it UNCHANGED.
Output ONLY the final query, nothing else."""

        try:
            rewritten = await self._llm(prompt, temperature=0.0)
            rewritten = rewritten.strip().strip('"')
            if rewritten and len(rewritten) > 3 and rewritten != query:
                self.logger.info(f"Query rewritten: {query!r} → {rewritten!r}")
                return rewritten
        except Exception as e:
            self.logger.debug(f"Query rewrite خطا: {e}")

        return query

    # ── 5. Reranker ──────────────────────────────────────────────
    def rerank(
        self, query: str, results_text: str
    ) -> Tuple[str, float]:
        """
        نتیجه RAG را با Cross-Encoder rerank می‌کند.

        چون RAGAnything یک متن واحد برمی‌گرداند (نه لیست)،
        اینجا reranker را به شکل متفاوتی به کار می‌بریم:
        query را با نتیجه pair می‌کنیم و یک relevance score می‌گیریم.
        اگر score پایین بود، به کاربر هشدار می‌دهیم.

        خروجی: (results_text, relevance_score)
        """
        reranker = self._get_reranker()
        if reranker is None or not results_text:
            return results_text, 1.0

        try:
            score = float(reranker.predict([(query, results_text[:1000])]))
            self.logger.info(f"Reranker score: {score:.3f}")
            return results_text, score
        except Exception as e:
            self.logger.debug(f"Reranker خطا: {e}")
            return results_text, 1.0


# ══════════════════════════════════════════════════════════════════════
# IDK Detection + Source Citation
# ══════════════════════════════════════════════════════════════════════

class AnswerAnalyzer:
    """
    دو وظیفه دارد:
      1. IDK Detection — تشخیص اینکه RAG جواب مطمئنی نداشته
      2. Source Citation — استخراج منابع از متن پاسخ RAG

    RAG-Anything معمولاً مسیر فایل‌ها را در پاسخ ذکر می‌کند.
    این کلاس آن مسیرها را استخراج و به فرمت citation تبدیل می‌کند.
    """

    # regex استخراج مسیر فایل از پاسخ
    _RE_WIN_PATH  = re.compile(
        r'[A-Za-z]:[/\\][\w\s/\\.\.\-\u0600-\u06FF\u200c()،]+?'
        r'\.(?:pdf|docx|doc|pptx|ppt|xlsx|xls|txt|md|csv|jpg|jpeg|png|bmp|tiff|gif|webp'
        r'|sqlite|db|sqlite3|sql)',
        re.IGNORECASE | re.UNICODE,
    )
    _RE_UNIX_PATH = re.compile(
        r'(?:/[\w.\-\u0600-\u06FF\u200c،()]+)+'
        r'\.(?:pdf|docx|doc|pptx|ppt|xlsx|xls|txt|md|csv|jpg|jpeg|png|bmp|tiff|gif|webp'
        r'|sqlite|db|sqlite3|sql)',
        re.IGNORECASE | re.UNICODE,
    )
    # الگوی «صفحه N» یا «page N»
    _RE_PAGE      = re.compile(r'(?:صفحه|page)[\s:]+([\d]+)', re.IGNORECASE)
    # الگوی «جدول X» یا «table X» یا «sheet X»
    _RE_TABLE     = re.compile(r'(?:جدول|table|sheet)[\s:]+([\w]+)', re.IGNORECASE)

    def __init__(self, config: Config):
        self.config = config

    # ── IDK Detection ─────────────────────────────────────────────

    def is_idk(self, answer: str) -> bool:
        """
        True اگر پاسخ واقعاً محتوای مفیدی نداشته باشد.
        معیارها:
          - خیلی کوتاه است
          - حاوی عبارت‌های IDK است
        """
        if not self.config.enable_idk:
            return False

        stripped = answer.strip()
        if len(stripped) < self.config.idk_min_length:
            return True

        lower = stripped.lower()
        for phrase in self.config.idk_phrases:
            if phrase.lower() in lower:
                return True

        return False

    def format_idk_response(
        self, question: str, fallback_results: List[str]
    ) -> str:
        """
        پاسخ «نمی‌دانم» استاندارد با نزدیک‌ترین منابع.
        fallback_results: نتایج خام RAG (حتی اگر ناکافی باشند)
        """
        lines = [
            "❓ **در منابع موجود پاسخ قطعی پیدا نشد.**\n",
        ]

        # نمایش نزدیک‌ترین منابع
        valid = [r for r in fallback_results if r and len(r) > 10][:3]
        if valid:
            lines.append("📎 **نزدیک‌ترین منابع مرتبط:**\n")
            for i, res in enumerate(valid, 1):
                snippet = res.strip()[:200].replace("\n", " ")
                lines.append(f"{i}. {snippet}…")
        else:
            lines.append("هیچ منبع مرتبطی در پایگاه دانش پیدا نشد.")

        lines.append(
            "\n💡 *پیشنهاد: سوال را با کلمات متفاوت بپرسید "
            "یا اسناد مرتبط بیشتری اضافه کنید.*"
        )
        return "\n".join(lines)

    # ── Source Citation ────────────────────────────────────────────

    def extract_sources(self, answer: str) -> List[Dict[str, str]]:
        """
        مسیرها و اطلاعات منبع را از متن پاسخ استخراج می‌کند.
        خروجی: لیست dict با کلیدهای path, name, ext, page?, table?
        """
        if not self.config.enable_source_citation:
            return []

        paths: List[str] = []
        for m in self._RE_WIN_PATH.finditer(answer):
            paths.append(m.group().replace("\\", "/").strip())
        for m in self._RE_UNIX_PATH.finditer(answer):
            p = m.group().strip()
            if p not in paths:
                paths.append(p)

        # dedup با حفظ ترتیب
        seen: set = set()
        unique: List[str] = []
        for p in paths:
            key = p.lower()
            if key not in seen:
                seen.add(key)
                unique.append(p)

        sources: List[Dict[str, str]] = []
        for path in unique[: self.config.citation_max_sources]:
            from pathlib import Path as _P
            p = _P(path)
            entry: Dict[str, str] = {
                "path": path,
                "name": p.name,
                "ext":  p.suffix.lower(),
            }
            # صفحه
            page_m = self._RE_PAGE.search(answer)
            if page_m:
                entry["page"] = page_m.group(1)
            # جدول/sheet
            tbl_m = self._RE_TABLE.search(answer)
            if tbl_m:
                entry["table"] = tbl_m.group(1)
            sources.append(entry)

        return sources

    def format_sources(self, sources: List[Dict[str, str]]) -> str:
        """
        فرمت نمایش منابع زیر پاسخ.
        """
        if not sources:
            return ""

        # آیکون بر اساس نوع فایل
        _icons = {
            ".pdf": "📄", ".docx": "📝", ".doc": "📝",
            ".pptx": "📊", ".ppt": "📊",
            ".xlsx": "📈", ".xls": "📈", ".csv": "📈",
            ".txt": "📃", ".md": "📃",
            ".jpg": "🖼️", ".jpeg": "🖼️", ".png": "🖼️",
            ".gif": "🖼️", ".webp": "🖼️", ".bmp": "🖼️", ".tiff": "🖼️",
            ".sqlite": "🗄️", ".db": "🗄️", ".sqlite3": "🗄️", ".sql": "🗄️",
        }

        lines = ["\n---", "**📌 منابع:**"]
        for i, src in enumerate(sources, 1):
            icon = _icons.get(src["ext"], "📎")
            line = f"{i}. {icon} **{src['name']}**"
            extras = []
            if src.get("page"):
                extras.append(f"صفحه {src['page']}")
            if src.get("table"):
                extras.append(f"جدول/Sheet: {src['table']}")
            if extras:
                line += " — " + " | ".join(extras)
            lines.append(line)

        return "\n".join(lines)

    def enrich_answer(
        self,
        question: str,
        answer: str,
        fallback_results: Optional[List[str]] = None,
    ) -> str:
        """
        نقطه ورودی اصلی:
          - اگر IDK → پاسخ «نمی‌دانم» با منابع نزدیک
          - در غیر این صورت → پاسخ اصلی + منابع
        """
        if self.is_idk(answer):
            return self.format_idk_response(question, fallback_results or [answer])

        sources = self.extract_sources(answer)
        if sources:
            return answer + self.format_sources(sources)
        return answer


# ══════════════════════════════════════════════════════════════════════
# جستجو — سه حالت
# ══════════════════════════════════════════════════════════════════════

async def do_query(
    rag: RAGAnything,
    question: str,
    mode: str,
    query_type: str,
    config: Config,
    logger: logging.Logger,
    enhancer: Optional["QueryEnhancer"] = None,
    chat_history: Optional[List[dict]] = None,
    multimodal_content: Optional[List[dict]] = None,
    vlm_enhanced: Optional[bool] = None,
    analyzer: Optional["AnswerAnalyzer"] = None,
) -> str:
    """
    query_type:
      "text"        → aquery معمولی (+ Enhancement اگر فعال باشد)
      "vlm"         → aquery با VLM Enhanced
      "multimodal"  → aquery_with_multimodal

    Enhancement pipeline (فقط برای text و vlm):
      1. Rewrite با تاریخچه
      2. Intent Router  → اگر "chat" بود، بدون RAG پاسخ می‌دهد
      3. Multi-Query    → چند query موازی
      4. HyDE           → query فرضی اضافه می‌شود
      5. RAG query      → بهترین نتیجه انتخاب می‌شود
      6. Reranker       → score اطمینان اضافه می‌شود
    """
    logger.info(f"🔍 Query [{query_type}/{mode}]: {question[:80]}...")

    # ── AnswerAnalyzer (IDK + Citation) ──────────────────────────
    _analyzer = analyzer or AnswerAnalyzer(config)

    async def _enrich(answer: str, fallback_results: Optional[List[str]] = None) -> str:
        """پاسخ را با IDK detection و source citation غنی می‌کند."""
        return _analyzer.enrich_answer(question, answer, fallback_results)

    try:
        # ── multimodal: بدون Enhancement ─────────────────────────
        if query_type == "multimodal" and multimodal_content:
            raw = await rag.aquery_with_multimodal(
                question,
                multimodal_content=multimodal_content,
                mode=mode,
            )
            return await _enrich(raw)

        # ── Enhancement pipeline ──────────────────────────────────
        if enhancer is not None and query_type in ("text", "vlm"):

            # ۱. بازنویسی با تاریخچه
            effective_query = await enhancer.rewrite_with_history(
                question, chat_history or []
            )

            # ۲. Intent Router
            intent = await enhancer.detect_intent(effective_query)
            logger.info(f"   Intent: {intent}")

            if intent == "chat":
                chat_prompt = (
                    "You are a helpful assistant. Answer the user's message naturally.\n"
                    "If they ask about documents or search, briefly explain what you can do.\n"
                    f"User: {effective_query}\nAnswer:"
                )
                try:
                    msgs = [{"role": "user", "content": chat_prompt}]
                    answer = await asyncio.to_thread(_call_llm_sync, msgs, config)
                    return _strip_thinking_tags(answer)
                except Exception as e:
                    logger.warning(f"Chat fallback خطا: {e}")

            # ۳. Multi-Query + HyDE (اجرای موازی)
            multi_task = enhancer.multi_query_rewrite(effective_query)
            hyde_task  = enhancer.hyde_generate(effective_query)
            all_queries_list, hyde_text = await asyncio.gather(multi_task, hyde_task)
            if hyde_text:
                all_queries_list.append(hyde_text)
            logger.info(f"   جمع query ها: {len(all_queries_list)} نسخه")

            # ۴. اجرای موازی همه query ها روی RAG
            _use_vlm = (query_type == "vlm") if vlm_enhanced is None else vlm_enhanced
            async def _single_query(q: str) -> str:
                try:
                    return await rag.aquery(
                        q, mode=mode,
                        top_k=config.top_k, chunk_top_k=config.chunk_top_k,
                        vlm_enhanced=_use_vlm,
                    )
                except Exception:
                    return ""

            rag_results = await asyncio.gather(*[_single_query(q) for q in all_queries_list])
            all_raw: List[str] = list(rag_results)

            # ۵. انتخاب بهترین نتیجه با Reranker
            valid_results = [r for r in rag_results if r and len(r) > 20]
            if not valid_results:
                # هیچ نتیجه‌ای نیامد → IDK قطعی
                return _analyzer.format_idk_response(effective_query, [])

            if len(valid_results) == 1:
                best_result = valid_results[0]
                best_score  = 1.0
            else:
                reranker = enhancer._get_reranker()
                if reranker is not None:
                    try:
                        pairs  = [(effective_query, r[:800]) for r in valid_results]
                        scores = reranker.predict(pairs)
                        best_idx    = int(max(range(len(scores)), key=lambda i: scores[i]))
                        best_result = valid_results[best_idx]
                        best_score  = float(scores[best_idx])
                        logger.info(f"   Reranker: idx={best_idx} score={best_score:.3f}")
                    except Exception as e:
                        logger.debug(f"Reranker خطا: {e}")
                        best_result = valid_results[0]
                        best_score  = 1.0
                else:
                    best_result = valid_results[0]
                    best_score  = 1.0

            # ۶. اگر score پایین → IDK با منابع
            if best_score < config.reranker_score_threshold:
                logger.warning(f"   ⚠️ score پایین: {best_score:.3f}")
                return _analyzer.format_idk_response(effective_query, valid_results)

            return await _enrich(best_result, all_raw)

        # ── بدون Enhancement (حالت ساده) ─────────────────────────
        _use_vlm = (query_type == "vlm") if vlm_enhanced is None else vlm_enhanced
        raw = await rag.aquery(
            question, mode=mode,
            top_k=config.top_k, chunk_top_k=config.chunk_top_k,
            vlm_enhanced=_use_vlm,
        )
        return await _enrich(raw)

    except Exception as e:
        logger.error(f"خطا در جستجو: {e}")
        return f"⚠️ خطا: {e}"


# ══════════════════════════════════════════════════════════════════════
# Session تعاملی
# ══════════════════════════════════════════════════════════════════════

HELP_TEXT = """
╔══════════════════════════════════════════════════════════════════╗
║  دستورات موجود:                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  /help              نمایش این راهنما                             ║
║  /mode <m>          hybrid|local|global|naive|mix                ║
║  /type <t>          text | vlm | multimodal                      ║
║  /vlm on|off        فعال/غیرفعال VLM Enhanced                    ║
║  /enhance on|off    فعال/غیرفعال Query Enhancement               ║
║  /history [clear]   تاریخچه مکالمه                              ║
║  ─────────────────────────────────────────────────────────────  ║
║  /imgsearch <path>  جستجوی تصویر مشابه با DINOv2                 ║
║  /imgbuild          ساخت/به‌روزرسانی image index                  ║
║  /imgbuild force    بازسازی کامل index از صفر                     ║
║  /imgstats          آمار image index                               ║
║  ─────────────────────────────────────────────────────────────  ║
║  /docs              لیست اسناد                                   ║
║  /db                لیست فایل‌های دیتابیسی                       ║
║  /stats             آمار کامل سیستم                              ║
║  /cache             وضعیت کش                                     ║
║  /insert            مثال درج مستقیم content list                 ║
║  /clear             پاک‌کردن صفحه                                ║
║  /exit              خروج                                          ║
╠══════════════════════════════════════════════════════════════════╣
║  multimodal JSON: {"q":"سوال","content":[{"type":"..."}]}       ║
╚══════════════════════════════════════════════════════════════════╝
"""

async def interactive_session(
    rag: RAGAnything,
    config: Config,
    logger: logging.Logger,
    file_cache: ProcessedFilesCache,
    img_engine=None,   # ImageSearchEngine | None
):
    query_mode    = "hybrid"   # hybrid | local | global | naive | mix
    query_type    = "text"     # text | vlm | multimodal
    vlm_override  = None       # None = auto | True | False
    chat_history: List[dict] = []

    # ساخت QueryEnhancer و AnswerAnalyzer
    enhancer    = QueryEnhancer(config, logger)
    analyzer    = AnswerAnalyzer(config)
    enhancement_on = True      # می‌توان با /enhance off غیرفعال کرد

    print("\n" + "═" * 66)
    print("🤖  RAG-Anything — سیستم جستجوی هوشمند چندوجهی")
    print("═" * 66)
    print("برای راهنما /help را وارد کنید\n")

    while True:
        try:
            prompt_label = f"[{query_type}/{query_mode}] ❓ "
            user_input = input(prompt_label).strip()

            if not user_input:
                continue

            cmd = user_input.lower()

            # ── دستورات ─────────────────────────────────────────────
            if cmd in ("/exit", "exit", "quit", "خروج"):
                print("👋 خداحافظ!")
                break

            if cmd == "/help":
                print(HELP_TEXT)
                continue

            if cmd == "/clear":
                os.system("clear" if os.name == "posix" else "cls")
                continue

            if cmd.startswith("/mode "):
                m = cmd.split()[1]
                if m in ("hybrid", "local", "global", "naive", "mix"):
                    query_mode = m
                    print(f"✅ حالت جستجو: {query_mode}")
                else:
                    print("⚠️ مقادیر مجاز: hybrid | local | global | naive | mix")
                continue

            if cmd.startswith("/type "):
                t = cmd.split()[1]
                if t in ("text", "vlm", "multimodal"):
                    query_type = t
                    print(f"✅ نوع query: {query_type}")
                else:
                    print("⚠️ مقادیر مجاز: text | vlm | multimodal")
                continue

            if cmd.startswith("/vlm "):
                v = cmd.split()[1]
                if v == "on":
                    vlm_override = True
                    print("✅ VLM Enhanced: روشن")
                elif v == "off":
                    vlm_override = False
                    print("✅ VLM Enhanced: خاموش")
                else:
                    vlm_override = None
                    print("✅ VLM Enhanced: خودکار")
                continue

            if cmd.startswith("/enhance "):
                v = cmd.split()[1]
                if v == "on":
                    enhancement_on = True
                    print("✅ Query Enhancement: روشن (Multi-Query + HyDE + Reranker + Intent)")
                elif v == "off":
                    enhancement_on = False
                    print("✅ Query Enhancement: خاموش (جستجوی ساده)")
                else:
                    print("⚠️ مقادیر مجاز: on | off")
                continue

            if cmd.startswith("/history"):
                parts = cmd.split()
                if len(parts) > 1 and parts[1] == "clear":
                    chat_history.clear()
                    print("✅ تاریخچه مکالمه پاک شد")
                else:
                    if not chat_history:
                        print("\n📜 تاریخچه خالی است\n")
                    else:
                        print(f"\n📜 تاریخچه مکالمه ({len(chat_history)} پیام):")
                        for i, m in enumerate(chat_history[-10:], 1):
                            role = "👤" if m["role"] == "user" else "🤖"
                            print(f"  {role} {m['content'][:120]}")
                        if len(chat_history) > 10:
                            print(f"  ... ({len(chat_history)-10} پیام قدیمی‌تر)")
                        print()
                continue

            if cmd == "/docs":
                files = []
                for ext in config.supported_extensions:
                    files.extend(config.data_dir.rglob(f"*{ext}"))
                    files.extend(config.data_dir.rglob(f"*{ext.upper()}"))
                files = sorted(set(files))
                print(f"\n📄 اسناد موجود ({len(files)}):")
                for f in files[:30]:
                    cached = "✅" if file_cache.is_processed(f) else "⬜"
                    size = f.stat().st_size / 1024
                    print(f"  {cached} {f.name}  ({size:.1f} KB)")
                if len(files) > 30:
                    print(f"  ... و {len(files)-30} فایل دیگر")
                continue

            if cmd == "/stats":
                files = [f for ext in config.supported_extensions for f in config.data_dir.rglob(f"*{ext}")]
                total_gb = sum(f.stat().st_size for f in files) / (1024**3)
                print(f"\n📊 آمار سیستم:")
                print(f"  📁 داده: {config.data_dir}")
                print(f"  💾 Storage: {config.working_dir}")
                print(f"  🖥️  LLM: {config.llm_model_name}")
                print(f"  🔤 Embedding: {config.embedding_model}")
                print(f"  🔍 حالت جستجو: {query_mode} | نوع: {query_type}")
                print(f"  📄 فایل‌ها: {len(files)} ({total_gb:.2f} GB)")
                print(f"  ✅ پردازش‌شده: {file_cache.count()}")
                print(f"  🔧 پارسر: {config.parser} | روش: {config.parse_method}")
                print(f"  🖥️  Device: {config.mineru_device}")
                print(f"  ✨ Enhancement: {'روشن' if enhancement_on else 'خاموش'}")
                if enhancement_on:
                    print(f"    • Multi-Query: {'✅' if config.enable_multi_query else '❌'} (n={config.multi_query_count})")
                    print(f"    • HyDE:        {'✅' if config.enable_hyde else '❌'}")
                    print(f"    • Reranker:    {'✅' if config.enable_reranker else '❌'} ({config.reranker_model.split('/')[-1]})")
                    print(f"    • Intent:      {'✅' if config.enable_intent_router else '❌'}")
                print(f"  💬 تاریخچه: {len(chat_history)} پیام")
                continue

            if cmd == "/cache":
                print(f"\n💾 کش: {file_cache.count()} فایل پردازش‌شده")
                print(f"  مسیر DB: {config.processed_files_db}")
                continue

            if cmd == "/insert":
                await insert_content_list_example(rag, logger)
                continue

            # ── جستجوی تصویر با تصویر ───────────────────────────
            if cmd.startswith("/imgsearch "):
                if img_engine is None:
                    print("⚠️ image search غیرفعال است — faiss/torch/transformers را نصب کنید")
                    continue
                if img_engine.total_indexed == 0:
                    print("⚠️ Index خالی است — ابتدا /imgbuild را اجرا کنید")
                    continue
                query_path_str = user_input[len("/imgsearch "):].strip().strip('"\'')
                query_path = Path(query_path_str)
                if not query_path.exists():
                    print(f"⚠️ فایل پیدا نشد: {query_path_str}")
                    continue
                print(f"\n🔍 جستجوی تصویر مشابه با DINOv2...")
                try:
                    t0 = time.time()
                    results = img_engine.search_by_path(query_path)
                    elapsed = time.time() - t0
                    print(img_engine.format_results(results))
                    print(f"\n  ⏱️  زمان جستجو: {elapsed*1000:.0f}ms")
                except Exception as e:
                    print(f"❌ خطا: {e}")
                continue

            if cmd.startswith("/imgbuild"):
                if img_engine is None:
                    print("⚠️ image search غیرفعال است")
                    continue
                force = "force" in cmd
                parser_out = config.parser_output_dir if config.index_document_images else None
                print(f"\n🏗️  ساخت image index{'  (force rebuild)' if force else ''}...")
                try:
                    t0 = time.time()
                    stats = img_engine.build_index(
                        data_dir          = config.data_dir,
                        parser_output_dir = parser_out,
                        force_rebuild     = force,
                    )
                    elapsed = time.time() - t0
                    print(f"✅ ساخت index تمام شد ({elapsed:.1f}s):")
                    print(f"   ایندکس شد:  {stats['indexed']}")
                    print(f"   رد شده:     {stats['skipped']}")
                    print(f"   خطا:        {stats['failed']}")
                    print(f"   مجموع:      {img_engine.total_indexed} تصویر\n")
                except Exception as e:
                    print(f"❌ خطا: {e}")
                continue

            if cmd == "/imgstats":
                if img_engine is None:
                    print("⚠️ image search غیرفعال است")
                else:
                    s = img_engine.stats()
                    print(f"\n🖼️  آمار image index:")
                    print(f"   مدل DINOv2:       {s['model']}")
                    print(f"   بعد embedding:     {s['dim']}")
                    print(f"   کل تصاویر:        {s['total']}")
                    print(f"   فایل مستقل:       {s['standalone']}")
                    print(f"   داخل سند:         {s['document_embed']}")
                    print(f"   بر اساس فرمت:     {s['by_extension']}")
                    print(f"   مسیر index:       {s['index_path']}\n")
                continue

            if cmd == "/db":
                db_files = []
                for ext in config.db_extensions:
                    db_files.extend(config.data_dir.rglob(f"*{ext}"))
                db_files = sorted(set(db_files))
                print(f"\n🗄️  فایل‌های دیتابیسی ({len(db_files)}):")
                for f in db_files:
                    cached = "✅" if file_cache.is_processed(f) else "⬜"
                    size = f.stat().st_size / 1024
                    print(f"  {cached} {f.name}  ({size:.1f} KB)  [{f.suffix}]")
                if config.mysql_conn:
                    print(f"  🔌 MySQL زنده: {config.mysql_conn.get('database')} @ {config.mysql_conn.get('host')}")
                if config.postgresql_conn:
                    print(f"  🔌 PostgreSQL زنده: {config.postgresql_conn.get('database')} @ {config.postgresql_conn.get('host')}")
                if not db_files and not config.mysql_conn and not config.postgresql_conn:
                    print("  (هیچ فایل یا اتصال دیتابیسی تعریف نشده)")
                print()
                continue

            # ── جستجوی چندوجهی ──────────────────────────────────────
            if query_type == "multimodal":
                multimodal_content = None
                question = user_input
                try:
                    parsed = json.loads(user_input)
                    question = parsed.get("q", user_input)
                    multimodal_content = parsed.get("content", None)
                except json.JSONDecodeError:
                    pass

                print("\n🤔 در حال پردازش چندوجهی...")
                answer = await do_query(
                    rag, question, query_mode, query_type, config, logger,
                    enhancer=None,   # multimodal از Enhancement استفاده نمی‌کند
                    multimodal_content=multimodal_content,
                    vlm_enhanced=vlm_override,
                
                    analyzer=analyzer,
                )
            else:
                enh_label = "✨ Enhancement" if enhancement_on else "⚡ ساده"
                print(f"\n🤔 در حال جستجو ({enh_label})...")
                answer = await do_query(
                    rag, user_input, query_mode, query_type, config, logger,
                    enhancer=enhancer if enhancement_on else None,
                    chat_history=chat_history if enhancement_on else None,
                    vlm_enhanced=vlm_override,
                
                    analyzer=analyzer,
                )

            # ذخیره در تاریخچه
            chat_history.append({"role": "user",      "content": user_input})
            chat_history.append({"role": "assistant", "content": answer})
            # نگه‌داشتن ۲۰ پیام آخر
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]

            # نمایش پاسخ
            print("\n" + "═" * 66)
            print("📝 پاسخ:")
            print("═" * 66)
            if len(answer) > 1500:
                print(answer[:1500])
                print(f"\n... (ادامه دارد — {len(answer)} کاراکتر)")
            else:
                print(answer)
            print("═" * 66 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 خداحافظ!")
            break
        except Exception as e:
            logger.error(f"خطای session: {e}")
            print(f"\n⚠️ خطا: {e}\n")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

async def main():
    config = Config()
    config.ensure_directories()
    logger = setup_logging(config)
    setup_cache(config.cache_dir)

    file_cache = ProcessedFilesCache(config.processed_files_db)

    print("\n" + "═" * 70)
    print("🚀  RAG-Anything Production System  —  v2.0")
    print("═" * 70)
    print(f"  📁 داده:      {config.data_dir}")
    print(f"  💾 Storage:   {config.working_dir}")
    print(f"  🖥️  LLM:       {config.llm_model_name}")
    print(f"  🔤 Embedding:  {config.embedding_model}")
    print(f"  🔧 پارسر:     {config.parser} / {config.parse_method}")
    print(f"  ✅ کش:        {file_cache.count()} فایل پردازش‌شده")
    print("═" * 70 + "\n")

    # جمع‌آوری فایل‌ها
    if not config.data_dir.exists():
        config.data_dir.mkdir()
        print(f"⚠️ پوشه data ساخته شد. فایل‌ها را اضافه کنید و دوباره اجرا کنید.\n")
        return

    all_files = []
    for ext in config.supported_extensions:
        all_files.extend(config.data_dir.rglob(f"*{ext}"))
        all_files.extend(config.data_dir.rglob(f"*{ext.upper()}"))
    all_files = sorted(set(all_files))

    # فایل‌های دیتابیسی (جدا)
    db_files = []
    for ext in config.db_extensions:
        db_files.extend(config.data_dir.rglob(f"*{ext}"))
        db_files.extend(config.data_dir.rglob(f"*{ext.upper()}"))
    db_files = sorted(set(db_files))

    if not all_files and not db_files and not config.mysql_conn and not config.postgresql_conn:
        print(f"⚠️ هیچ فایلی در {config.data_dir} پیدا نشد.\n")
    else:
        if all_files:
            total_gb = sum(f.stat().st_size for f in all_files) / (1024**3)
            new_files = [f for f in all_files if not file_cache.is_processed(f)]
            print(f"📄 اسناد: {len(all_files)} ({total_gb:.2f} GB)")
            print(f"   🆕 پردازش جدید: {len(new_files)} | ✅ کش: {len(all_files)-len(new_files)}")
        if db_files:
            new_db = [f for f in db_files if not file_cache.is_processed(f)]
            print(f"🗄️  دیتابیس: {len(db_files)} فایل")
            print(f"   🆕 پردازش جدید: {len(new_db)} | ✅ کش: {len(db_files)-len(new_db)}")
        if config.mysql_conn:
            print(f"🔌 MySQL: {config.mysql_conn.get('database')} @ {config.mysql_conn.get('host')}")
        if config.postgresql_conn:
            print(f"🔌 PostgreSQL: {config.postgresql_conn.get('database')} @ {config.postgresql_conn.get('host')}")
        print()

    # ساخت RAG
    embedding_service = EmbeddingService(config, logger)
    logger.info("در حال راه‌اندازی RAG-Anything...")
    rag = await create_rag(config, embedding_service)
    logger.info("✓ RAG-Anything آماده است")

    # ساخت Image Search Engine
    img_engine = None
    if config.enable_image_search and _IMG_SEARCH_AVAILABLE:
        logger.info("در حال راه‌اندازی Image Search (DINOv2 + FAISS)...")
        img_engine = build_image_search_engine(config, logger)
        if img_engine is not None:
            parser_out = config.parser_output_dir if config.index_document_images else None
            ready = img_engine.load_or_build(config.data_dir, parser_out)
            if ready:
                logger.info(f"✓ Image Search آماده — {img_engine.total_indexed} تصویر در index")
            else:
                logger.info("ℹ️  Image index خالی است — از /imgbuild برای ساخت استفاده کنید")

    # سوال پردازش فایل‌های سند
    new_files = [f for f in all_files if not file_cache.is_processed(f)]
    if all_files and new_files:
        print("═" * 50)
        ans = input(f"آیا {len(new_files)} سند جدید پردازش شوند؟ (y/n): ").strip().lower()
        print("═" * 50 + "\n")

        if ans in ("y", "yes", "بله", "آره"):
            force = input("پردازش مجدد اسناد قبلی هم؟ (y/n): ").strip().lower() in ("y", "yes")
            result = await process_documents(rag, all_files, config, logger, file_cache, force_reprocess=force)
            print("\n" + "═" * 60)
            print("📊 خلاصه پردازش اسناد:")
            print(f"  کل: {result['total']} | ✅ {result['success']} | ⏭️ {result['skipped']} | ❌ {result['failed']}")
            if result["failed_files"]:
                for f in result["failed_files"][:10]:
                    print(f"    - {f}")
            print("═" * 60 + "\n")
        else:
            print("⚠️ پردازش اسناد رد شد.\n")

    # سوال پردازش دیتابیس‌ها
    has_db = db_files or config.mysql_conn or config.postgresql_conn
    if has_db and _DB_AVAILABLE:
        new_db_files = [f for f in db_files if not file_cache.is_processed(f)]
        live_conns = (1 if config.mysql_conn else 0) + (1 if config.postgresql_conn else 0)
        pending = len(new_db_files) + live_conns

        if pending > 0:
            print("═" * 50)
            db_desc = []
            if new_db_files:
                db_desc.append(f"{len(new_db_files)} فایل دیتابیس")
            if live_conns:
                db_desc.append(f"{live_conns} اتصال زنده")
            ans_db = input(f"آیا {' و '.join(db_desc)} پردازش شوند؟ (y/n): ").strip().lower()
            print("═" * 50 + "\n")

            if ans_db in ("y", "yes", "بله", "آره"):
                force_db = input("پردازش مجدد دیتابیس‌های قبلی هم؟ (y/n): ").strip().lower() in ("y", "yes")
                db_result = await process_databases(rag, config, logger, file_cache, force_reprocess=force_db)
                print("═" * 60)
                print("📊 خلاصه پردازش دیتابیس:")
                print(f"  فایل‌ها: ✅ {db_result['files_processed']} | ⏭️ {db_result['files_skipped']} | ❌ {db_result['files_failed']}")
                print(f"  اتصال زنده: {db_result['live_connections']}")
                print(f"  جداول: {db_result['total_tables']} | آیتم‌های RAG: {db_result['total_items']}")
                print("═" * 60 + "\n")
            else:
                print("⚠️ پردازش دیتابیس رد شد.\n")
        else:
            print("✅ همه دیتابیس‌ها قبلاً پردازش شده‌اند.\n")
    elif has_db and not _DB_AVAILABLE:
        print("⚠️ db_processor.py پیدا نشد — فایل‌های دیتابیسی پردازش نمی‌شوند.\n")

    # شروع session تعاملی
    await interactive_session(rag, config, logger, file_cache, img_engine=img_engine)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 خداحافظ!")
    except Exception as e:
        print(f"\n❌ خطای بحرانی: {e}")
        sys.exit(1)