"""
╔══════════════════════════════════════════════════════════════════════╗
║  image_search.py  —  جستجوی تصویر به تصویر (Google Lens mode)      ║
╠══════════════════════════════════════════════════════════════════════╣
║  معماری:                                                             ║
║    Index-time:                                                        ║
║      فایل تصویری مستقل  ──┐                                          ║
║      تصویر داخل سند      ──┼──→ DINOv2 embed → FAISS index          ║
║      (parser_output_dir) ──┘                                          ║
║                                                                      ║
║    Query-time:                                                        ║
║      کاربر تصویر می‌دهد → DINOv2 embed → جستجو در FAISS             ║
║      → نتایج مشابه با score + metadata                               ║
║                                                                      ║
║  وابستگی‌ها:                                                          ║
║    pip install torch transformers faiss-cpu pillow opencv-python      ║
║    (faiss-gpu اگر GPU دارید: pip install faiss-gpu)                  ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# ── torch و transformers (DINOv2) ────────────────────────────────────
try:
    import torch
    from transformers import AutoImageProcessor, AutoModel
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ── FAISS ────────────────────────────────────────────────────────────
try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════
# تنظیمات
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ImageSearchConfig:
    # مدل DINOv2 — گزینه‌ها:
    #   facebook/dinov2-small   (dim=384,  سریع‌تر)
    #   facebook/dinov2-base    (dim=768,  متوازن)
    #   facebook/dinov2-large   (dim=1024, دقیق‌تر)
    dino_model_id: str = "facebook/dinov2-base"
    dino_dim: int = 768               # باید با مدل match باشد

    # مسیر ذخیره index
    index_dir: Path = Path("./image_index")

    # پسوندهای تصویری که index می‌شوند
    image_extensions: List[str] = field(default_factory=lambda: [
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
        ".gif", ".webp",
    ])

    # حداکثر تصاویر برای index (None = بدون محدودیت)
    max_images: Optional[int] = None

    # اندازه batch برای embed کردن
    batch_size: int = 16

    # حداکثر نتایج جستجو
    default_top_k: int = 10

    # حداقل score برای نمایش (cosine similarity: 0.0 تا 1.0)
    min_score: float = 0.0

    # آیا تصاویر داخل parser_output_dir هم index شوند؟
    index_parser_output: bool = True

    # cache مسیر مدل
    model_cache_dir: Optional[str] = None

    def ensure_dirs(self):
        self.index_dir.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# بارگذاری تصویر (سازگار با مسیر فارسی / Unicode)
# ══════════════════════════════════════════════════════════════════════

def _load_pil(path: Path) -> Image.Image:
    """
    بارگذاری تصویر — اول با OpenCV (سازگار با مسیر Unicode ویندوز)،
    و در صورت شکست، fallback به PIL مستقیم.

    دلیل fallback: OpenCV گاهی روی PNG هایی با پروفایل رنگی غیراستاندارد
    (iCCP warning) یا برخی JPEG های دارای داده اضافی شکست می‌خورد،
    در حالی که PIL آن‌ها را به‌خوبی می‌خواند.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    img_cv = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_cv is not None:
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)

    # ── fallback به PIL ──────────────────────────────────────────
    try:
        img_pil = Image.open(str(path))
        img_pil.load()   # بارگذاری کامل قبل از convert (تشخیص فایل‌های واقعاً خراب)
        return img_pil.convert("RGB")
    except Exception as e:
        raise RuntimeError(
            f"نه OpenCV و نه PIL نتوانستند تصویر را decode کنند: {path} ({e})"
        )


def _load_pil_from_bytes(raw: bytes) -> Image.Image:
    """بارگذاری از bytes خام (مثلاً از HTTP upload)."""
    import io
    return Image.open(io.BytesIO(raw)).convert("RGB")


# ══════════════════════════════════════════════════════════════════════
# DINOv2 Encoder
# ══════════════════════════════════════════════════════════════════════

class DINOv2Encoder:
    """
    تولید embedding با DINOv2.
    از CLS token آخرین لایه استفاده می‌کند و L2 normalize می‌کند.
    """

    def __init__(self, cfg: ImageSearchConfig, logger: logging.Logger):
        self.cfg    = cfg
        self.logger = logger
        self._processor = None
        self._model     = None
        self._device    = "cpu"

    def load(self):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "torch و transformers لازم است:\n"
                "  pip install torch transformers"
            )

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"بارگذاری DINOv2: {self.cfg.dino_model_id}  (device={self._device})")

        kwargs = {}
        if self.cfg.model_cache_dir:
            kwargs["cache_dir"] = self.cfg.model_cache_dir

        self._processor = AutoImageProcessor.from_pretrained(
            self.cfg.dino_model_id, **kwargs
        )
        self._model = AutoModel.from_pretrained(
            self.cfg.dino_model_id, **kwargs
        ).to(self._device)
        self._model.eval()
        self.logger.info(f"✓ DINOv2 آماده — dim={self.cfg.dino_dim}")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def encode_single(self, img_pil: Image.Image) -> np.ndarray:
        """یک تصویر → بردار numpy به شکل (dim,)"""
        return self.encode_batch([img_pil])[0]

    def encode_batch(self, images: List[Image.Image]) -> np.ndarray:
        """لیستی از تصاویر → ماتریس numpy به شکل (N, dim)"""
        if not self.is_loaded:
            self.load()

        inputs = self._processor(images=images, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
            # CLS token — اولین token از last_hidden_state
            cls_vecs = outputs.last_hidden_state[:, 0, :]

        # L2 normalize
        norms = cls_vecs.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        cls_vecs = cls_vecs / norms
        return cls_vecs.cpu().numpy().astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# FAISS Index Manager
# ══════════════════════════════════════════════════════════════════════

class FAISSImageIndex:
    """
    نگه‌داری و جستجو در FAISS index.

    ساختار فایل‌ها:
        index_dir/
          image_index.faiss   ← بردارها
          image_meta.pkl      ← metadata (مسیر، منبع، ...)
    """

    INDEX_FILE = "image_index.faiss"
    META_FILE  = "image_meta.pkl"

    def __init__(self, cfg: ImageSearchConfig, logger: logging.Logger):
        self.cfg    = cfg
        self.logger = logger
        self._index: Optional[faiss.Index] = None
        self._meta:  List[Dict[str, Any]]  = []   # یک dict به ازای هر بردار

    # ── بارگذاری / ذخیره ────────────────────────────────────────────

    @property
    def index_path(self) -> Path:
        return self.cfg.index_dir / self.INDEX_FILE

    @property
    def meta_path(self) -> Path:
        return self.cfg.index_dir / self.META_FILE

    def load(self) -> bool:
        """بارگذاری index موجود. اگر نبود False برمی‌گرداند."""
        if not _FAISS_AVAILABLE:
            raise ImportError("faiss لازم است:\n  pip install faiss-cpu")

        if not self.index_path.exists() or not self.meta_path.exists():
            return False

        self._index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, "rb") as f:
            self._meta = pickle.load(f)

        self.logger.info(
            f"✓ Image index بارگذاری شد — {self._index.ntotal} تصویر"
        )
        return True

    def save(self):
        """ذخیره index روی دیسک."""
        if self._index is None:
            return
        self.cfg.ensure_dirs()
        faiss.write_index(self._index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump(self._meta, f)
        self.logger.info(
            f"💾 Image index ذخیره شد — {self._index.ntotal} تصویر"
        )

    # ── ساخت و افزودن ────────────────────────────────────────────────

    def _ensure_index(self, dim: int):
        """اگر index وجود نداشت یکی می‌سازیم."""
        if self._index is None:
            # IndexFlatIP = inner product روی L2-normalized vectors = cosine similarity
            self._index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray, metas: List[Dict[str, Any]]):
        """
        vectors: (N, dim) float32
        metas:   N dict — هر dict حاوی اطلاعات تصویر
        """
        if not _FAISS_AVAILABLE:
            raise ImportError("faiss لازم است:\n  pip install faiss-cpu")

        dim = vectors.shape[1]
        self._ensure_index(dim)

        self._index.add(vectors)
        self._meta.extend(metas)

    def indexed_paths(self) -> set:
        """مجموعه مسیرهای قبلاً index شده."""
        return {m["path"] for m in self._meta}

    @property
    def total(self) -> int:
        return self._index.ntotal if self._index else 0

    # ── جستجو ───────────────────────────────────────────────────────

    def search(
        self, query_vec: np.ndarray, top_k: int = 10, min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        query_vec: (dim,) float32 — L2 normalized
        خروجی: لیست dict مرتب‌شده بر اساس score نزولی
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        query = query_vec.reshape(1, -1).astype(np.float32)
        k = min(top_k, self._index.ntotal)

        scores, indices = self._index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            score_f = float(score)
            if score_f < min_score:
                continue
            entry = dict(self._meta[idx])
            entry["score"] = round(score_f, 4)
            results.append(entry)

        return results


# ══════════════════════════════════════════════════════════════════════
# Image Search Engine — نقطه ورودی اصلی
# ══════════════════════════════════════════════════════════════════════

class ImageSearchEngine:
    """
    مدیریت کامل index و جستجوی تصویر.

    استفاده:
        engine = ImageSearchEngine(cfg, logger)
        engine.load_or_build(data_dir, parser_output_dir)

        # جستجو با فایل
        results = engine.search_by_path(Path("query.jpg"))

        # جستجو با bytes (از HTTP upload)
        results = engine.search_by_bytes(raw_bytes)
    """

    def __init__(self, cfg: ImageSearchConfig, logger: logging.Logger):
        self.cfg     = cfg
        self.logger  = logger
        self.encoder = DINOv2Encoder(cfg, logger)
        self.index   = FAISSImageIndex(cfg, logger)

    # ── جمع‌آوری مسیر تصاویر ────────────────────────────────────────

    def _collect_images(
        self,
        data_dir: Path,
        parser_output_dir: Optional[Path] = None,
    ) -> List[Tuple[Path, str]]:
        """
        جمع‌آوری همه تصاویر از:
          1. data_dir (فایل‌های مستقل)
          2. parser_output_dir (تصاویر استخراج‌شده از اسناد)

        خروجی: لیست (path, source)
          source: "standalone" | "document_embed"
        """
        found: List[Tuple[Path, str]] = []
        exts = {e.lower() for e in self.cfg.image_extensions}

        def _scan(directory: Path, source: str):
            if not directory.exists():
                return
            for ext in exts:
                for p in directory.rglob(f"*{ext}"):
                    found.append((p, source))
                for p in directory.rglob(f"*{ext.upper()}"):
                    found.append((p, source))

        _scan(data_dir, "standalone")

        if self.cfg.index_parser_output and parser_output_dir:
            _scan(parser_output_dir, "document_embed")

        # dedup بر اساس resolved path
        seen = set()
        unique = []
        for p, s in found:
            rp = str(p.resolve())
            if rp not in seen:
                seen.add(rp)
                unique.append((p, s))

        return unique

    # ── ساخت index ──────────────────────────────────────────────────

    def build_index(
        self,
        data_dir: Path,
        parser_output_dir: Optional[Path] = None,
        force_rebuild: bool = False,
    ) -> Dict[str, int]:
        """
        ساخت یا به‌روزرسانی index.
        تصاویری که قبلاً index شده‌اند رد می‌شوند (مگر force_rebuild=True).
        """
        if not _FAISS_AVAILABLE:
            raise ImportError("faiss لازم است:\n  pip install faiss-cpu")

        self.encoder.load()

        all_images = self._collect_images(data_dir, parser_output_dir)
        self.logger.info(f"📸 تصاویر یافت‌شده: {len(all_images)}")

        if not all_images:
            self.logger.warning("هیچ تصویری یافت نشد.")
            return {"total": 0, "indexed": 0, "skipped": 0, "failed": 0}

        # فیلتر تصاویر جدید
        if not force_rebuild:
            already = self.index.indexed_paths()
            to_index = [(p, s) for p, s in all_images if str(p.resolve()) not in already]
        else:
            # ساخت index از صفر
            self.index._index = None
            self.index._meta  = []
            to_index = all_images

        # محدودیت تعداد
        if self.cfg.max_images:
            to_index = to_index[: self.cfg.max_images]

        self.logger.info(
            f"🆕 جدید: {len(to_index)} | "
            f"⏭️ رد‌شده: {len(all_images)-len(to_index)}"
        )

        stats = {"total": len(all_images), "indexed": 0, "skipped": len(all_images) - len(to_index), "failed": 0}

        # پردازش به‌صورت batch
        batch_imgs:  List[Image.Image]   = []
        batch_metas: List[Dict[str, Any]] = []

        def _flush():
            if not batch_imgs:
                return
            try:
                vecs = self.encoder.encode_batch(batch_imgs)
                self.index.add(vecs, batch_metas)
                stats["indexed"] += len(batch_imgs)
            except Exception as e:
                self.logger.error(f"Batch embed خطا: {e}")
                stats["failed"] += len(batch_imgs)
            batch_imgs.clear()
            batch_metas.clear()

        for i, (img_path, source) in enumerate(to_index, 1):
            try:
                img_pil = _load_pil(img_path)

                # تشخیص اینکه تصویر داخل کدام سند بوده
                doc_path = _infer_document_source(img_path, parser_output_dir)

                batch_imgs.append(img_pil)
                batch_metas.append({
                    "path":       str(img_path.resolve()),
                    "name":       img_path.name,
                    "source":     source,
                    "doc_source": doc_path,
                    "size_kb":    round(img_path.stat().st_size / 1024, 1),
                    "ext":        img_path.suffix.lower(),
                })

                if len(batch_imgs) >= self.cfg.batch_size:
                    _flush()
                    pct = stats["indexed"] / len(to_index) * 100
                    self.logger.info(
                        f"  📊 {stats['indexed']}/{len(to_index)} ({pct:.0f}%)"
                    )

            except Exception as e:
                self.logger.warning(f"  ⚠️ {img_path.name}: {e}")
                stats["failed"] += 1

        _flush()   # باقی‌مانده

        # ذخیره
        self.index.save()
        self.logger.info(
            f"✅ Index ساخته شد — "
            f"indexed={stats['indexed']} | failed={stats['failed']}"
        )
        return stats

    # ── بارگذاری یا ساخت ────────────────────────────────────────────

    def load_or_build(
        self,
        data_dir: Path,
        parser_output_dir: Optional[Path] = None,
        force_rebuild: bool = False,
    ) -> bool:
        """
        اگر index موجود باشد بارگذاری می‌کند.
        اگر نبود یا force_rebuild=True بود، می‌سازد.
        خروجی: True اگر engine آماده باشد.
        """
        if not _FAISS_AVAILABLE:
            self.logger.warning(
                "faiss نصب نیست — image search غیرفعال:\n"
                "  pip install faiss-cpu"
            )
            return False

        if not force_rebuild and self.index.load():
            # index موجود است — تصاویر جدید را اضافه می‌کنیم
            all_imgs = self._collect_images(data_dir, parser_output_dir)
            new_imgs = [
                (p, s) for p, s in all_imgs
                if str(p.resolve()) not in self.index.indexed_paths()
            ]
            if new_imgs:
                self.logger.info(f"📸 {len(new_imgs)} تصویر جدید به index اضافه می‌شود")
                self.encoder.load()
                batch_imgs:  List[Image.Image]    = []
                batch_metas: List[Dict[str, Any]] = []

                def _flush_inc():
                    if not batch_imgs:
                        return
                    try:
                        vecs = self.encoder.encode_batch(batch_imgs)
                        self.index.add(vecs, batch_metas)
                    except Exception as e:
                        self.logger.warning(f"Incremental index خطا: {e}")
                    batch_imgs.clear()
                    batch_metas.clear()

                for img_path, source in new_imgs:
                    try:
                        doc_path = _infer_document_source(img_path, parser_output_dir)
                        batch_imgs.append(_load_pil(img_path))
                        batch_metas.append({
                            "path":       str(img_path.resolve()),
                            "name":       img_path.name,
                            "source":     source,
                            "doc_source": doc_path,
                            "size_kb":    round(img_path.stat().st_size / 1024, 1),
                            "ext":        img_path.suffix.lower(),
                        })
                        if len(batch_imgs) >= self.cfg.batch_size:
                            _flush_inc()
                    except Exception as e:
                        self.logger.warning(f"  ⚠️ {img_path.name}: {e}")

                _flush_inc()
                self.index.save()
        else:
            stats = self.build_index(data_dir, parser_output_dir, force_rebuild)
            if stats["indexed"] == 0 and stats["total"] == 0:
                return False

        return self.index.total > 0

    # ── جستجو ───────────────────────────────────────────────────────

    def _do_search(
        self, img_pil: Image.Image, top_k: int, min_score: float
    ) -> List[Dict[str, Any]]:
        if not self.encoder.is_loaded:
            self.encoder.load()
        vec = self.encoder.encode_single(img_pil)
        return self.index.search(vec, top_k=top_k, min_score=min_score)

    def search_by_path(
        self,
        img_path: Path,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """جستجو با مسیر فایل تصویری."""
        img_pil = _load_pil(img_path)
        return self._do_search(
            img_pil,
            top_k    = top_k    or self.cfg.default_top_k,
            min_score= min_score or self.cfg.min_score,
        )

    def search_by_bytes(
        self,
        raw_bytes: bytes,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """جستجو با bytes خام (مثلاً از HTTP upload یا clipboard)."""
        img_pil = _load_pil_from_bytes(raw_bytes)
        return self._do_search(
            img_pil,
            top_k    = top_k    or self.cfg.default_top_k,
            min_score= min_score or self.cfg.min_score,
        )

    # ── آمار ────────────────────────────────────────────────────────

    @property
    def total_indexed(self) -> int:
        return self.index.total

    def stats(self) -> Dict[str, Any]:
        meta = self.index._meta
        standalone = sum(1 for m in meta if m.get("source") == "standalone")
        embedded   = sum(1 for m in meta if m.get("source") == "document_embed")
        exts: Dict[str, int] = {}
        for m in meta:
            e = m.get("ext", "unknown")
            exts[e] = exts.get(e, 0) + 1
        return {
            "total":          self.index.total,
            "standalone":     standalone,
            "document_embed": embedded,
            "by_extension":   exts,
            "index_path":     str(self.index.index_path),
            "model":          self.cfg.dino_model_id,
            "dim":            self.cfg.dino_dim,
        }

    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """نمایش نتایج جستجو به صورت متن."""
        if not results:
            return "❌ هیچ تصویر مشابهی یافت نشد."

        lines = [f"📸 {len(results)} تصویر مشابه یافت شد:\n"]
        for i, r in enumerate(results, 1):
            source_icon = "📄" if r.get("source") == "document_embed" else "🖼️"
            doc_info = ""
            if r.get("doc_source"):
                doc_info = f"\n     📎 سند منبع: {r['doc_source']}"

            lines.append(
                f"  {i}. {source_icon} {r['name']}\n"
                f"     📊 شباهت: {r['score']:.1%}\n"
                f"     📁 مسیر: {r['path']}\n"
                f"     💾 حجم: {r.get('size_kb', '?')} KB"
                f"{doc_info}"
            )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# تابع کمکی — تشخیص سند منبع تصویر
# ══════════════════════════════════════════════════════════════════════

def _infer_document_source(
    img_path: Path,
    parser_output_dir: Optional[Path],
) -> Optional[str]:
    """
    MinerU تصاویر را در ساختار زیر ذخیره می‌کند:
        parser_output_dir/
          <doc_name>/
            images/
              figure_0.png
              figure_1.jpg

    با بررسی مسیر، نام سند اصلی را استنباط می‌کنیم.
    """
    if parser_output_dir is None:
        return None

    try:
        rel = img_path.resolve().relative_to(parser_output_dir.resolve())
        parts = rel.parts
        # parts[0] = نام پوشه سند
        if len(parts) >= 2:
            return parts[0]
    except ValueError:
        pass

    return None