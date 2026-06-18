"""
╔══════════════════════════════════════════════════════════════════════╗
║  app.py  —  RAG-Anything Web UI  v2                                 ║
║  اجرا:  streamlit run app.py                                         ║
╚══════════════════════════════════════════════════════════════════════╝

اصلاحات v2:
  ✅ ۱. نام فایل اصلی قابل تنظیم (RAG_SYSTEM_FILE env var)
  ✅ ۲. vlm_enhanced از query_type استنتاج می‌شود
  ✅ ۳. multimodal — JSON parse و پاس‌دادن content
  ✅ ۴. chat_history قبل از append ساخته می‌شود
  ✅ ۵. تب آپلود و پردازش سند
  ✅ ۶. تب مدیریت دیتابیس
  ✅ ۷. بعد از پردازش سند → پیشنهاد به‌روزرسانی image index
  ✅ ۸. تنظیمات per-session (کپی Config برای هر session)
"""

import streamlit as st
import time

st.set_page_config(
    page_title="RAG-Anything",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════
# استایل سراسری
# ════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Vazirmatn', sans-serif !important;
    direction: rtl;
}
.stApp { background: #f5f7fa; color: #1a1d23; }

[data-testid="stSidebar"] { background:#fff; border-left:1px solid #e2e8f0; }
[data-testid="stSidebar"] * { direction:rtl; text-align:right; }

.sidebar-section {
    font-size:13px; font-weight:700; color:#2563eb;
    padding:10px 0 8px 0; border-bottom:1px solid #e2e8f0; margin-bottom:12px;
}

[data-testid="stChatMessageContent"] { direction:rtl; text-align:right; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background:#eff6ff; border:1px solid #bfdbfe; border-radius:12px; margin-bottom:8px;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background:#fff; border:1px solid #e2e8f0; border-radius:12px; margin-bottom:8px;
}
[data-testid="stChatInput"] textarea {
    background:#fff !important; color:#1a1d23 !important;
    border:1px solid #cbd5e1 !important; border-radius:12px !important;
    direction:rtl; font-family:'Vazirmatn',sans-serif !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color:#2563eb !important; box-shadow:0 0 0 3px rgba(37,99,235,.12) !important;
}

.badge {
    display:inline-flex; align-items:center; gap:4px;
    padding:2px 10px; border-radius:20px; font-size:12px; font-weight:500;
}
.badge-search { background:#dbeafe; color:#1d4ed8; border:1px solid #93c5fd; }
.badge-chat   { background:#fef9c3; color:#854d0e; border:1px solid #fde047; }
.badge-vlm    { background:#f3e8ff; color:#6b21a8; border:1px solid #d8b4fe; }
.badge-mm     { background:#dcfce7; color:#166534; border:1px solid #86efac; }

.pipe-step {
    display:inline-flex; align-items:center; gap:4px;
    padding:2px 9px; background:#f1f5f9; border-radius:6px;
    font-size:11px; color:#94a3b8; margin:2px; border:1px solid #e2e8f0;
}
.pipe-step.done { color:#166534; background:#dcfce7; border-color:#86efac; }

.result-card {
    background:#fff; border:1px solid #e2e8f0; border-radius:10px;
    padding:12px 14px; margin:5px 0; direction:rtl;
    transition:border-color .2s, box-shadow .2s;
}
.result-card:hover { border-color:#2563eb; box-shadow:0 2px 8px rgba(37,99,235,.1); }

[data-testid="stMetric"] {
    background:#fff; border:1px solid #e2e8f0; border-radius:10px;
    padding:10px 14px; direction:rtl; text-align:right;
}
[data-testid="stMetricLabel"] { color:#64748b !important; font-size:12px !important; }
[data-testid="stMetricValue"] { color:#1a1d23 !important; font-size:22px !important; }

.stButton button {
    background:#fff; color:#374151; border:1px solid #d1d5db;
    border-radius:8px; font-family:'Vazirmatn',sans-serif; transition:all .2s;
}
.stButton button:hover { background:#eff6ff; border-color:#2563eb; color:#2563eb; }

.stAlert { direction:rtl; text-align:right; }
[data-testid="stSlider"] * { direction:ltr; }
hr { border-color:#e2e8f0 !important; }
.main-title { font-size:26px; font-weight:700; color:#1a1d23; margin-bottom:2px; }
.main-sub   { font-size:13px; color:#64748b; margin-bottom:18px; }
.accent     { color:#2563eb; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# ① بارگذاری engine — یک‌بار، مشترک بین همه session‌ها
# ════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _load_engine():
    """
    فایل اصلی را پیدا کرده و همه اجزا را initialize می‌کند.

    اولویت پیدا کردن فایل اصلی:
      1. متغیر محیطی RAG_SYSTEM_FILE
      2. فایل rag_system.py در همان پوشه app.py
      3. فایل main.py در همان پوشه
    """
    import importlib.util, sys as _sys, os, asyncio

    ph = st.empty()
    ph.info("⏳ در حال بارگذاری RAG-Anything...")

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # ① پیدا کردن فایل اصلی
        candidates = [
            os.environ.get("RAG_SYSTEM_FILE", ""),
            os.path.join(base_dir, "rag_system.py"),
            os.path.join(base_dir, "main.py"),
        ]
        rag_file = next((p for p in candidates if p and os.path.isfile(p)), None)
        if rag_file is None:
            raise FileNotFoundError(
                "فایل اصلی پیدا نشد.\n"
                "یا فایل را rag_system.py نام‌گذاری کنید،\n"
                "یا متغیر محیطی RAG_SYSTEM_FILE را تنظیم کنید."
            )

        spec = importlib.util.spec_from_file_location("rag_system", rag_file)
        mod  = importlib.util.module_from_spec(spec)
        _sys.modules["rag_system"] = mod
        spec.loader.exec_module(mod)

        # ② ساخت Config و اجزای مشترک
        config        = mod.Config()
        config.ensure_directories()
        logger        = mod.setup_logging(config)
        mod.setup_cache(config.cache_dir)
        file_cache    = mod.ProcessedFilesCache(config.processed_files_db)

        embed_svc     = mod.EmbeddingService(config, logger)
        loop          = asyncio.new_event_loop()
        rag           = loop.run_until_complete(mod.create_rag(config, embed_svc))

        img_engine = None
        if mod._IMG_SEARCH_AVAILABLE and config.enable_image_search:
            img_engine = mod.build_image_search_engine(config, logger)
            if img_engine:
                parser_out = config.parser_output_dir if config.index_document_images else None
                img_engine.load_or_build(config.data_dir, parser_out)

        enhancer = mod.QueryEnhancer(config, logger)
        analyzer = mod.AnswerAnalyzer(config)

        ph.empty()
        return {
            "mod": mod, "config": config, "logger": logger,
            "file_cache": file_cache, "rag": rag,
            "img_engine": img_engine, "enhancer": enhancer, "analyzer": analyzer, "loop": loop,
        }

    except Exception as e:
        ph.error(f"❌ {e}")
        st.exception(e)
        st.stop()


_engine    = _load_engine()
mod        = _engine["mod"]
_config    = _engine["config"]   # config مشترک — فقط برای read
logger     = _engine["logger"]
file_cache = _engine["file_cache"]
rag        = _engine["rag"]
img_engine = _engine["img_engine"]
enhancer   = _engine["enhancer"]
analyzer   = _engine["analyzer"]
loop       = _engine["loop"]


# ════════════════════════════════════════════════════════
# ⑧ تنظیمات per-session — هر session کپی جداگانه دارد
# ════════════════════════════════════════════════════════

def _get_session_cfg():
    """
    یک نسخه Config مختص این session.
    اولین بار از config مشترک کپی می‌شود؛ بعد فقط از session خوانده می‌شود.
    """
    if "_cfg" not in st.session_state:
        import copy
        st.session_state["_cfg"] = copy.copy(_config)
    return st.session_state["_cfg"]

cfg = _get_session_cfg()   # config این session


# ════════════════════════════════════════════════════════
# توابع کمکی
# ════════════════════════════════════════════════════════

def _run(coro):
    return loop.run_until_complete(coro)


def _query_rag(question: str, mode: str, query_type: str, chat_history: list) -> tuple:
    """
    ② vlm_enhanced از query_type استنتاج می‌شود.
    ③ اگر query_type == multimodal، JSON parse می‌شود.
    خروجی: (answer, intent, effective_question)
    """
    multimodal_content = None
    effective_question = question
    vlm_enhanced       = (query_type == "vlm")   # ← اصلاح ایراد ②

    # ③ parse JSON برای multimodal
    if query_type == "multimodal":
        import json
        try:
            parsed = json.loads(question)
            effective_question  = parsed.get("q", question)
            multimodal_content  = parsed.get("content", None)
        except (json.JSONDecodeError, ValueError):
            pass   # ورودی ساده — بدون content

    async def _run_inner():
        intent = "search"
        if cfg.enable_intent_router and query_type != "multimodal":
            intent = await enhancer.detect_intent(effective_question)

        answer = await mod.do_query(
            rag               = rag,
            question          = effective_question,
            mode              = mode,
            query_type        = query_type,
            config            = cfg,
            logger            = logger,
            enhancer          = enhancer if st.session_state.get("enhancement_on", True) else None,
            chat_history      = chat_history,
            multimodal_content= multimodal_content,
            vlm_enhanced      = vlm_enhanced,
            analyzer          = analyzer,
        )
        return answer, intent

    answer, intent = _run(_run_inner())
    return answer, intent, effective_question


def _get_doc_files():
    files = []
    for ext in cfg.supported_extensions:
        files.extend(cfg.data_dir.rglob(f"*{ext}"))
        files.extend(cfg.data_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(files))


def _get_db_files():
    files = []
    for ext in cfg.db_extensions:
        files.extend(cfg.data_dir.rglob(f"*{ext}"))
        files.extend(cfg.data_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(files))


# ════════════════════════════════════════════════════════
# Session state defaults
# ════════════════════════════════════════════════════════
_defaults = {
    "messages":        [],
    "session_stats":   {"total": 0, "search": 0, "chat": 0, "image": 0},
    "query_mode":      "hybrid",
    "query_type":      "text",
    "enhancement_on":  True,
    "doc_proc_log":    [],    # لاگ پردازش اسناد
    "db_proc_log":     [],    # لاگ پردازش دیتابیس
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ════════════════════════════════════════════════════════
# سایدبار
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div style="font-size:18px;font-weight:700;color:#2563eb;'
        'padding-bottom:12px;border-bottom:1px solid #e2e8f0;margin-bottom:16px">'
        '🧠 RAG-Anything</div>',
        unsafe_allow_html=True,
    )

    # ── تنظیمات جستجو ───────────────────────────────────
    st.markdown('<div class="sidebar-section">🔍 تنظیمات جستجو</div>', unsafe_allow_html=True)

    st.session_state["query_mode"] = st.selectbox(
        "حالت RAG",
        ["hybrid", "local", "global", "naive", "mix"],
        index=["hybrid", "local", "global", "naive", "mix"]
              .index(st.session_state["query_mode"]),
        help="hybrid: ترکیب local+global | local: جزئیات | global: دید کلی",
    )
    st.session_state["query_type"] = st.selectbox(
        "نوع Query",
        ["text", "vlm", "multimodal"],
        index=["text", "vlm", "multimodal"].index(st.session_state["query_type"]),
        help=(
            "text: جستجوی متنی | "
            "vlm: با تحلیل تصویر | "
            "multimodal: JSON با فیلد 'q' و 'content'"
        ),
    )
    if st.session_state["query_type"] == "multimodal":
        st.info(
            '📌 فرمت ورودی:\n'
            '`{"q":"سوال","content":[{"type":"equation","latex":"..."}]}`',
            icon="ℹ️",
        )

    cfg.top_k = st.slider("Top-K بازیابی", 10, 80, cfg.top_k, 5)
    cfg.reranker_score_threshold = st.slider(
        "آستانه Reranker", -5.0, 2.0, float(cfg.reranker_score_threshold), 0.1,
    )

    # ── Query Enhancement ────────────────────────────────
    st.markdown('<div class="sidebar-section">✨ Query Enhancement</div>', unsafe_allow_html=True)
    st.session_state["enhancement_on"] = st.toggle(
        "فعال", value=st.session_state["enhancement_on"],
    )
    if st.session_state["enhancement_on"]:
        cfg.enable_multi_query = st.checkbox("Multi-Query", value=cfg.enable_multi_query)
        if cfg.enable_multi_query:
            cfg.multi_query_count = st.slider("تعداد بازنویسی", 1, 6, cfg.multi_query_count)
        cfg.enable_hyde        = st.checkbox("HyDE",          value=cfg.enable_hyde)
        cfg.enable_reranker    = st.checkbox("Reranker",       value=cfg.enable_reranker)
        cfg.enable_intent_router = st.checkbox("Intent Router", value=cfg.enable_intent_router)

    # ── آمار سیستم ──────────────────────────────────────
    st.markdown('<div class="sidebar-section">📊 آمار سیستم</div>', unsafe_allow_html=True)
    all_docs = _get_doc_files()
    n_proc   = file_cache.count()
    total_mb = sum(f.stat().st_size for f in all_docs) / (1024**2) if all_docs else 0
    c1, c2 = st.columns(2)
    with c1:
        st.metric("اسناد", len(all_docs))
        st.metric("پردازش‌شده", n_proc)
    with c2:
        st.metric("حجم (MB)", f"{total_mb:.0f}")
        img_n = img_engine.total_indexed if img_engine else 0
        st.metric("📸 Image Index", img_n)

    st.markdown('<div class="sidebar-section">📈 این جلسه</div>', unsafe_allow_html=True)
    s = st.session_state["session_stats"]
    c1, c2 = st.columns(2)
    with c1:
        st.metric("کل", s["total"]); st.metric("جستجو", s["search"])
    with c2:
        st.metric("گفتگو", s["chat"]); st.metric("تصویر", s["image"])

    st.markdown("---")
    if st.button("🗑️ پاک‌کردن تاریخچه چت", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["session_stats"] = {"total":0,"search":0,"chat":0,"image":0}
        st.rerun()

    st.markdown('<div class="sidebar-section">📌 راهنما</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:12px;color:#64748b;line-height:1.9">'
        '🔍 <b>جستجو:</b> سوال درباره اسناد<br>'
        '💬 <b>گفتگو:</b> سوال عمومی<br>'
        '🖼️ <b>تصویر:</b> آپلود → پیدا کردن مشابه<br>'
        '📎 <b>multimodal:</b> JSON با content<br>'
        '📤 <b>سند:</b> آپلود و پردازش در تب مدیریت<br>'
        '🗄️ <b>دیتابیس:</b> SQLite/SQL در تب دیتابیس'
        '</div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════
# هدر اصلی + تب‌ها
# ════════════════════════════════════════════════════════
st.markdown(
    '<div style="text-align:right;padding:6px 0 14px 0">'
    '<div class="main-title">🧠 RAG<span class="accent">-Anything</span></div>'
    '<div class="main-sub">سیستم جستجوی هوشمند — متن · تصویر · جدول · معادله · دیتابیس</div>'
    '</div>',
    unsafe_allow_html=True,
)

tab_chat, tab_img, tab_docs, tab_db = st.tabs([
    "💬 جستجوی متنی",
    "🖼️ جستجوی تصویری",
    "📤 مدیریت اسناد",
    "🗄️ مدیریت دیتابیس",
])


# ════════════════════════════════════════════════════════
# ابزارهای UI مشترک
# ════════════════════════════════════════════════════════

def _intent_badge(intent: str) -> str:
    m = {
        "search":     ("🔍", "جستجو",     "search"),
        "chat":       ("💬", "گفتگو",     "chat"),
        "vlm":        ("🎨", "تصویر",     "vlm"),
        "multimodal": ("🌐", "چندوجهی",   "mm"),
    }
    icon, label, cls = m.get(intent, ("❓", intent, "search"))
    return f'<span class="badge badge-{cls}">{icon} {label}</span>'


def _mode_badge(mode: str, qtype: str) -> str:
    return (
        f'<span style="font-size:11px;color:#94a3b8">[{mode}/{qtype}]</span>'
    )


PIPELINE_STEPS = ["بازنویسی", "Embedding", "بازیابی", "Reranking", "تولید پاسخ"]

def _pipeline_html(done: bool) -> str:
    cls  = "pipe-step done" if done else "pipe-step"
    icon = "✓"             if done else "⏳"
    return (
        '<div style="margin-bottom:10px">'
        + "".join(f'<span class="{cls}">{icon} {s}</span>' for s in PIPELINE_STEPS)
        + "</div>"
    )


def _render_sources(sources: list):
    """نمایش بصری منابع در expander."""
    if not sources:
        return
    _icons = {
        ".pdf":"📄",".docx":"📝",".doc":"📝",".pptx":"📊",".ppt":"📊",
        ".xlsx":"📈",".xls":"📈",".csv":"📈",".txt":"📃",".md":"📃",
        ".jpg":"🖼️",".jpeg":"🖼️",".png":"🖼️",".gif":"🖼️",".webp":"🖼️",
        ".bmp":"🖼️",".tiff":"🖼️",".sqlite":"🗄️",".db":"🗄️",".sql":"🗄️",
    }
    with st.expander(f"📌 منابع ({len(sources)} مورد)", expanded=False):
        for i, src in enumerate(sources, 1):
            icon = _icons.get(src.get("ext",""), "📎")
            extras = []
            if src.get("page"):  extras.append(f"صفحه {src['page']}")
            if src.get("table"): extras.append(f"جدول: {src['table']}")
            extra_str = "  —  " + " | ".join(extras) if extras else ""
            st.markdown(
                f'<div class="result-card" style="padding:7px 12px;margin:3px 0">' +
                f'<span style="font-size:12px">{i}. {icon} <b>{src["name"]}</b>{extra_str}</span>' +
                f'<div style="font-family:monospace;font-size:10px;color:#94a3b8;direction:ltr;word-break:break-all">{src["path"]}</div>' +
                '</div>',
                unsafe_allow_html=True,
            )


def _render_msg(msg: dict):
    role = msg["role"]
    with st.chat_message(role, avatar="👤" if role == "user" else "🤖"):
        if role == "user":
            st.markdown(msg["content"])
        else:
            parts = []
            if msg.get("intent"):   parts.append(_intent_badge(msg["intent"]))
            if msg.get("mode"):     parts.append(_mode_badge(msg["mode"], msg.get("qtype","")))
            if msg.get("elapsed"):  parts.append(f'<span style="font-size:11px;color:#64748b">⏱ {msg["elapsed"]}</span>')
            if parts:
                st.markdown('<div style="margin-bottom:8px">' + "  ".join(parts) + "</div>",
                            unsafe_allow_html=True)
            st.markdown(msg["content"])
            # نمایش منابع اگر موجود باشند
            if msg.get("sources"):
                _render_sources(msg["sources"])


# ════════════════════════════════════════════════════════
# تب ۱ — جستجوی متنی
# ════════════════════════════════════════════════════════
with tab_chat:

    for msg in st.session_state["messages"]:
        _render_msg(msg)

    if not st.session_state["messages"]:
        st.markdown("---")
        examples = [
            ("🔍", "جستجو در اسناد", [
                "معادله مشتق‌گیری در فصل سه چیست؟",
                "جدول مقایسه مدل‌ها را نشان بده",
                "تصاویر مربوط به معماری شبکه کجاست؟",
            ]),
            ("📊", "چندوجهی", [
                "فرمول‌های صفحه اول را توضیح بده",
                "داده‌های جدول فروش را تحلیل کن",
                "محتوای دیتابیس محصولات را بگو",
            ]),
            ("💬", "گفتگو", [
                "سلام، چه اسنادی داری؟",
                "آمار کلی سیستم چیست؟",
                "چه فرمت‌هایی پشتیبانی می‌شود؟",
            ]),
        ]
        for col, (icon, title, items) in zip(st.columns(3), examples):
            with col:
                st.markdown(
                    f'<div class="result-card">'
                    f'<div style="font-size:18px;margin-bottom:6px">{icon}</div>'
                    f'<div style="font-weight:600;margin-bottom:8px">{title}</div>'
                    + "".join(
                        f'<div style="font-size:12px;color:#64748b;padding:3px 0;'
                        f'border-bottom:1px solid #f1f5f9">"{i}"</div>'
                        for i in items
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )

    if user_input := st.chat_input("سوال یا جستجوی خود را بنویسید..."):
        mode  = st.session_state["query_mode"]
        qtype = st.session_state["query_type"]
        enh   = st.session_state["enhancement_on"]

        # ④ تاریخچه را قبل از append پیام فعلی می‌سازیم
        chat_history_for_rag = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state["messages"]
            if m["role"] in ("user", "assistant")
        ]

        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        st.session_state["messages"].append({"role": "user", "content": user_input})

        with st.chat_message("assistant", avatar="🤖"):
            pipe_ph = st.empty()
            if enh:
                pipe_ph.markdown(_pipeline_html(done=False), unsafe_allow_html=True)

            wait_ph = st.empty()
            wait_ph.markdown(
                '<div style="color:#94a3b8;font-size:13px">⏳ در حال پردازش...</div>',
                unsafe_allow_html=True,
            )

            t0 = time.time()
            try:
                answer, intent, _ = _query_rag(user_input, mode, qtype, chat_history_for_rag)
            except Exception as e:
                answer, intent = f"❌ خطا: {e}", "search"
            elapsed = f"{time.time()-t0:.1f}s"

            if enh:
                pipe_ph.markdown(_pipeline_html(done=True), unsafe_allow_html=True)
            wait_ph.empty()

            badge_row = _intent_badge(intent) + "  " + _mode_badge(mode, qtype)
            if enh:
                badge_row += '  <span style="font-size:11px;color:#64748b">✨ Enhancement</span>'
            badge_row += f'  <span style="font-size:11px;color:#64748b">⏱ {elapsed}</span>'
            st.markdown(f'<div style="margin-bottom:8px">{badge_row}</div>',
                        unsafe_allow_html=True)
            st.markdown(answer)

        # استخراج منابع برای نمایش
        _sources = []
        try:
            _sources = analyzer.extract_sources(answer)
        except Exception:
            pass

        # نمایش منابع زیر پاسخ (در همان chat message)
        if _sources:
            _render_sources(_sources)

        st.session_state["messages"].append({
            "role": "assistant", "content": answer,
            "intent": intent, "elapsed": elapsed, "mode": mode, "qtype": qtype,
            "sources": _sources,
        })
        st.session_state["session_stats"]["total"] += 1
        st.session_state["session_stats"]["chat" if intent == "chat" else "search"] += 1


# ════════════════════════════════════════════════════════
# تب ۲ — جستجوی تصویری
# ════════════════════════════════════════════════════════
with tab_img:
    import os as _os

    if not img_engine or img_engine.total_indexed == 0 and not (
        mod._IMG_SEARCH_AVAILABLE and cfg.enable_image_search
    ):
        st.warning(
            "⚠️ جستجوی تصویر فعال نیست.\n\n"
            "نصب: `pip install faiss-cpu torch transformers`\n\n"
            "سپس `enable_image_search = True` در Config."
        )
    else:
        st.markdown(
            '<div style="font-weight:600;color:#1d4ed8;font-size:15px;margin-bottom:12px">'
            '🖼️ جستجوی تصویر مشابه با DINOv2</div>',
            unsafe_allow_html=True,
        )

        col_up, col_set = st.columns([3, 1])
        with col_set:
            lens_k    = st.slider("تعداد نتایج",   2, 20, cfg.image_search_top_k, 1)
            min_score = st.slider("حداقل شباهت", 0.0, 1.0, float(cfg.image_search_min_score), 0.05)
            if st.button("🔄 بازسازی Index"):
                with st.spinner("در حال index‌سازی..."):
                    try:
                        parser_out = cfg.parser_output_dir if cfg.index_document_images else None
                        st_r = img_engine.build_index(cfg.data_dir, parser_out, force_rebuild=True)
                        st.success(f"✅ {st_r['indexed']} تصویر  ({st_r['failed']} خطا)")
                    except Exception as e:
                        st.error(f"❌ {e}")

        with col_up:
            uploaded = st.file_uploader(
                "تصویر خود را آپلود کنید",
                type=["jpg","jpeg","png","webp","bmp","tiff"],
                key="lens_up",
            )
            if uploaded:
                st.image(uploaded, caption="تصویر شما", width=280)

        if uploaded and img_engine:
            if st.button("🔍 جستجوی مشابه", type="primary", use_container_width=True):
                with st.spinner("⏳ تحلیل با DINOv2..."):
                    try:
                        t0      = time.time()
                        results = img_engine.search_by_bytes(uploaded.getvalue(), top_k=lens_k)
                        elapsed = time.time() - t0
                        results = [r for r in results if r.get("score",0) >= min_score]
                        st.session_state["session_stats"]["image"] += 1
                        st.session_state["session_stats"]["total"] += 1
                    except Exception as e:
                        st.error(f"❌ {e}"); results = []; elapsed = 0.0

                if not results:
                    st.warning("⚠️ نتیجه‌ای پیدا نشد.")
                else:
                    st.markdown(
                        f'<div style="font-weight:600;color:#166534;margin:10px 0">'
                        f'✅ {len(results)} تصویر مشابه  ⏱ {elapsed*1000:.0f}ms</div>',
                        unsafe_allow_html=True,
                    )
                    for row_i in range(0, len(results), 2):
                        cols = st.columns(2)
                        for ci, col in enumerate(cols):
                            idx = row_i + ci
                            if idx >= len(results): break
                            r = results[idx]
                            with col:
                                path = r.get("path","")
                                if path and _os.path.exists(path):
                                    st.image(path, use_container_width=True)
                                else:
                                    st.info("🖼️ پیش‌نمایش موجود نیست")
                                src = "📄" if r.get("source") == "document_embed" else "🖼️"
                                doc = f'<br><b>سند:</b> {r["doc_source"]}' if r.get("doc_source") else ""
                                st.markdown(
                                    f'<div class="result-card"><div style="font-size:12px;color:#64748b;line-height:1.8">'
                                    f'{src} <b>شباهت:</b> {r.get("score",0):.1%}<br>'
                                    f'<b>نام:</b> {r.get("name","—")}{doc}<br>'
                                    f'<span style="font-family:monospace;font-size:10px;direction:ltr;'
                                    f'display:block;word-break:break-all;color:#94a3b8">{path}</span>'
                                    f'</div></div>',
                                    unsafe_allow_html=True,
                                )

        if img_engine:
            with st.expander("📊 آمار image index"):
                s = img_engine.stats()
                c1,c2,c3 = st.columns(3)
                c1.metric("کل", s["total"])
                c2.metric("فایل مستقل", s["standalone"])
                c3.metric("داخل سند", s["document_embed"])
                st.caption(f"مدل: {s['model']} | بعد: {s['dim']}")
                for ext, cnt in sorted(s.get("by_extension",{}).items(), key=lambda x:-x[1]):
                    st.markdown(f"- `{ext}`: {cnt}")


# ════════════════════════════════════════════════════════
# ⑤ تب ۳ — مدیریت اسناد (آپلود + پردازش)
# ════════════════════════════════════════════════════════
with tab_docs:
    import shutil

    st.markdown(
        '<div style="font-weight:600;color:#1d4ed8;font-size:15px;margin-bottom:12px">'
        '📤 آپلود و پردازش اسناد</div>',
        unsafe_allow_html=True,
    )

    # ── آپلود سند ───────────────────────────────────────
    with st.expander("📎 آپلود سند جدید", expanded=True):
        accept_types = [e.lstrip(".") for e in cfg.supported_extensions
                        if e not in {".sqlite",".db",".sqlite3",".db3",".sql"}]
        uploaded_docs = st.file_uploader(
            "فایل‌های خود را انتخاب کنید",
            type=accept_types,
            accept_multiple_files=True,
            key="doc_upload",
        )
        if uploaded_docs:
            if st.button("💾 ذخیره در پوشه data", type="primary"):
                saved = []
                for uf in uploaded_docs:
                    dest = cfg.data_dir / uf.name
                    dest.write_bytes(uf.getvalue())
                    saved.append(uf.name)
                st.success(f"✅ {len(saved)} فایل ذخیره شد: {', '.join(saved)}")
                st.rerun()

    # ── لیست فایل‌های موجود ─────────────────────────────
    st.markdown("---")
    all_doc_files = _get_doc_files()
    st.markdown(
        f'<div style="font-weight:600;margin-bottom:10px">'
        f'📄 فایل‌های موجود ({len(all_doc_files)})</div>',
        unsafe_allow_html=True,
    )

    if not all_doc_files:
        st.info("هنوز فایلی در پوشه data وجود ندارد.")
    else:
        new_files  = [f for f in all_doc_files if not file_cache.is_processed(f)]
        done_files = [f for f in all_doc_files if file_cache.is_processed(f)]

        for f in all_doc_files[:40]:
            is_done = file_cache.is_processed(f)
            icon    = "✅" if is_done else "🆕"
            size_kb = f.stat().st_size / 1024
            st.markdown(
                f'<div class="result-card" style="padding:8px 12px">'
                f'<span style="font-size:12px">{icon} <b>{f.name}</b>'
                f'  <span style="color:#94a3b8">({size_kb:.0f} KB)</span>'
                f'  <span style="color:#94a3b8;font-size:11px">{f.suffix.upper()}</span>'
                f'</span></div>',
                unsafe_allow_html=True,
            )
        if len(all_doc_files) > 40:
            st.caption(f"... و {len(all_doc_files)-40} فایل دیگر")

    # ── پردازش اسناد ────────────────────────────────────
    st.markdown("---")
    col_proc1, col_proc2 = st.columns([2,1])
    with col_proc1:
        st.markdown(
            f'<div style="font-weight:600;margin-bottom:6px">⚙️ پردازش اسناد</div>'
            f'<div style="font-size:13px;color:#64748b">'
            f'{len(new_files) if all_doc_files else 0} فایل جدید آماده پردازش است.</div>',
            unsafe_allow_html=True,
        )
    with col_proc2:
        force_reprocess = st.checkbox("پردازش مجدد همه", value=False)

    proc_btn = st.button(
        "▶️ شروع پردازش",
        type="primary",
        disabled=not all_doc_files,
        use_container_width=True,
    )

    if proc_btn and all_doc_files:
        log_ph  = st.empty()
        prog_ph = st.progress(0)
        logs    = []

        total   = len(all_doc_files)
        success = 0
        failed  = []

        mineru_kw = mod._mineru_kwargs(cfg)

        for i, fp in enumerate(all_doc_files, 1):
            if not force_reprocess and file_cache.is_processed(fp):
                logs.append(f"⏭️ رد شد (کش): {fp.name}")
                prog_ph.progress(i / total)
                log_ph.markdown(
                    '<div style="font-family:monospace;font-size:12px;'
                    'background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;'
                    'padding:10px;max-height:200px;overflow-y:auto">'
                    + "<br>".join(logs[-12:]) + "</div>",
                    unsafe_allow_html=True,
                )
                continue

            try:
                _run(rag.process_document_complete(
                    file_path     = str(fp),
                    output_dir    = str(cfg.output_dir),
                    parse_method  = cfg.parse_method,
                    display_stats = False,
                    **mineru_kw,
                ))
                file_cache.mark_processed(fp)
                success += 1
                logs.append(f"✅ {fp.name}")
            except Exception as e:
                failed.append(fp.name)
                logs.append(f"❌ {fp.name}: {str(e)[:60]}")

            prog_ph.progress(i / total)
            log_ph.markdown(
                '<div style="font-family:monospace;font-size:12px;'
                'background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;'
                'padding:10px;max-height:200px;overflow-y:auto">'
                + "<br>".join(logs[-12:]) + "</div>",
                unsafe_allow_html=True,
            )

        prog_ph.progress(1.0)
        st.session_state["doc_proc_log"] = logs

        c1,c2,c3 = st.columns(3)
        c1.metric("✅ موفق",  success)
        c2.metric("❌ ناموفق", len(failed))
        c3.metric("کل", total)

        if failed:
            with st.expander("فایل‌های ناموفق"):
                for f in failed:
                    st.markdown(f"- `{f}`")

        # ⑦ پیشنهاد به‌روزرسانی image index بعد از پردازش سند
        if success > 0 and img_engine is not None:
            st.info(
                "📸 اسناد جدید ممکن است حاوی تصویر باشند.\n"
                "آیا image index به‌روزرسانی شود؟"
            )
            if st.button("🔄 به‌روزرسانی Image Index", key="post_proc_img"):
                with st.spinner("در حال index‌سازی تصاویر..."):
                    try:
                        parser_out = cfg.parser_output_dir if cfg.index_document_images else None
                        ir = img_engine.load_or_build(cfg.data_dir, parser_out)
                        st.success(f"✅ Image index به‌روز شد — {img_engine.total_indexed} تصویر")
                    except Exception as e:
                        st.error(f"❌ {e}")


# ════════════════════════════════════════════════════════
# ⑥ تب ۴ — مدیریت دیتابیس
# ════════════════════════════════════════════════════════
with tab_db:

    st.markdown(
        '<div style="font-weight:600;color:#1d4ed8;font-size:15px;margin-bottom:12px">'
        '🗄️ مدیریت دیتابیس</div>',
        unsafe_allow_html=True,
    )

    if not mod._DB_AVAILABLE:
        st.warning(
            "⚠️ db_processor.py پیدا نشد.\n\n"
            "فایل db_processor.py را در همان پوشه app.py قرار دهید."
        )
    else:
        # ── لیست فایل‌های دیتابیسی ──────────────────────
        db_files = _get_db_files()
        st.markdown(
            f'<div style="font-weight:600;margin-bottom:10px">'
            f'📋 فایل‌های دیتابیسی ({len(db_files)})</div>',
            unsafe_allow_html=True,
        )

        if not db_files:
            st.info(f"فایل SQLite یا SQL در پوشه `{cfg.data_dir}` پیدا نشد.")
        else:
            for f in db_files:
                is_done = file_cache.is_processed(f)
                icon    = "✅" if is_done else "🆕"
                size_kb = f.stat().st_size / 1024
                st.markdown(
                    f'<div class="result-card" style="padding:8px 12px">'
                    f'<span style="font-size:12px">{icon} <b>{f.name}</b>'
                    f'  <span style="color:#94a3b8">({size_kb:.0f} KB)</span>'
                    f'  <span style="color:#94a3b8;font-size:11px">{f.suffix.upper()}</span>'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )

        # ── وضعیت اتصال‌های زنده ─────────────────────────
        st.markdown("---")
        st.markdown(
            '<div style="font-weight:600;margin-bottom:8px">🔌 اتصال‌های زنده</div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            mysql_status = "✅ تنظیم شده" if cfg.mysql_conn else "❌ تنظیم نشده"
            st.markdown(
                f'<div class="result-card" style="padding:8px 12px">'
                f'<span style="font-size:12px">🐬 MySQL: {mysql_status}</span></div>',
                unsafe_allow_html=True,
            )
        with c2:
            pg_status = "✅ تنظیم شده" if cfg.postgresql_conn else "❌ تنظیم نشده"
            st.markdown(
                f'<div class="result-card" style="padding:8px 12px">'
                f'<span style="font-size:12px">🐘 PostgreSQL: {pg_status}</span></div>',
                unsafe_allow_html=True,
            )

        # ── پردازش دیتابیس ──────────────────────────────
        st.markdown("---")
        has_pending = any(not file_cache.is_processed(f) for f in db_files)
        has_live    = bool(cfg.mysql_conn or cfg.postgresql_conn)

        force_db = st.checkbox("پردازش مجدد دیتابیس‌های قبلی", value=False)
        db_btn   = st.button(
            "▶️ پردازش دیتابیس‌ها",
            type    = "primary",
            disabled= not (db_files or has_live),
        )

        if db_btn:
            log_ph = st.empty()
            logs   = []

            async def _process_db_ui():
                return await mod.process_databases(
                    rag, cfg, logger, file_cache, force_reprocess=force_db
                )

            with st.spinner("در حال پردازش دیتابیس‌ها..."):
                try:
                    t0  = time.time()
                    res = _run(_process_db_ui())
                    elapsed = time.time() - t0
                except Exception as e:
                    st.error(f"❌ خطا: {e}")
                    res = {}

            if res and "error" not in res:
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("✅ فایل‌ها",      res.get("files_processed", 0))
                c2.metric("⏭️ رد شده",       res.get("files_skipped", 0))
                c3.metric("📋 جداول",        res.get("total_tables", 0))
                c4.metric("📦 آیتم‌های RAG",  res.get("total_items", 0))
                st.success(f"✅ پردازش تمام شد در {elapsed:.1f}s")

                # جزئیات هر دیتابیس
                for detail in res.get("details", []):
                    with st.expander(f"📊 {detail.get('file','?')} ({detail.get('type','?')})"):
                        st.markdown(f"**کل ردیف:** {detail.get('total_rows',0)}")
                        st.markdown(f"**کل آیتم RAG:** {detail.get('total_items',0)}")
                        if detail.get("tables"):
                            for t in detail["tables"]:
                                st.markdown(
                                    f"- `{t['name']}`: "
                                    f"{t['columns']} ستون، {t['rows_processed']} ردیف، "
                                    f"{t['items_generated']} آیتم"
                                )
            elif res:
                st.error(f"❌ {res.get('error')}")