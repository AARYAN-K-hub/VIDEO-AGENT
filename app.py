"""
AI Video Assistant — Production-Ready Streamlit App
"""

import time
import re
import io
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Backend imports ─────────────────────────────────────────────────────────
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
def inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ── Google Font ── */
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

        /* ── Root palette ── */
        :root {
            --bg-base:        #080c14;
            --bg-surface:     #0d1421;
            --bg-card:        #111827;
            --bg-card-hover:  #151f30;
            --border:         rgba(99,179,237,0.12);
            --border-glow:    rgba(99,179,237,0.35);
            --accent-blue:    #4fa3e3;
            --accent-cyan:    #22d3ee;
            --accent-purple:  #a78bfa;
            --accent-green:   #34d399;
            --accent-orange:  #fb923c;
            --text-primary:   #e8f0fe;
            --text-secondary: #8ba3c7;
            --text-muted:     #4a6080;
            --radius-sm:      8px;
            --radius-md:      14px;
            --radius-lg:      22px;
            --shadow-card:    0 4px 32px rgba(0,0,0,0.45), 0 1px 0 rgba(255,255,255,0.04);
            --transition:     all 0.22s cubic-bezier(0.4,0,0.2,1);
        }

        /* ── Base reset ── */
        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
            background-color: var(--bg-base) !important;
            color: var(--text-primary) !important;
        }

        /* ── Animated gradient mesh background ── */
        .stApp {
            background:
                radial-gradient(ellipse 80% 50% at 10% 0%, rgba(79,163,227,0.06) 0%, transparent 60%),
                radial-gradient(ellipse 60% 40% at 90% 80%, rgba(167,139,250,0.05) 0%, transparent 55%),
                radial-gradient(ellipse 50% 60% at 50% 50%, rgba(34,211,238,0.02) 0%, transparent 70%),
                var(--bg-base) !important;
        }

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1120 0%, #080c14 100%) !important;
            border-right: 1px solid var(--border) !important;
        }
        section[data-testid="stSidebar"] * {
            color: var(--text-primary) !important;
        }
        section[data-testid="stSidebar"] .stMarkdown p {
            color: var(--text-secondary) !important;
        }

        /* ── Remove default Streamlit chrome ── */
        #MainMenu, footer, header { visibility: hidden; }
        .block-container {
            padding: 1.5rem 2.5rem 3rem !important;
            max-width: 1380px !important;
        }

        /* ── Typography ── */
        h1, h2, h3, h4 {
            font-family: 'Syne', sans-serif !important;
            letter-spacing: -0.02em;
        }

        /* ── Hero header ── */
        .hero-wrap {
            display: flex;
            align-items: center;
            gap: 20px;
            margin-bottom: 2.5rem;
            padding: 2rem 2.5rem;
            background: linear-gradient(135deg,
                rgba(13,20,33,0.9) 0%,
                rgba(11,17,32,0.7) 100%);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            position: relative;
            overflow: hidden;
        }
        .hero-wrap::before {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(90deg,
                rgba(79,163,227,0.07) 0%,
                rgba(167,139,250,0.04) 50%,
                transparent 100%);
            pointer-events: none;
        }
        .hero-wrap::after {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 1px;
            background: linear-gradient(90deg,
                transparent, rgba(79,163,227,0.5), rgba(167,139,250,0.4), transparent);
        }
        .hero-icon {
            font-size: 3.2rem;
            line-height: 1;
            filter: drop-shadow(0 0 18px rgba(79,163,227,0.5));
        }
        .hero-title {
            font-family: 'Syne', sans-serif !important;
            font-size: 2rem !important;
            font-weight: 800 !important;
            background: linear-gradient(135deg, #e8f0fe 30%, var(--accent-cyan) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0 !important;
            padding: 0 !important;
        }
        .hero-sub {
            color: var(--text-secondary) !important;
            font-size: 0.95rem;
            margin-top: 4px;
        }

        /* ── Glass card ── */
        .glass-card {
            background: linear-gradient(135deg,
                rgba(17,24,39,0.95) 0%,
                rgba(13,20,33,0.85) 100%);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 1.6rem 1.8rem;
            margin-bottom: 1.2rem;
            box-shadow: var(--shadow-card);
            position: relative;
            overflow: hidden;
            transition: var(--transition);
        }
        .glass-card:hover {
            border-color: var(--border-glow);
            box-shadow: 0 8px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(79,163,227,0.1);
        }
        .glass-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 1px;
            background: linear-gradient(90deg,
                transparent, rgba(255,255,255,0.08), transparent);
        }

        /* ── Card section header ── */
        .section-label {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 1rem;
        }
        .section-label .icon {
            font-size: 1.2rem;
        }
        .section-label .title {
            font-family: 'Syne', sans-serif;
            font-weight: 700;
            font-size: 0.9rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--text-secondary);
        }
        .section-label .line {
            flex: 1;
            height: 1px;
            background: var(--border);
        }

        /* ── Generated title display ── */
        .result-title {
            font-family: 'Syne', sans-serif;
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-primary);
            line-height: 1.35;
            margin: 0;
        }

        /* ── Bullet list items ── */
        .bullet-item {
            display: flex;
            gap: 12px;
            align-items: flex-start;
            padding: 0.65rem 0.8rem;
            border-radius: var(--radius-sm);
            margin-bottom: 0.45rem;
            background: rgba(255,255,255,0.025);
            border: 1px solid transparent;
            transition: var(--transition);
        }
        .bullet-item:hover {
            background: rgba(79,163,227,0.06);
            border-color: rgba(79,163,227,0.12);
        }
        .bullet-dot {
            width: 7px; height: 7px;
            border-radius: 50%;
            margin-top: 7px;
            flex-shrink: 0;
        }
        .dot-blue   { background: var(--accent-blue);   box-shadow: 0 0 6px var(--accent-blue); }
        .dot-green  { background: var(--accent-green);  box-shadow: 0 0 6px var(--accent-green); }
        .dot-orange { background: var(--accent-orange); box-shadow: 0 0 6px var(--accent-orange); }
        .dot-purple { background: var(--accent-purple); box-shadow: 0 0 6px var(--accent-purple); }
        .bullet-text {
            color: var(--text-primary);
            font-size: 0.92rem;
            line-height: 1.55;
        }

        /* ── Metric pill ── */
        .metric-row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 0.5rem;
        }
        .metric-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 13px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 500;
            border: 1px solid;
        }
        .pill-blue   { background: rgba(79,163,227,0.1);  border-color: rgba(79,163,227,0.3);  color: var(--accent-blue);   }
        .pill-green  { background: rgba(52,211,153,0.1);  border-color: rgba(52,211,153,0.3);  color: var(--accent-green);  }
        .pill-purple { background: rgba(167,139,250,0.1); border-color: rgba(167,139,250,0.3); color: var(--accent-purple); }
        .pill-cyan   { background: rgba(34,211,238,0.1);  border-color: rgba(34,211,238,0.3);  color: var(--accent-cyan);   }
        .pill-orange { background: rgba(251,146,60,0.1);  border-color: rgba(251,146,60,0.3);  color: var(--accent-orange); }

        /* ── Processing timeline ── */
        .timeline-wrap { display: flex; flex-direction: column; gap: 6px; }
        .timeline-step {
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 10px 14px;
            border-radius: var(--radius-sm);
            font-size: 0.88rem;
            transition: var(--transition);
        }
        .timeline-step.done  { background: rgba(52,211,153,0.07);  color: var(--accent-green);  }
        .timeline-step.active{ background: rgba(79,163,227,0.1);   color: var(--accent-blue);   animation: pulse 1.2s ease-in-out infinite; }
        .timeline-step.wait  { background: rgba(255,255,255,0.03); color: var(--text-muted);    }
        @keyframes pulse {
            0%,100% { opacity:1; }
            50%      { opacity:0.55; }
        }
        .step-icon { font-size: 1rem; flex-shrink: 0; }

        /* ── Chat bubbles ── */
        .chat-wrap { display: flex; flex-direction: column; gap: 14px; }
        .chat-bubble {
            max-width: 88%;
            padding: 12px 16px;
            border-radius: var(--radius-md);
            font-size: 0.92rem;
            line-height: 1.6;
            position: relative;
        }
        .chat-user {
            align-self: flex-end;
            background: linear-gradient(135deg,
                rgba(79,163,227,0.18) 0%, rgba(79,163,227,0.08) 100%);
            border: 1px solid rgba(79,163,227,0.25);
            color: var(--text-primary);
            border-bottom-right-radius: 4px;
        }
        .chat-assistant {
            align-self: flex-start;
            background: rgba(17,24,39,0.95);
            border: 1px solid var(--border);
            color: var(--text-primary);
            border-bottom-left-radius: 4px;
        }
        .chat-meta {
            font-size: 0.72rem;
            color: var(--text-muted);
            margin-top: 5px;
        }
        .chat-container {
            max-height: 480px;
            overflow-y: auto;
            padding: 1rem;
            background: rgba(8,12,20,0.6);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            scroll-behavior: smooth;
        }
        .chat-container::-webkit-scrollbar      { width: 5px; }
        .chat-container::-webkit-scrollbar-track { background: transparent; }
        .chat-container::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }

        /* ── Streamlit widget overrides ── */
        .stTextInput > div > div > input,
        .stSelectbox > div > div,
        .stTextArea textarea {
            background: rgba(13,20,33,0.9) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-sm) !important;
            color: var(--text-primary) !important;
            transition: var(--transition) !important;
        }
        .stTextInput > div > div > input:focus,
        .stTextArea textarea:focus {
            border-color: var(--accent-blue) !important;
            box-shadow: 0 0 0 3px rgba(79,163,227,0.12) !important;
        }
        .stButton > button {
            background: linear-gradient(135deg, var(--accent-blue) 0%, #3b82f6 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: var(--radius-sm) !important;
            font-family: 'Syne', sans-serif !important;
            font-weight: 600 !important;
            letter-spacing: 0.03em !important;
            padding: 0.55rem 1.4rem !important;
            transition: var(--transition) !important;
            box-shadow: 0 4px 20px rgba(79,163,227,0.25) !important;
        }
        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 28px rgba(79,163,227,0.4) !important;
        }
        .stButton > button:active { transform: translateY(0) !important; }

        /* ── Download button ── */
        .stDownloadButton > button {
            background: rgba(52,211,153,0.1) !important;
            color: var(--accent-green) !important;
            border: 1px solid rgba(52,211,153,0.3) !important;
            border-radius: var(--radius-sm) !important;
            font-weight: 500 !important;
            transition: var(--transition) !important;
        }
        .stDownloadButton > button:hover {
            background: rgba(52,211,153,0.18) !important;
            box-shadow: 0 4px 16px rgba(52,211,153,0.15) !important;
        }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(13,20,33,0.6) !important;
            border-radius: var(--radius-md) !important;
            padding: 4px !important;
            gap: 2px !important;
            border: 1px solid var(--border) !important;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            color: var(--text-muted) !important;
            border-radius: 10px !important;
            font-family: 'DM Sans', sans-serif !important;
            font-weight: 500 !important;
            font-size: 0.88rem !important;
            padding: 7px 18px !important;
            transition: var(--transition) !important;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(79,163,227,0.15) !important;
            color: var(--accent-blue) !important;
        }
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 1.2rem !important;
        }

        /* ── Expander ── */
        .stExpander {
            background: rgba(13,20,33,0.7) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
        }
        .stExpander > div > div > div > div > p {
            color: var(--text-secondary) !important;
        }

        /* ── Success / error / info ── */
        .stSuccess  { background: rgba(52,211,153,0.08) !important; border-left: 3px solid var(--accent-green) !important;  border-radius: var(--radius-sm) !important; }
        .stError    { background: rgba(239,68,68,0.08)  !important; border-left: 3px solid #ef4444 !important;               border-radius: var(--radius-sm) !important; }
        .stInfo     { background: rgba(79,163,227,0.08) !important; border-left: 3px solid var(--accent-blue) !important;   border-radius: var(--radius-sm) !important; }
        .stWarning  { background: rgba(251,146,60,0.08) !important; border-left: 3px solid var(--accent-orange) !important; border-radius: var(--radius-sm) !important; }

        /* ── File uploader ── */
        [data-testid="stFileUploadDropzone"] {
            background: rgba(13,20,33,0.7) !important;
            border: 2px dashed rgba(79,163,227,0.25) !important;
            border-radius: var(--radius-md) !important;
            transition: var(--transition) !important;
        }
        [data-testid="stFileUploadDropzone"]:hover {
            border-color: rgba(79,163,227,0.5) !important;
            background: rgba(79,163,227,0.04) !important;
        }

        /* ── Divider ── */
        hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

        /* ── Empty state ── */
        .empty-state {
            text-align: center;
            padding: 4rem 2rem;
            color: var(--text-muted);
        }
        .empty-state .es-icon { font-size: 3.5rem; margin-bottom: 1rem; opacity: 0.5; }
        .empty-state .es-title {
            font-family: 'Syne', sans-serif;
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 0.4rem;
        }
        .empty-state .es-sub { font-size: 0.88rem; line-height: 1.6; }

        /* ── Sidebar nav item ── */
        .nav-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 9px 12px;
            border-radius: var(--radius-sm);
            font-size: 0.88rem;
            font-weight: 500;
            color: var(--text-secondary);
            cursor: default;
            transition: var(--transition);
            margin-bottom: 3px;
        }
        .nav-item:hover { background: rgba(79,163,227,0.07); color: var(--text-primary); }
        .nav-item.active { background: rgba(79,163,227,0.12); color: var(--accent-blue); }

        /* ── Status badge ── */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .status-ready   { background: rgba(52,211,153,0.12); color: var(--accent-green);  border: 1px solid rgba(52,211,153,0.3); }
        .status-idle    { background: rgba(74,96,128,0.15);  color: var(--text-muted);    border: 1px solid rgba(74,96,128,0.3); }
        .status-working { background: rgba(79,163,227,0.12); color: var(--accent-blue);   border: 1px solid rgba(79,163,227,0.3); }

        /* ── Scrollbar global ── */
        ::-webkit-scrollbar       { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }

        /* ── Footer ── */
        .app-footer {
            text-align: center;
            padding: 2rem 0 0.5rem;
            color: var(--text-muted);
            font-size: 0.78rem;
            border-top: 1px solid var(--border);
            margin-top: 3rem;
        }
        .app-footer span { color: var(--accent-blue); }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Session state initialisation ─────────────────────────────────────────────
def init_session() -> None:
    defaults = {
        "processed":    False,
        "processing":   False,
        "result":       None,
        "chat_history": [],
        "rag_chain":    None,
        "proc_time":    None,
        "input_source": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar() -> None:
    with st.sidebar:
        # Logo
        st.markdown(
            """
            <div style="padding:1.2rem 0 1rem; text-align:center;">
                <div style="font-size:2.4rem; margin-bottom:6px;">🎥</div>
                <div style="font-family:'Syne',sans-serif; font-size:1.1rem;
                            font-weight:800; background:linear-gradient(135deg,#e8f0fe,#4fa3e3);
                            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                            background-clip:text;">AI Video Assistant</div>
                <div style="font-size:0.72rem; color:#4a6080; margin-top:3px;">v1.0 · Powered by Claude</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Status indicator
        if st.session_state.processing:
            badge = '<span class="status-badge status-working">⚙ Processing…</span>'
        elif st.session_state.processed:
            badge = '<span class="status-badge status-ready">✓ Results Ready</span>'
        else:
            badge = '<span class="status-badge status-idle">◦ Idle</span>'
        st.markdown(f"<div style='margin-bottom:1rem;'>{badge}</div>", unsafe_allow_html=True)

        # Nav
        st.markdown(
            """
            <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em;
                        color:#4a6080; margin-bottom:8px; padding-left:4px;">Navigation</div>
            <div class="nav-item active">🏠&nbsp; Dashboard</div>
            <div class="nav-item">📄&nbsp; Transcripts</div>
            <div class="nav-item">💬&nbsp; Q&amp;A Chat</div>
            <div class="nav-item">📊&nbsp; Insights</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Processing stats (only when result available)
        if st.session_state.processed and st.session_state.result:
            res = st.session_state.result
            transcript_words = len(res["transcript"].split())
            transcript_chars = len(res["transcript"])

            st.markdown(
                """
                <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em;
                            color:#4a6080; margin-bottom:10px; padding-left:4px;">Session Stats</div>
                """,
                unsafe_allow_html=True,
            )

            for label, value, icon in [
                ("Words",    f"{transcript_words:,}", "📝"),
                ("Chars",    f"{transcript_chars:,}", "🔤"),
                ("Duration", f"{st.session_state.proc_time:.1f}s" if st.session_state.proc_time else "—", "⏱"),
                ("Messages", str(len(st.session_state.chat_history)), "💬"),
            ]:
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between; align-items:center;
                                padding:7px 10px; border-radius:8px; margin-bottom:4px;
                                background:rgba(255,255,255,0.03); border:1px solid rgba(99,179,237,0.08);">
                        <span style="color:#8ba3c7; font-size:0.82rem;">{icon} {label}</span>
                        <span style="color:#e8f0fe; font-weight:600; font-size:0.82rem;">{value}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("---")

        # Tips
        with st.expander("💡 Tips & Examples", expanded=False):
            st.markdown(
                """
                **YouTube URLs**
                - Full: `https://youtube.com/watch?v=…`
                - Short: `https://youtu.be/…`

                **File formats**
                - Audio: `.mp3`, `.wav`, `.m4a`, `.ogg`
                - Video: `.mp4`, `.mov`, `.mkv`, `.webm`

                **Language**
                - *English* — pure English content
                - *Hinglish* — Hindi–English mix
                """
            )

        # Reset
        if st.session_state.processed:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.button("🔄 Reset Session", use_container_width=True):
                for key in ["processed", "processing", "result",
                            "chat_history", "rag_chain", "proc_time", "input_source"]:
                    st.session_state[key] = (
                        [] if key == "chat_history"
                        else False if key in ("processed", "processing")
                        else None
                    )
                st.rerun()

        # Footer in sidebar
        st.markdown(
            """
            <div style="position:absolute; bottom:1.2rem; left:0; right:0;
                        text-align:center; font-size:0.72rem; color:#2d4060;">
                Built with ❤ using Streamlit
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Helper renderers ──────────────────────────────────────────────────────────
def section_header(icon: str, title: str) -> None:
    st.markdown(
        f"""
        <div class="section-label">
            <span class="icon">{icon}</span>
            <span class="title">{title}</span>
            <span class="line"></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def bullet_list(items: list, dot_class: str = "dot-blue") -> None:
    if not items:
        st.markdown(
            '<p style="color:var(--text-muted); font-size:0.88rem;">— None identified</p>',
            unsafe_allow_html=True,
        )
        return
    for item in items:
        text = item.strip().lstrip("-•*").strip()
        if text:
            st.markdown(
                f"""
                <div class="bullet-item">
                    <div class="bullet-dot {dot_class}"></div>
                    <div class="bullet-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def parse_list_output(raw: str) -> list:
    """Convert LLM bullet/numbered output to a clean Python list."""
    lines = raw.strip().splitlines()
    items = []
    for line in lines:
        clean = re.sub(r"^[\s\-\*\•\d\.\)]+", "", line).strip()
        if clean:
            items.append(clean)
    return items


def card(content_fn, *args, **kwargs) -> None:
    """Wraps content_fn output in a glass-card div."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    content_fn(*args, **kwargs)
    st.markdown("</div>", unsafe_allow_html=True)


# ── Input panel ───────────────────────────────────────────────────────────────
def render_input_panel() -> tuple:
    """Returns (source_value: str|None, language: str, submitted: bool)."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    section_header("📥", "Input Source")

    tab_yt, tab_file = st.tabs(["  🔗 YouTube URL  ", "  📂 Upload File  "])

    source_value = None
    with tab_yt:
        yt_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=…",
            label_visibility="collapsed",
            key="yt_url_input",
        )
        if yt_url:
            source_value = yt_url.strip()

    with tab_file:
        uploaded = st.file_uploader(
            "Drop audio or video file here",
            type=["mp3", "wav", "m4a", "ogg", "mp4", "mov", "mkv", "webm"],
            label_visibility="collapsed",
            key="file_uploader",
        )
        if uploaded:
            # Save to temp path for backend
            import tempfile, os
            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                source_value = tmp.name

    col_lang, col_btn, col_spacer = st.columns([2, 2, 4])
    with col_lang:
        language = st.selectbox(
            "Language",
            ["english", "hinglish"],
            format_func=lambda x: "🇬🇧 English" if x == "english" else "🇮🇳 Hinglish",
            label_visibility="visible",
        )

    with col_btn:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        submitted = st.button(
            "⚡ Process Video",
            disabled=st.session_state.processing or not source_value,
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
    return source_value, language, submitted


# ── Processing timeline ───────────────────────────────────────────────────────
PIPELINE_STEPS = [
    ("🎙", "Extracting audio chunks"),
    ("📝", "Transcribing audio"),
    ("🏷",  "Generating title"),
    ("📋", "Summarising content"),
    ("✅", "Extracting action items"),
    ("⚖",  "Identifying key decisions"),
    ("❓", "Finding open questions"),
    ("🔗", "Building RAG chain"),
]


def render_timeline(current_step: int) -> None:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    section_header("⚙", "Processing Pipeline")
    st.markdown('<div class="timeline-wrap">', unsafe_allow_html=True)
    for i, (icon, label) in enumerate(PIPELINE_STEPS):
        if i < current_step:
            css = "done"
            prefix = "✓"
        elif i == current_step:
            css = "active"
            prefix = "›"
        else:
            css = "wait"
            prefix = "○"
        st.markdown(
            f'<div class="timeline-step {css}"><span class="step-icon">{icon}</span>'
            f'<span>{prefix}&nbsp;&nbsp;{label}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div></div>", unsafe_allow_html=True)


# ── Run backend pipeline ──────────────────────────────────────────────────────
def run_pipeline_with_progress(source: str, language: str) -> dict | None:
    """Executes pipeline steps one by one, updating timeline UI."""
    timeline_slot = st.empty()

    def update(step: int) -> None:
        with timeline_slot.container():
            render_timeline(step)

    try:
        update(0)
        chunks = process_input(source)

        update(1)
        transcript = transcribe_all(chunks, language)
        if not transcript or not transcript.strip():
            st.error("⚠️ Transcription returned empty. Please check your input source.")
            return None

        update(2)
        title = generate_title(transcript)

        update(3)
        summary = summarize(transcript)

        update(4)
        action_items_raw = extract_action_items(transcript)

        update(5)
        decisions_raw = extract_key_decisions(transcript)

        update(6)
        questions_raw = extract_questions(transcript)

        update(7)
        rag_chain = build_rag_chain(transcript)

        # Final: all done
        update(len(PIPELINE_STEPS))
        time.sleep(0.3)
        timeline_slot.empty()

        return {
            "title":        title,
            "transcript":   transcript,
            "summary":      summary,
            "action_items": parse_list_output(action_items_raw) if isinstance(action_items_raw, str) else action_items_raw,
            "key_decisions": parse_list_output(decisions_raw)   if isinstance(decisions_raw, str)   else decisions_raw,
            "open_questions": parse_list_output(questions_raw)  if isinstance(questions_raw, str)   else questions_raw,
            "rag_chain":    rag_chain,
        }

    except Exception as exc:  # noqa: BLE001
        timeline_slot.empty()
        st.error(f"❌ Pipeline error: {exc}")
        return None


# ── Results UI ────────────────────────────────────────────────────────────────
def render_results(res: dict) -> None:
    # ── Title card ──────────────────────────────────────────────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    section_header("🏷", "Generated Title")
    st.markdown(
        f'<p class="result-title">{res["title"]}</p>',
        unsafe_allow_html=True,
    )

    transcript_words = len(res["transcript"].split())
    n_actions   = len(res["action_items"])
    n_decisions = len(res["key_decisions"])
    n_questions = len(res["open_questions"])
    proc_secs   = st.session_state.proc_time or 0

    st.markdown(
        f"""
        <div class="metric-row" style="margin-top:1rem;">
            <span class="metric-pill pill-blue">📝 {transcript_words:,} words</span>
            <span class="metric-pill pill-green">✅ {n_actions} actions</span>
            <span class="metric-pill pill-purple">⚖ {n_decisions} decisions</span>
            <span class="metric-pill pill-orange">❓ {n_questions} questions</span>
            <span class="metric-pill pill-cyan">⏱ {proc_secs:.1f}s</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Download row
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    dl_col1, dl_col2, _ = st.columns([2, 2, 4])
    with dl_col1:
        summary_text = (
            f"# {res['title']}\n\n"
            f"## Summary\n{res['summary']}\n\n"
            f"## Action Items\n" + "\n".join(f"- {a}" for a in res["action_items"]) + "\n\n"
            f"## Key Decisions\n" + "\n".join(f"- {d}" for d in res["key_decisions"]) + "\n\n"
            f"## Open Questions\n" + "\n".join(f"- {q}" for q in res["open_questions"])
        )
        st.download_button(
            "📥 Download Summary",
            data=summary_text.encode(),
            file_name="summary.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with dl_col2:
        st.download_button(
            "📥 Download Transcript",
            data=res["transcript"].encode(),
            file_name="transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Tabbed analysis ──────────────────────────────────────────────────────
    tab_sum, tab_act, tab_dec, tab_q, tab_tr = st.tabs([
        "  📋 Summary  ",
        "  ✅ Actions  ",
        "  ⚖ Decisions  ",
        "  ❓ Questions  ",
        "  📄 Transcript  ",
    ])

    with tab_sum:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section_header("📋", "Summary")
        st.markdown(
            f'<p style="line-height:1.8; color:var(--text-primary); font-size:0.93rem;">'
            f'{res["summary"]}</p>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_act:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section_header("✅", "Action Items")
        bullet_list(res["action_items"], dot_class="dot-green")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_dec:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section_header("⚖", "Key Decisions")
        bullet_list(res["key_decisions"], dot_class="dot-purple")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_q:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section_header("❓", "Open Questions")
        bullet_list(res["open_questions"], dot_class="dot-orange")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_tr:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section_header("📄", "Full Transcript")
        st.markdown(
            f'<div style="max-height:460px; overflow-y:auto; padding:1rem;'
            f'background:rgba(8,12,20,0.6); border-radius:10px;'
            f'border:1px solid var(--border); font-size:0.88rem; line-height:1.8;'
            f'color:var(--text-secondary); white-space:pre-wrap; font-family:monospace;">'
            f'{res["transcript"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ── RAG Chat ──────────────────────────────────────────────────────────────────
def render_chat(rag_chain) -> None:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    section_header("💬", "Ask Anything About the Video")

    # Render history
    if st.session_state.chat_history:
        st.markdown('<div class="chat-container"><div class="chat-wrap">', unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            role = msg["role"]
            text = msg["content"]
            ts   = msg.get("ts", "")
            css  = "chat-user" if role == "user" else "chat-assistant"
            icon = "👤" if role == "user" else "🤖"
            st.markdown(
                f"""
                <div class="chat-bubble {css}">
                    <div>{icon}&nbsp;&nbsp;{text}</div>
                    <div class="chat-meta">{ts}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div style="text-align:center; padding:2rem 1rem; color:var(--text-muted);">
                <div style="font-size:2rem; margin-bottom:8px;">💬</div>
                <div style="font-size:0.88rem;">Ask a question about the video content.<br>
                Try: <em>"What are the main topics discussed?"</em></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Input row
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    q_col, btn_col = st.columns([8, 2])
    with q_col:
        question = st.text_input(
            "question",
            placeholder="Ask a question about the video…",
            label_visibility="collapsed",
            key="chat_input",
        )
    with btn_col:
        send = st.button("Send ➤", use_container_width=True, key="chat_send")

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑 Clear Conversation", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    if (send or question) and question and question.strip():
        with st.spinner("Thinking…"):
            try:
                answer = ask_question(rag_chain, question.strip())
            except Exception as exc:  # noqa: BLE001
                answer = f"⚠️ Error retrieving answer: {exc}"

        ts_now = datetime.now().strftime("%H:%M")
        st.session_state.chat_history.append({"role": "user",      "content": question.strip(), "ts": ts_now})
        st.session_state.chat_history.append({"role": "assistant", "content": answer,           "ts": ts_now})
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ── Empty state ───────────────────────────────────────────────────────────────
def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state glass-card">
            <div class="es-icon">🎬</div>
            <div class="es-title">No video processed yet</div>
            <div class="es-sub">
                Paste a YouTube URL or upload an audio/video file above,<br>
                select the language, and hit <strong>Process Video</strong> to begin.
            </div>
            <div class="metric-row" style="justify-content:center; margin-top:1.4rem;">
                <span class="metric-pill pill-blue">🔗 YouTube URLs</span>
                <span class="metric-pill pill-green">🎵 Audio files</span>
                <span class="metric-pill pill-purple">🎞 Video files</span>
                <span class="metric-pill pill-cyan">🌐 Hinglish support</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    inject_css()
    init_session()
    render_sidebar()

    # ── Hero header ─────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-icon">🎥</div>
            <div>
                <div class="hero-title">AI Video Assistant</div>
                <div class="hero-sub">
                    Transcribe · Summarise · Extract Insights · Chat with any video
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Input panel ─────────────────────────────────────────────────────────
    source_value, language, submitted = render_input_panel()

    # ── Trigger pipeline ────────────────────────────────────────────────────
    if submitted and source_value and not st.session_state.processing:
        st.session_state.processing   = True
        st.session_state.processed    = False
        st.session_state.result       = None
        st.session_state.chat_history = []
        st.session_state.rag_chain    = None
        st.session_state.input_source = source_value

        t_start = time.time()
        result = run_pipeline_with_progress(source_value, language)
        elapsed = time.time() - t_start

        st.session_state.processing = False

        if result:
            st.session_state.result    = result
            st.session_state.rag_chain = result["rag_chain"]
            st.session_state.proc_time = elapsed
            st.session_state.processed = True
            st.success(
                f"✅ Processing complete in **{elapsed:.1f}s** — "
                f"{len(result['transcript'].split()):,} words transcribed."
            )
            st.rerun()

    # ── Results ─────────────────────────────────────────────────────────────
    if st.session_state.processed and st.session_state.result:
        render_results(st.session_state.result)
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        render_chat(st.session_state.rag_chain)
    elif not st.session_state.processing:
        render_empty_state()

    # ── Footer ──────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="app-footer">
            <span>AI Video Assistant</span> · Built with Streamlit ·
            Transcription × Summarisation × RAG Q&A ·
            <span>2025</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()