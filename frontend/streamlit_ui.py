"""
frontend/streamlit_ui.py - Personalized Networking Assistant UI

A modern, responsive Streamlit interface that communicates exclusively
with the FastAPI backend. All AI processing happens server-side.

Features:
- Generate personalized conversation starters
- Analyze event themes
- Fact-check topics via Wikipedia
- Review conversation history (with favorites & sorting)
- Submit and view feedback
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = "http://127.0.0.1:8000/api/v1"
REQUEST_TIMEOUT = 120  # seconds – model inference can be slow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Personalized Networking Assistant",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #e8e8f0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Cards */
    .card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.2);
    }

    /* Starter card */
    .starter-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.15));
        border: 1px solid rgba(102, 126, 234, 0.4);
        border-radius: 12px;
        padding: 18px 22px;
        margin: 10px 0;
        position: relative;
        transition: all 0.25s ease;
    }
    .starter-card:hover {
        border-color: rgba(102, 126, 234, 0.8);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.25), rgba(118, 75, 162, 0.25));
        transform: translateX(4px);
    }
    .starter-number {
        font-size: 11px;
        font-weight: 600;
        color: #a78bfa;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 6px;
    }
    .starter-text {
        font-size: 16px;
        color: #e8e8f0;
        line-height: 1.6;
        font-weight: 400;
    }

    /* Theme badge */
    .theme-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 4px;
        letter-spacing: 0.5px;
    }

    /* Fact check card */
    .fact-verified {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 12px;
        padding: 18px;
    }
    .fact-unverified {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 12px;
        padding: 18px;
    }

    /* History entry */
    .history-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        transition: all 0.2s ease;
    }
    .history-card:hover {
        border-color: rgba(102, 126, 234, 0.5);
        background: rgba(255, 255, 255, 0.07);
    }
    .history-card.favorite {
        border-left: 3px solid #f59e0b;
    }

    /* Section header */
    .section-header {
        font-size: 24px;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 8px;
    }
    .section-sub {
        color: #9ca3af;
        font-size: 14px;
        margin-bottom: 24px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        padding: 12px 28px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.35) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5) !important;
    }

    /* Text inputs & areas */
    .stTextArea textarea, .stTextInput input {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        color: #e8e8f0 !important;
        font-size: 14px !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #a78bfa !important;
        font-weight: 700 !important;
    }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.08) !important; }

    /* Spinner text */
    .stSpinner > div { color: #a78bfa !important; }

    /* Score bar */
    .score-bar-bg {
        background: rgba(255,255,255,0.1);
        border-radius: 6px;
        height: 8px;
        margin: 4px 0 12px 0;
        overflow: hidden;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 6px;
        background: linear-gradient(90deg, #667eea, #a78bfa);
    }

    /* Hero */
    .hero-title {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    }
    .hero-sub {
        font-size: 17px;
        color: #9ca3af;
        margin-top: 10px;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Initialize all required session state keys."""
    defaults: dict[str, Any] = {
        "generated_result": None,
        "analysis_result": None,
        "fact_result": None,
        "history": [],
        "feedback_map": {},
        "active_tab": "generate",
        "last_conversation_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# API Client Helpers
# ---------------------------------------------------------------------------

def api_post(endpoint: str, payload: dict) -> dict | None:
    """Send a POST request to the FastAPI backend."""
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            resp = client.post(f"{API_BASE}{endpoint}", json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        st.error("❌ Cannot connect to the backend. Make sure FastAPI is running on port 8000.")
        return None
    except httpx.TimeoutException:
        st.error("⏱️ Request timed out. The model may still be loading — please try again.")
        return None
    except httpx.HTTPStatusError as exc:
        detail = exc.response.json().get("detail", str(exc))
        st.error(f"❌ API Error {exc.response.status_code}: {detail}")
        return None


def api_get(endpoint: str, params: dict | None = None) -> dict | None:
    """Send a GET request to the FastAPI backend."""
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            resp = client.get(f"{API_BASE}{endpoint}", params=params or {})
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        st.error("❌ Cannot connect to the backend. Make sure FastAPI is running on port 8000.")
        return None
    except httpx.TimeoutException:
        st.error("⏱️ Request timed out.")
        return None
    except httpx.HTTPStatusError as exc:
        detail = exc.response.json().get("detail", str(exc))
        st.error(f"❌ API Error {exc.response.status_code}: {detail}")
        return None


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    """Render the sidebar with navigation and app info."""
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 20px 0 10px;">
                <span style="font-size:52px;">🤝</span>
                <div style="font-size:20px; font-weight:700; color:#e8e8f0; margin-top:8px;">
                    Networking<br>Assistant
                </div>
                <div style="font-size:12px; color:#6b7280; margin-top:4px;">
                    AI-Powered v1.0
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # Navigation
        st.markdown("### 🧭 Navigation")
        tabs = {
            "generate": "✨ Generate Starters",
            "analyze": "🔍 Analyze Event",
            "factcheck": "📚 Fact Check",
            "history": "📜 History",
            "feedback": "💬 Feedback",
        }
        for key, label in tabs.items():
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.active_tab = key
                st.rerun()

        st.markdown("---")

        # API Status
        st.markdown("### 📡 API Status")
        try:
            with httpx.Client(timeout=3) as client:
                r = client.get(f"{API_BASE}/health")
                if r.status_code == 200:
                    st.success("Backend Online ✅")
                else:
                    st.warning("Backend returned non-200")
        except Exception:
            st.error("Backend Offline ❌")
            st.info("Run: `uvicorn backend.main:app --reload`")

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size:12px; color:#4b5563; text-align:center; padding:10px 0;">
                Built with FastAPI · GPT-2 · DistilBERT<br>
                Wikipedia · Streamlit · Pydantic
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_theme_badges(themes: list[str], scores: list[float]) -> None:
    """Render theme badges with score progress bars."""
    cols = st.columns(min(len(themes), 3))
    for i, (theme, score) in enumerate(zip(themes, scores)):
        with cols[i % 3]:
            pct = int(score * 100)
            st.markdown(
                f"""
                <div style="text-align:center; padding:8px;">
                    <div class="theme-badge">{theme}</div>
                    <div style="font-size:12px; color:#9ca3af; margin-top:4px;">{pct}%</div>
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="width:{pct}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_starters(starters: list[str]) -> None:
    """Render conversation starter cards."""
    for i, starter in enumerate(starters, 1):
        st.markdown(
            f"""
            <div class="starter-card">
                <div class="starter-number">Starter {i}</div>
                <div class="starter-text">💬 {starter}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Tab: Generate Conversation Starters
# ---------------------------------------------------------------------------

def tab_generate() -> None:
    """Main tab: generate personalized networking conversation starters."""
    st.markdown('<div class="hero-title">✨ Generate Conversation Starters</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Describe your networking event and personalize with your interests to get AI-crafted openers.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    with st.container():
        col1, col2 = st.columns([3, 2], gap="large")

        with col1:
            st.markdown("##### 📝 Event Description")
            event_desc = st.text_area(
                label="event_description",
                placeholder="e.g., AI for Sustainable Cities summit bringing together tech leaders, city planners, and climate scientists...",
                height=160,
                label_visibility="collapsed",
                key="gen_event_desc",
            )

        with col2:
            st.markdown("##### 🎯 Your Interests")
            interests_raw = st.text_input(
                label="user_interests",
                placeholder="e.g., climate change, urban planning, AI ethics",
                label_visibility="collapsed",
                key="gen_interests",
            )
            st.markdown(
                '<div style="font-size:12px; color:#6b7280; margin-top:-10px;">Separate interests with commas</div>',
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                generate_btn = st.button("🚀 Generate", key="btn_generate", use_container_width=True)
            with col_btn2:
                clear_btn = st.button("🗑️ Clear", key="btn_clear_gen", use_container_width=True)

            if clear_btn:
                st.session_state.generated_result = None
                st.rerun()

    if generate_btn:
        if not event_desc.strip():
            st.warning("⚠️ Please enter an event description.")
            return

        interests = [i.strip() for i in interests_raw.split(",") if i.strip()]

        with st.spinner("🤖 Analyzing event themes and generating starters... (first run may take ~30s for model loading)"):
            result = api_post(
                "/generate-conversation",
                {
                    "event_description": event_desc,
                    "user_interests": interests,
                },
            )

        if result:
            st.session_state.generated_result = result
            st.session_state.last_conversation_id = result.get("conversation_id")
            st.success("✅ Conversation starters generated successfully!")

    # Display results
    if st.session_state.generated_result:
        result = st.session_state.generated_result
        st.markdown("---")

        # Themes
        st.markdown('<div class="section-header">🏷️ Extracted Themes</div>', unsafe_allow_html=True)
        render_theme_badges(result.get("themes", []), result.get("scores", []))

        st.markdown("<br>", unsafe_allow_html=True)

        # Starters
        st.markdown('<div class="section-header">💬 Your Conversation Starters</div>', unsafe_allow_html=True)
        render_starters(result.get("starters", []))

        # Copy section
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📋 Copy All Starters as Text"):
            text = "\n\n".join(
                [f"{i}. {s}" for i, s in enumerate(result.get("starters", []), 1)]
            )
            st.code(text, language=None)

        # Metadata
        with st.expander("ℹ️ Session Metadata"):
            st.json(
                {
                    "conversation_id": result.get("conversation_id"),
                    "timestamp": result.get("timestamp"),
                    "themes": result.get("themes"),
                    "scores": result.get("scores"),
                }
            )


# ---------------------------------------------------------------------------
# Tab: Analyze Event
# ---------------------------------------------------------------------------

def tab_analyze() -> None:
    """Tab: analyze event description for themes only."""
    st.markdown('<div class="section-header">🔍 Analyze Event Themes</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Extract the key themes from any event description using zero-shot classification.</div>',
        unsafe_allow_html=True,
    )

    event_desc = st.text_area(
        "Event Description",
        placeholder="e.g., Healthcare Innovation Summit focused on AI diagnostics and blockchain record management...",
        height=150,
        key="analyze_event_desc",
    )

    if st.button("🔎 Analyze Event", key="btn_analyze"):
        if not event_desc.strip():
            st.warning("⚠️ Please enter an event description.")
            return

        with st.spinner("🧠 Running zero-shot classification..."):
            result = api_post("/analyze-event", {"event_description": event_desc})

        if result:
            st.session_state.analysis_result = result
            st.success("✅ Analysis complete!")

    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        st.markdown("---")
        st.markdown("#### 🏷️ Extracted Themes & Confidence Scores")
        render_theme_badges(result.get("themes", []), result.get("scores", []))

        st.markdown("#### 📊 Detailed Scores")
        for theme, score in zip(result.get("themes", []), result.get("scores", [])):
            cols = st.columns([3, 1])
            cols[0].markdown(f"**{theme}**")
            cols[1].markdown(f"`{score:.2%}`")
            st.progress(score)


# ---------------------------------------------------------------------------
# Tab: Fact Check
# ---------------------------------------------------------------------------

def tab_factcheck() -> None:
    """Tab: Wikipedia fact checking."""
    st.markdown('<div class="section-header">📚 Fact Check via Wikipedia</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Verify any topic against Wikipedia to get a reliable summary.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([4, 1], gap="medium")
    with col1:
        topic = st.text_input(
            "Topic to Verify",
            placeholder="e.g., Blockchain in Healthcare, Climate Change, Machine Learning",
            key="fact_topic",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        check_btn = st.button("🔍 Check", key="btn_fact_check", use_container_width=True)

    if check_btn:
        if not topic.strip():
            st.warning("⚠️ Please enter a topic to verify.")
            return

        with st.spinner(f"🔎 Searching Wikipedia for '{topic}'..."):
            result = api_get("/fact-check", {"topic": topic})

        if result:
            st.session_state.fact_result = result

    if st.session_state.fact_result:
        result = st.session_state.fact_result
        st.markdown("---")

        verified = result.get("verified", False)
        css_class = "fact-verified" if verified else "fact-unverified"
        status_icon = "✅ Verified" if verified else "⚠️ Not Found"
        badge_color = "#10b981" if verified else "#f59e0b"

        st.markdown(
            f"""
            <div class="{css_class}">
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:14px;">
                    <div style="font-size:22px; font-weight:700; color:#e8e8f0;">{result.get('title', 'N/A')}</div>
                    <span style="background:{badge_color}; color:white; padding:3px 12px; border-radius:20px; font-size:12px; font-weight:600;">
                        {status_icon}
                    </span>
                </div>
                <div style="color:#d1d5db; font-size:15px; line-height:1.7;">{result.get('summary', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if result.get("url"):
            st.markdown(
                f"<br>🔗 [Read full article on Wikipedia]({result['url']})",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Tab: History
# ---------------------------------------------------------------------------

def tab_history() -> None:
    """Tab: conversation history viewer."""
    st.markdown('<div class="section-header">📜 Conversation History</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Review all previously generated conversation starters.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([2, 2, 2], gap="small")
    with col1:
        sort_order = st.selectbox("Sort by", ["newest", "oldest"], key="hist_sort")
    with col2:
        favs_only = st.checkbox("⭐ Favorites Only", key="hist_favs")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh_btn = st.button("🔄 Refresh", key="btn_hist_refresh", use_container_width=True)

    if refresh_btn or not st.session_state.history:
        with st.spinner("Loading history..."):
            result = api_get(
                "/history",
                {"sort": sort_order, "favorites_only": str(favs_only).lower()},
            )
        if result:
            st.session_state.history = result.get("entries", [])

    entries = st.session_state.history
    if not entries:
        st.info("📭 No history found. Generate some conversations first!")
        return

    st.metric("Total Conversations", len(entries))
    st.markdown("---")

    for entry in entries:
        is_fav = entry.get("favorite", False)
        fav_icon = "⭐" if is_fav else "☆"
        ts = entry.get("timestamp", "")
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                ts = dt.strftime("%B %d, %Y · %H:%M UTC")
            except Exception:
                pass

        with st.expander(
            f"{fav_icon} {entry.get('event_description', '')[:80]}...  —  {ts}"
        ):
            # Themes
            themes = entry.get("themes", [])
            theme_html = "".join(f'<span class="theme-badge">{t}</span>' for t in themes)
            st.markdown(f'<div style="margin-bottom:12px;">{theme_html}</div>', unsafe_allow_html=True)

            # Starters
            st.markdown("**Conversation Starters:**")
            for i, s in enumerate(entry.get("starters", []), 1):
                st.markdown(f"&nbsp;&nbsp;**{i}.** {s}")

            # Interests
            interests = entry.get("user_interests", [])
            if interests:
                st.markdown(f"**Interests:** {', '.join(interests)}")

            st.caption(f"ID: `{entry.get('conversation_id', 'N/A')}`")


# ---------------------------------------------------------------------------
# Tab: Feedback
# ---------------------------------------------------------------------------

def tab_feedback() -> None:
    """Tab: submit and view feedback."""
    st.markdown('<div class="section-header">💬 Feedback</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Rate conversations and help improve future generations.</div>',
        unsafe_allow_html=True,
    )

    # Submit feedback
    with st.container():
        st.markdown("#### 📝 Submit Feedback")
        col1, col2 = st.columns([3, 2], gap="large")

        with col1:
            conv_id = st.text_input(
                "Conversation ID",
                value=st.session_state.last_conversation_id or "",
                placeholder="Paste conversation ID here",
                key="fb_conv_id",
            )
            comment = st.text_area(
                "Comment (optional)",
                placeholder="What did you like or dislike?",
                height=100,
                key="fb_comment",
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Your Rating:**")
            col_up, col_down = st.columns(2)
            with col_up:
                thumbs_up = st.button("👍 Helpful", key="btn_thumbs_up", use_container_width=True)
            with col_down:
                thumbs_down = st.button("👎 Not Helpful", key="btn_thumbs_down", use_container_width=True)

        if thumbs_up or thumbs_down:
            if not conv_id.strip():
                st.warning("⚠️ Please enter a Conversation ID.")
            else:
                payload = {
                    "conversation_id": conv_id.strip(),
                    "thumbs_up": bool(thumbs_up),
                    "thumbs_down": bool(thumbs_down),
                    "comment": comment.strip() or None,
                }
                with st.spinner("Submitting feedback..."):
                    result = api_post("/feedback", payload)
                if result:
                    st.success("✅ Thank you for your feedback!")
                    st.session_state.feedback_map[result["feedback_id"]] = result

    st.markdown("---")

    # View all feedback
    st.markdown("#### 📋 All Feedback")
    col_r1, col_r2 = st.columns([3, 1])
    with col_r1:
        filter_id = st.text_input(
            "Filter by Conversation ID (optional)",
            placeholder="Leave blank to see all",
            key="fb_filter",
        )
    with col_r2:
        st.markdown("<br>", unsafe_allow_html=True)
        load_fb_btn = st.button("📥 Load Feedback", key="btn_load_fb", use_container_width=True)

    if load_fb_btn:
        params = {"conversation_id": filter_id.strip()} if filter_id.strip() else {}
        with st.spinner("Loading feedback..."):
            result = api_get("/feedback", params)

        if result:
            entries = result.get("entries", [])
            if not entries:
                st.info("No feedback found.")
            else:
                st.metric("Total Feedback Records", len(entries))
                for fb in entries:
                    rating = "👍 Helpful" if fb.get("thumbs_up") else ("👎 Not Helpful" if fb.get("thumbs_down") else "—")
                    ts = fb.get("timestamp", "")
                    if ts:
                        try:
                            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                            ts = dt.strftime("%b %d %Y %H:%M")
                        except Exception:
                            pass

                    with st.expander(f"{rating} — {ts}"):
                        st.markdown(f"**Conversation ID:** `{fb.get('conversation_id', 'N/A')}`")
                        st.markdown(f"**Feedback ID:** `{fb.get('feedback_id', 'N/A')}`")
                        if fb.get("comment"):
                            st.markdown(f"**Comment:** {fb['comment']}")


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point – render the full application."""
    init_session_state()
    render_sidebar()

    # Main content area
    tab = st.session_state.active_tab

    if tab == "generate":
        tab_generate()
    elif tab == "analyze":
        tab_analyze()
    elif tab == "factcheck":
        tab_factcheck()
    elif tab == "history":
        tab_history()
    elif tab == "feedback":
        tab_feedback()
    else:
        tab_generate()


if __name__ == "__main__":
    main()
