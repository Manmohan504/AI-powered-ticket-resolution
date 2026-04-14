import streamlit as st
import sys
import os
import time
import plotly.graph_objects as go
import plotly.express as px

# Ensure app directory is on the path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import auth_service
import ticket_service
import database
import rag_engine

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Ticket Resolution System",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — Dark Mode with Inter Font & Blue Accents
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }

    /* Header styling */
    h1 {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    /* Card styling */
    .ticket-card {
        background: rgba(30, 30, 46, 0.6);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.1));
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }

    /* Status badges */
    .status-resolved {
        background: #065f46;
        color: #6ee7b7;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-tentative {
        background: #78350f;
        color: #fcd34d;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-unresolved {
        background: #7f1d1d;
        color: #fca5a5;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Confidence bar */
    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        background: rgba(255,255,255,0.1);
        margin-top: 4px;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    }

    /* Form styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
    }

    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.3), transparent);
        margin: 1.5rem 0;
    }

    /* ---- Analytics KPI Cards ---- */
    .kpi-card {
        background: rgba(30, 30, 46, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.4rem 1.2rem;
        text-align: center;
        backdrop-filter: blur(12px);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15);
    }
    .kpi-icon {
        font-size: 1.6rem;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.7;
        margin-top: 0.3rem;
    }
    .kpi-accent-blue   { border-top: 3px solid #3b82f6; }
    .kpi-accent-green  { border-top: 3px solid #22c55e; }
    .kpi-accent-amber  { border-top: 3px solid #f59e0b; }
    .kpi-accent-red    { border-top: 3px solid #ef4444; }
    .kpi-accent-purple { border-top: 3px solid #8b5cf6; }
    .kpi-accent-cyan   { border-top: 3px solid #06b6d4; }

    /* Helpful rate progress */
    .rate-bar-bg {
        height: 12px;
        border-radius: 6px;
        background: rgba(255,255,255,0.08);
        overflow: hidden;
        margin-top: 6px;
    }
    .rate-bar-fill {
        height: 100%;
        border-radius: 6px;
        background: linear-gradient(90deg, #22c55e, #16a34a);
        transition: width 0.8s ease;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1.5rem 0 0.8rem 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Initialize System
# ---------------------------------------------------------------------------
@st.cache_resource
def init_system():
    """Initialize DB and check model availability (runs once)."""
    ticket_service.initialize_system()
    return True


# ---------------------------------------------------------------------------
# Session State Defaults
# ---------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "logged_in": False,
        "username": None,
        "role": None,
        "page": "login",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Auth Pages
# ---------------------------------------------------------------------------
def render_login_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("# 🎫 AI Ticket Resolution")
        st.markdown("*Intelligent IT support powered by local AI*")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["🔑 Login", "📝 Sign Up"])

        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submitted = st.form_submit_button("Login", width='stretch')

                if submitted:
                    if not username or not password:
                        st.error("Please fill in all fields.")
                    else:
                        user = auth_service.login_user(username, password)
                        if user:
                            st.session_state.logged_in = True
                            st.session_state.username = user["username"]
                            st.session_state.role = user["role"]
                            st.success(f"Welcome back, {user['username']}!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("Invalid username or password.")

            st.caption("Default accounts: `admin` / `admin123` or `testuser` / `user123`")

        with tab_signup:
            with st.form("signup_form"):
                new_user = st.text_input("Choose a Username", placeholder="Pick a username")
                new_pass = st.text_input("Choose a Password", type="password", placeholder="Min 4 characters")
                confirm_pass = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
                signup_submitted = st.form_submit_button("Create Account", width='stretch')

                if signup_submitted:
                    if not new_user or not new_pass:
                        st.error("Please fill in all fields.")
                    elif len(new_pass) < 4:
                        st.error("Password must be at least 4 characters.")
                    elif new_pass != confirm_pass:
                        st.error("Passwords do not match.")
                    else:
                        if auth_service.register_user(new_user, new_pass):
                            st.success("Account created! You can now log in.")
                        else:
                            st.error("Username already exists.")


# ---------------------------------------------------------------------------
# Confidence Badge Helper
# ---------------------------------------------------------------------------
def confidence_badge(score):
    """Returns colored confidence display."""
    pct = int(score * 100)
    if score >= 0.65:
        color = "#22c55e"
    elif score >= 0.4:
        color = "#eab308"
    else:
        color = "#ef4444"

    return f"""
    <div style="display: flex; align-items: center; gap: 8px;">
        <span style="font-weight: 600; color: {color};">{pct}%</span>
        <div class="confidence-bar" style="flex: 1;">
            <div class="confidence-fill" style="width: {pct}%; background: {color};"></div>
        </div>
    </div>
    """


def status_badge(status):
    """Returns a styled status badge."""
    return f'<span class="status-{status}">{status.upper()}</span>'


# ---------------------------------------------------------------------------
# Main App — New Incident Tab
# ---------------------------------------------------------------------------
def render_new_incident():
    st.markdown("### 🔥 Submit a New Incident")
    st.markdown("Describe your issue below and AI will generate a resolution.")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    with st.form("ticket_form", clear_on_submit=True):
        title = st.text_input("Issue Title", placeholder="e.g., Cannot access email")

        description = st.text_area(
            "Description",
            placeholder="Describe the issue in detail...",
            height=120,
        )

        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox(
                "Category",
                ["Network", "Hardware", "Software", "Access/Permissions", "Email", "Other"],
            )
        with col2:
            priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])

        submitted = st.form_submit_button("🚀 Submit Ticket", width='stretch')

    if submitted:
        if not title or not description:
            st.error("Please provide both a title and description.")
            return

        with st.spinner("🤖 AI is analyzing your ticket..."):
            try:
                result = ticket_service.submit_ticket(
                    title=title,
                    description=description,
                    category=category,
                    priority=priority,
                    user_id=st.session_state.username,
                )

                st.success("Ticket submitted successfully!")
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

                # Resolution display
                st.markdown("#### 🤖 AI Resolution")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Status:** {status_badge(result.get('resolution_status', 'unresolved'))}", unsafe_allow_html=True)
                with col2:
                    st.markdown("**Confidence:**")
                    st.markdown(confidence_badge(result.get("confidence_score", 0)), unsafe_allow_html=True)
                with col3:
                    kb = "✅ Found" if result.get("kb_context_found") else "❌ Not Found"
                    st.markdown(f"**KB Context:** {kb}")

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown(result.get("ai_resolution", "No resolution generated."))

                # Feedback buttons
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown("**Was this resolution helpful?**")
                fb_col1, fb_col2, _ = st.columns([1, 1, 4])
                with fb_col1:
                    if st.button("👍 Helpful", key="fb_helpful"):
                        ticket_service.submit_feedback(result["id"], "helpful", st.session_state.username)
                        st.success("Thanks for your feedback!")
                with fb_col2:
                    if st.button("👎 Not Helpful", key="fb_not_helpful"):
                        ticket_service.submit_feedback(result["id"], "not_helpful", st.session_state.username)
                        st.info("Thanks — we'll work on improving.")

            except Exception as e:
                st.error(f"Error submitting ticket: {e}")


# ---------------------------------------------------------------------------
# Main App — My History Tab
# ---------------------------------------------------------------------------
def render_history():
    st.markdown("### 📂 My Ticket History")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    tickets = ticket_service.get_user_tickets(st.session_state.username)

    if tickets.empty:
        st.info("You haven't submitted any tickets yet. Go to **New Incident** to get started!")
        return

    st.caption(f"Showing {len(tickets)} tickets")

    for _, ticket in tickets.iterrows():
        with st.expander(f"🎫 {ticket['title']}  —  {ticket.get('category', 'N/A')}  |  {ticket.get('priority', 'N/A')}", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Status:** {status_badge(ticket.get('resolution_status', 'unresolved'))}", unsafe_allow_html=True)
            with col2:
                st.markdown("**Confidence:**")
                st.markdown(confidence_badge(ticket.get("confidence_score", 0)), unsafe_allow_html=True)
            with col3:
                st.markdown(f"**Created:** {ticket.get('created_at', 'N/A')}")

            st.markdown(f"**Description:** {ticket['description']}")
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("**🤖 AI Resolution:**")
            st.markdown(ticket.get("ai_resolution", "No resolution available."))

            # Feedback
            feedback = ticket.get("feedback_value")
            if feedback:
                emoji = "👍" if feedback == "helpful" else "👎"
                st.caption(f"Your feedback: {emoji} {feedback}")
            else:
                fb_col1, fb_col2, _ = st.columns([1, 1, 4])
                with fb_col1:
                    if st.button("👍 Helpful", key=f"hist_helpful_{ticket['id']}"):
                        ticket_service.submit_feedback(ticket["id"], "helpful", st.session_state.username)
                        st.rerun()
                with fb_col2:
                    if st.button("👎 Not Helpful", key=f"hist_not_{ticket['id']}"):
                        ticket_service.submit_feedback(ticket["id"], "not_helpful", st.session_state.username)
                        st.rerun()


# ---------------------------------------------------------------------------
# Admin Dashboard
# ---------------------------------------------------------------------------
def _plotly_layout(title="", height=350):
    """Shared Plotly layout for dark-themed charts."""
    return dict(
        title=dict(text=title, font=dict(size=15, family="Inter", color="#e2e8f0"), x=0.02),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#94a3b8", size=12),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )


def render_admin_dashboard():
    header_col1, header_col2 = st.columns([6, 1])
    with header_col1:
        st.markdown("### 📊 Admin Dashboard")
    with header_col2:
        if st.button("🔄 Refresh", key="refresh_dashboard"):
            st.rerun()
    st.caption("📡 Live data — charts update automatically after each ticket submission")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ---- KPI Metrics ----
    try:
        kpis = ticket_service.get_admin_kpis()
        total   = kpis.get("total_tickets", 0) or 0
        resolved = kpis.get("resolved_tickets", 0) or 0
        tentative = kpis.get("tentative_tickets", 0) or 0
        unresolved = kpis.get("unresolved_tickets", 0) or 0
        avg_conf = kpis.get("avg_confidence", 0) or 0
        helpful_rate = kpis.get("helpful_rate", 0) or 0
        helpful_count = kpis.get("helpful_count", 0) or 0
        not_helpful_count = kpis.get("not_helpful_count", 0) or 0

        def kpi_html(icon, value, label, accent):
            return f"""
            <div class="kpi-card kpi-accent-{accent}">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-label">{label}</div>
            </div>
            """

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.markdown(kpi_html("🎫", total, "Total Tickets", "blue"), unsafe_allow_html=True)
        with col2:
            st.markdown(kpi_html("✅", resolved, "Resolved", "green"), unsafe_allow_html=True)
        with col3:
            st.markdown(kpi_html("⏳", tentative, "Tentative", "amber"), unsafe_allow_html=True)
        with col4:
            st.markdown(kpi_html("❌", unresolved, "Unresolved", "red"), unsafe_allow_html=True)
        with col5:
            st.markdown(kpi_html("🎯", f"{avg_conf:.0%}", "Avg Confidence", "purple"), unsafe_allow_html=True)
        with col6:
            st.markdown(kpi_html("👍", f"{helpful_rate:.0%}", "Helpful Rate", "cyan"), unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"Could not load KPIs: {e}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ---- Analytics Charts ----
    try:
        analytics = ticket_service.get_analytics_data()

        # Row 1: Status Donut + Feedback donut
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            status_df = analytics["status_distribution"]
            if not status_df.empty:
                colors_map = {"resolved": "#22c55e", "tentative": "#f59e0b", "unresolved": "#ef4444"}
                colors = [colors_map.get(s, "#64748b") for s in status_df["resolution_status"]]
                fig = go.Figure(data=[go.Pie(
                    labels=status_df["resolution_status"].str.title(),
                    values=status_df["count"],
                    hole=0.55,
                    marker=dict(colors=colors, line=dict(color="rgba(0,0,0,0.3)", width=2)),
                    textinfo="label+value",
                    textfont=dict(size=12),
                )])
                fig.update_layout(**_plotly_layout("Resolution Status", 320))
                st.plotly_chart(fig, use_container_width=True)

        with chart_col2:
            fb_df = analytics["feedback_distribution"]
            if not fb_df.empty:
                fb_helpful = int(fb_df["helpful"].iloc[0] or 0)
                fb_not = int(fb_df["not_helpful"].iloc[0] or 0)
                fb_pending = int(fb_df["pending"].iloc[0] or 0)
                fb_total = fb_helpful + fb_not + fb_pending
                fig = go.Figure(data=[go.Pie(
                    labels=["Helpful", "Not Helpful", "Pending"],
                    values=[fb_helpful, fb_not, fb_pending],
                    hole=0.55,
                    marker=dict(
                        colors=["#22c55e", "#ef4444", "#64748b"],
                        line=dict(color="rgba(0,0,0,0.3)", width=2),
                    ),
                    textinfo="label+value",
                    textfont=dict(size=12),
                )])
                fig.update_layout(**_plotly_layout("Feedback Analysis", 320))
                st.plotly_chart(fig, use_container_width=True)

        # ---- Helpful Rate — Dedicated Section ----
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        fb_df = analytics["feedback_distribution"]
        if not fb_df.empty:
            fb_helpful = int(fb_df["helpful"].iloc[0] or 0)
            fb_not = int(fb_df["not_helpful"].iloc[0] or 0)
            fb_pending = int(fb_df["pending"].iloc[0] or 0)
            fb_responded = fb_helpful + fb_not
            fb_total = fb_helpful + fb_not + fb_pending
            rate_pct = (fb_helpful / fb_responded * 100) if fb_responded else 0
            response_rate = (fb_responded / fb_total * 100) if fb_total else 0

            gauge_col, detail_col = st.columns([1, 1])

            with gauge_col:
                # Gauge chart for helpful rate
                if rate_pct >= 70:
                    gauge_color = "#22c55e"
                elif rate_pct >= 40:
                    gauge_color = "#f59e0b"
                else:
                    gauge_color = "#ef4444"

                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=rate_pct,
                    number=dict(suffix="%", font=dict(size=42, family="Inter", color="#e2e8f0")),
                    title=dict(text="User Satisfaction Rate", font=dict(size=16, family="Inter", color="#94a3b8")),
                    gauge=dict(
                        axis=dict(range=[0, 100], tickcolor="#475569", tickfont=dict(color="#64748b")),
                        bar=dict(color=gauge_color, thickness=0.7),
                        bgcolor="rgba(30,30,46,0.5)",
                        borderwidth=0,
                        steps=[
                            dict(range=[0, 40], color="rgba(239,68,68,0.15)"),
                            dict(range=[40, 70], color="rgba(245,158,11,0.15)"),
                            dict(range=[70, 100], color="rgba(34,197,94,0.15)"),
                        ],
                        threshold=dict(line=dict(color="#e2e8f0", width=2), thickness=0.8, value=rate_pct),
                    ),
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color="#94a3b8"),
                    height=280,
                    margin=dict(l=30, r=30, t=60, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

            with detail_col:
                st.markdown("")
                # Detailed breakdown cards
                def feedback_stat_html(emoji, label, count, total, color):
                    pct = (count / total * 100) if total else 0
                    return f"""
                    <div style="padding: 0.75rem 1rem; background: rgba(30,30,46,0.4); border-radius: 10px;
                                border-left: 3px solid {color}; margin-bottom: 0.6rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 1rem;">{emoji} <strong>{label}</strong></span>
                            <span style="font-size: 1.3rem; font-weight: 700; color: {color};">{count}</span>
                        </div>
                        <div class="rate-bar-bg" style="margin-top: 6px;">
                            <div style="height: 100%; width: {pct:.1f}%; background: {color};
                                        border-radius: 6px; transition: width 0.8s ease;"></div>
                        </div>
                        <div style="text-align: right; font-size: 0.75rem; opacity: 0.6; margin-top: 3px;">{pct:.1f}% of total</div>
                    </div>
                    """

                st.markdown(feedback_stat_html("👍", "Helpful", fb_helpful, fb_total, "#22c55e"), unsafe_allow_html=True)
                st.markdown(feedback_stat_html("👎", "Not Helpful", fb_not, fb_total, "#ef4444"), unsafe_allow_html=True)
                st.markdown(feedback_stat_html("⏳", "Awaiting Feedback", fb_pending, fb_total, "#64748b"), unsafe_allow_html=True)

                # Response rate
                st.markdown(f"""
                <div style="padding: 0.6rem 1rem; background: rgba(59,130,246,0.08); border-radius: 8px;
                            border: 1px solid rgba(59,130,246,0.2); text-align: center; margin-top: 0.3rem;">
                    <span style="font-size: 0.85rem; opacity: 0.8;">Feedback Response Rate:</span>
                    <span style="font-weight: 700; color: #3b82f6; font-size: 1.1rem; margin-left: 8px;">
                        {response_rate:.1f}%</span>
                    <span style="font-size: 0.75rem; opacity: 0.6; margin-left: 4px;">({fb_responded}/{fb_total} tickets rated)</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Row 2: Category bar + Priority bar
        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            cat_df = analytics["category_distribution"]
            if not cat_df.empty:
                fig = go.Figure(data=[go.Bar(
                    x=cat_df["count"],
                    y=cat_df["category"],
                    orientation="h",
                    marker=dict(
                        color=cat_df["count"],
                        colorscale=[[0, "#3b82f6"], [1, "#8b5cf6"]],
                        line=dict(width=0),
                        cornerradius=5,
                    ),
                    text=cat_df["count"],
                    textposition="auto",
                    textfont=dict(color="white", size=12, family="Inter"),
                )])
                fig.update_layout(**_plotly_layout("Tickets by Category", 300))
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=False)
                st.plotly_chart(fig, use_container_width=True)

        with chart_col4:
            pri_df = analytics["priority_distribution"]
            if not pri_df.empty:
                pri_colors = {"Critical": "#ef4444", "High": "#f59e0b", "Medium": "#3b82f6", "Low": "#22c55e"}
                colors = [pri_colors.get(p, "#64748b") for p in pri_df["priority"]]
                fig = go.Figure(data=[go.Bar(
                    x=pri_df["priority"],
                    y=pri_df["count"],
                    marker=dict(color=colors, cornerradius=5),
                    text=pri_df["count"],
                    textposition="auto",
                    textfont=dict(color="white", size=12, family="Inter"),
                )])
                fig.update_layout(**_plotly_layout("Tickets by Priority", 300))
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=False)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Row 3: Daily trend + Confidence histogram
        chart_col5, chart_col6 = st.columns(2)

        with chart_col5:
            trend_df = analytics["daily_trend"]
            if not trend_df.empty:
                fig = go.Figure(data=[go.Scatter(
                    x=trend_df["date"],
                    y=trend_df["count"],
                    mode="lines+markers",
                    line=dict(color="#3b82f6", width=2.5, shape="spline"),
                    marker=dict(size=7, color="#3b82f6", line=dict(width=2, color="white")),
                    fill="tozeroy",
                    fillcolor="rgba(59,130,246,0.1)",
                )])
                fig.update_layout(**_plotly_layout("Ticket Volume (Last 30 Days)", 300))
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
                st.plotly_chart(fig, use_container_width=True)

        with chart_col6:
            conf_df = analytics["confidence_distribution"]
            if not conf_df.empty:
                fig = go.Figure(data=[go.Bar(
                    x=conf_df["bucket"],
                    y=conf_df["count"],
                    marker=dict(
                        color=conf_df["count"],
                        colorscale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#22c55e"]],
                        cornerradius=5,
                    ),
                    text=conf_df["count"],
                    textposition="auto",
                    textfont=dict(color="white", size=12, family="Inter"),
                )])
                fig.update_layout(**_plotly_layout("Confidence Score Distribution", 300))
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=False)
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not load analytics: {e}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ---- All Tickets Table ----
    st.markdown("#### 📋 All Tickets")
    all_tickets = ticket_service.get_all_tickets()

    if all_tickets.empty:
        st.info("No tickets in the system yet.")
    else:
        display_cols = ["id", "title", "category", "priority", "user_id",
                        "resolution_status", "confidence_score", "created_at"]
        available_cols = [c for c in display_cols if c in all_tickets.columns]
        st.dataframe(all_tickets[available_cols], use_container_width=True, hide_index=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ---- Top Questions ----
    st.markdown("#### 🔝 Top Recurring Questions")
    try:
        top_q = ticket_service.get_top_questions(limit=10)
        if top_q.empty:
            st.info("No recurring questions detected yet.")
        else:
            st.dataframe(top_q, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Could not load top questions: {e}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ---- Knowledge Gaps ----
    st.markdown("#### ⚠️ Knowledge Gaps")
    try:
        conn = database.get_db_connection()
        import pandas as pd
        gaps = pd.read_sql_query(
            "SELECT display_query, category, occurrence_count, avg_confidence_score, "
            "suggested_kb_filename, last_seen_at FROM knowledge_gap_events "
            "ORDER BY occurrence_count DESC LIMIT 20",
            conn,
        )
        conn.close()

        if gaps.empty:
            st.success("No knowledge gaps detected — your KB is covering queries well!")
        else:
            st.dataframe(gaps, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Could not load knowledge gaps: {e}")


# ---------------------------------------------------------------------------
# Upload Documents Tab
# ---------------------------------------------------------------------------
def render_upload_documents():
    st.markdown("### 📤 Upload Knowledge Base Documents")
    st.markdown("Upload PDF or TXT files to add to the AI knowledge base. Documents will be processed automatically.")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT",
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
        for uf in uploaded_files:
            size_kb = uf.size / 1024
            st.caption(f"📄 {uf.name} ({size_kb:.1f} KB)")

        if st.button("🚀 Process Documents"):
            import shutil
            from langchain_community.vectorstores import FAISS
            from langchain_community.document_loaders import TextLoader

            progress = st.progress(0, text="Starting...")
            status_container = st.container()

            total_steps = len(uploaded_files) + 3  # files + embed + save + cleanup
            current_step = 0

            all_chunks = []
            saved_files = []

            # Ensure directories exist
            os.makedirs(rag_engine.DATA_RAW_DIR, exist_ok=True)
            os.makedirs(rag_engine.DATA_PROCESSED_DIR, exist_ok=True)

            # Step 1: Save and process each file
            for uf in uploaded_files:
                current_step += 1
                progress.progress(current_step / total_steps, text=f"Processing {uf.name}...")

                # Save uploaded file to data/raw/
                file_path = os.path.join(rag_engine.DATA_RAW_DIR, uf.name)
                with open(file_path, "wb") as f:
                    f.write(uf.getbuffer())
                saved_files.append(uf.name)

                try:
                    if uf.name.lower().endswith(".pdf"):
                        with status_container:
                            st.info(f"📄 {uf.name}: Layout-aware extraction + chunking...")
                        docs = rag_engine._load_pdf_layout_aware(file_path)
                        chunks = rag_engine.chunk_documents(docs, source_type="pdf")
                        all_chunks.extend(chunks)
                        with status_container:
                            st.success(f"✅ {uf.name}: {len(chunks)} chunks created")

                    elif uf.name.lower().endswith(".txt"):
                        with status_container:
                            st.info(f"📝 {uf.name}: Loading text + chunking...")
                        loader = TextLoader(file_path)
                        docs = loader.load()
                        chunks = rag_engine.chunk_documents(docs, source_type="txt")
                        all_chunks.extend(chunks)
                        with status_container:
                            st.success(f"✅ {uf.name}: {len(chunks)} chunks created")

                except Exception as e:
                    with status_container:
                        st.error(f"❌ {uf.name}: Error — {e}")

            if not all_chunks:
                st.error("No valid content extracted from uploaded files.")
                return

            # Step 2: Generate embeddings and update FAISS index
            current_step += 1
            progress.progress(current_step / total_steps, text="Generating embeddings...")

            try:
                embeddings = rag_engine.get_embeddings()

                if os.path.exists(rag_engine.FAISS_INDEX_PATH):
                    db = FAISS.load_local(
                        rag_engine.FAISS_INDEX_PATH, embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    db.add_documents(all_chunks)
                    with status_container:
                        st.info("Updated existing FAISS index")
                else:
                    db = FAISS.from_documents(all_chunks, embeddings)
                    with status_container:
                        st.info("Created new FAISS index")

                # Step 3: Save indexes
                current_step += 1
                progress.progress(current_step / total_steps, text="Saving indexes...")

                db.save_local(rag_engine.FAISS_INDEX_PATH)

                # Update BM25 index
                rag_engine._save_bm25_corpus(all_chunks)

                with status_container:
                    st.info("FAISS + BM25 indexes saved")

            except Exception as e:
                st.error(f"Embedding/indexing failed: {e}")
                return

            # Step 4: Move files to processed
            current_step += 1
            progress.progress(current_step / total_steps, text="Cleaning up...")

            for fname in saved_files:
                src = os.path.join(rag_engine.DATA_RAW_DIR, fname)
                dst = os.path.join(rag_engine.DATA_PROCESSED_DIR, fname)
                try:
                    shutil.move(src, dst)
                except Exception:
                    pass

            progress.progress(1.0, text="Complete!")

            # Summary
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.success(f"✅ **Ingestion Complete!**")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Files Processed", len(saved_files))
            with col2:
                st.metric("Chunks Created", len(all_chunks))
            with col3:
                chunk_sizes = [len(c.page_content) for c in all_chunks]
                st.metric("Avg Chunk Size", f"{sum(chunk_sizes)//len(chunk_sizes)} chars")

            # Chunk method breakdown
            methods = {}
            for c in all_chunks:
                m = c.metadata.get("chunking", "header_only")
                methods[m] = methods.get(m, 0) + 1
            st.caption(f"Chunking methods: {methods}")
            st.caption("New documents are now searchable in the knowledge base!")


# ---------------------------------------------------------------------------
# Main App (Post-Login)
# ---------------------------------------------------------------------------
def render_main_app():
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")
        st.caption(f"Role: {st.session_state.role}")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        if st.button("🚪 Logout"):
            for key in ["logged_in", "username", "role"]:
                st.session_state[key] = None if key != "logged_in" else False
            st.rerun()

    # Title
    st.markdown("# 🎫 AI Ticket Resolution System")

    # Tabs based on role
    if st.session_state.role == "admin":
        tab1, tab2, tab3, tab4 = st.tabs(["🔥 New Incident", "📂 My History", "📤 Upload Documents", "📊 Admin Dashboard"])
        with tab1:
            render_new_incident()
        with tab2:
            render_history()
        with tab3:
            render_upload_documents()
        with tab4:
            render_admin_dashboard()
    else:
        tab1, tab2, tab3 = st.tabs(["🔥 New Incident", "📂 My History", "📤 Upload Documents"])
        with tab1:
            render_new_incident()
        with tab2:
            render_history()
        with tab3:
            render_upload_documents()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def main():
    init_session_state()
    init_system()

    if st.session_state.logged_in:
        render_main_app()
    else:
        render_login_page()


if __name__ == "__main__":
    main()
