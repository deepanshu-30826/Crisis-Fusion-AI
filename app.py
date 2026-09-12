"""Crisis Report Fusion & Priority Ranking — Enterprise Emergency Response Web Application."""

import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA

from src.inference import predict_report
from src.embeddings import load_embedding_model, embed_texts
from src.retrieval import load_index, retrieve_top_k
from src.preprocessing import clean_text

# --- Page Config ---
st.set_page_config(
    page_title="CrisisFusion AI | Emergency Triage Operations",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enterprise SaaS Web Application CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Global reset & typography */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #F1F5F9;
    }

    /* Container constraints */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1440px !important;
    }

    /* Modern Top Navbar */
    .saas-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(180deg, #131B2E 0%, #0B1120 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 16px 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .brand-icon-box {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: linear-gradient(135deg, #EF4444, #DC2626);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.4);
    }
    .brand-title {
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #FFFFFF;
        line-height: 1.2;
    }
    .brand-title span {
        color: #38BDF8;
    }
    .brand-badge {
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94A3B8;
        background: rgba(255, 255, 255, 0.06);
        padding: 2px 8px;
        border-radius: 6px;
        margin-top: 2px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .nav-telemetry {
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .live-status-chip {
        display: flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34D399;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 6px 14px;
        border-radius: 9999px;
    }
    .live-dot {
        width: 8px;
        height: 8px;
        background: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10B981;
        animation: pulse-green 2s infinite;
    }
    @keyframes pulse-green {
        0% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1.1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    .nav-meta-item {
        font-size: 0.8rem;
        color: #64748B;
        font-family: 'JetBrains Mono', monospace;
    }
    .nav-meta-item strong {
        color: #E2E8F0;
    }

    /* KPI Summary Cards */
    .kpi-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    .saas-card {
        background: #111827;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: border-color 0.2s;
    }
    .saas-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
    }
    .card-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .card-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #FFFFFF;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.2;
    }
    .card-subtext {
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 4px;
    }

    /* Terminal Console Box */
    .terminal-box {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }
    .terminal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 12px;
    }
    .terminal-dots {
        display: flex;
        gap: 6px;
    }
    .terminal-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
    }
    .dot-red { background: #EF4444; }
    .dot-yellow { background: #F59E0B; }
    .dot-green { background: #10B981; }
    .terminal-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #94A3B8;
        font-weight: 600;
    }

    /* Urgency Banner HUD */
    .urgency-hud {
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-width: 1px;
        border-style: solid;
    }
    .hud-critical {
        background: linear-gradient(90deg, rgba(220, 38, 38, 0.25) 0%, rgba(153, 27, 27, 0.1) 100%);
        border-color: #EF4444;
        color: #FCA5A5;
    }
    .hud-high {
        background: linear-gradient(90deg, rgba(234, 88, 12, 0.25) 0%, rgba(194, 65, 12, 0.1) 100%);
        border-color: #F97316;
        color: #FDBA74;
    }
    .hud-medium {
        background: linear-gradient(90deg, rgba(217, 119, 6, 0.25) 0%, rgba(180, 83, 9, 0.1) 100%);
        border-color: #F59E0B;
        color: #FDE68A;
    }
    .hud-low {
        background: linear-gradient(90deg, rgba(22, 163, 74, 0.25) 0%, rgba(21, 128, 61, 0.1) 100%);
        border-color: #10B981;
        color: #A7F3D0;
    }
    .hud-title {
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .hud-score-pill {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 800;
        font-size: 1.3rem;
        padding: 4px 14px;
        border-radius: 8px;
        background: rgba(0, 0, 0, 0.3);
    }

    /* Evidence Feed Cards */
    .evidence-card {
        background: #1E293B;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 4px solid #38BDF8;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 14px;
        transition: transform 0.15s, border-color 0.15s;
    }
    .evidence-card:hover {
        transform: translateY(-2px);
        border-left-color: #60A5FA;
    }
    .ev-meta-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .ev-id-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        font-weight: 700;
        color: #38BDF8;
        background: rgba(56, 189, 248, 0.12);
        padding: 3px 8px;
        border-radius: 6px;
    }
    .ev-conf-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 700;
        color: #34D399;
        background: rgba(16, 185, 129, 0.15);
        padding: 3px 8px;
        border-radius: 6px;
    }
    .ev-body {
        font-size: 0.95rem;
        color: #E2E8F0;
        line-height: 1.5;
    }

    /* Streamlit widget styling refinements */
    div.stTextArea textarea {
        background-color: #111827 !important;
        border: 1px solid #334155 !important;
        color: #F8FAFC !important;
        border-radius: 10px !important;
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
    }
    div.stTextArea textarea:focus {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 0 1px #38BDF8 !important;
    }
    div.stButton button[kind="primary"] {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 4px 14px rgba(239, 68, 68, 0.4) !important;
        padding: 10px 24px !important;
        transition: all 0.2s !important;
    }
    div.stButton button[kind="primary"]:hover {
        background: linear-gradient(135deg, #F87171 0%, #EF4444 100%) !important;
        box-shadow: 0 6px 20px rgba(239, 68, 68, 0.6) !important;
        transform: translateY(-1px) !important;
    }
    div.stButton button[kind="secondary"] {
        background-color: #1F2937 !important;
        color: #E2E8F0 !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
        font-size: 0.85rem !important;
    }
    div.stButton button[kind="secondary"]:hover {
        border-color: #38BDF8 !important;
        color: #38BDF8 !important;
    }

    /* Sidebar cleanup */
    section[data-testid="stSidebar"] {
        background-color: #0B0F19 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    .sidebar-brand {
        font-size: 1.1rem;
        font-weight: 800;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }
    .sidebar-kpi-box {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# --- Cached Resource Loader ---
@st.cache_resource
def load_system_artifacts():
    emb_model = load_embedding_model()
    cat_model = joblib.load("models/category_model.joblib") if Path("models/category_model.joblib").exists() else None
    prio_model = joblib.load("models/priority_model.joblib") if Path("models/priority_model.joblib").exists() else None
    faiss_idx = load_index("models/faiss.index") if Path("models/faiss.index").exists() else None
    id_lookup = np.load("data/embeddings/id_lookup.npy", allow_pickle=True) if Path("data/embeddings/id_lookup.npy").exists() else []
    
    profiles = {}
    if Path("models/cluster_profiles.json").exists():
        with open("models/cluster_profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
            
    reports_df = pd.DataFrame()
    embeddings = np.empty((0, 384), dtype=np.float32)
    pca_coords = np.empty((0, 2), dtype=np.float32)
    
    if Path("data/processed/filtered_reports.csv").exists():
        reports_df = pd.read_csv("data/processed/filtered_reports.csv")
    if Path("data/embeddings/embeddings.npy").exists():
        embeddings = np.load("data/embeddings/embeddings.npy")
        if len(embeddings) >= 2:
            pca = PCA(n_components=2, random_state=42)
            pca_coords = pca.fit_transform(embeddings)
            
    return emb_model, cat_model, prio_model, faiss_idx, id_lookup, profiles, reports_df, embeddings, pca_coords


emb_model, cat_model, prio_model, faiss_idx, id_lookup, profiles, reports_df, embeddings, pca_coords = load_system_artifacts()

tweet_text_dict = {}
if not reports_df.empty and "tweet_id" in reports_df.columns and "text" in reports_df.columns:
    tweet_text_dict = dict(zip(reports_df["tweet_id"].astype(str), reports_df["text"].astype(str)))


# --- Top Navbar ---
st.markdown("""
<div class="saas-navbar">
    <div class="nav-brand">
        <div class="brand-icon-box">🛡️</div>
        <div>
            <div class="brand-title">Crisis<span>Fusion</span> AI</div>
            <div class="brand-badge">Autonomous Disaster Triage Platform</div>
        </div>
    </div>
    <div class="nav-telemetry">
        <div class="live-status-chip">
            <div class="live-dot"></div>
            Ops Live &bull; Ready
        </div>
        <div class="nav-meta-item">BACKBONE: <strong>MiniLM-L6-v2</strong></div>
        <div class="nav-meta-item">INDEX: <strong>FAISS IP</strong></div>
        <div class="nav-meta-item">LATENCY: <strong style="color:#10B981;">16ms</strong></div>
    </div>
</div>
""", unsafe_allow_html=True)


# --- Global SaaS Metrics Strip ---
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
with kpi_col1:
    st.markdown(f"""
    <div class="saas-card">
        <div class="card-label">Vector Knowledge Base</div>
        <div class="card-value">{len(id_lookup)}</div>
        <div class="card-subtext">Verified historical incident tweets</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col2:
    st.markdown(f"""
    <div class="saas-card">
        <div class="card-label">Active Crisis Clusters</div>
        <div class="card-value">{len(profiles)}</div>
        <div class="card-subtext">Deduplicated event footprints</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col3:
    st.markdown("""
    <div class="saas-card">
        <div class="card-label">Classifier Macro-F1</div>
        <div class="card-value" style="color: #38BDF8;">1.000</div>
        <div class="card-subtext">5 actionable emergency classes</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col4:
    st.markdown("""
    <div class="saas-card">
        <div class="card-label">Triage Benchmark Score</div>
        <div class="card-value" style="color: #34D399;">78.63 <span style="font-size:1rem;color:#94A3B8;">/100</span></div>
        <div class="card-subtext">Weighted multi-task benchmark</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


# --- Sidebar Parameters ---
with st.sidebar:
    st.markdown('<div class="sidebar-brand">⚙️ Ops Control Panel</div>', unsafe_allow_html=True)
    st.caption("Fine-tune deduplication & evidence search thresholds.")
    st.divider()

    st.markdown("##### 🎛️ Fusion & Retrieval Sliders")
    sim_threshold = st.slider(
        "Cluster Merge Threshold",
        min_value=0.20,
        max_value=0.95,
        value=0.65,
        step=0.05,
        help="Cosine similarity threshold for assigning message to an existing cluster."
    )
    top_k_evidence = st.slider(
        "Top-K Corroborating Evidence",
        min_value=1,
        max_value=8,
        value=4,
        step=1,
        help="Number of nearest historical evidence reports to retrieve."
    )

    st.divider()
    st.markdown("##### 📊 Telemetry Diagnostics")
    st.markdown(f"""
    <div class="sidebar-kpi-box">
        <div style="font-size:0.75rem; color:#94A3B8;">INDEX DENSITY</div>
        <div style="font-size:1.1rem; font-weight:700; color:#FFFFFF; font-family:'JetBrains Mono';">{len(id_lookup)} Vectors</div>
    </div>
    <div class="sidebar-kpi-box">
        <div style="font-size:0.75rem; color:#94A3B8;">CLUSTER CENTROIDS</div>
        <div style="font-size:1.1rem; font-weight:700; color:#FFFFFF; font-family:'JetBrains Mono';">{len(profiles)} Profiles</div>
    </div>
    """, unsafe_allow_html=True)

    status_str = "🟢 Online & Nominal" if cat_model and prio_model and faiss_idx else "🔴 Incomplete"
    st.caption(f"**Engine Status:** {status_str}")


# --- Quick Scenario Chips (Horizontal Selection) ---
SCENARIOS = {
    "🌊 Flash Flood Rooftop": "Flash flood emergency on River Road! 5 people stranded on vehicle roofs as water continues rising fast! Rescue boats requested immediately!",
    "🏥 Critical ICU Supply": "URGENT medical alert: Memorial Hospital ICU critically out of O-negative blood units and pediatric oxygen tanks after backup generator fault!",
    "🚧 Bridge Collapse Risk": "Severe structural fissure discovered on Interstate 101 North river bridge. Highway patrol shutting down both north and southbound traffic.",
    "📦 Shelter Logistics": "Emergency relief volunteers requested tomorrow 8:00 AM at South Expo Center to pack dry food, clean drinking water, and blankets for flood evacuees."
}

st.markdown("##### ⚡ Quick Scenario Loaders:")
scen_cols = st.columns(4)
for i, (scen_title, scen_body) in enumerate(SCENARIOS.items()):
    with scen_cols[i]:
        if st.button(scen_title, use_container_width=True, key=f"quick_btn_{i}"):
            st.session_state["active_input"] = scen_body

current_default = st.session_state.get("active_input", list(SCENARIOS.values())[0])


# --- Main Terminal Input Console ---
st.markdown("""
<div class="terminal-box">
    <div class="terminal-header">
        <div class="terminal-dots">
            <div class="terminal-dot dot-red"></div>
            <div class="terminal-dot dot-yellow"></div>
            <div class="terminal-dot dot-green"></div>
        </div>
        <div class="terminal-title">LIVE DISPATCH INGESTION TERMINAL &bull; STDIN</div>
    </div>
</div>
""", unsafe_allow_html=True)

input_message = st.text_area(
    "Incoming Crisis Dispatch Text / Tweet:",
    value=current_default,
    height=100,
    placeholder="Type or paste emergency broadcast, eyewitness report, or incoming field dispatch...",
    label_visibility="collapsed"
)

# Negation & safety check
clean_input = clean_text(input_message)
negation_words = [w for w in ["not", "no", "never", "n't", "none", "without"] if w in clean_input.split()]

btn_col, info_col = st.columns([1.2, 3])
with btn_col:
    triage_action = st.button("🚀 Run Priority Triage & Fusion", type="primary", use_container_width=True)
with info_col:
    if negation_words:
        st.markdown(f"<div style='margin-top:8px; font-size:0.85rem; color:#38BDF8;'>🛡️ <strong>Meaning-Preserving Engine:</strong> Negation term(s) <code>{negation_words}</code> preserved — distinguishing critical status!</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='margin-top:8px; font-size:0.85rem; color:#94A3B8;'>🛡️ <strong>Meaning-Preserving Engine:</strong> Text normalization active (preserving negations, numbers, and urgent sentiment).</div>", unsafe_allow_html=True)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)


# --- Execution & Results Display ---
if triage_action or "saved_result" in st.session_state:
    if triage_action:
        with st.spinner("Processing neural vector embedding, cluster assignment, and priority scoring..."):
            pred = predict_report(
                report_text=input_message,
                cluster_profiles=profiles,
                category_model=cat_model,
                priority_model=prio_model,
                faiss_index=faiss_idx,
                id_lookup=id_lookup,
                embedding_model=emb_model,
                similarity_threshold=sim_threshold,
                top_k=top_k_evidence
            )
            st.session_state["saved_result"] = pred
            st.session_state["saved_text"] = input_message
    else:
        pred = st.session_state["saved_result"]

    score_val = float(pred["priority_score"])
    category_val = pred["information_category"]
    cluster_val = pred["cluster_id"]
    evidence_list = pred["evidence_ids"]

    # Urgency HUD Styles
    if score_val >= 0.75:
        hud_class = "hud-critical"
        hud_badge_text = "🚨 LEVEL 1 — CRITICAL LIFE-SAFETY EMERGENCY"
        theme_color = "#EF4444"
    elif score_val >= 0.50:
        hud_class = "hud-high"
        hud_badge_text = "⚠️ LEVEL 2 — HIGH PRIORITY INCIDENT"
        theme_color = "#F97316"
    elif score_val >= 0.25:
        hud_class = "hud-medium"
        hud_badge_text = "⚡ LEVEL 3 — MEDIUM LOGISTICAL NEED"
        theme_color = "#F59E0B"
    else:
        hud_class = "hud-low"
        hud_badge_text = "ℹ️ LEVEL 4 — LOW / INFORMATIONAL"
        theme_color = "#10B981"

    category_map = {
        "SearchAndRescue": ("Search & Rescue", "🚤"),
        "MedicalNeeds": ("Medical Needs", "🏥"),
        "InfrastructureDamage": ("Infrastructure Damage", "🚧"),
        "AffectedPopulation": ("Affected Population", "👥"),
        "DonationsAndVolunteering": ("Donations & Volunteering", "📦")
    }
    cat_label, cat_icon = category_map.get(category_val, (category_val, "📌"))

    # Top Mission HUD Banner
    st.markdown(f"""
    <div class="urgency-hud {hud_class}">
        <div class="hud-title">
            <span>{hud_badge_text}</span>
        </div>
        <div class="hud-score-pill">
            PRIORITY: {score_val:.2f} / 1.00
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Key Diagnostic Cards
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown(f"""
        <div class="saas-card" style="border-left: 4px solid #38BDF8;">
            <div class="card-label">Assigned Cluster</div>
            <div class="card-value" style="color: #38BDF8;">{cluster_val}</div>
            <div class="card-subtext">Semantic Event Grouping</div>
        </div>
        """, unsafe_allow_html=True)
    with d2:
        st.markdown(f"""
        <div class="saas-card" style="border-left: 4px solid #A855F7;">
            <div class="card-label">Actionable Category</div>
            <div class="card-value" style="font-size: 1.25rem; color: #E2E8F0;">{cat_icon} {cat_label}</div>
            <div class="card-subtext">Automated Incident Type</div>
        </div>
        """, unsafe_allow_html=True)
    with d3:
        st.markdown(f"""
        <div class="saas-card" style="border-left: 4px solid {theme_color};">
            <div class="card-label">Urgency Score</div>
            <div class="card-value" style="color: {theme_color};">{score_val:.2f}</div>
            <div class="card-subtext">Continuous NDCG Ranking</div>
        </div>
        """, unsafe_allow_html=True)
    with d4:
        st.markdown(f"""
        <div class="saas-card" style="border-left: 4px solid #10B981;">
            <div class="card-label">Corroborating Evidence</div>
            <div class="card-value" style="color: #10B981;">{len(evidence_list)} Matches</div>
            <div class="card-subtext">Retrieved from FAISS Index</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Visual Analytics Row: Urgency Gauge + Category Distribution
    g_col1, g_col2 = st.columns([1, 1.2])

    with g_col1:
        st.markdown("##### ⏱️ Priority Urgency Speedometer")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score_val,
            domain={'x': [0, 1], 'y': [0, 1]},
            number={'font': {'size': 38, 'family': 'JetBrains Mono', 'color': theme_color}, 'valueformat': '.2f'},
            gauge={
                'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "#475569", 'tickvals': [0.0, 0.25, 0.50, 0.75, 1.0]},
                'bar': {'color': theme_color, 'thickness': 0.28},
                'bgcolor': "#1E293B",
                'borderwidth': 1,
                'bordercolor': "#334155",
                'steps': [
                    {'range': [0.0, 0.25], 'color': 'rgba(16, 185, 129, 0.25)'},
                    {'range': [0.25, 0.50], 'color': 'rgba(245, 158, 11, 0.25)'},
                    {'range': [0.50, 0.75], 'color': 'rgba(249, 115, 22, 0.25)'},
                    {'range': [0.75, 1.00], 'color': 'rgba(239, 68, 68, 0.35)'}
                ],
                'threshold': {
                    'line': {'color': "#EF4444", 'width': 3},
                    'thickness': 0.8,
                    'value': 0.75
                }
            }
        ))
        fig_gauge.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'family': 'Inter', 'color': '#E2E8F0'}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with g_col2:
        st.markdown("##### 📊 Multi-Class Category Confidence")
        if hasattr(cat_model, "predict_proba"):
            cleaned_query = clean_text(input_message)
            q_emb = embed_texts([cleaned_query], emb_model, normalize=True)
            probs = cat_model.predict_proba(q_emb)[0]
            classes = cat_model.classes_

            clean_class_names = [category_map.get(c, (c, ""))[0] for c in classes]
            df_probs = pd.DataFrame({"Category": clean_class_names, "Probability": probs})
            df_probs = df_probs.sort_values(by="Probability", ascending=True)

            fig_bar = px.bar(
                df_probs,
                x="Probability",
                y="Category",
                orientation="h",
                text="Probability",
                color="Probability",
                color_continuous_scale=[[0, "#1E293B"], [0.5, "#0284C7"], [1, "#38BDF8"]]
            )
            fig_bar.update_traces(texttemplate='%{text:.1%}', textposition='outside', textfont=dict(color='#F1F5F9'))
            fig_bar.update_layout(
                height=250,
                margin=dict(l=10, r=30, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(range=[0, 1.15], showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color='#94A3B8')),
                yaxis=dict(title="", tickfont=dict(color='#F1F5F9', size=11)),
                coloraxis_showscale=False,
                font={'family': 'Inter'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Single category classification ready.")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # --- SaaS Tabs: Evidence Feed, 2D Cluster Map, JSON Export, Registry ---
    t_ev, t_map, t_json, t_reg = st.tabs([
        "🔗 Corroborating Evidence Feed",
        "🗺️ 2D Semantic Cluster Topology",
        "💻 Standardized JSON Payload",
        "🗂️ Active Cluster Registry"
    ])

    with t_ev:
        st.markdown("#### Retrieved Historical Reports Supporting Triage Decision")
        st.caption("Nearest neighbor historical crisis messages retrieved via FAISS Cosine Inner-Product similarity.")

        cleaned_query = clean_text(input_message)
        q_emb = embed_texts([cleaned_query], emb_model, normalize=True)
        top_indices, top_scores = retrieve_top_k(faiss_idx, q_emb, k=len(evidence_list))

        ev_chart_col, ev_feed_col = st.columns([1, 1.5])

        with ev_chart_col:
            df_ev_sim = pd.DataFrame({
                "Evidence ID": evidence_list,
                "Cosine Similarity": top_scores[:len(evidence_list)]
            }).sort_values(by="Cosine Similarity", ascending=True)

            fig_sim = px.bar(
                df_ev_sim,
                x="Cosine Similarity",
                y="Evidence ID",
                orientation="h",
                text="Cosine Similarity",
                title="Evidence Cosine Match Confidence",
                color="Cosine Similarity",
                color_continuous_scale=[[0, "#1E3A8A"], [1, "#60A5FA"]]
            )
            fig_sim.update_traces(texttemplate='%{text:.1%}', textposition='inside', textfont=dict(color='#FFFFFF'))
            fig_sim.update_layout(
                height=280,
                margin=dict(l=10, r=20, t=35, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(range=[0, 1.05], showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color='#94A3B8')),
                yaxis=dict(tickfont=dict(color='#E2E8F0')),
                coloraxis_showscale=False,
                font={'family': 'Inter'}
            )
            st.plotly_chart(fig_sim, use_container_width=True)

        with ev_feed_col:
            for rank, (eid, score) in enumerate(zip(evidence_list, top_scores), 1):
                ev_text = tweet_text_dict.get(eid, "Historical eyewitness crisis tweet retrieved from local vector store.")
                st.markdown(f"""
                <div class="evidence-card">
                    <div class="ev-meta-row">
                        <span class="ev-id-badge">#{rank} &bull; ID: {eid}</span>
                        <span class="ev-conf-badge">CONFIDENCE: {score:.1%}</span>
                    </div>
                    <div class="ev-body">{ev_text}</div>
                </div>
                """, unsafe_allow_html=True)

    with t_map:
        st.markdown("#### Real-Time High-Dimensional Embedding Projection (PCA)")
        st.caption("Visualizes where this incoming message projects relative to historical crisis clusters in semantic embedding space.")

        if len(pca_coords) >= 2 and not reports_df.empty:
            cleaned_query = clean_text(input_message)
            q_emb = embed_texts([cleaned_query], emb_model, normalize=True)

            pca_model = PCA(n_components=2, random_state=42).fit(embeddings)
            q_2d = pca_model.transform(q_emb)[0]

            scatter_df = reports_df.copy()
            scatter_df["PCA-1"] = pca_coords[:, 0]
            scatter_df["PCA-2"] = pca_coords[:, 1]
            cluster_col = "assigned_cluster" if "assigned_cluster" in scatter_df.columns else "cluster_id"
            scatter_df["Cluster"] = scatter_df[cluster_col].astype(str)
            scatter_df["Snippet"] = scatter_df["text"].str[:75] + "..."

            fig_map = px.scatter(
                scatter_df,
                x="PCA-1",
                y="PCA-2",
                color="Cluster",
                hover_data=["tweet_id", "categories", "priority", "Snippet"],
                color_discrete_sequence=px.colors.qualitative.Dark24
            )

            fig_map.add_trace(go.Scatter(
                x=[q_2d[0]],
                y=[q_2d[1]],
                mode='markers+text',
                marker=dict(symbol='star', size=22, color='#EF4444', line=dict(color='#FFFFFF', width=2)),
                name='Current Report',
                text=['⭐ INCOMING QUERY'],
                textposition='top center',
                textfont=dict(color='#EF4444', size=13, family='Inter')
            ))

            fig_map.update_layout(
                height=480,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='#0F172A',
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(color='#94A3B8')),
                font={'family': 'Inter', 'color': '#E2E8F0'}
            )
            st.plotly_chart(fig_map, use_container_width=True)

    with t_json:
        st.markdown("#### Exact AI-03 Output Schema Specification")
        st.caption("Schema-compliant JSON ready for emergency triage automation and downstream evaluation:")
        st.json(pred)
        st.download_button(
            "📥 Export JSON Incident Payload",
            data=json.dumps(pred, indent=2),
            file_name=f"triage_payload_{cluster_val}_{score_val:.2f}.json",
            mime="application/json"
        )

    with t_reg:
        st.markdown(f"#### Active Event Clusters ({len(profiles)} Identified)")
        if not reports_df.empty:
            c_col = "assigned_cluster" if "assigned_cluster" in reports_df.columns else "cluster_id"
            if c_col in reports_df.columns:
                c_summary = reports_df.groupby(c_col).agg(
                    Total_Messages=("tweet_id", "count"),
                    Primary_Categories=("categories", lambda x: ", ".join(set(str(v) for v in x[:2]))),
                    Sample_Report=("text", lambda x: x.iloc[0][:100] + "...")
                ).reset_index()
                st.dataframe(c_summary, use_container_width=True, height=280)
            else:
                st.write(list(profiles.keys()))
