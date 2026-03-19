import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Placement And Salary Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLES
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0c14; color: #e8eaf0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem 2.5rem; max-width: 1400px; }

[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2535 !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem; }

.sidebar-logo {
    font-family: 'Syne', sans-serif; font-size: 1.3rem; font-weight: 800;
    background: linear-gradient(90deg, #a5b4fc, #e879f9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.sidebar-tagline { font-size: 0.72rem; color: #4b5563; margin-bottom: 1.5rem; }
.sidebar-divider { border: none; border-top: 1px solid #1e2535; margin: 1.2rem 0; }
.sidebar-stat {
    background: #111827; border: 1px solid #1e2535;
    border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 0.6rem;
}
.sidebar-stat-label { font-size: 0.68rem; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; }
.sidebar-stat-value { font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700; color: #e5e7eb; }

.hero-wrap {
    background: linear-gradient(135deg, #0d1117 0%, #111827 50%, #0d1117 100%);
    border: 1px solid #1e2535; border-radius: 20px;
    padding: 2.2rem 2.8rem; margin-bottom: 2rem;
    position: relative; overflow: hidden;
}
.hero-wrap::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-row { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem; }
.hero-right { display: flex; gap: 1.2rem; flex-wrap: wrap; }
.hero-stat-pill {
    background: rgba(255,255,255,0.04); border: 1px solid #1e2535;
    border-radius: 12px; padding: 0.7rem 1.2rem; text-align: center; min-width: 90px;
}
.hero-stat-num { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 800; color: #f9fafb; line-height: 1; }
.hero-stat-lbl { font-size: 0.65rem; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; margin-top: 3px; }
.hero-title {
    font-family: 'Syne', sans-serif; font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(90deg, #a5b4fc 0%, #e879f9 50%, #34d399 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem 0; line-height: 1.1;
}
.hero-sub { font-size: 0.9rem; color: #6b7280; font-weight: 300; }
.badge {
    display: inline-block; background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.35); color: #a5b4fc;
    font-size: 0.68rem; font-weight: 600; letter-spacing: 1.5px;
    text-transform: uppercase; padding: 3px 10px; border-radius: 20px; margin-bottom: 0.8rem;
}

.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2rem; }
.kpi-card {
    background: #111827; border: 1px solid #1e2535; border-radius: 16px;
    padding: 1.4rem 1.6rem; position: relative; overflow: hidden;
    transition: border-color 0.25s, transform 0.2s;
}
.kpi-card:hover { border-color: #374151; transform: translateY(-2px); }
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 3px; border-radius: 16px 16px 0 0;
}
.kpi-card.indigo::before  { background: linear-gradient(90deg, #6366f1, #8b5cf6); }
.kpi-card.emerald::before { background: linear-gradient(90deg, #10b981, #34d399); }
.kpi-card.violet::before  { background: linear-gradient(90deg, #8b5cf6, #ec4899); }
.kpi-card.amber::before   { background: linear-gradient(90deg, #f59e0b, #f97316); }
.kpi-top { display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 0.4rem; }
.kpi-icon {
    width: 38px; height: 38px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center; font-size: 1.1rem;
}
.kpi-icon.indigo  { background: rgba(99,102,241,0.15); }
.kpi-icon.emerald { background: rgba(16,185,129,0.15); }
.kpi-icon.violet  { background: rgba(139,92,246,0.15); }
.kpi-icon.amber   { background: rgba(245,158,11,0.15); }
.kpi-label { font-size: 0.68rem; font-weight: 600; letter-spacing: 1.4px; text-transform: uppercase; color: #6b7280; margin-bottom: 0.4rem; }
.kpi-value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #f9fafb; line-height: 1; }
.kpi-delta { font-size: 0.75rem; color: #34d399; margin-top: 0.4rem; }

.sec-head {
    font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700;
    color: #e8eaf0; margin: 0 0 1.2rem 0; display: flex; align-items: center; gap: 0.5rem;
}
.sec-head span { font-size: 0.68rem; font-weight: 500; letter-spacing: 1px; color: #4b5563; text-transform: uppercase; }

.chart-box {
    background: #111827; border: 1px solid #1e2535;
    border-radius: 16px; padding: 1.4rem 1.6rem; height: 100%;
}

.stTabs [data-baseweb="tab-list"] {
    background: #111827; border-radius: 12px;
    padding: 4px; gap: 4px; border: 1px solid #1e2535;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif; font-weight: 500;
    font-size: 0.88rem; color: #6b7280; border-radius: 9px; padding: 8px 20px; border: none;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important; color: #fff !important;
}

.stSlider > div > div > div > div { background: #6366f1; }

.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #fff; border: none; border-radius: 10px;
    font-family: 'DM Sans', sans-serif; font-weight: 600;
    font-size: 0.9rem; padding: 0.65rem 2rem; width: 100%;
    transition: opacity 0.2s, transform 0.15s;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

[data-testid="stMetric"] { background: #111827; border: 1px solid #1e2535; border-radius: 14px; padding: 1rem 1.2rem; }
[data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { font-family: 'Syne', sans-serif !important; color: #f9fafb !important; font-size: 1.6rem !important; }

.stDataFrame { border-radius: 12px; overflow: hidden; }
.streamlit-expanderHeader { background: #111827 !important; border-radius: 10px !important; color: #9ca3af !important; }
hr { border-color: #1e2535 !important; }

.stProgress > div > div > div > div { background: linear-gradient(90deg, #6366f1, #ec4899) !important; border-radius: 10px; }
.stProgress > div > div > div { background: #1e2535 !important; border-radius: 10px; }

.result-card {
    background: linear-gradient(135deg, #0f1923, #111827);
    border: 1px solid #1e2535; border-radius: 16px; padding: 1.6rem; text-align: center;
}
.result-placed { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800; color: #34d399; }
.result-not    { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800; color: #f87171; }
.result-label  { font-size: 0.8rem; color: #6b7280; margin-top: 0.3rem; }

.skill-row { margin-bottom: 0.9rem; }
.skill-label { display: flex; justify-content: space-between; font-size: 0.82rem; color: #9ca3af; margin-bottom: 0.3rem; }
.skill-track { background: #1e2535; border-radius: 6px; height: 7px; }
.skill-fill  { height: 7px; border-radius: 6px; }

.gauge-wrap {
    background: #0d1117; border: 1px solid #1e2535;
    border-radius: 16px; padding: 1.2rem; text-align: center; margin-bottom: 1rem;
}
.gauge-title { font-size: 0.72rem; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.8rem; }
.gauge-pct { font-family: 'Syne', sans-serif; font-size: 2.6rem; font-weight: 800; line-height: 1; }
.gauge-pct.high { color: #34d399; }
.gauge-pct.mid  { color: #f59e0b; }
.gauge-pct.low  { color: #f87171; }
.gauge-sub { font-size: 0.75rem; color: #6b7280; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# MATPLOTLIB DARK THEME
# ============================================================
plt.rcParams.update({
    "figure.facecolor": "#111827",
    "axes.facecolor"  : "#111827",
    "axes.edgecolor"  : "#1e2535",
    "axes.labelcolor" : "#9ca3af",
    "axes.titlecolor" : "#e5e7eb",
    "axes.titlesize"  : 10,
    "axes.labelsize"  : 8,
    "xtick.color"     : "#6b7280",
    "ytick.color"     : "#6b7280",
    "xtick.labelsize" : 7,
    "ytick.labelsize" : 7,
    "grid.color"      : "#1e2535",
    "grid.linewidth"  : 0.6,
    "text.color"      : "#9ca3af",
    "font.family"     : "DejaVu Sans",
    "legend.facecolor": "#111827",
    "legend.edgecolor": "#1e2535",
})

COLORS = ["#6366f1", "#8b5cf6", "#ec4899", "#10b981", "#f59e0b", "#3b82f6"]

# ============================================================
# LOAD DATA AND MODELS
# ============================================================
@st.cache_data(show_spinner=False)
def load_data():
    try:
        conn = sqlite3.connect("placement.db")
        df   = pd.read_sql("SELECT * FROM student_data", conn)
        conn.close()
        return df
    except Exception as e:
        st.error("Could not load database: " + str(e))
        st.stop()

@st.cache_data(show_spinner=False)
def load_perf():
    try:
        return pd.read_csv("model_performance.csv")
    except Exception as e:
        st.error("Could not load model_performance.csv: " + str(e))
        st.stop()

@st.cache_data(show_spinner=False)
def load_salary_perf():
    if os.path.exists("salary_performance.csv"):
        return pd.read_csv("salary_performance.csv")
    return None

@st.cache_data(show_spinner=False)
def load_history():
    if os.path.exists("model_history.csv"):
        return pd.read_csv("model_history.csv")
    return None

@st.cache_resource(show_spinner=False)
def load_models():
    missing = []
    if not os.path.exists("models/best_model.pkl") : missing.append("models/best_model.pkl")
    if not os.path.exists("models/salary_model.pkl"): missing.append("models/salary_model.pkl")
    if not os.path.exists("scaler.pkl")             : missing.append("scaler.pkl")

    if missing:
        st.error("Missing files: " + str(missing) + "\n\nPlease run train.py first.")
        st.stop()

    placement_model = pickle.load(open("models/best_model.pkl",  "rb"))
    salary_model    = pickle.load(open("models/salary_model.pkl","rb"))
    scaler          = pickle.load(open("scaler.pkl",              "rb"))
    return placement_model, salary_model, scaler

df                              = load_data()
perf                            = load_perf()
salary_perf                     = load_salary_perf()
history                         = load_history()
placement_model, salary_model, scaler = load_models()

# ============================================================
# BASIC STATS
# ============================================================
features = [
    "cgpa",
    "internships_completed",
    "projects_completed",
    "coding_skill_rating",
    "communication_skill_rating",
    "aptitude_skill_rating"
]

# Handle placement_status whether it is string or number
if df["placement_status"].dtype == object:
    placed_mask = df["placement_status"] == "Placed"
else:
    placed_mask = df["placement_status"] == 1

total        = len(df)
placed_count = int(placed_mask.sum())
placed_pct   = round(placed_count / total * 100, 1)
avg_cgpa     = round(df["cgpa"].mean(), 2)
avg_salary   = round(df[placed_mask]["salary_lpa"].mean(), 2)
best_acc     = round(perf["Accuracy"].max() * 100, 1)
best_name    = perf.loc[perf["Accuracy"].idxmax(), "Model"]

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🎓 PlaceIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Placement & Salary Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown("**📂 Dataset Overview**")
    st.markdown(f"""
    <div class="sidebar-stat">
      <div class="sidebar-stat-label">Total Students</div>
      <div class="sidebar-stat-value">{total:,}</div>
    </div>
    <div class="sidebar-stat">
      <div class="sidebar-stat-label">Placement Rate</div>
      <div class="sidebar-stat-value">{placed_pct}%</div>
    </div>
    <div class="sidebar-stat">
      <div class="sidebar-stat-label">Avg CGPA</div>
      <div class="sidebar-stat-value">{avg_cgpa}</div>
    </div>
    <div class="sidebar-stat">
      <div class="sidebar-stat-label">Avg Salary (Placed)</div>
      <div class="sidebar-stat-value">₹ {avg_salary} LPA</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown("**🤖 Active Model**")
    st.markdown(f"""
    <div class="sidebar-stat">
      <div class="sidebar-stat-label">Best Model</div>
      <div class="sidebar-stat-value">{best_name}</div>
    </div>
    <div class="sidebar-stat">
      <div class="sidebar-stat-label">Accuracy</div>
      <div class="sidebar-stat-value">{best_acc}%</div>
    </div>
    """, unsafe_allow_html=True)

    if history is not None and len(history) >= 1:
        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
        st.markdown("**🔄 Retrain History**")
        st.markdown(
            f'<div style="font-size:0.78rem;color:#6b7280;padding:3px 0;">'
            f'Retrained {len(history)} time(s)</div>',
            unsafe_allow_html=True
        )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown("**⚙️ Features Used**")
    for f in ["CGPA", "Internships", "Projects", "Coding Skill", "Communication", "Aptitude"]:
        st.markdown(f'<div style="font-size:0.78rem;color:#6b7280;padding:3px 0;">✓ &nbsp;{f}</div>',
                    unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.68rem;color:#374151;text-align:center;margin-top:2rem;">v2.0 · 2025</div>',
                unsafe_allow_html=True)

# ============================================================
# HERO
# ============================================================
st.markdown(f"""
<div class="hero-wrap">
  <div class="hero-row">
    <div>
      <div class="badge">AI-Powered Analytics</div>
      <div class="hero-title">Placement & Salary<br>Prediction System</div>
      <div class="hero-sub">Data analytics · ML models · Career outcome prediction</div>
    </div>
    <div class="hero-right">
      <div class="hero-stat-pill">
        <div class="hero-stat-num">{total:,}</div>
        <div class="hero-stat-lbl">Students</div>
      </div>
      <div class="hero-stat-pill">
        <div class="hero-stat-num">{placed_pct}%</div>
        <div class="hero-stat-lbl">Placed</div>
      </div>
      <div class="hero-stat-pill">
        <div class="hero-stat-num">{best_acc}%</div>
        <div class="hero-stat-lbl">Accuracy</div>
      </div>
      <div class="hero-stat-pill">
        <div class="hero-stat-num">₹{avg_salary}</div>
        <div class="hero-stat-lbl">Avg Salary</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# KPI CARDS
# ============================================================
st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card indigo">
    <div class="kpi-top">
      <div>
        <div class="kpi-label">Total Students</div>
        <div class="kpi-value">{total:,}</div>
        <div class="kpi-delta">↑ Active cohort</div>
      </div>
      <div class="kpi-icon indigo">👥</div>
    </div>
  </div>
  <div class="kpi-card emerald">
    <div class="kpi-top">
      <div>
        <div class="kpi-label">Placement Rate</div>
        <div class="kpi-value">{placed_pct}%</div>
        <div class="kpi-delta">↑ {placed_count} students placed</div>
      </div>
      <div class="kpi-icon emerald">🎯</div>
    </div>
  </div>
  <div class="kpi-card violet">
    <div class="kpi-top">
      <div>
        <div class="kpi-label">Avg CGPA</div>
        <div class="kpi-value">{avg_cgpa}</div>
        <div class="kpi-delta">Across cohort</div>
      </div>
      <div class="kpi-icon violet">📊</div>
    </div>
  </div>
  <div class="kpi-card amber">
    <div class="kpi-top">
      <div>
        <div class="kpi-label">Best Accuracy</div>
        <div class="kpi-value">{best_acc}%</div>
        <div class="kpi-delta">↑ {best_name}</div>
      </div>
      <div class="kpi-icon amber">🏆</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "  📊  Analytics  ",
    "  🤖  Placement Model  ",
    "  💰  Salary Model  ",
    "  🎯  Predict  "
])

# ============================================================
# TAB 1 — ANALYTICS
# ============================================================
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)

    df_plot = df.copy()
    if df_plot["placement_status"].dtype == object:
        df_plot["placement_num"] = df_plot["placement_status"].map({"Placed": 1, "Not Placed": 0})
    else:
        df_plot["placement_num"] = df_plot["placement_status"]

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown('<p class="sec-head">Placement Distribution <span>count</span></p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4.5, 2.8))
        counts  = df_plot["placement_num"].value_counts()
        bars    = ax.bar(
            ["Not Placed", "Placed"],
            [counts.get(0, 0), counts.get(1, 0)],
            color=["#f87171", "#34d399"], width=0.5, zorder=2
        )
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.005,
                str(int(bar.get_height())),
                ha="center", va="bottom", fontsize=8, color="#e5e7eb", fontweight="bold"
            )
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown('<p class="sec-head">CGPA Distribution <span>kde</span></p>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(4.5, 2.8))
        sns.histplot(df["cgpa"], kde=True, ax=ax2, color="#6366f1", alpha=0.55, edgecolor="none", bins=20)
        ax2.lines[0].set_color("#a5b4fc")
        ax2.lines[0].set_linewidth(2)
        ax2.spines[["top", "right", "left"]].set_visible(False)
        ax2.set_axisbelow(True)
        ax2.yaxis.grid(True)
        ax2.set_xlabel("CGPA", fontsize=8)
        st.pyplot(fig2)
        plt.close(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c3, c4 = st.columns([1.1, 0.9], gap="medium")

    with c3:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown('<p class="sec-head">Feature Correlation Heatmap <span>pearson</span></p>', unsafe_allow_html=True)
        num_cols = features + ["salary_lpa"]
        fig3, ax3 = plt.subplots(figsize=(5.5, 3.4))
        sns.heatmap(
            df_plot[num_cols].corr(),
            cmap=sns.diverging_palette(240, 10, as_cmap=True),
            annot=True, fmt=".2f", annot_kws={"size": 6},
            linewidths=0.5, linecolor="#0a0c14",
            ax=ax3, cbar_kws={"shrink": 0.7}
        )
        ax3.tick_params(axis="x", rotation=35)
        st.pyplot(fig3)
        plt.close(fig3)
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown('<p class="sec-head">CGPA by Placement <span>boxplot</span></p>', unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(4, 3.4))
        bp = ax4.boxplot(
            [df_plot[df_plot["placement_num"] == 0]["cgpa"],
             df_plot[df_plot["placement_num"] == 1]["cgpa"]],
            labels=["Not Placed", "Placed"],
            patch_artist=True,
            medianprops=dict(color="#f9fafb", linewidth=2),
            whiskerprops=dict(color="#4b5563"),
            capprops=dict(color="#4b5563"),
            flierprops=dict(marker="o", markerfacecolor="#374151", markersize=3, markeredgecolor="none")
        )
        for patch, color in zip(bp["boxes"], ["#f87171", "#34d399"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax4.spines[["top", "right", "left"]].set_visible(False)
        ax4.yaxis.grid(True)
        ax4.set_axisbelow(True)
        st.pyplot(fig4)
        plt.close(fig4)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c5, c6 = st.columns(2, gap="medium")

    with c5:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown('<p class="sec-head">Salary Distribution <span>placed students only</span></p>', unsafe_allow_html=True)
        fig5, ax5 = plt.subplots(figsize=(4.5, 2.8))
        placed_salaries = df_plot[df_plot["placement_num"] == 1]["salary_lpa"]
        sns.histplot(placed_salaries, kde=True, ax=ax5, color="#10b981", alpha=0.55, edgecolor="none", bins=20)
        ax5.lines[0].set_color("#34d399")
        ax5.lines[0].set_linewidth(2)
        ax5.spines[["top", "right", "left"]].set_visible(False)
        ax5.set_axisbelow(True)
        ax5.yaxis.grid(True)
        ax5.set_xlabel("Salary (LPA)", fontsize=8)
        st.pyplot(fig5)
        plt.close(fig5)
        st.markdown('</div>', unsafe_allow_html=True)

    with c6:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown('<p class="sec-head">Skill Ratings Comparison <span>placed vs not placed</span></p>', unsafe_allow_html=True)
        skill_cols  = ["coding_skill_rating", "communication_skill_rating", "aptitude_skill_rating"]
        placed_g    = df_plot[df_plot["placement_num"] == 1]
        notplaced_g = df_plot[df_plot["placement_num"] == 0]
        fig6, ax6   = plt.subplots(figsize=(4.5, 2.8))
        x = np.arange(3)
        w = 0.32
        ax6.bar(x - w/2, [notplaced_g[c].mean() for c in skill_cols], width=w, color="#f87171", alpha=0.85, label="Not Placed")
        ax6.bar(x + w/2, [placed_g[c].mean()    for c in skill_cols], width=w, color="#34d399", alpha=0.85, label="Placed")
        ax6.set_xticks(x)
        ax6.set_xticklabels(["Coding", "Communication", "Aptitude"], fontsize=7)
        ax6.set_ylim(0, 10)
        ax6.yaxis.grid(True)
        ax6.set_axisbelow(True)
        ax6.spines[["top", "right", "left"]].set_visible(False)
        ax6.legend(frameon=False, fontsize=8)
        st.pyplot(fig6)
        plt.close(fig6)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB 2 — PLACEMENT MODEL
# ============================================================
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)

    best_row = perf.loc[perf["Accuracy"].idxmax()].to_dict()

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0f172a,#1e1b4b);
                border:1px solid rgba(99,102,241,0.4); border-radius:16px;
                padding:1.4rem 2rem; margin-bottom:1.5rem;
                display:flex; align-items:center; gap:1.5rem;">
      <div style="font-size:2.5rem">🏆</div>
      <div>
        <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;color:#a5b4fc;">
          {best_row['Model']}
        </div>
        <div style="font-size:0.82rem;color:#6b7280;margin-top:0.2rem;">
          Best placement model · {round(best_row['Accuracy']*100, 2)}% accuracy
        </div>
      </div>
      <div style="margin-left:auto;text-align:right;">
        <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;
                    background:linear-gradient(90deg,#6366f1,#a78bfa);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
          {round(best_row['Accuracy']*100, 2)}%
        </div>
        <div style="font-size:0.72rem;color:#6b7280;letter-spacing:1px;text-transform:uppercase;">Accuracy</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 0.8], gap="medium")

    with c1:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown('<p class="sec-head">Model Accuracy Comparison</p>', unsafe_allow_html=True)
        sorted_perf = perf.sort_values("Accuracy", ascending=True)
        fig7, ax7   = plt.subplots(figsize=(5.5, 3.2))
        bars7       = ax7.barh(
            sorted_perf["Model"], sorted_perf["Accuracy"],
            color=[COLORS[i % len(COLORS)] for i in range(len(sorted_perf))],
            alpha=0.85, height=0.55
        )
        for bar, val in zip(bars7, sorted_perf["Accuracy"]):
            ax7.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                     str(round(val, 3)), va="center", fontsize=7.5, color="#e5e7eb", fontweight="bold")
        ax7.set_xlim(0, 1.08)
        ax7.spines[["top", "right", "bottom"]].set_visible(False)
        ax7.xaxis.grid(True)
        ax7.set_axisbelow(True)
        ax7.set_xlabel("Accuracy")
        st.pyplot(fig7)
        plt.close(fig7)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown('<p class="sec-head">Best Model Metrics</p>', unsafe_allow_html=True)
        metric_colors = {
            "Accuracy" : "linear-gradient(90deg,#6366f1,#8b5cf6)",
            "Precision": "linear-gradient(90deg,#10b981,#34d399)",
            "Recall"   : "linear-gradient(90deg,#f59e0b,#f97316)",
            "F1"       : "linear-gradient(90deg,#ec4899,#f87171)",
            "ROC-AUC"  : "linear-gradient(90deg,#3b82f6,#6366f1)",
        }
        for m in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
            if m in best_row:
                val  = round(best_row[m] * 100, 2)
                grad = metric_colors.get(m, "linear-gradient(90deg,#6366f1,#a78bfa)")
                st.markdown(f"""
                <div class="skill-row">
                  <div class="skill-label">
                    <span>{m}</span>
                    <span style="color:#e5e7eb;font-weight:600">{val}%</span>
                  </div>
                  <div class="skill-track">
                    <div class="skill-fill" style="width:{round(val)}%;background:{grad}"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋 View Full Model Comparison Table"):
        st.dataframe(perf, use_container_width=True)

    if history is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🔄 View Retrain History"):
            st.dataframe(history, use_container_width=True)

# ============================================================
# TAB 3 — SALARY MODEL
# ============================================================
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)

    if salary_perf is not None:
        sal_row = salary_perf.iloc[0].to_dict()

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0f1f14,#0f2318);
                    border:1px solid rgba(16,185,129,0.4); border-radius:16px;
                    padding:1.4rem 2rem; margin-bottom:1.5rem;
                    display:flex; align-items:center; gap:1.5rem;">
          <div style="font-size:2.5rem">💰</div>
          <div>
            <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;color:#34d399;">
              {sal_row['Model']}
            </div>
            <div style="font-size:0.82rem;color:#6b7280;margin-top:0.2rem;">
              Salary prediction model · R² score: {sal_row['R2']}
            </div>
          </div>
          <div style="margin-left:auto;text-align:right;">
            <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#34d399;">
              {sal_row['R2']}
            </div>
            <div style="font-size:0.72rem;color:#6b7280;letter-spacing:1px;text-transform:uppercase;">R² Score</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4, gap="medium")
        metrics_display = [
            (c1, "MSE",  sal_row["MSE"],  "Mean Squared Error",       "#f87171"),
            (c2, "RMSE", sal_row["RMSE"], "Root Mean Squared Error",   "#f59e0b"),
            (c3, "MAE",  sal_row["MAE"],  "Mean Absolute Error",       "#6366f1"),
            (c4, "R²",   sal_row["R2"],   "R² Score (closer to 1 = better)", "#34d399"),
        ]
        for col, label, value, description, color in metrics_display:
            with col:
                st.markdown(f"""
                <div style="background:#111827;border:1px solid #1e2535;border-radius:14px;
                            padding:1.2rem;text-align:center;">
                  <div style="font-size:0.68rem;color:#6b7280;text-transform:uppercase;
                              letter-spacing:1px;margin-bottom:0.6rem;">{label}</div>
                  <div style="font-family:'Syne',sans-serif;font-size:1.8rem;
                              font-weight:800;color:{color};">{value}</div>
                  <div style="font-size:0.7rem;color:#4b5563;margin-top:0.4rem;">{description}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown('<p class="sec-head">Salary Distribution by Branch <span>placed students</span></p>', unsafe_allow_html=True)
        placed_df_tab = df_plot[df_plot["placement_num"] == 1]
        if "branch" in placed_df_tab.columns:
            branch_salary = placed_df_tab.groupby("branch")["salary_lpa"].mean().sort_values(ascending=True)
            fig8, ax8 = plt.subplots(figsize=(9, 3))
            bars8 = ax8.barh(branch_salary.index, branch_salary.values,
                             color="#10b981", alpha=0.8, height=0.55)
            for bar, val in zip(bars8, branch_salary.values):
                ax8.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                         f"₹{round(val, 2)} LPA", va="center", fontsize=7.5, color="#e5e7eb")
            ax8.spines[["top", "right", "bottom"]].set_visible(False)
            ax8.xaxis.grid(True)
            ax8.set_axisbelow(True)
            ax8.set_xlabel("Average Salary (LPA)")
            st.pyplot(fig8)
            plt.close(fig8)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Salary performance metrics not found. Please run train.py first.")

# ============================================================
# TAB 4 — PREDICT
# ============================================================
with tab4:
    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<p class="sec-head">🎛️ Student Profile <span>enter details below</span></p>', unsafe_allow_html=True)

        cgpa          = st.slider("CGPA",                       0.0, 10.0, 7.0, 0.1)
        internships   = st.slider("Internships Completed",      0,   5,    1)
        projects      = st.slider("Projects Completed",         0,   10,   2)
        coding        = st.slider("Coding Skill Rating",        1,   10,   5)
        communication = st.slider("Communication Skill Rating", 1,   10,   5)
        aptitude      = st.slider("Aptitude Skill Rating",      1,   10,   5)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Live probability gauge (updates on every slider change)
        # CGPA below 3.5 is considered fail — auto not placed
        if cgpa < 3.5:
            _live_prob = 0.0
            _gcls      = "low"
            _gemoji    = "🔴"
        else:
            _live_input = pd.DataFrame([[cgpa, internships, projects, coding, communication, aptitude]],
                                       columns=features)
            _live_scaled = scaler.transform(_live_input)
            _live_prob   = round(float(placement_model.predict_proba(_live_scaled)[0][1]) * 100, 1)
            if _live_prob >= 65:
                _gcls = "high"; _gemoji = "🟢"
            elif _live_prob >= 45:
                _gcls = "mid";  _gemoji = "🟡"
            else:
                _gcls = "low";  _gemoji = "🔴"

        skill_avg = round((coding + communication + aptitude) / 3, 1)

        st.markdown(f"""
        <div class="gauge-wrap">
          <div class="gauge-title">Live Placement Probability</div>
          <div class="gauge-pct {_gcls}">{_gemoji} {_live_prob}%</div>
          <div class="gauge-sub">Updates as you move the sliders</div>
        </div>
        <div style="background:#0d1117;border:1px solid #1e2535;border-radius:12px;
                    padding:1rem 1.3rem;font-size:0.82rem;color:#9ca3af;">
          <b style="color:#e5e7eb">Profile Summary</b><br><br>
          CGPA <b style="color:#a5b4fc">{cgpa}</b> &nbsp;·&nbsp;
          Internships <b style="color:#a5b4fc">{internships}</b> &nbsp;·&nbsp;
          Projects <b style="color:#a5b4fc">{projects}</b><br>
          Avg Skill Score <b style="color:#34d399">{skill_avg} / 10</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡  Predict Placement Outcome")

    with right:
        st.markdown('<p class="sec-head">📈 Prediction Result</p>', unsafe_allow_html=True)

        if predict_btn:

            # ── CGPA below 3.5 = hard rule, not placed
            if cgpa < 3.5:
                st.markdown("""
                <div class="result-card">
                  <div style="font-size:2rem;margin-bottom:0.5rem">⚠️</div>
                  <div class="result-not">NOT PLACED</div>
                  <div class="result-label">CGPA below 3.5</div>
                  <div style="margin-top:1rem;padding-top:1rem;border-top:1px solid #1e2535;
                              font-size:0.84rem;color:#9ca3af;line-height:1.6;">
                    A CGPA below 3.5 is generally considered below the
                    minimum eligibility criteria for campus placements.
                    Focus on improving your academic performance first.
                  </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                # ── Normal model prediction
                input_data = pd.DataFrame(
                    [[cgpa, internships, projects, coding, communication, aptitude]],
                    columns=features
                )
                data_scaled = scaler.transform(input_data)
                prob        = placement_model.predict_proba(data_scaled)[0][1]
                placement   = placement_model.predict(data_scaled)[0]

                # Probability bar
                st.markdown(f"""
                <div style="margin-bottom:0.4rem;display:flex;
                            justify-content:space-between;font-size:0.78rem;color:#6b7280;">
                  <span>Placement Probability</span>
                  <span style="color:#e5e7eb;font-weight:700">{round(prob*100, 1)}%</span>
                </div>
                """, unsafe_allow_html=True)
                st.progress(float(prob))
                st.markdown("<br>", unsafe_allow_html=True)

                if placement == 1:
                    salary = salary_model.predict(data_scaled)[0]

                    if prob >= 0.75:
                        conf_label = "High Confidence"
                        conf_color = "#34d399"
                    elif prob >= 0.55:
                        conf_label = "Moderate Confidence"
                        conf_color = "#f59e0b"
                    else:
                        conf_label = "Low Confidence"
                        conf_color = "#f87171"

                    st.markdown(f"""
                    <div class="result-card">
                      <div style="font-size:2rem;margin-bottom:0.5rem">🎉</div>
                      <div class="result-placed">PLACED</div>
                      <div class="result-label">Placement Probability: {round(prob*100, 1)}%</div>
                      <div style="margin-top:1rem;padding-top:1rem;border-top:1px solid #1e2535;">
                        <div style="font-size:0.75rem;color:#6b7280;text-transform:uppercase;letter-spacing:1px;">
                          Expected Salary Package
                        </div>
                        <div style="font-family:'Syne',sans-serif;font-size:2rem;
                                    font-weight:800;color:#f9fafb;margin:0.3rem 0;">
                          ₹ {round(salary, 2)} LPA
                        </div>
                      </div>
                      <div style="margin-top:0.8rem;">
                        <span style="background:rgba(0,0,0,0.3);
                                     border:1px solid {conf_color}33;
                                     color:{conf_color};font-size:0.72rem;font-weight:600;
                                     letter-spacing:1px;text-transform:uppercase;
                                     padding:4px 12px;border-radius:20px;">
                          {conf_label}
                        </span>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.markdown(f"""
                    <div class="result-card">
                      <div style="font-size:2rem;margin-bottom:0.5rem">📚</div>
                      <div class="result-not">NOT PLACED</div>
                      <div class="result-label">Placement Probability: {round(prob*100, 1)}%</div>
                      <div style="margin-top:1rem;padding-top:1rem;border-top:1px solid #1e2535;
                                  font-size:0.84rem;color:#9ca3af;line-height:1.6;">
                        Focus on improving your skills and adding more
                        projects and internships to boost your chances.
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Profile breakdown chart
                st.markdown("<br>", unsafe_allow_html=True)
                fig9, ax9    = plt.subplots(figsize=(5, 2.2))
                skill_names  = ["CGPA", "Internships", "Projects", "Coding", "Comm.", "Aptitude"]
                skill_values = [cgpa, internships, projects, coding, communication, aptitude]
                bar_colors   = ["#6366f1", "#8b5cf6", "#ec4899", "#10b981", "#f59e0b", "#3b82f6"]
                bars9        = ax9.barh(skill_names, skill_values, color=bar_colors, alpha=0.8, height=0.55)
                for bar, val in zip(bars9, skill_values):
                    ax9.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                             str(round(val, 1)), va="center", fontsize=7, color="#e5e7eb")
                ax9.set_xlim(0, 12)
                ax9.spines[["top", "right", "bottom"]].set_visible(False)
                ax9.xaxis.grid(True)
                ax9.set_axisbelow(True)
                ax9.set_title("Your Profile", fontsize=9, color="#9ca3af", pad=8)
                st.pyplot(fig9)
                plt.close(fig9)

        else:
            st.markdown("""
            <div style="background:#0d1117;border:1px dashed #1e2535;
                        border-radius:16px;padding:3rem 2rem;text-align:center;
                        color:#4b5563;margin-top:1rem;">
              <div style="font-size:2.5rem;margin-bottom:1rem">🎯</div>
              <div style="font-size:0.9rem;">
                Adjust the sliders on the left and click<br>
                <b style="color:#6366f1">Predict Placement Outcome</b>
              </div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:1.5rem;border-top:1px solid #1e2535;
            font-size:0.75rem;color:#374151;letter-spacing:0.5px;">
  Placement And Salary Prediction System &nbsp;·&nbsp; Built with Streamlit & Scikit-learn
  &nbsp;·&nbsp; <span style="color:#6366f1">AI-Powered</span>
</div>
""", unsafe_allow_html=True)
