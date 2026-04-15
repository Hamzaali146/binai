"""
binai — CV Robustness Testing Framework
Streamlit Dashboard  |  v2
Run: streamlit run dashboard/app.py
"""
import io as _io
import json

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="binai · CV Testing",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Theme tokens
# ─────────────────────────────────────────────────────────────────────────────
ACCENT  = "#58a6ff"
SUCCESS = "#3fb950"
DANGER  = "#f78166"
WARN    = "#d29922"
PURPLE  = "#bc8cff"
CARD    = "#161b22"
BORDER  = "#30363d"
MUTED   = "#8b949e"
FG      = "#f0f6fc"
BG      = "#0d1117"

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color=FG,
)

CSS = f"""
<style>
/* ── global ──────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {{
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background-color: {BG};
}}
.main .block-container {{
    padding-top: 1.4rem;
    padding-bottom: 2.5rem;
    max-width: 1400px;
}}

/* ── sidebar ─────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: #0d1117 !important;
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] * {{ color: #c9d1d9 !important; }}
section[data-testid="stSidebar"] .stRadio > label {{ display: none; }}
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {{
    display: flex !important;
    align-items: center;
    padding: 0.55rem 0.9rem;
    border-radius: 7px;
    margin: 2px 0;
    font-size: 0.88rem;
    cursor: pointer;
    transition: background 0.15s;
}}
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {{
    background: #21262d !important;
}}

/* ── KPI cards ───────────────────────────────────────────────────────────── */
.kpi-row {{
    display: flex;
    gap: 1rem;
    margin-bottom: 1.4rem;
    flex-wrap: wrap;
}}
.kpi-card {{
    flex: 1;
    min-width: 160px;
    background: {CARD};
    border: 1px solid {BORDER};
    border-top: 3px solid var(--c);
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
}}
.kpi-label {{
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {MUTED};
    margin-bottom: 0.25rem;
}}
.kpi-value {{ font-size: 2rem; font-weight: 700; color: {FG}; line-height: 1.1; }}
.kpi-sub   {{ font-size: 0.75rem; color: {MUTED}; margin-top: 0.2rem; }}

/* ── status badges ───────────────────────────────────────────────────────── */
.badge {{
    display: inline-block;
    padding: 0.18em 0.65em;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}
.badge-completed {{ background: #1a4731; color: {SUCCESS}; }}
.badge-running   {{ background: #341a00; color: {WARN}; }}
.badge-failed    {{ background: #3d1a1a; color: {DANGER}; }}
.badge-pending   {{ background: #21262d; color: {MUTED}; }}

/* ── scenario cards ──────────────────────────────────────────────────────── */
.sc-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-left: 4px solid {ACCENT};
    border-radius: 9px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.7rem;
    transition: border-left-color 0.2s;
}}
.sc-card:hover {{ border-left-color: {PURPLE}; }}
.sc-name {{ font-size: 0.92rem; font-weight: 600; color: {FG}; }}
.sc-desc {{ font-size: 0.78rem; color: {MUTED}; margin-top: 0.2rem; }}

/* ── model / dataset cards ───────────────────────────────────────────────── */
.reg-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 9px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
}}
.reg-title {{ font-size: 1rem; font-weight: 600; color: {FG}; }}
.reg-sub   {{ font-size: 0.78rem; color: {MUTED}; margin-top: 0.15rem; }}
.reg-tag {{
    display: inline-block;
    background: #21262d;
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 0.1em 0.5em;
    font-size: 0.72rem;
    color: {ACCENT};
    margin-right: 0.3rem;
    margin-top: 0.4rem;
}}

/* ── eval list rows ──────────────────────────────────────────────────────── */
.eval-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.7rem 1rem;
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-bottom: 0.5rem;
}}
.eval-name {{ font-size: 0.9rem; font-weight: 600; color: {FG}; }}
.eval-meta {{ font-size: 0.75rem; color: {MUTED}; margin-top: 0.15rem; }}

/* ── page header ─────────────────────────────────────────────────────────── */
.page-title {{ font-size: 1.65rem; font-weight: 700; color: {FG}; margin-bottom: 0.1rem; }}
.page-sub   {{ font-size: 0.85rem; color: {MUTED}; margin-bottom: 1.3rem; }}

/* ── tabs ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background: {CARD};
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
    border: 1px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 6px;
    color: {MUTED};
    font-size: 0.85rem;
}}
.stTabs [aria-selected="true"] {{
    background: #21262d !important;
    color: {FG} !important;
}}

/* ── divider ─────────────────────────────────────────────────────────────── */
hr {{ border-color: {BORDER} !important; margin: 1.1rem 0 !important; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def api_get(path: str, **params):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API server — make sure the backend is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, data: dict):
    try:
        r = requests.post(f"{API_BASE}{path}", json=data, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def fmt(val, spec=".4f"):
    return "—" if val is None else f"{val:{spec}}"


def badge(status: str) -> str:
    return f"<span class='badge badge-{status}'>{status}</span>"


def kpi_card(label: str, value, sub: str = "", color: str = ACCENT) -> str:
    return (
        f"<div class='kpi-card' style='--c:{color};'>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value'>{value}</div>"
        f"<div class='kpi-sub'>{sub}</div>"
        f"</div>"
    )


SCENARIO_ICONS: dict[str, str] = {
    "gaussian_blur": "🌫️",      "motion_blur": "💨",      "median_blur": "🔆",
    "gaussian_noise": "📡",      "salt_pepper_noise": "✨", "speckle_noise": "🔊",
    "low_light": "🌙",           "overexposure": "☀️",     "shadow": "🌑",
    "fog": "🌁",                 "rotation": "🔄",          "flip": "↔️",
    "perspective": "📐",         "zoom": "🔍",              "random_occlusion": "⬛",
    "grid_occlusion": "🔲",      "watermark_occlusion": "🏷️",
}


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"<div style='padding:0.6rem 0.2rem 1rem;'>"
        f"<div style='font-size:1.55rem;font-weight:800;color:{FG};letter-spacing:-0.02em;'>🧬 binai</div>"
        f"<div style='font-size:0.7rem;color:{MUTED};margin-top:0.1rem;letter-spacing:0.1em;text-transform:uppercase;'>"
        f"CV Robustness Testing</div></div>",
        unsafe_allow_html=True,
    )

    page = st.radio(
        "nav",
        [
            "📊  Overview",
            "🧪  Evaluations",
            "🌩️  Scenarios",
            "⚖️  Model Comparison",
            "❌  Failure Analysis",
            "📁  Datasets",
            "🤖  Models",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        f"<hr style='margin:1rem 0 0.7rem;border-color:{BORDER};'>"
        f"<div style='font-size:0.7rem;color:{MUTED};'>binai v1.0 · MLOps Platform</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Overview
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊  Overview":
    hdr, refresh_col = st.columns([9, 1])
    hdr.markdown("<div class='page-title'>Overview</div><div class='page-sub'>Real-time summary of your CV evaluation pipeline</div>", unsafe_allow_html=True)
    if refresh_col.button("⟳", help="Refresh"):
        st.rerun()

    evals         = api_get("/evaluations", limit=200) or []
    models_list   = api_get("/models") or []
    datasets_list = api_get("/datasets") or []

    completed = [e for e in evals if e["status"] == "completed"]
    failed    = [e for e in evals if e["status"] == "failed"]
    running   = [e for e in evals if e["status"] == "running"]

    st.markdown(
        "<div class='kpi-row'>"
        + kpi_card("Total Evaluations", len(evals),    f"{len(running)} running",    ACCENT)
        + kpi_card("Completed",          len(completed), f"{len(failed)} failed",    SUCCESS)
        + kpi_card("Models",             len(models_list), "registered",             PURPLE)
        + kpi_card("Datasets",           len(datasets_list), "registered",           WARN)
        + "</div>",
        unsafe_allow_html=True,
    )

    if not evals:
        st.info("No evaluations yet. Register a model and dataset to get started.")
        st.stop()

    col_left, col_right = st.columns([5, 3])

    with col_left:
        st.markdown(f"<div style='font-size:0.95rem;font-weight:600;color:{FG};margin-bottom:0.5rem;'>Status Distribution</div>", unsafe_allow_html=True)
        status_df = pd.Series([e["status"] for e in evals]).value_counts().reset_index()
        status_df.columns = ["Status", "Count"]
        color_map = {"completed": SUCCESS, "failed": DANGER, "running": WARN, "pending": MUTED}
        fig_pie = px.pie(
            status_df, values="Count", names="Status",
            color="Status", color_discrete_map=color_map,
            hole=0.55,
        )
        fig_pie.update_traces(textposition="outside", textinfo="label+percent")
        fig_pie.update_layout(**PLOTLY_THEME, margin=dict(t=10, b=10, l=10, r=10), height=260, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

        if completed:
            st.markdown(f"<div style='font-size:0.95rem;font-weight:600;color:{FG};margin-bottom:0.5rem;'>Primary Metric Trend</div>", unsafe_allow_html=True)
            trend_rows = []
            for ev in sorted(completed, key=lambda e: e.get("created_at", "")):
                m = ev.get("metrics") or {}
                primary = m.get("accuracy") or m.get("map_50") or m.get("mean_iou")
                if primary is not None:
                    trend_rows.append({"Evaluation": ev["name"], "Score": primary, "Date": ev.get("created_at", "")[:10]})
            if trend_rows:
                fig_trend = px.line(
                    pd.DataFrame(trend_rows), x="Date", y="Score",
                    text="Evaluation", markers=True,
                )
                fig_trend.update_traces(
                    textposition="top center", textfont_size=9,
                    line_color=ACCENT, marker_color=ACCENT,
                )
                fig_trend.update_layout(**PLOTLY_THEME, height=240, margin=dict(t=10, b=10))
                st.plotly_chart(fig_trend, use_container_width=True)

    with col_right:
        st.markdown(f"<div style='font-size:0.95rem;font-weight:600;color:{FG};margin-bottom:0.5rem;'>Recent Evaluations</div>", unsafe_allow_html=True)
        for ev in sorted(evals, key=lambda e: e.get("created_at", ""), reverse=True)[:10]:
            s = ev["status"]
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"padding:0.5rem 0.75rem;background:{CARD};border:1px solid {BORDER};"
                f"border-radius:7px;margin-bottom:0.4rem;'>"
                f"<span style='font-size:0.84rem;color:{FG};font-weight:500;'>{ev['name'][:30]}</span>"
                f"{badge(s)}</div>",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluations
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🧪  Evaluations":
    st.markdown("<div class='page-title'>Evaluations</div>", unsafe_allow_html=True)
    tab_list, tab_detail, tab_new = st.tabs(["📋 All", "🔍 Detail", "➕ New"])

    with tab_list:
        evals = api_get("/evaluations", limit=100) or []
        if not evals:
            st.info("No evaluations found.")
        else:
            for ev in sorted(evals, key=lambda e: e.get("created_at", ""), reverse=True):
                s   = ev["status"]
                m   = ev.get("metrics") or {}
                primary = m.get("accuracy") or m.get("map_50") or m.get("mean_iou")
                pri_str = f"{primary:.3f}" if primary is not None else "—"
                dur     = ev.get("duration_seconds")
                dur_str = f"{dur:.1f}s" if dur else "—"
                st.markdown(
                    f"<div class='eval-row'>"
                    f"<div>"
                    f"  <div class='eval-name'>{ev['name']}</div>"
                    f"  <div class='eval-meta'>"
                    f"    {ev.get('task_type','?')} · {ev.get('num_samples',0)} samples"
                    f"    · {ev.get('num_failures',0)} failures · {dur_str}"
                    f"  </div>"
                    f"</div>"
                    f"<div style='display:flex;align-items:center;gap:1rem;'>"
                    f"  <span style='font-size:1rem;font-weight:700;color:{ACCENT};'>{pri_str}</span>"
                    f"  {badge(s)}"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    with tab_detail:
        evals = api_get("/evaluations", limit=100) or []
        if not evals:
            st.info("No evaluations found.")
        else:
            selected_name = st.selectbox("Select evaluation", [e["name"] for e in evals], key="detail_sel")
            ev = next((e for e in evals if e["name"] == selected_name), None)
            if ev:
                info_col, status_col = st.columns([4, 1])
                info_col.markdown(
                    f"**Task:** `{ev.get('task_type','')}` &nbsp;&nbsp;"
                    f"**Samples:** `{ev.get('num_samples',0)}` &nbsp;&nbsp;"
                    f"**Failures:** `{ev.get('num_failures',0)}` &nbsp;&nbsp;"
                    f"**Duration:** `{ev.get('duration_seconds') or '—'}s`"
                )
                status_col.markdown(f"Status: {badge(ev['status'])}", unsafe_allow_html=True)

                m = ev.get("metrics") or {}
                metric_defs = [
                    ("accuracy",       "Accuracy",       ACCENT),
                    ("f1_score",       "F1 Score",       SUCCESS),
                    ("precision",      "Precision",      PURPLE),
                    ("recall",         "Recall",         WARN),
                    ("map_50",         "mAP@50",         ACCENT),
                    ("map_75",         "mAP@75",         PURPLE),
                    ("mean_iou",       "Mean IoU",       SUCCESS),
                    ("pixel_accuracy", "Pixel Accuracy", WARN),
                ]
                present = [(k, lbl, c) for k, lbl, c in metric_defs if m.get(k) is not None]
                if present:
                    st.markdown("#### Baseline Metrics")
                    gcols = st.columns(min(len(present), 4))
                    for i, (k, lbl, c) in enumerate(present):
                        fig_g = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=m[k],
                            number={"font": {"color": FG, "size": 26}, "valueformat": ".3f"},
                            gauge={
                                "axis": {"range": [0, 1], "tickcolor": MUTED, "tickfont": {"color": MUTED}},
                                "bar": {"color": c},
                                "bgcolor": CARD,
                                "bordercolor": BORDER,
                                "steps": [
                                    {"range": [0, 0.5], "color": "#21262d"},
                                    {"range": [0.5, 1], "color": "#1f2d3d"},
                                ],
                                "threshold": {"line": {"color": SUCCESS, "width": 2}, "thickness": 0.75, "value": 0.8},
                            },
                            title={"text": lbl, "font": {"color": MUTED, "size": 11}},
                        ))
                        fig_g.update_layout(**PLOTLY_THEME, height=160, margin=dict(t=30, b=5, l=10, r=10))
                        gcols[i % 4].plotly_chart(fig_g, use_container_width=True)

                sr = ev.get("scenario_results") or []
                if sr:
                    st.markdown("#### Scenario Robustness")
                    names, scores, degrades = [], [], []
                    for s_res in sr:
                        sc_m = s_res.get("metrics") or {}
                        p    = sc_m.get("accuracy") or sc_m.get("map_50") or sc_m.get("mean_iou") or 0
                        names.append(s_res["scenario_name"])
                        scores.append(p)
                        degrades.append(s_res.get("degradation_pct") or 0)

                    fig_sc = go.Figure([
                        go.Bar(name="Score",          x=names, y=scores,   marker_color=ACCENT,
                               text=[f"{v:.3f}" for v in scores],   textposition="outside"),
                        go.Bar(name="Degradation %",  x=names, y=degrades, marker_color=DANGER,
                               text=[f"{v:.1f}%" for v in degrades], textposition="outside"),
                    ])
                    fig_sc.update_layout(
                        **PLOTLY_THEME,
                        barmode="group",
                        height=300,
                        margin=dict(t=10, b=10),
                        legend=dict(orientation="h", y=-0.18),
                    )
                    st.plotly_chart(fig_sc, use_container_width=True)

                if m.get("per_class_metrics"):
                    st.markdown("#### Per-Class Metrics")
                    pc_rows = [{"Class": cls, **vals} for cls, vals in m["per_class_metrics"].items()]
                    st.dataframe(pd.DataFrame(pc_rows), use_container_width=True, hide_index=True)

    with tab_new:
        st.markdown(f"<div style='font-size:0.85rem;color:{MUTED};margin-bottom:1rem;'>Configure and launch a new evaluation run</div>", unsafe_allow_html=True)
        models_list   = api_get("/models") or []
        datasets_list = api_get("/datasets") or []
        if not models_list:
            st.warning("No models registered yet — go to Models first.")
        elif not datasets_list:
            st.warning("No datasets registered yet — go to Datasets first.")
        else:
            with st.form("new_eval"):
                r1c1, r1c2 = st.columns(2)
                eval_name      = r1c1.text_input("Evaluation Name", placeholder="e.g. ResNet50 Baseline")
                model_choice   = r1c2.selectbox("Model", [f"{m['name']} v{m['version']}" for m in models_list])
                r2c1, r2c2     = st.columns(2)
                dataset_choice = r2c1.selectbox("Dataset", [d["name"] for d in datasets_list])
                max_samples    = r2c2.number_input("Max Samples (0 = all)", min_value=0, value=0)
                r3c1, r3c2     = st.columns(2)
                conf_thr       = r3c1.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
                iou_thr        = r3c2.slider("IoU Threshold",         0.0, 1.0, 0.5, 0.05)

                st.markdown(f"<div style='font-size:0.85rem;font-weight:600;color:{FG};margin:0.6rem 0 0.4rem;'>Scenarios to apply</div>", unsafe_allow_html=True)
                sc_options = api_get("/scenarios") or []
                selected_scenarios = []
                sc_cols = st.columns(3)
                for i, sc in enumerate(sc_options):
                    icon = SCENARIO_ICONS.get(sc["name"], "🔧")
                    if sc_cols[i % 3].checkbox(f"{icon} {sc['name']}", help=sc.get("description", "")):
                        selected_scenarios.append({"name": sc["name"], "enabled": True, "params": {}})

                if st.form_submit_button("🚀 Launch Evaluation", use_container_width=True):
                    if not eval_name:
                        st.error("Please provide an evaluation name.")
                    else:
                        model_obj   = next(m for m in models_list if f"{m['name']} v{m['version']}" == model_choice)
                        dataset_obj = next(d for d in datasets_list if d["name"] == dataset_choice)
                        result = api_post("/evaluations", {
                            "name": eval_name,
                            "model_id": model_obj["id"],
                            "dataset_id": dataset_obj["id"],
                            "config": {
                                "confidence_threshold": conf_thr,
                                "iou_threshold": iou_thr,
                                "max_samples": max_samples if max_samples > 0 else None,
                                "scenarios": selected_scenarios,
                            },
                        })
                        if result:
                            st.success(f"✅ Evaluation **{eval_name}** launched! ID: `{result['id'][:8]}…`")


# ─────────────────────────────────────────────────────────────────────────────
# Scenarios
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🌩️  Scenarios":
    st.markdown("<div class='page-title'>Robustness Scenarios</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Test model behaviour under real-world image corruptions</div>", unsafe_allow_html=True)

    scenarios = api_get("/scenarios") or []
    if not scenarios:
        st.info("No scenarios available.")
    else:
        grid_cols = st.columns(3)
        for i, sc in enumerate(scenarios):
            icon = SCENARIO_ICONS.get(sc["name"], "🔧")
            grid_cols[i % 3].markdown(
                f"<div class='sc-card'>"
                f"<div class='sc-name'>{icon} {sc['name'].replace('_',' ').title()}</div>"
                f"<div class='sc-desc'>{sc.get('description','')}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown(f"<div style='font-size:1.05rem;font-weight:600;color:{FG};margin-bottom:0.8rem;'>Live Scenario Preview</div>", unsafe_allow_html=True)

    up_col, sc_col = st.columns(2)
    uploaded = up_col.file_uploader("Upload image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    sc_names = [s["name"] for s in scenarios] if scenarios else []
    sc_name  = sc_col.selectbox(
        "Scenario",
        sc_names,
        format_func=lambda n: f"{SCENARIO_ICONS.get(n, '🔧')} {n.replace('_', ' ').title()}",
    )

    # Per-scenario parameter controls
    params: dict = {}
    if sc_name:
        pc1, pc2 = st.columns(2)
        if sc_name == "gaussian_blur":
            params["kernel_size"] = pc1.slider("Kernel size", 3, 51, 15, 2)
            params["sigma"]       = pc2.slider("Sigma", 0, 20, 0)
        elif sc_name == "motion_blur":
            params["kernel_size"] = pc1.slider("Kernel size", 3, 51, 15, 2)
            params["angle"]       = pc2.slider("Angle °", 0, 360, 0)
        elif sc_name == "median_blur":
            params["ksize"] = pc1.slider("Kernel size", 3, 21, 5, 2)
        elif sc_name == "gaussian_noise":
            params["std"]  = pc1.slider("Std dev", 1, 100, 25)
            params["mean"] = pc2.slider("Mean", -20, 20, 0)
        elif sc_name == "salt_pepper_noise":
            params["density"] = pc1.slider("Density", 0.01, 0.30, 0.05)
        elif sc_name == "speckle_noise":
            params["intensity"] = pc1.slider("Intensity", 0.01, 1.0, 0.2)
        elif sc_name == "low_light":
            params["gamma"] = pc1.slider("Gamma", 1.0, 5.0, 2.5)
        elif sc_name == "overexposure":
            params["factor"] = pc1.slider("Factor", 1.0, 4.0, 1.8)
        elif sc_name == "shadow":
            params["darkness"] = pc1.slider("Darkness", 0.0, 1.0, 0.4)
        elif sc_name == "fog":
            params["intensity"] = pc1.slider("Intensity", 0.0, 1.0, 0.4)
        elif sc_name == "rotation":
            params["angle"] = pc1.slider("Angle °", -180, 180, 15)
            params["scale"] = pc2.slider("Scale", 0.5, 2.0, 1.0)
        elif sc_name == "flip":
            direction = pc1.selectbox("Direction", ["Horizontal", "Vertical", "Both"])
            params["flip_code"] = {"Horizontal": 1, "Vertical": 0, "Both": -1}[direction]
        elif sc_name == "perspective":
            params["distortion"] = pc1.slider("Distortion", 0.01, 0.30, 0.05)
        elif sc_name == "zoom":
            params["zoom_factor"] = pc1.slider("Zoom factor", 1.1, 3.0, 1.5)
        elif sc_name == "random_occlusion":
            params["num_patches"] = pc1.slider("Patches",     1,    10,   3)
            params["patch_ratio"] = pc2.slider("Patch ratio", 0.05, 0.50, 0.15)
        elif sc_name == "grid_occlusion":
            params["num_tiles"]  = pc1.slider("Tiles",      3,   10,  5)
            params["drop_ratio"] = pc2.slider("Drop ratio", 0.1, 0.8, 0.3)
        elif sc_name == "watermark_occlusion":
            params["alpha"]      = pc1.slider("Opacity",    0.1, 1.0, 0.5)
            params["font_scale"] = pc2.slider("Font scale", 1.0, 4.0, 2.0)

    if uploaded and sc_name:
        uploaded.seek(0)
        image_bytes = uploaded.read()
        r = None
        try:
            r = requests.post(
                f"{API_BASE}/scenarios/preview",
                params={"scenario_name": sc_name, "params": json.dumps(params)},
                files={"image": (uploaded.name, image_bytes, uploaded.type)},
                timeout=15,
            )
        except Exception as e:
            st.error(f"Could not reach API: {e}")

        if r is not None:
            if r.status_code == 200:
                img_col1, img_col2 = st.columns(2)
                img_col1.markdown(f"<div style='text-align:center;font-size:0.8rem;color:{MUTED};margin-bottom:0.3rem;'>Original</div>", unsafe_allow_html=True)
                img_col1.image(image_bytes, use_column_width=True)
                img_col2.markdown(f"<div style='text-align:center;font-size:0.8rem;color:{ACCENT};margin-bottom:0.3rem;'>After: {sc_name.replace('_',' ').title()}</div>", unsafe_allow_html=True)
                img_col2.image(_io.BytesIO(r.content), use_column_width=True)
            else:
                st.error(f"Preview failed: {r.text}")


# ─────────────────────────────────────────────────────────────────────────────
# Model Comparison
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚖️  Model Comparison":
    st.markdown("<div class='page-title'>Model Comparison</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Compare metrics across completed evaluation runs</div>", unsafe_allow_html=True)

    evals     = api_get("/evaluations", limit=100) or []
    completed = [e for e in evals if e["status"] == "completed"]

    if len(completed) < 2:
        st.info("You need at least 2 completed evaluations to compare models.")
    else:
        selected_evals = st.multiselect(
            "Select evaluations",
            options=[e["name"] for e in completed],
            default=[e["name"] for e in completed[:min(3, len(completed))]],
        )
        selected_ids = [e["id"] for e in completed if e["name"] in selected_evals]

        if len(selected_ids) >= 2:
            result = api_post("/evaluations/compare", {
                "evaluation_ids": selected_ids,
                "metrics": ["accuracy", "f1_score", "map_50", "mean_iou", "precision", "recall"],
            })
            if result:
                best = result.get("best_model", "—")
                st.markdown(
                    f"<div style='background:{CARD};border:1px solid {BORDER};"
                    f"border-left:4px solid {SUCCESS};border-radius:8px;"
                    f"padding:0.8rem 1.2rem;margin-bottom:1.2rem;'>"
                    f"<div style='font-size:0.72rem;color:{MUTED};text-transform:uppercase;"
                    f"letter-spacing:0.06em;'>Best Model</div>"
                    f"<div style='font-size:1.1rem;font-weight:700;color:{SUCCESS};'>🏆 {best}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                metric_keys = ["accuracy", "f1_score", "map_50", "mean_iou", "precision", "recall"]
                palette = [ACCENT, SUCCESS, DANGER, WARN, PURPLE, "#58d68d"]

                radar_col, bar_col = st.columns(2)

                with radar_col:
                    st.markdown(f"<div style='font-size:0.9rem;font-weight:600;color:{FG};margin-bottom:0.4rem;'>Radar</div>", unsafe_allow_html=True)
                    fig_r = go.Figure()
                    for idx, ev_data in enumerate(result["evaluations"]):
                        m    = ev_data.get("metrics") or {}
                        vals = [m.get(k, 0) or 0 for k in metric_keys] + [m.get(metric_keys[0], 0) or 0]
                        fig_r.add_trace(go.Scatterpolar(
                            r=vals,
                            theta=metric_keys + [metric_keys[0]],
                            fill="toself",
                            opacity=0.6,
                            name=f"{ev_data['model_name']} v{ev_data.get('model_version','?')}",
                            line_color=palette[idx % len(palette)],
                        ))
                    fig_r.update_layout(
                        **PLOTLY_THEME,
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 1], color=MUTED),
                            angularaxis=dict(color=MUTED),
                            bgcolor=CARD,
                        ),
                        height=380,
                        margin=dict(t=20, b=30),
                        legend=dict(orientation="h", y=-0.15, font_size=11),
                    )
                    st.plotly_chart(fig_r, use_container_width=True)

                with bar_col:
                    st.markdown(f"<div style='font-size:0.9rem;font-weight:600;color:{FG};margin-bottom:0.4rem;'>Side-by-Side</div>", unsafe_allow_html=True)
                    bar_rows = []
                    for ev_data in result["evaluations"]:
                        m     = ev_data.get("metrics") or {}
                        label = f"{ev_data['model_name']} v{ev_data.get('model_version','?')}"
                        for mk in metric_keys:
                            v = m.get(mk)
                            if v is not None:
                                bar_rows.append({"Model": label, "Metric": mk, "Value": v})
                    if bar_rows:
                        fig_b = px.bar(
                            pd.DataFrame(bar_rows), x="Metric", y="Value", color="Model",
                            barmode="group",
                            color_discrete_sequence=palette,
                        )
                        fig_b.update_layout(
                            **PLOTLY_THEME,
                            height=380,
                            margin=dict(t=10, b=30),
                            yaxis_range=[0, 1],
                            legend=dict(orientation="h", y=-0.2, font_size=11),
                        )
                        st.plotly_chart(fig_b, use_container_width=True)

                # Summary table
                rows = []
                for ev_data in result["evaluations"]:
                    m = ev_data.get("metrics") or {}
                    rows.append({
                        "Model":     ev_data["model_name"],
                        "Version":   ev_data.get("model_version", "?"),
                        "Accuracy":  fmt(m.get("accuracy")),
                        "F1":        fmt(m.get("f1_score")),
                        "Precision": fmt(m.get("precision")),
                        "Recall":    fmt(m.get("recall")),
                        "mAP@50":    fmt(m.get("map_50")),
                        "mIoU":      fmt(m.get("mean_iou")),
                        "Failures":  ev_data.get("num_failures", 0),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # Delta cards
                if result.get("metric_deltas"):
                    st.markdown("#### Metric Deltas vs Baseline")
                    for pair, deltas in result["metric_deltas"].items():
                        st.markdown(f"<div style='font-size:0.85rem;color:{MUTED};margin-bottom:0.4rem;'>{pair}</div>", unsafe_allow_html=True)
                        delta_cols = st.columns(len(deltas))
                        for i, (mk, dv) in enumerate(deltas.items()):
                            color = SUCCESS if dv >= 0 else DANGER
                            arrow = "▲" if dv >= 0 else "▼"
                            delta_cols[i].markdown(
                                f"<div style='background:{CARD};border:1px solid {BORDER};"
                                f"border-radius:7px;padding:0.5rem;text-align:center;'>"
                                f"<div style='font-size:0.7rem;color:{MUTED};'>{mk}</div>"
                                f"<div style='font-size:1.05rem;font-weight:700;color:{color};'>"
                                f"{arrow} {abs(dv):.4f}</div></div>",
                                unsafe_allow_html=True,
                            )


# ─────────────────────────────────────────────────────────────────────────────
# Failure Analysis
# ─────────────────────────────────────────────────────────────────────────────
elif page == "❌  Failure Analysis":
    st.markdown("<div class='page-title'>Failure Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Drill into what's going wrong and why</div>", unsafe_allow_html=True)

    evals     = api_get("/evaluations", limit=100) or []
    completed = [e for e in evals if e["status"] == "completed"]

    if not completed:
        st.info("No completed evaluations yet.")
    else:
        eval_name = st.selectbox("Evaluation", [e["name"] for e in completed])
        ev        = next(e for e in completed if e["name"] == eval_name)
        failures  = api_get(f"/evaluations/{ev['id']}/failures", limit=200) or []

        total    = ev.get("num_failures", 0)
        samples  = ev.get("num_samples", 0) or 1
        fail_pct = total / samples * 100

        st.markdown(
            "<div class='kpi-row'>"
            + kpi_card("Total Failures", total,          f"{fail_pct:.1f}% failure rate",   DANGER)
            + kpi_card("Samples",         ev.get("num_samples", 0), "evaluated",             ACCENT)
            + kpi_card("Passed",          ev.get("num_samples", 0) - total, f"{100 - fail_pct:.1f}% pass rate", SUCCESS)
            + "</div>",
            unsafe_allow_html=True,
        )

        if not failures:
            st.success("✅ No failure cases recorded for this evaluation!")
        else:
            df = pd.DataFrame([{
                "Image ID":   f.get("image_id", ""),
                "Error Type": f.get("error_type", "unknown"),
                "Confidence": f.get("confidence"),
                "IoU":        f.get("iou_score"),
                "Scenario":   f.get("scenario") or "baseline",
            } for f in failures])

            ch1, ch2 = st.columns(2)

            with ch1:
                ec = df["Error Type"].value_counts().reset_index()
                ec.columns = ["Error Type", "Count"]
                fig_e = px.bar(
                    ec, x="Count", y="Error Type", orientation="h",
                    color="Count",
                    color_continuous_scale=[[0, CARD], [1, DANGER]],
                )
                fig_e.update_layout(**PLOTLY_THEME, title="Failures by Error Type",
                                    height=280, margin=dict(t=30, b=10), coloraxis_showscale=False)
                st.plotly_chart(fig_e, use_container_width=True)

            with ch2:
                sc_cnt = df["Scenario"].value_counts().reset_index()
                sc_cnt.columns = ["Scenario", "Count"]
                fig_s = px.pie(
                    sc_cnt, values="Count", names="Scenario", hole=0.5,
                    color_discrete_sequence=[ACCENT, DANGER, WARN, PURPLE, SUCCESS],
                )
                fig_s.update_layout(**PLOTLY_THEME, title="Failures by Scenario",
                                    height=280, margin=dict(t=30, b=10))
                st.plotly_chart(fig_s, use_container_width=True)

            # Confidence / IoU histograms
            conf_series = df["Confidence"].dropna()
            iou_series  = df["IoU"].dropna()
            if not conf_series.empty or not iou_series.empty:
                h1, h2 = st.columns(2)
                if not conf_series.empty:
                    fig_c = px.histogram(conf_series, nbins=20, color_discrete_sequence=[WARN])
                    fig_c.update_layout(**PLOTLY_THEME, title="Confidence Distribution",
                                        height=220, margin=dict(t=30, b=10), showlegend=False)
                    h1.plotly_chart(fig_c, use_container_width=True)
                if not iou_series.empty:
                    fig_i = px.histogram(iou_series, nbins=20, color_discrete_sequence=[PURPLE])
                    fig_i.update_layout(**PLOTLY_THEME, title="IoU Distribution",
                                        height=220, margin=dict(t=30, b=10), showlegend=False)
                    h2.plotly_chart(fig_i, use_container_width=True)

            st.markdown("---")
            f1_col, f2_col = st.columns(2)
            error_filter    = f1_col.selectbox("Filter by error type", ["All"] + sorted(df["Error Type"].unique().tolist()))
            scenario_filter = f2_col.selectbox("Filter by scenario",   ["All"] + sorted(df["Scenario"].unique().tolist()))

            filtered = df.copy()
            if error_filter != "All":
                filtered = filtered[filtered["Error Type"] == error_filter]
            if scenario_filter != "All":
                filtered = filtered[filtered["Scenario"] == scenario_filter]

            filtered = filtered.copy()
            filtered["Confidence"] = filtered["Confidence"].apply(lambda v: fmt(v, ".3f"))
            filtered["IoU"]        = filtered["IoU"].apply(lambda v: fmt(v, ".3f"))
            st.dataframe(filtered, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📁  Datasets":
    st.markdown("<div class='page-title'>Datasets</div>", unsafe_allow_html=True)
    tab_list, tab_new = st.tabs(["📋 Registered", "➕ Register New"])

    with tab_list:
        datasets_list = api_get("/datasets") or []
        if not datasets_list:
            st.info("No datasets registered yet.")
        else:
            task_color = {"classification": ACCENT, "detection": SUCCESS, "segmentation": PURPLE}
            for d in datasets_list:
                c = task_color.get(d["task_type"], MUTED)
                st.markdown(
                    f"<div class='reg-card' style='border-left:3px solid {c};'>"
                    f"<div class='reg-title'>📁 {d['name']}</div>"
                    f"<div class='reg-sub'>{(d.get('description') or 'No description')[:90]}</div>"
                    f"<div>"
                    f"  <span class='reg-tag'>{d['task_type']}</span>"
                    f"  <span class='reg-tag'>{d['num_samples']} samples</span>"
                    f"  <span class='reg-tag'>{d['num_classes']} classes</span>"
                    f"  <span class='reg-tag' style='color:{MUTED};'>{d.get('created_at','')[:10]}</span>"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    with tab_new:
        with st.form("new_dataset"):
            c1, c2       = st.columns(2)
            name         = c1.text_input("Dataset Name")
            task_type    = c2.selectbox("Task Type", ["classification", "detection", "segmentation"])
            path         = st.text_input("Dataset Path", placeholder="/data/my_dataset")
            c3, c4       = st.columns(2)
            num_samples  = c3.number_input("Samples", min_value=0, value=0)
            num_classes  = c4.number_input("Classes", min_value=1, value=2)
            classes_raw  = st.text_input("Class Names (comma-separated)", placeholder="cat, dog, bird")
            description  = st.text_area("Description (optional)")
            if st.form_submit_button("Register Dataset", use_container_width=True):
                if name and path:
                    classes = [c.strip() for c in classes_raw.split(",") if c.strip()]
                    result = api_post("/datasets", {
                        "name": name, "task_type": task_type, "path": path,
                        "description": description, "num_samples": num_samples,
                        "num_classes": num_classes, "classes": classes, "extra_metadata": {},
                    })
                    if result:
                        st.success(f"✅ Dataset **{name}** registered!")
                else:
                    st.error("Name and path are required.")


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖  Models":
    st.markdown("<div class='page-title'>Model Registry</div>", unsafe_allow_html=True)
    tab_list, tab_new = st.tabs(["📋 Registered", "➕ Register New"])

    with tab_list:
        models_list = api_get("/models") or []
        if not models_list:
            st.info("No models registered yet.")
        else:
            task_color = {"classification": ACCENT, "detection": SUCCESS, "segmentation": PURPLE}
            fw_icon    = {"pytorch": "🔥", "tensorflow": "🟢", "onnx": "⚙️", "other": "📦"}
            for m in models_list:
                c    = task_color.get(m["task_type"], MUTED)
                icon = fw_icon.get(m.get("framework", "other"), "📦")
                st.markdown(
                    f"<div class='reg-card' style='border-left:3px solid {c};'>"
                    f"<div class='reg-title'>{icon} {m['name']} "
                    f"<span style='color:{MUTED};font-weight:400;font-size:0.85rem;'>v{m['version']}</span></div>"
                    f"<div class='reg-sub'>{(m.get('description') or 'No description')[:90]}</div>"
                    f"<div>"
                    f"  <span class='reg-tag'>{m['task_type']}</span>"
                    f"  <span class='reg-tag'>{m.get('framework','other')}</span>"
                    f"  <span class='reg-tag' style='color:{MUTED};'>{m.get('created_at','')[:10]}</span>"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    with tab_new:
        with st.form("new_model"):
            c1, c2      = st.columns(2)
            name        = c1.text_input("Model Name",  placeholder="e.g. ResNet50")
            version     = c2.text_input("Version",     placeholder="e.g. 2.1.0")
            c3, c4      = st.columns(2)
            task_type   = c3.selectbox("Task Type",  ["classification", "detection", "segmentation"])
            framework   = c4.selectbox("Framework",  ["pytorch", "tensorflow", "onnx", "other"])
            model_path  = st.text_input("Model Path (optional)", placeholder="/models/resnet50.pt")
            description = st.text_area("Description (optional)")
            if st.form_submit_button("Register Model", use_container_width=True):
                if name and version:
                    result = api_post("/models", {
                        "name": name, "version": version, "task_type": task_type,
                        "framework": framework, "model_path": model_path or None,
                        "description": description, "config": {},
                    })
                    if result:
                        st.success(f"✅ Model **{name} v{version}** registered!")
                else:
                    st.error("Name and version are required.")
