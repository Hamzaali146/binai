"""
CV Testing Framework — Streamlit Dashboard
Run: streamlit run dashboard/app.py
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="CV Testing Framework",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: white;
        margin-bottom: 0.5rem;
    }
    .metric-card .label { font-size: 0.75rem; opacity: 0.8; text-transform: uppercase; letter-spacing: 0.1em; }
    .metric-card .value { font-size: 2rem; font-weight: 700; }
    .status-completed { color: #2e7d32; font-weight: 600; }
    .status-running   { color: #e65100; font-weight: 600; }
    .status-failed    { color: #c62828; font-weight: 600; }
    .status-pending   { color: #555; font-weight: 600; }
    section[data-testid="stSidebar"] { background: #0d1117; }
    section[data-testid="stSidebar"] .stRadio label { color: #e0e0e0 !important; }
</style>
""", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────
def api_get(path: str, **params):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API server. Make sure the backend is running on port 8000.")
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


def fmt_metric(val, fmt=".4f"):
    if val is None:
        return "—"
    return f"{val:{fmt}}"


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 CV Testing")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Overview", "🧪 Evaluations", "🌩️ Scenarios", "⚖️ Model Comparison",
         "❌ Failure Analysis", "📁 Datasets", "🤖 Models"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("CV Testing Framework v1.0")
    st.caption("© 2024 MLOps Platform")


# ── Overview ──────────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title("📊 CV Testing Overview")

    evals = api_get("/evaluations", limit=200) or []
    models_list = api_get("/models") or []
    datasets_list = api_get("/datasets") or []

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    completed = [e for e in evals if e["status"] == "completed"]
    failed = [e for e in evals if e["status"] == "failed"]
    running = [e for e in evals if e["status"] == "running"]

    col1.metric("Total Evaluations", len(evals))
    col2.metric("Completed", len(completed), delta=None)
    col3.metric("Models Registered", len(models_list))
    col4.metric("Datasets", len(datasets_list))

    if not evals:
        st.info("No evaluations yet. Start by registering a model and dataset.")
        st.stop()

    st.markdown("---")
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Evaluation Status Distribution")
        status_counts = pd.Series([e["status"] for e in evals]).value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        color_map = {"completed": "#2e7d32", "failed": "#c62828", "running": "#e65100", "pending": "#888"}
        fig = px.pie(status_counts, values="Count", names="Status",
                     color="Status", color_discrete_map=color_map,
                     hole=0.45)
        fig.update_layout(margin=dict(t=0, b=0), height=280)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Recent Evaluations")
        recent = sorted(evals, key=lambda e: e.get("created_at", ""), reverse=True)[:8]
        for ev in recent:
            status = ev["status"]
            css_class = f"status-{status}"
            st.markdown(
                f"**{ev['name']}** — "
                f"<span class='{css_class}'>{status.upper()}</span>",
                unsafe_allow_html=True,
            )

    # Metrics trend for completed evaluations
    if completed:
        st.markdown("---")
        st.subheader("Primary Metric Trend (Completed Evaluations)")
        rows = []
        for ev in sorted(completed, key=lambda e: e.get("created_at", "")):
            m = ev.get("metrics") or {}
            primary = (m.get("accuracy") or m.get("map_50") or m.get("mean_iou"))
            if primary is not None:
                rows.append({"Evaluation": ev["name"], "Score": primary,
                             "Created": ev.get("created_at", "")[:10]})
        if rows:
            df = pd.DataFrame(rows)
            fig2 = px.line(df, x="Created", y="Score", text="Evaluation",
                           markers=True, labels={"Score": "Primary Metric"})
            fig2.update_traces(textposition="top center", textfont_size=10)
            fig2.update_layout(height=300, margin=dict(t=20, b=20))
            st.plotly_chart(fig2, use_container_width=True)


# ── Evaluations ───────────────────────────────────────────────────────────────
elif page == "🧪 Evaluations":
    st.title("🧪 Evaluations")
    tab_list, tab_new = st.tabs(["📋 All Evaluations", "➕ New Evaluation"])

    with tab_list:
        evals = api_get("/evaluations", limit=100) or []
        if not evals:
            st.info("No evaluations found.")
        else:
            df = pd.DataFrame([{
                "ID": e["id"][:8] + "...",
                "Name": e["name"],
                "Task": e.get("task_type", ""),
                "Status": e["status"].upper(),
                "Samples": e.get("num_samples", 0),
                "Failures": e.get("num_failures", 0),
                "Duration (s)": e.get("duration_seconds") or "—",
                "Created": e.get("created_at", "")[:16],
            } for e in evals])
            st.dataframe(df, use_container_width=True)

            st.markdown("---")
            st.subheader("Evaluation Details")
            selected_name = st.selectbox("Select evaluation", [e["name"] for e in evals])
            selected = next((e for e in evals if e["name"] == selected_name), None)
            if selected:
                st.markdown(f"**Status:** `{selected['status']}`")
                m = selected.get("metrics") or {}
                if m:
                    st.markdown("#### Baseline Metrics")
                    metric_cols = st.columns(4)
                    metric_keys = [
                        ("accuracy", "Accuracy"),
                        ("f1_score", "F1 Score"),
                        ("map_50", "mAP@50"),
                        ("mean_iou", "Mean IoU"),
                        ("precision", "Precision"),
                        ("recall", "Recall"),
                    ]
                    col_idx = 0
                    for key, label in metric_keys:
                        val = m.get(key)
                        if val is not None:
                            metric_cols[col_idx % 4].metric(label, f"{val:.4f}")
                            col_idx += 1

                sr = selected.get("scenario_results") or []
                if sr:
                    st.markdown("#### Scenario Results")
                    sc_rows = []
                    for s in sr:
                        sc_m = s.get("metrics") or {}
                        primary = (sc_m.get("accuracy") or sc_m.get("map_50") or sc_m.get("mean_iou") or 0)
                        sc_rows.append({
                            "Scenario": s["scenario_name"],
                            "Primary Metric": f"{primary:.4f}",
                            "Degradation %": f"{s.get('degradation_pct', 0) or 0:.1f}%",
                            "Failures": s.get("num_failures", 0),
                        })
                    st.dataframe(pd.DataFrame(sc_rows), use_container_width=True)

                if m.get("per_class_metrics"):
                    st.markdown("#### Per-Class Metrics")
                    pc = m["per_class_metrics"]
                    pc_rows = [{"Class": cls, **vals} for cls, vals in pc.items()]
                    st.dataframe(pd.DataFrame(pc_rows), use_container_width=True)

    with tab_new:
        st.subheader("Create New Evaluation")
        models_list = api_get("/models") or []
        datasets_list = api_get("/datasets") or []

        if not models_list:
            st.warning("No models registered. Go to the Models section first.")
        elif not datasets_list:
            st.warning("No datasets registered. Go to the Datasets section first.")
        else:
            with st.form("new_eval"):
                eval_name = st.text_input("Evaluation Name", placeholder="e.g. ResNet50 v2 Baseline")
                model_choice = st.selectbox("Model", [f"{m['name']} v{m['version']}" for m in models_list])
                dataset_choice = st.selectbox("Dataset", [d["name"] for d in datasets_list])
                conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
                iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, 0.5, 0.05)
                max_samples = st.number_input("Max Samples (0 = all)", min_value=0, value=0)

                st.markdown("**Scenarios**")
                sc_options = api_get("/scenarios") or []
                selected_scenarios = []
                for sc in sc_options:
                    if st.checkbox(sc["name"], help=sc.get("description", "")):
                        selected_scenarios.append({"name": sc["name"], "enabled": True, "params": {}})

                submitted = st.form_submit_button("🚀 Run Evaluation")
                if submitted and eval_name:
                    model_obj = next(m for m in models_list
                                     if f"{m['name']} v{m['version']}" == model_choice)
                    dataset_obj = next(d for d in datasets_list if d["name"] == dataset_choice)
                    payload = {
                        "name": eval_name,
                        "model_id": model_obj["id"],
                        "dataset_id": dataset_obj["id"],
                        "config": {
                            "confidence_threshold": conf_threshold,
                            "iou_threshold": iou_threshold,
                            "max_samples": max_samples if max_samples > 0 else None,
                            "scenarios": selected_scenarios,
                        },
                    }
                    result = api_post("/evaluations", payload)
                    if result:
                        st.success(f"Evaluation '{eval_name}' started! ID: {result['id'][:8]}...")


# ── Scenarios ─────────────────────────────────────────────────────────────────
elif page == "🌩️ Scenarios":
    st.title("🌩️ Robustness Scenarios")
    scenarios = api_get("/scenarios") or []

    if not scenarios:
        st.info("No scenarios available.")
    else:
        cols = st.columns(3)
        for i, sc in enumerate(scenarios):
            with cols[i % 3]:
                st.markdown(f"""
                <div style='background:white;border-radius:8px;padding:1rem;
                     box-shadow:0 2px 8px rgba(0,0,0,.08);margin-bottom:1rem;
                     border-left:4px solid #3f51b5;'>
                  <strong>🔧 {sc['name']}</strong><br>
                  <small style='color:#555;'>{sc.get('description','')}</small>
                </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Preview Scenario on Image")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    sc_name = st.selectbox("Scenario", [s["name"] for s in scenarios] if scenarios else [])
    if uploaded and sc_name:
        import io as _io
        files = {"image": (uploaded.name, uploaded.read(), uploaded.type)}
        try:
            r = requests.post(
                f"{API_BASE}/scenarios/preview",
                params={"scenario_name": sc_name, "params": "{}"},
                files=files, timeout=15,
            )
            if r.status_code == 200:
                col1, col2 = st.columns(2)
                uploaded.seek(0)
                col1.image(uploaded.read(), caption="Original", use_container_width=True)
                col2.image(_io.BytesIO(r.content), caption=f"After {sc_name}", use_container_width=True)
            else:
                st.error(f"Preview failed: {r.text}")
        except Exception as e:
            st.error(f"Could not reach API: {e}")


# ── Model Comparison ──────────────────────────────────────────────────────────
elif page == "⚖️ Model Comparison":
    st.title("⚖️ Model Comparison")
    evals = api_get("/evaluations", limit=100) or []
    completed = [e for e in evals if e["status"] == "completed"]
    if len(completed) < 2:
        st.info("You need at least 2 completed evaluations to compare models.")
    else:
        selected_evals = st.multiselect(
            "Select evaluations to compare",
            options=[e["name"] for e in completed],
            default=[e["name"] for e in completed[:2]],
        )
        selected_ids = [e["id"] for e in completed if e["name"] in selected_evals]

        if len(selected_ids) >= 2:
            result = api_post("/evaluations/compare", {
                "evaluation_ids": selected_ids,
                "metrics": ["accuracy", "f1_score", "map_50", "mean_iou", "precision", "recall"],
            })
            if result:
                st.success(f"Best model: **{result.get('best_model', '—')}**")

                # Radar chart
                metric_keys = ["accuracy", "f1_score", "map_50", "mean_iou"]
                fig = go.Figure()
                for ev_data in result["evaluations"]:
                    m = ev_data.get("metrics") or {}
                    values = [m.get(k, 0) or 0 for k in metric_keys]
                    values += values[:1]  # close the loop
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=metric_keys + metric_keys[:1],
                        fill="toself",
                        name=f"{ev_data['model_name']} v{ev_data['model_version']}",
                    ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title="Metric Comparison Radar",
                    height=420,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Table
                rows = []
                for ev_data in result["evaluations"]:
                    m = ev_data.get("metrics") or {}
                    rows.append({
                        "Model": ev_data["model_name"],
                        "Version": ev_data.get("model_version", "?"),
                        "Accuracy": fmt_metric(m.get("accuracy")),
                        "F1": fmt_metric(m.get("f1_score")),
                        "mAP@50": fmt_metric(m.get("map_50")),
                        "mIoU": fmt_metric(m.get("mean_iou")),
                        "Failures": ev_data.get("num_failures", 0),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ── Failure Analysis ──────────────────────────────────────────────────────────
elif page == "❌ Failure Analysis":
    st.title("❌ Failure Analysis")
    evals = api_get("/evaluations", limit=100) or []
    completed = [e for e in evals if e["status"] == "completed"]
    if not completed:
        st.info("No completed evaluations yet.")
    else:
        eval_name = st.selectbox("Select evaluation", [e["name"] for e in completed])
        ev = next(e for e in completed if e["name"] == eval_name)
        failures = api_get(f"/evaluations/{ev['id']}/failures", limit=100) or []

        if not failures:
            st.success("No failure cases recorded for this evaluation!")
        else:
            st.metric("Total Failures", ev.get("num_failures", 0))
            df = pd.DataFrame([{
                "Image ID": f.get("image_id", ""),
                "Error Type": f.get("error_type", ""),
                "Confidence": fmt_metric(f.get("confidence"), ".3f"),
                "IoU": fmt_metric(f.get("iou_score"), ".3f"),
                "Scenario": f.get("scenario") or "baseline",
            } for f in failures])

            col1, col2 = st.columns(2)
            with col1:
                error_counts = df["Error Type"].value_counts().reset_index()
                error_counts.columns = ["Error Type", "Count"]
                fig = px.bar(error_counts, x="Error Type", y="Count",
                             color="Error Type", title="Failure Types")
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                sc_counts = df["Scenario"].value_counts().reset_index()
                sc_counts.columns = ["Scenario", "Count"]
                fig2 = px.pie(sc_counts, values="Count", names="Scenario",
                              title="Failures by Scenario", hole=0.4)
                fig2.update_layout(height=300)
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("---")
            st.subheader("Failure Case Table")
            error_filter = st.selectbox("Filter by error type", ["All"] + df["Error Type"].unique().tolist())
            filtered = df if error_filter == "All" else df[df["Error Type"] == error_filter]
            st.dataframe(filtered, use_container_width=True)


# ── Datasets ──────────────────────────────────────────────────────────────────
elif page == "📁 Datasets":
    st.title("📁 Datasets")
    tab_list, tab_new = st.tabs(["📋 Registered Datasets", "➕ Register New"])

    with tab_list:
        datasets_list = api_get("/datasets") or []
        if not datasets_list:
            st.info("No datasets registered yet.")
        else:
            df = pd.DataFrame([{
                "Name": d["name"],
                "Task": d["task_type"],
                "Samples": d["num_samples"],
                "Classes": d["num_classes"],
                "Path": d["path"][:40] + "...",
                "Created": d.get("created_at", "")[:10],
            } for d in datasets_list])
            st.dataframe(df, use_container_width=True)

    with tab_new:
        with st.form("new_dataset"):
            name = st.text_input("Dataset Name")
            task_type = st.selectbox("Task Type", ["classification", "detection", "segmentation"])
            path = st.text_input("Dataset Path", placeholder="/data/my_dataset")
            description = st.text_area("Description (optional)")
            num_samples = st.number_input("Number of Samples", min_value=0, value=0)
            num_classes = st.number_input("Number of Classes", min_value=1, value=2)
            classes_raw = st.text_input("Class Names (comma-separated)", placeholder="cat, dog, bird")
            submitted = st.form_submit_button("Register Dataset")
            if submitted and name and path:
                classes = [c.strip() for c in classes_raw.split(",") if c.strip()]
                payload = {
                    "name": name, "task_type": task_type, "path": path,
                    "description": description, "num_samples": num_samples,
                    "num_classes": num_classes, "classes": classes, "extra_metadata": {},
                }
                result = api_post("/datasets", payload)
                if result:
                    st.success(f"Dataset '{name}' registered with ID: {result['id'][:8]}...")


# ── Models ────────────────────────────────────────────────────────────────────
elif page == "🤖 Models":
    st.title("🤖 Model Registry")
    tab_list, tab_new = st.tabs(["📋 Registered Models", "➕ Register New"])

    with tab_list:
        models_list = api_get("/models") or []
        if not models_list:
            st.info("No models registered yet.")
        else:
            df = pd.DataFrame([{
                "Name": m["name"],
                "Version": m["version"],
                "Task": m["task_type"],
                "Framework": m.get("framework") or "—",
                "Description": (m.get("description") or "")[:40],
                "Created": m.get("created_at", "")[:10],
            } for m in models_list])
            st.dataframe(df, use_container_width=True)

    with tab_new:
        with st.form("new_model"):
            name = st.text_input("Model Name", placeholder="e.g. ResNet50")
            version = st.text_input("Version", placeholder="e.g. 2.1.0")
            task_type = st.selectbox("Task Type", ["classification", "detection", "segmentation"])
            framework = st.selectbox("Framework", ["pytorch", "tensorflow", "onnx", "other"])
            model_path = st.text_input("Model Path (optional)", placeholder="/models/resnet50.pt")
            description = st.text_area("Description (optional)")
            submitted = st.form_submit_button("Register Model")
            if submitted and name and version:
                payload = {
                    "name": name, "version": version, "task_type": task_type,
                    "framework": framework, "model_path": model_path or None,
                    "description": description, "config": {},
                }
                result = api_post("/models", payload)
                if result:
                    st.success(f"Model '{name} v{version}' registered!")
