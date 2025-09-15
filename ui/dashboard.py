# import streamlit as st
# # ensure project root (the folder that contains "app/") is importable
# import sys, pathlib
# ROOT = pathlib.Path(__file__).resolve().parents[1]
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))


# from app.graph import build_graph, AgentState


# st.title('Agentic Incident Responder')
# scenario = st.text_input('Scenario name', 'bearing_wear_03')
# desc = st.text_area('Description', 'High vibration spike reported on sensor_A.')
# if st.button('Run'):
#     app = build_graph()
#     out = app.invoke(AgentState(scenario=scenario, description=desc))
#     st.subheader('Final Report')
#     st.markdown(out.get("report_md", "*(no report)*"))

#----------------
# ui/dashboard.py
# ensure project root (the folder that contains "app/") is importable
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import streamlit as st
import pathlib, yaml, os
from app.graph import build_graph, AgentState

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCEN_DIR = ROOT / "data" / "scenarios"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "eval" / "results"

st.set_page_config(page_title="Agentic Incident Responder", layout="wide")
st.title("🛠️ Agentic Incident Responder (Industrial IoT)")

# Sidebar controls
with st.sidebar:
    st.header("Run Settings")
    # Scenario selector from available YAMLs
    scen_files = sorted(p.stem for p in SCEN_DIR.glob("*.yaml"))
    scenario = st.selectbox("Scenario", options=scen_files, index=0 if scen_files else None)
    # Load description from YAML by default
    default_desc = ""
    if scenario:
        meta = yaml.safe_load((SCEN_DIR / f"{scenario}.yaml").read_text(encoding="utf-8"))
        default_desc = meta.get("description", "")
    description = st.text_area("Description", value=default_desc, height=80)

    force_ticket = st.checkbox("Force ticket (demo mode)", value=False,
                               help="Bypass guardrails; always write a ticket.")
    # Let user pick a HF model (optional)
    hf_model = st.text_input("HuggingFace model (optional)", value=os.getenv("HUGGINGFACE_MODEL", "distilgpt2"),
                             help="Set HUGGINGFACE_MODEL env var for persistent change.")
    run_btn = st.button("▶ Run Agent")

    st.divider()
    eval_btn = st.button("📊 Run Evaluation (all scenarios)")

# Run agent
if run_btn and scenario:
    # Optional: set HUGGINGFACE_MODEL for this session/run
    os.environ["HUGGINGFACE_MODEL"] = hf_model or os.environ.get("HUGGINGFACE_MODEL", "distilgpt2")
    # Optional: force ticket
    os.environ["FORCE_TICKET"] = "1" if force_ticket else "0"

    st.write(f"**Running**: `{scenario}`")
    app = build_graph()
    out = app.invoke(AgentState(scenario=scenario, description=description))

    # Top-level stats
    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", f"{float(out.get('confidence', 0.0)):.2f}")
    ev = out.get("evidence", {}) or {}
    sources = sum([
        1 if float(ev.get("anomaly_score", 0.0)) >= 0.6 else 0,
        1 if len(ev.get("logs", []) or []) > 0 else 0,
        1 if len(ev.get("kb", []) or []) > 0 else 0,
    ])
    c2.metric("Evidence sources", str(sources))
    # Try to find a fresh ticket in MODELS_DIR (by mtime)
    ticket_path = None
    if MODELS_DIR.exists():
        tickets = sorted(MODELS_DIR.glob("ticket_*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        ticket_path = tickets[0] if tickets else None
    c3.metric("Ticket", "created" if ticket_path else "skipped")

    # Tabs: Evidence / RAG / Report / Trace
    t_ev, t_rag, t_rep, t_trace = st.tabs(["Evidence", "RAG", "Final Report", "Trace"])

    with t_ev:
        st.subheader("Timeseries anomaly")
        st.write(f"**Anomaly Score:** {float(ev.get('anomaly_score', 0.0)):.2f}")
        st.subheader("Log hits (trimmed)")
        st.json(ev.get("logs", [])[:5])

    with t_rag:
        st.subheader("Citations")
        kb = ev.get("kb", [])
        if not kb:
            st.info("No KB notes retrieved.")
        else:
            for n in kb:
                st.markdown(f"- **{n.get('id','?')}** (score={float(n.get('score',0)):.3f})")
                st.code(n.get("snippet","")[:300])

    with t_rep:
        st.subheader("Incident Report")
        st.markdown(out.get("report_md", "*(no report)*"))
        if ticket_path and ticket_path.exists():
            with open(ticket_path, "r", encoding="utf-8") as f:
                st.download_button(
                    "⬇ Download ticket (.md)",
                    data=f.read(),
                    file_name=ticket_path.name,
                    mime="text/markdown",
                )

    with t_trace:
        st.subheader("Trace")
        # The trace is embedded in the report markdown by write_node; we also try to surface tool logs:
        st.markdown("*(Trace appended at the bottom of the report.)*")

# Optional: run evaluation
if eval_btn:
    st.write("Running evaluation across all scenarios…")
    import subprocess, sys
    # Run module so imports resolve from project root
    cmd = [sys.executable, "-m", "eval.harness"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    st.code(proc.stdout or "(no stdout)")
    if proc.returncode != 0:
        st.error(proc.stderr)
    # Show latest artifacts if present
    csv_path = RESULTS_DIR / "metrics.csv"
    png_path = RESULTS_DIR / "confusion_matrix.png"
    if csv_path.exists():
        import pandas as pd
        st.success(f"Wrote {csv_path}")
        st.dataframe(pd.read_csv(csv_path))
    if png_path.exists():
        st.image(str(png_path), caption="Confusion Matrix")
