# from __future__ import annotations
# import pathlib, yaml, csv
# from app.graph import build_graph, AgentState
# import matplotlib.pyplot as plt

# SCEN_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "scenarios"
# RES_DIR = pathlib.Path(__file__).resolve().parents[1] / "eval" / "results"
# RES_DIR.mkdir(parents=True, exist_ok=True)

# # Load scenario labels
# labels = {}
# for p in SCEN_DIR.glob("*.yaml"):
#     y = yaml.safe_load(p.read_text(encoding="utf-8"))
#     labels[p.stem] = y["label"]

# app = build_graph()
# rows = []
# for name, label in labels.items():
#     state = AgentState(scenario=name, description=f"Scenario {name}")
#     out = app.invoke(state)  # dict-like state (AddableValuesDict)

#     report = (out.get("report_md") or "").lower()
#     conf = float(out.get("confidence", 0.0))

#     # super-simple heuristic classifier off the report text
#     if "bearing" in report or "vibration" in report:
#         pred = "bearing_wear"
#     elif "packet" in report or "gateway" in report or "backhaul" in report:
#         pred = "network_packet_loss"
#     else:
#         pred = "unknown"

#     rows.append({
#         "scenario": name,
#         "label": label,
#         "pred": pred,
#         "correct": int(pred == label),
#         "confidence": conf
#     })

# # Write CSV
# metrics_csv = RES_DIR / "metrics.csv"
# with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
#     w = csv.DictWriter(f, fieldnames=rows[0].keys())
#     w.writeheader(); w.writerows(rows)

# # Confusion matrix (tiny demo)
# cats = sorted({r["label"] for r in rows} | {r["pred"] for r in rows})
# cm = {a: {b: 0 for b in cats} for a in cats}
# for r in rows:
#     cm[r["label"]][r["pred"]] += 1

# fig = plt.figure(figsize=(4,4))
# plt.imshow([[cm[a][b] for b in cats] for a in cats])
# plt.xticks(range(len(cats)), cats, rotation=45)
# plt.yticks(range(len(cats)), cats)
# plt.title("Confusion Matrix")
# plt.colorbar()
# plt.tight_layout()
# fig.savefig(RES_DIR / "confusion_matrix.png")
# print("Wrote", metrics_csv, "and confusion_matrix.png")

#---------------
# eval/harness.py
from __future__ import annotations
import argparse
import csv
import pathlib
import yaml
import matplotlib.pyplot as plt

from app.graph import build_graph, AgentState


SCEN_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "scenarios"
RES_DIR_DEFAULT = pathlib.Path(__file__).resolve().parents[1] / "eval" / "results"


def classify_from_evidence(state: dict) -> str:
    """
    Evidence-first labeler:
      1) Look at structured log hits (most reliable)
      2) Fall back to report text if needed
    """
    report = (state.get("report_md") or "").lower()
    ev = state.get("evidence") or {}
    logs = ev.get("logs") or []

    def in_logs(keywords) -> bool:
        for h in logs:
            s = str(h).lower()
            if any(k in s for k in keywords):
                return True
        return False

    # 1) Prioritize logs
    if in_logs(["packet", "gateway", "backhaul", "retry", "loss"]):
        return "network_packet_loss"
    if in_logs(["vibration", "bearing", "lubric", "temp"]):
        return "bearing_wear"

    # 2) Fall back to report text
    if any(k in report for k in ["packet", "gateway", "backhaul", "loss"]):
        return "network_packet_loss"
    if any(k in report for k in ["vibration", "bearing"]):
        return "bearing_wear"

    return "unknown"


def load_labels() -> dict[str, str]:
    labels: dict[str, str] = {}
    for p in sorted(SCEN_DIR.glob("*.yaml")):
        y = yaml.safe_load(p.read_text(encoding="utf-8"))
        labels[p.stem] = y["label"]
    return labels


def write_metrics(rows: list[dict], outdir: pathlib.Path) -> pathlib.Path:
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "label", "pred", "correct", "confidence"])
        w.writeheader()
        w.writerows(rows)
    return csv_path


def save_confusion_matrix(rows: list[dict], outdir: pathlib.Path) -> pathlib.Path:
    cats = sorted({r["label"] for r in rows} | {r["pred"] for r in rows})
    if not cats:
        return outdir / "confusion_matrix.png"

    # Build matrix
    index = {c: i for i, c in enumerate(cats)}
    mat = [[0 for _ in cats] for _ in cats]
    for r in rows:
        mat[index[r["label"]]][index[r["pred"]]] += 1

    # Plot
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(mat)
    plt.xticks(range(len(cats)), cats, rotation=45, ha="right")
    plt.yticks(range(len(cats)), cats)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.tight_layout()

    path = outdir / "confusion_matrix.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def run_eval(outdir: pathlib.Path) -> None:
    labels = load_labels()
    app = build_graph()

    rows = []
    for scen, gold in labels.items():
        state = AgentState(scenario=scen, description=f"Scenario {scen}")
        out = app.invoke(state)  # LangGraph dict-like state

        pred = classify_from_evidence(out)
        conf = float(out.get("confidence", 0.0))
        rows.append(
            {
                "scenario": scen,
                "label": gold,
                "pred": pred,
                "correct": int(pred == gold),
                "confidence": conf,
            }
        )

    csv_path = write_metrics(rows, outdir)
    img_path = save_confusion_matrix(rows, outdir)
    print(f"Wrote {csv_path} and {img_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate agent over scenarios and save metrics.")
    parser.add_argument(
        "--outdir",
        type=pathlib.Path,
        default=RES_DIR_DEFAULT,
        help="Directory to write results (CSV and PNG).",
    )
    args = parser.parse_args()
    run_eval(args.outdir)


if __name__ == "__main__":
    main()

