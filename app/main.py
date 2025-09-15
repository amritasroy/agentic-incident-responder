from __future__ import annotations
import argparse, yaml, pathlib
from rich.console import Console
from rich.markdown import Markdown
from app.graph import build_graph, AgentState

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "scenarios"
console = Console()

def run_scenario(name: str, verbose: bool = False):
    with open(DATA_DIR / f"{name}.yaml") as f:
        meta = yaml.safe_load(f)
    state = AgentState(scenario=name, description=meta["description"])

    app = build_graph()
    console.rule(f"[bold]Incident: {name}")

    result = app.invoke(state)        # <- returns a dict-like state
    report_md = result.get("report_md", "")
    console.rule("[bold]Final Report")
    console.print(Markdown(report_md))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    run_scenario(args.scenario, args.verbose)
