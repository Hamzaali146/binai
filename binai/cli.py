"""
binai CLI
----------
Usage examples:
  binai info
  binai scenarios
  binai evaluate --dataset ./data/myds --task classification --report html
  binai server
  binai dashboard
"""
import sys
import json
import click


@click.group()
@click.version_option(package_name="binai")
def main():
    """binai — Automated CV Model Testing Framework"""
    pass


@main.command()
def info():
    """Show installed version and available components."""
    from binai._version import __version__
    from binai.scenarios import list_scenarios

    click.echo(f"\nbinai v{__version__}")
    click.echo(f"Python {sys.version.split()[0]}\n")
    click.echo(f"Registered scenarios : {len(list_scenarios())}")

    try:
        import fastapi
        click.echo(f"FastAPI              : {fastapi.__version__} (server available)")
    except ImportError:
        click.echo("FastAPI              : not installed  (pip install binai[server])")

    try:
        import streamlit
        click.echo(f"Streamlit            : {streamlit.__version__} (dashboard available)")
    except ImportError:
        click.echo("Streamlit            : not installed  (pip install binai[dashboard])")
    click.echo()


@main.command()
def scenarios():
    """List all available robustness scenarios."""
    from binai.scenarios import list_scenarios

    click.echo("\nAvailable scenarios:\n")
    groups = {
        "Blur":      ["gaussian_blur", "motion_blur", "median_blur"],
        "Noise":     ["gaussian_noise", "salt_pepper_noise", "speckle_noise"],
        "Lighting":  ["low_light", "overexposure", "shadow", "fog"],
        "Occlusion": ["random_occlusion", "grid_occlusion", "watermark_occlusion"],
        "Geometry":  ["rotation", "flip", "perspective", "zoom"],
    }
    sc_map = {s["name"]: s["description"] for s in list_scenarios()}

    for group, names in groups.items():
        click.echo(click.style(f"  {group}", bold=True))
        for name in names:
            desc = sc_map.get(name, "")
            click.echo(f"    {name:<28} {desc}")
        click.echo()


@main.command()
@click.option("--dataset", required=True, help="Path to dataset directory (must contain annotations.json)")
@click.option("--task", required=True, type=click.Choice(["classification", "detection", "segmentation"]), help="Task type")
@click.option("--scenarios", "scenario_list", multiple=True, default=[], help="Scenario names to test (repeat for multiple)")
@click.option("--max-samples", default=None, type=int, help="Limit number of samples")
@click.option("--report", type=click.Choice(["json", "html", "none"]), default="json", help="Report format")
@click.option("--out", default=None, help="Output path for report (default: auto-generated)")
def evaluate(dataset, task, scenario_list, max_samples, report, out):
    """Run an evaluation against a dataset using a stub predictor."""
    import numpy as np
    from binai import EvaluationEngine
    from binai.reporting import save_report

    click.echo(f"\n▶ Starting {task} evaluation on: {dataset}")

    # Stub predictor — replace with your real model
    def stub_predict(images):
        results = []
        for img in images:
            if task == "classification":
                scores = np.random.dirichlet(np.ones(3)).tolist()
                results.append({"class_id": int(np.argmax(scores)), "scores": scores})
            elif task == "detection":
                h, w = img.shape[:2]
                results.append([{
                    "class_id": 0,
                    "bbox": [10, 10, w - 10, h - 10],
                    "score": float(np.random.uniform(0.5, 0.95)),
                }])
            else:
                h, w = img.shape[:2]
                results.append(np.zeros((h, w), dtype=np.uint8))
        return results

    scenarios_cfg = [{"name": s, "enabled": True, "params": {}} for s in scenario_list]
    config = {
        "max_samples": max_samples,
        "scenarios": scenarios_cfg,
    }

    engine = EvaluationEngine(
        task_type=task,
        dataset_path=dataset,
        predict_fn=stub_predict,
    )

    try:
        results = engine.run(config)
    except FileNotFoundError as e:
        click.echo(click.style(f"\nERROR: {e}", fg="red"))
        click.echo("Make sure the dataset path contains an 'annotations.json' file.")
        sys.exit(1)

    # Print metrics
    m = results.get("baseline_metrics", {})
    click.echo("\nBaseline metrics:")
    for k, v in m.items():
        if isinstance(v, float):
            click.echo(f"  {k:<25} {v:.4f}")

    click.echo(f"\n  Samples evaluated : {results['num_samples']}")
    click.echo(f"  Failures logged   : {results['num_failures']}")
    click.echo(f"  Duration          : {results['duration_seconds']}s")

    # Scenario summary
    for sr in results.get("scenario_results", []):
        deg = sr.get("degradation_pct")
        deg_str = f"  ▼ {deg:.1f}%" if deg else ""
        click.echo(f"  {sr['scenario_name']:<28}{deg_str}")

    # Save report
    if report != "none":
        report_data = {
            "id": "cli-run",
            "name": f"{task} evaluation",
            "metrics": m,
            "scenario_results": results.get("scenario_results", []),
            "num_failures": results["num_failures"],
            "failure_cases": results.get("all_failures", [])[:20],
        }
        path = save_report(report_data, fmt=report)
        click.echo(f"\nReport saved: {path}")

    click.echo()


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to listen on")
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload")
def server(host, port, reload):
    """Start the REST API server."""
    try:
        import uvicorn
    except ImportError:
        click.echo("uvicorn not installed. Run: pip install binai[server]")
        sys.exit(1)

    click.echo(f"\nStarting API server on http://{host}:{port}")
    click.echo("API docs: http://localhost:{port}/docs\n")
    uvicorn.run("backend.main:app", host=host, port=port, reload=reload)


@main.command()
@click.option("--port", default=8501, help="Port to listen on")
def dashboard(port):
    """Launch the Streamlit analytics dashboard."""
    try:
        import streamlit.web.cli as stcli
    except ImportError:
        click.echo("Streamlit not installed. Run: pip install binai[dashboard]")
        sys.exit(1)

    import os
    from pathlib import Path

    dashboard_path = Path(__file__).parent.parent / "dashboard" / "app.py"
    sys.argv = [
        "streamlit", "run", str(dashboard_path),
        f"--server.port={port}",
        "--server.headless=true",
    ]
    sys.exit(stcli.main())
