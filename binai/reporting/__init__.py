"""
binai.reporting
----------------
Generate HTML and JSON reports from evaluation results.

Examples
--------
>>> from binai.reporting import save_report, generate_html_report

>>> # Save a JSON report to disk
>>> path = save_report(results, fmt="json")

>>> # Get HTML string (embed in a web page, email, etc.)
>>> html = generate_html_report(results, model_info="ResNet50 v2")
"""

from backend.core.reporting.report_generator import (
    save_report,
    generate_html_report,
)

__all__ = ["save_report", "generate_html_report"]
