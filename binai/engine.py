"""
Public EvaluationEngine — thin re-export of the backend engine
with a friendlier, library-facing interface.
"""
from backend.core.evaluation.engine import EvaluationEngine

__all__ = ["EvaluationEngine"]
