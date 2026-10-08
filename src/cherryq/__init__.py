"""CherryQ: predictive cash scenarios plus classical/quantum payment optimisation."""

from .benchmark import ScenarioBenchmark, benchmark_scenario, benchmark_scenarios
from .classical import ClassicalBenchmark, benchmark_classical, solve_milp
from .pipeline import CherryQPipelineReport, run_demo_pipeline
from .problem import (
    Invoice,
    PaymentDependency,
    PaymentProblem,
    PaymentSolution,
    QuboModel,
    build_example_problem,
)
from .recommendation import (
    CherryQRecommendation,
    ScenarioPlan,
    build_explainable_recommendation,
)
from .scenarios import (
    CashScenario,
    CustomerReceivable,
    PredictedReceipt,
    build_cash_scenarios,
    predict_receivables,
)

__all__ = [
    "CashScenario",
    "CherryQPipelineReport",
    "CherryQRecommendation",
    "ClassicalBenchmark",
    "CustomerReceivable",
    "Invoice",
    "PaymentDependency",
    "PaymentProblem",
    "PaymentSolution",
    "PredictedReceipt",
    "QuboModel",
    "ScenarioBenchmark",
    "ScenarioPlan",
    "benchmark_classical",
    "benchmark_scenario",
    "benchmark_scenarios",
    "build_cash_scenarios",
    "build_example_problem",
    "build_explainable_recommendation",
    "predict_receivables",
    "run_demo_pipeline",
    "solve_milp",
]
