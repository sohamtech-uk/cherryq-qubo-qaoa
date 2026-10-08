"""CherryQ: predictive cash scenarios plus classical/quantum payment optimisation.

The package root keeps imports lazy so lightweight hardware utilities can run in
CSC's IQM/Qiskit environment without importing the optional forecasting stack.
"""

from importlib import import_module

_EXPORTS = {
    "ScenarioBenchmark": ("benchmark", "ScenarioBenchmark"),
    "benchmark_scenario": ("benchmark", "benchmark_scenario"),
    "benchmark_scenarios": ("benchmark", "benchmark_scenarios"),
    "ClassicalBenchmark": ("classical", "ClassicalBenchmark"),
    "benchmark_classical": ("classical", "benchmark_classical"),
    "solve_milp": ("classical", "solve_milp"),
    "CherryQPipelineReport": ("pipeline", "CherryQPipelineReport"),
    "run_demo_pipeline": ("pipeline", "run_demo_pipeline"),
    "Invoice": ("problem", "Invoice"),
    "PaymentDependency": ("problem", "PaymentDependency"),
    "PaymentProblem": ("problem", "PaymentProblem"),
    "PaymentSolution": ("problem", "PaymentSolution"),
    "QuboModel": ("problem", "QuboModel"),
    "build_example_problem": ("problem", "build_example_problem"),
    "build_ten_invoice_problem": ("problem", "build_ten_invoice_problem"),
    "CherryQRecommendation": ("recommendation", "CherryQRecommendation"),
    "ScenarioPlan": ("recommendation", "ScenarioPlan"),
    "build_explainable_recommendation": (
        "recommendation",
        "build_explainable_recommendation",
    ),
    "CashScenario": ("scenarios", "CashScenario"),
    "CustomerReceivable": ("scenarios", "CustomerReceivable"),
    "PredictedReceipt": ("scenarios", "PredictedReceipt"),
    "build_cash_scenarios": ("scenarios", "build_cash_scenarios"),
    "predict_receivables": ("scenarios", "predict_receivables"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(f".{module_name}", __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
