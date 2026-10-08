from __future__ import annotations

from dataclasses import dataclass

from .benchmark import ScenarioBenchmark, benchmark_scenarios
from .forecast import ForecastReport, generate_synthetic_payment_history, train_payment_classifier
from .problem import PaymentProblem, build_example_problem
from .recommendation import CherryQRecommendation, build_explainable_recommendation
from .scenarios import (
    CashScenario,
    PredictedReceipt,
    build_cash_scenarios,
    build_demo_receivables,
    predict_receivables,
)


@dataclass(frozen=True)
class CherryQPipelineReport:
    base_problem: PaymentProblem
    forecast: ForecastReport
    predicted_receipts: tuple[PredictedReceipt, ...]
    scenarios: tuple[CashScenario, ...]
    benchmarks: tuple[ScenarioBenchmark, ...]
    recommendation: CherryQRecommendation


def run_demo_pipeline(
    include_quantum: bool = False,
    p_values: tuple[int, ...] = (1, 2),
    maxiter: int = 150,
    shots: int = 4096,
    seed: int = 42,
    quantum_scenario_limit: int | None = None,
) -> CherryQPipelineReport:
    base_problem = build_example_problem()

    dataset = generate_synthetic_payment_history(n_samples=1200, seed=seed)
    forecast = train_payment_classifier(dataset, seed=seed)

    predicted_receipts = predict_receivables(
        forecast.model,
        build_demo_receivables(),
    )
    scenarios = build_cash_scenarios(predicted_receipts)

    benchmarks = benchmark_scenarios(
        base_problem=base_problem,
        scenarios=scenarios,
        include_quantum=include_quantum,
        p_values=p_values,
        maxiter=maxiter,
        shots=shots,
        seed=seed,
        quantum_scenario_limit=quantum_scenario_limit,
    )
    recommendation = build_explainable_recommendation(base_problem, benchmarks)

    return CherryQPipelineReport(
        base_problem=base_problem,
        forecast=forecast,
        predicted_receipts=predicted_receipts,
        scenarios=scenarios,
        benchmarks=benchmarks,
        recommendation=recommendation,
    )
