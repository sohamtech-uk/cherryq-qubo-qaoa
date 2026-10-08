from __future__ import annotations

from dataclasses import dataclass

from .classical import ClassicalBenchmark, benchmark_classical
from .problem import PaymentProblem
from .quantum import QaoaRunResult, run_qaoa
from .scenarios import CashScenario


@dataclass(frozen=True)
class ScenarioBenchmark:
    scenario: CashScenario
    available_budget_gbp: int
    classical: ClassicalBenchmark
    qaoa_runs: tuple[QaoaRunResult, ...]

    @property
    def best_quantum_business_loss_gbp(self) -> int | None:
        losses = [
            run.best_sampled_business_loss_gbp
            for run in self.qaoa_runs
            if run.best_sampled_business_loss_gbp is not None
        ]
        return min(losses) if losses else None

    @property
    def best_quantum_gap_gbp(self) -> int | None:
        best = self.best_quantum_business_loss_gbp
        if best is None:
            return None
        return best - self.classical.exact_solution.business_loss_gbp

    def as_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario.scenario_id,
            "scenario_probability": self.scenario.probability,
            "arrived_receipts": self.scenario.arrived_receipt_ids,
            "extra_cash_gbp": self.scenario.extra_cash_gbp,
            "available_budget_gbp": self.available_budget_gbp,
            "classical": self.classical.as_dict(),
            "qaoa": [run.as_dict() for run in self.qaoa_runs],
            "best_quantum_business_loss_gbp": self.best_quantum_business_loss_gbp,
            "best_quantum_gap_gbp": self.best_quantum_gap_gbp,
        }


def benchmark_scenario(
    base_problem: PaymentProblem,
    scenario: CashScenario,
    include_quantum: bool = True,
    p_values: tuple[int, ...] = (1, 2),
    maxiter: int = 150,
    shots: int = 4096,
    seed: int = 42,
    penalty_gbp: float = 2000.0,
    relaxation_multistart: int = 64,
) -> ScenarioBenchmark:
    problem = scenario.to_payment_problem(base_problem)
    classical = benchmark_classical(problem)

    qaoa_runs: list[QaoaRunResult] = []
    if include_quantum:
        qubo = problem.build_qubo(penalty_gbp=penalty_gbp)
        for p in p_values:
            for method in ("standard", "warm-start"):
                qaoa_runs.append(
                    run_qaoa(
                        problem,
                        qubo,
                        method=method,
                        p=p,
                        maxiter=maxiter,
                        shots=shots,
                        seed=seed + p,
                        relaxation_multistart=relaxation_multistart,
                    )
                )

    return ScenarioBenchmark(
        scenario=scenario,
        available_budget_gbp=problem.available_budget_gbp,
        classical=classical,
        qaoa_runs=tuple(qaoa_runs),
    )


def benchmark_scenarios(
    base_problem: PaymentProblem,
    scenarios: tuple[CashScenario, ...],
    include_quantum: bool = True,
    p_values: tuple[int, ...] = (1, 2),
    maxiter: int = 150,
    shots: int = 4096,
    seed: int = 42,
    penalty_gbp: float = 2000.0,
    relaxation_multistart: int = 64,
    quantum_scenario_limit: int | None = None,
) -> tuple[ScenarioBenchmark, ...]:
    results: list[ScenarioBenchmark] = []

    quantum_ids: set[str] | None = None
    if include_quantum and quantum_scenario_limit is not None:
        if quantum_scenario_limit < 1:
            raise ValueError("quantum_scenario_limit must be positive when supplied")
        ranked = sorted(scenarios, key=lambda item: item.probability, reverse=True)
        selected = ranked[:quantum_scenario_limit]

        # Always benchmark the currently-cleared-cash scenario if it exists.
        current = next(
            (scenario for scenario in scenarios if not scenario.arrived_receipt_ids),
            None,
        )
        if current is not None and current not in selected:
            selected = selected[:-1] + [current] if selected else [current]
        quantum_ids = {scenario.scenario_id for scenario in selected}

    for index, scenario in enumerate(scenarios):
        scenario_quantum = include_quantum and (
            quantum_ids is None or scenario.scenario_id in quantum_ids
        )
        results.append(
            benchmark_scenario(
                base_problem=base_problem,
                scenario=scenario,
                include_quantum=scenario_quantum,
                p_values=p_values,
                maxiter=maxiter,
                shots=shots,
                seed=seed + index * 100,
                penalty_gbp=penalty_gbp,
                relaxation_multistart=relaxation_multistart,
            )
        )

    return tuple(results)
