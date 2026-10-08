from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Sequence

from .classical import benchmark_classical
from .problem import (
    PaymentProblem,
    build_example_problem,
    build_ten_invoice_problem,
)


@dataclass(frozen=True)
class PenaltyCheck:
    penalty_gbp: float
    best_feasible_energy: float
    best_infeasible_energy: float
    feasibility_margin: float
    safe: bool


@dataclass(frozen=True)
class ScalingBenchmark:
    label: str
    invoice_count: int
    business_decision_space: int
    available_budget_gbp: int
    exact_paid_invoice_ids: tuple[str, ...]
    exact_spend_gbp: int
    exact_business_loss_gbp: int
    exact_runtime_ms: float
    milp_paid_invoice_ids: tuple[str, ...]
    milp_business_loss_gbp: int
    milp_runtime_ms: float
    milp_gap: float | None
    greedy_paid_invoice_ids: tuple[str, ...]
    greedy_business_loss_gbp: int
    greedy_gap_gbp: int
    greedy_runtime_ms: float
    selected_penalty_gbp: float
    penalty_checks: tuple[PenaltyCheck, ...]
    qubo_variable_count: int
    slack_qubit_count: int
    qubo_state_space: int
    qubo_quadratic_terms: int
    p1_logical_cx_estimate: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _check_penalty(problem: PaymentProblem, penalty_gbp: float) -> PenaltyCheck:
    qubo = problem.build_qubo(penalty_gbp=penalty_gbp)

    best_feasible = float("inf")
    best_infeasible = float("inf")
    for bits in product((0, 1), repeat=len(qubo.variable_names)):
        values = dict(zip(qubo.variable_names, bits, strict=True))
        energy = qubo.energy(values)
        if abs(qubo.constraint_residual(values)) < 1e-9:
            best_feasible = min(best_feasible, energy)
        else:
            best_infeasible = min(best_infeasible, energy)

    margin = best_infeasible - best_feasible
    return PenaltyCheck(
        penalty_gbp=float(penalty_gbp),
        best_feasible_energy=float(best_feasible),
        best_infeasible_energy=float(best_infeasible),
        feasibility_margin=float(margin),
        safe=bool(margin > 0),
    )


def benchmark_problem(
    problem: PaymentProblem,
    *,
    label: str,
    penalty_grid: Sequence[float] = (100, 200, 250, 500, 1000),
) -> ScalingBenchmark:
    classical = benchmark_classical(problem)

    greedy_started = perf_counter()
    greedy = problem.solve_greedy()
    greedy_runtime_ms = (perf_counter() - greedy_started) * 1000.0

    checks = tuple(_check_penalty(problem, penalty) for penalty in penalty_grid)
    selected = next((item for item in checks if item.safe), None)
    if selected is None:
        raise RuntimeError("No safe QUBO penalty found in the supplied grid")

    qubo = problem.build_qubo(penalty_gbp=selected.penalty_gbp)
    qubo_values, _ = qubo.solve_exact()
    qubo_decision = problem.supplier_decision_from_values(
        qubo_values,
        qubo.variable_names,
    )
    qubo_paid = tuple(
        invoice_id
        for invoice_id in problem.invoice_ids
        if qubo_decision[invoice_id]
    )
    if qubo_paid != classical.exact_solution.paid_invoice_ids:
        raise RuntimeError(
            "Selected QUBO penalty does not reproduce the exact business optimum"
        )

    quadratic_terms = sum(
        1 for coefficient in qubo.quadratic.values() if abs(coefficient) > 1e-12
    )

    return ScalingBenchmark(
        label=label,
        invoice_count=len(problem.invoices),
        business_decision_space=1 << len(problem.invoices),
        available_budget_gbp=problem.available_budget_gbp,
        exact_paid_invoice_ids=classical.exact_solution.paid_invoice_ids,
        exact_spend_gbp=classical.exact_solution.spend_gbp,
        exact_business_loss_gbp=classical.exact_solution.business_loss_gbp,
        exact_runtime_ms=classical.exact_runtime_ms,
        milp_paid_invoice_ids=classical.milp_solution.paid_invoice_ids,
        milp_business_loss_gbp=classical.milp_solution.business_loss_gbp,
        milp_runtime_ms=classical.milp_runtime_ms,
        milp_gap=classical.milp_gap,
        greedy_paid_invoice_ids=greedy.paid_invoice_ids,
        greedy_business_loss_gbp=greedy.business_loss_gbp,
        greedy_gap_gbp=(
            greedy.business_loss_gbp
            - classical.exact_solution.business_loss_gbp
        ),
        greedy_runtime_ms=greedy_runtime_ms,
        selected_penalty_gbp=selected.penalty_gbp,
        penalty_checks=checks,
        qubo_variable_count=len(qubo.variable_names),
        slack_qubit_count=(
            len(qubo.variable_names) - len(problem.invoices)
        ),
        qubo_state_space=1 << len(qubo.variable_names),
        qubo_quadratic_terms=quadratic_terms,
        # CherryQ's current p=1 cost layer implements each ZZ term as
        # CX-RZ-CX. This is a logical resource count, not a hardware count.
        p1_logical_cx_estimate=2 * quadratic_terms,
    )


def run_scaling_benchmark() -> tuple[ScalingBenchmark, ScalingBenchmark]:
    return (
        benchmark_problem(build_example_problem(), label="5-invoice"),
        benchmark_problem(build_ten_invoice_problem(), label="10-invoice"),
    )


def _print_report(results: Sequence[ScalingBenchmark]) -> None:
    print("CherryQ scaling benchmark")
    print("========================")
    for result in results:
        print()
        print(f"{result.label}:")
        print(
            f"  business combinations: {result.business_decision_space:,}"
        )
        print(
            f"  exact/MILP optimum: {'+'.join(result.exact_paid_invoice_ids)}; "
            f"spend £{result.exact_spend_gbp:,}; "
            f"loss £{result.exact_business_loss_gbp:,}"
        )
        print(
            f"  greedy: {'+'.join(result.greedy_paid_invoice_ids)}; "
            f"loss £{result.greedy_business_loss_gbp:,}; "
            f"gap £{result.greedy_gap_gbp:,}"
        )
        print(
            f"  runtimes: exact {result.exact_runtime_ms:.3f} ms; "
            f"MILP {result.milp_runtime_ms:.3f} ms; "
            f"greedy {result.greedy_runtime_ms:.3f} ms"
        )
        print(
            f"  QUBO: {result.qubo_variable_count} binary variables "
            f"({result.slack_qubit_count} slack), "
            f"{result.qubo_state_space:,} states, "
            f"{result.qubo_quadratic_terms} quadratic couplings"
        )
        print(
            f"  p=1 logical entangling estimate: "
            f"{result.p1_logical_cx_estimate} CX"
        )
        print(
            f"  smallest safe tested penalty: "
            f"£{result.selected_penalty_gbp:,.0f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare CherryQ's verified 5- and 10-invoice scaling cases"
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for a machine-readable scaling report",
    )
    args = parser.parse_args()

    results = run_scaling_benchmark()
    _print_report(results)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps([item.as_dict() for item in results], indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
