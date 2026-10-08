from __future__ import annotations

import argparse
import json
from pathlib import Path

from .problem import build_ten_invoice_problem
from .quantum import run_qaoa
from .scaling import benchmark_problem


def run_ten_invoice_experiment(
    *,
    include_qaoa: bool = False,
    maxiter: int = 80,
    shots: int = 2048,
    seed: int = 42,
    relaxation_multistart: int = 32,
) -> dict[str, object]:
    problem = build_ten_invoice_problem()
    scaling = benchmark_problem(problem, label="10-invoice")

    report: dict[str, object] = {
        "problem": scaling.as_dict(),
        "qaoa": None,
    }

    if include_qaoa:
        qubo = problem.build_qubo(
            penalty_gbp=scaling.selected_penalty_gbp
        )
        qaoa = run_qaoa(
            problem=problem,
            qubo=qubo,
            method="warm-start-x",
            p=1,
            maxiter=maxiter,
            shots=shots,
            seed=seed,
            relaxation_multistart=relaxation_multistart,
        )
        report["qaoa"] = qaoa.as_dict()

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the CherryQ 10-invoice SME scaling experiment"
    )
    parser.add_argument(
        "--qaoa",
        action="store_true",
        help="Also run the ideal statevector warm-start-X p=1 QAOA benchmark",
    )
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--relaxation-multistart", type=int, default=32)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    report = run_ten_invoice_experiment(
        include_qaoa=args.qaoa,
        maxiter=args.maxiter,
        shots=args.shots,
        seed=args.seed,
        relaxation_multistart=args.relaxation_multistart,
    )

    problem = report["problem"]
    print("CherryQ 10-invoice SME scaling case")
    print("==================================")
    print(
        "Exact/MILP optimum: "
        + "+".join(problem["exact_paid_invoice_ids"])
        + f"; spend £{problem['exact_spend_gbp']:,}; "
        + f"loss £{problem['exact_business_loss_gbp']:,}"
    )
    print(
        "Greedy baseline: "
        + "+".join(problem["greedy_paid_invoice_ids"])
        + f"; loss £{problem['greedy_business_loss_gbp']:,}; "
        + f"gap £{problem['greedy_gap_gbp']:,}"
    )
    print(
        f"Decision space: {problem['business_decision_space']:,}; "
        f"QUBO variables: {problem['qubo_variable_count']}; "
        f"QUBO states: {problem['qubo_state_space']:,}"
    )
    print(
        f"Quadratic couplings: {problem['qubo_quadratic_terms']}; "
        f"p=1 logical CX estimate: {problem['p1_logical_cx_estimate']}"
    )
    print(
        f"Smallest safe tested penalty: "
        f"£{problem['selected_penalty_gbp']:,.0f}"
    )

    if report["qaoa"] is not None:
        qaoa = report["qaoa"]
        print("\nIdeal warm-start-X p=1 QAOA")
        print("-----------------------------")
        print(
            f"P(optimal business plan): "
            f"{qaoa['optimal_business_plan_probability']:.2%}"
        )
        print(
            f"Business feasible: {qaoa['business_feasible_probability']:.2%}; "
            f"QUBO feasible: {qaoa['qubo_feasible_probability']:.2%}"
        )
        print(
            f"Logical depth: {qaoa['circuit_depth']}; "
            f"two-qubit gates: {qaoa['two_qubit_gate_count']}"
        )
        print(
            f"Best sampled plan: "
            f"{'+'.join(qaoa['best_sampled_paid_invoice_ids'])}; "
            f"loss £{qaoa['best_sampled_business_loss_gbp']}"
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2, default=list),
            encoding="utf-8",
        )
        print(f"\nWrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
