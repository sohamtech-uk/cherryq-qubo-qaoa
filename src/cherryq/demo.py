from __future__ import annotations

import argparse
import json

from .pipeline import run_demo_pipeline


def _print_plan(plan: dict[str, object]) -> None:
    print(f"Scenario: {plan['scenario_id']} (probability {plan['probability']:.1%})")
    print(f"Condition: {plan['condition']}")
    print(f"Available budget: £{plan['available_budget_gbp']:,}")
    print(f"Pay: {', '.join(plan['paid_invoice_ids']) or 'none'}")
    print(f"Defer: {', '.join(plan['deferred_invoice_ids']) or 'none'}")
    print(
        f"Spend £{plan['spend_gbp']:,}; remaining modelled loss "
        f"£{plan['modelled_loss_gbp']:,}"
    )
    for reason in plan["explanation"]:
        print(f"  - {reason}")


def run(
    skip_quantum: bool,
    maxiter: int,
    shots: int,
    quantum_scenario_limit: int | None,
) -> int:
    report = run_demo_pipeline(
        include_quantum=not skip_quantum,
        p_values=(1, 2),
        maxiter=maxiter,
        shots=shots,
        seed=42,
        quantum_scenario_limit=quantum_scenario_limit,
    )

    print("CherryQ decision pipeline")
    print("=========================")
    print(
        "AI payment probability -> cash-arrival scenarios -> QUBO/MILP/exact "
        "benchmark -> standard/warm-start QAOA -> explainable recommendation"
    )

    print("\nPredictive AI prototype")
    print("-----------------------")
    print(json.dumps(report.forecast.as_dict(), indent=2))
    print(
        "NOTE: forecast metrics use synthetic data and validate engineering only; "
        "they are not customer-validation evidence."
    )

    print("\nPredicted customer receipts")
    print("---------------------------")
    for receipt in report.predicted_receipts:
        print(
            f"{receipt.receipt_id} {receipt.customer}: £{receipt.amount_gbp:,}; "
            f"P(paid before run)={receipt.probability_before_payment_run:.1%}"
        )

    print("\nCash scenarios and classical benchmarks")
    print("---------------------------------------")
    for benchmark in report.benchmarks:
        exact = benchmark.classical.exact_solution
        milp = benchmark.classical.milp_solution
        print(
            f"{benchmark.scenario.scenario_id}: p={benchmark.scenario.probability:.1%}, "
            f"budget=£{benchmark.available_budget_gbp:,}, "
            f"exact={exact.paid_invoice_ids}/£{exact.business_loss_gbp:,} loss, "
            f"MILP={milp.paid_invoice_ids}/£{milp.business_loss_gbp:,} loss, "
            f"agree={benchmark.classical.solutions_agree}"
        )
        for qaoa in benchmark.qaoa_runs:
            print(
                f"  QAOA {qaoa.method} p={qaoa.p}: "
                f"best loss={qaoa.best_sampled_business_loss_gbp}, "
                f"optimal-plan probability={qaoa.optimal_business_plan_probability:.1%}, "
                f"business-feasible probability={qaoa.business_feasible_probability:.1%}"
            )

    recommendation = report.recommendation.as_dict()
    print("\nExplainable CherryQ recommendation")
    print("---------------------------------")
    print(recommendation["decision_policy"])
    print("\nACTIONABLE NOW")
    _print_plan(recommendation["current_plan"])

    if recommendation["conditional_plans"]:
        print("\nCONDITIONAL AFTER RECEIPTS SETTLE")
        for plan in recommendation["conditional_plans"]:
            print()
            _print_plan(plan)

    print(
        "\nProbability-weighted modelled loss across conditional scenarios: "
        f"£{recommendation['probability_weighted_modelled_loss_gbp']:.2f}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the CherryQ decision pipeline")
    parser.add_argument(
        "--skip-quantum",
        action="store_true",
        help="Run AI scenarios plus exact/MILP classical benchmarks without QAOA",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=150,
        help="COBYLA iterations per QAOA run (default: 150)",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=4096,
        help="Finite-shot sample size used for final QAOA candidate metrics",
    )
    parser.add_argument(
        "--quantum-scenario-limit",
        type=int,
        default=None,
        help=(
            "Benchmark QAOA on only the N highest-probability scenarios while always "
            "including current cash. Omit to benchmark every scenario."
        ),
    )
    args = parser.parse_args()
    return run(
        skip_quantum=args.skip_quantum,
        maxiter=args.maxiter,
        shots=args.shots,
        quantum_scenario_limit=args.quantum_scenario_limit,
    )


if __name__ == "__main__":
    raise SystemExit(main())
