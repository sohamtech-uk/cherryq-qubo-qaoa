from __future__ import annotations

from dataclasses import dataclass

from .benchmark import ScenarioBenchmark
from .problem import PaymentProblem, PaymentSolution


@dataclass(frozen=True)
class ScenarioPlan:
    scenario_id: str
    probability: float
    condition: str
    available_budget_gbp: int
    paid_invoice_ids: tuple[str, ...]
    deferred_invoice_ids: tuple[str, ...]
    spend_gbp: int
    modelled_loss_gbp: int
    explanation: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "probability": self.probability,
            "condition": self.condition,
            "available_budget_gbp": self.available_budget_gbp,
            "paid_invoice_ids": self.paid_invoice_ids,
            "deferred_invoice_ids": self.deferred_invoice_ids,
            "spend_gbp": self.spend_gbp,
            "modelled_loss_gbp": self.modelled_loss_gbp,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class CherryQRecommendation:
    current_plan: ScenarioPlan
    conditional_plans: tuple[ScenarioPlan, ...]
    probability_weighted_modelled_loss_gbp: float
    all_classical_benchmarks_agree: bool
    decision_policy: str

    def as_dict(self) -> dict[str, object]:
        return {
            "decision_policy": self.decision_policy,
            "all_classical_benchmarks_agree": self.all_classical_benchmarks_agree,
            "probability_weighted_modelled_loss_gbp": (
                self.probability_weighted_modelled_loss_gbp
            ),
            "current_plan": self.current_plan.as_dict(),
            "conditional_plans": [plan.as_dict() for plan in self.conditional_plans],
        }


def _explain_solution(
    problem: PaymentProblem,
    benchmark: ScenarioBenchmark,
    solution: PaymentSolution,
) -> tuple[str, ...]:
    scenario = benchmark.scenario
    messages: list[str] = []

    if scenario.arrived_receipt_ids:
        joined = ", ".join(scenario.arrived_receipt_ids)
        messages.append(
            f"Conditional on confirmed receipt(s) {joined}, £{scenario.extra_cash_gbp:,} "
            "additional cleared cash is available."
        )
    else:
        messages.append(
            "No forecast customer receipt is assumed spendable; this plan uses cleared cash only."
        )

    messages.append(
        f"£{problem.protected_cash_gbp:,} remains protected for payroll, tax and minimum reserves."
    )

    for dependency in problem.dependencies:
        paid = all(solution.decision[invoice_id] for invoice_id in dependency.invoice_ids)
        ids = " + ".join(dependency.invoice_ids)
        if paid:
            messages.append(
                f"Paying {ids} together avoids the modelled £"
                f"{dependency.loss_if_not_all_paid_gbp:,} dependency loss: "
                f"{dependency.description}."
            )
        else:
            messages.append(
                f"The plan does not fully fund {ids}; the model therefore retains the £"
                f"{dependency.loss_if_not_all_paid_gbp:,} dependency loss."
            )

    for invoice in problem.invoices:
        if solution.decision[invoice.invoice_id]:
            messages.append(
                f"Pay {invoice.invoice_id} ({invoice.supplier}) £{invoice.amount_gbp:,}; "
                f"this avoids £{invoice.deferral_loss_gbp:,} standalone deferral loss."
            )

    deferred = [
        invoice
        for invoice in problem.invoices
        if not solution.decision[invoice.invoice_id]
    ]
    if deferred:
        messages.append(
            "Defer "
            + ", ".join(
                f"{invoice.invoice_id} ({invoice.supplier})" for invoice in deferred
            )
            + " under this scenario to stay within the available payment budget."
        )

    if benchmark.classical.solutions_agree:
        messages.append(
            "Exact enumeration and the MILP baseline agree on the modelled optimum."
        )
    else:
        messages.append(
            "Exact enumeration and the MILP baseline disagree; do not rely on this recommendation."
        )

    if benchmark.qaoa_runs:
        best_gap = benchmark.best_quantum_gap_gbp
        if best_gap == 0:
            messages.append(
                "At least one sampled QAOA run found a plan matching the exact classical "
                "business-loss optimum; this is a benchmark result, not evidence of quantum advantage."
            )
        elif best_gap is not None:
            messages.append(
                f"The best sampled QAOA plan was £{best_gap:,} above the exact modelled optimum."
            )

    messages.append("Human approval is required before any supplier payment is executed.")
    return tuple(messages)


def _to_plan(
    base_problem: PaymentProblem,
    benchmark: ScenarioBenchmark,
) -> ScenarioPlan:
    problem = benchmark.scenario.to_payment_problem(base_problem)
    solution = benchmark.classical.exact_solution
    arrived = benchmark.scenario.arrived_receipt_ids

    if arrived:
        condition = (
            "Only use this plan after the listed forecast receipt(s) have actually settled: "
            + ", ".join(arrived)
            + "."
        )
    else:
        condition = "Actionable now using currently cleared cash only."

    deferred = tuple(
        invoice_id
        for invoice_id in problem.invoice_ids
        if not solution.decision[invoice_id]
    )

    return ScenarioPlan(
        scenario_id=benchmark.scenario.scenario_id,
        probability=benchmark.scenario.probability,
        condition=condition,
        available_budget_gbp=problem.available_budget_gbp,
        paid_invoice_ids=solution.paid_invoice_ids,
        deferred_invoice_ids=deferred,
        spend_gbp=solution.spend_gbp,
        modelled_loss_gbp=solution.business_loss_gbp,
        explanation=_explain_solution(problem, benchmark, solution),
    )


def build_explainable_recommendation(
    base_problem: PaymentProblem,
    benchmarks: tuple[ScenarioBenchmark, ...],
) -> CherryQRecommendation:
    if not benchmarks:
        raise ValueError("At least one scenario benchmark is required")

    current = next(
        (
            benchmark
            for benchmark in benchmarks
            if not benchmark.scenario.arrived_receipt_ids
        ),
        None,
    )
    if current is None:
        raise ValueError("A current-cleared-cash scenario is required")

    current_plan = _to_plan(base_problem, current)
    conditional_plans = tuple(
        _to_plan(base_problem, benchmark)
        for benchmark in sorted(
            (
                item
                for item in benchmarks
                if item.scenario.arrived_receipt_ids
            ),
            key=lambda item: item.scenario.probability,
            reverse=True,
        )
    )

    weighted_loss = sum(
        benchmark.scenario.probability
        * benchmark.classical.exact_solution.business_loss_gbp
        for benchmark in benchmarks
    )

    return CherryQRecommendation(
        current_plan=current_plan,
        conditional_plans=conditional_plans,
        probability_weighted_modelled_loss_gbp=float(weighted_loss),
        all_classical_benchmarks_agree=all(
            benchmark.classical.solutions_agree for benchmark in benchmarks
        ),
        decision_policy=(
            "Forecast probabilities create conditional cash scenarios. CherryQ never treats "
            "uncleared expected receipts as spendable cash. The prototype recommendation uses "
            "the exact/MILP classical optimum; QAOA is benchmarked separately."
        ),
    )
