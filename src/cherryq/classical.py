from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .problem import PaymentProblem, PaymentSolution


@dataclass(frozen=True)
class ClassicalBenchmark:
    exact_solution: PaymentSolution
    milp_solution: PaymentSolution
    exact_runtime_ms: float
    milp_runtime_ms: float
    milp_success: bool
    milp_message: str
    milp_gap: float | None

    @property
    def solutions_agree(self) -> bool:
        return (
            self.exact_solution.business_loss_gbp
            == self.milp_solution.business_loss_gbp
            and self.exact_solution.spend_gbp == self.milp_solution.spend_gbp
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "exact_paid": self.exact_solution.paid_invoice_ids,
            "exact_spend_gbp": self.exact_solution.spend_gbp,
            "exact_business_loss_gbp": self.exact_solution.business_loss_gbp,
            "milp_paid": self.milp_solution.paid_invoice_ids,
            "milp_spend_gbp": self.milp_solution.spend_gbp,
            "milp_business_loss_gbp": self.milp_solution.business_loss_gbp,
            "exact_runtime_ms": self.exact_runtime_ms,
            "milp_runtime_ms": self.milp_runtime_ms,
            "milp_success": self.milp_success,
            "milp_message": self.milp_message,
            "milp_gap": self.milp_gap,
            "solutions_agree": self.solutions_agree,
        }


def solve_milp(problem: PaymentProblem) -> tuple[PaymentSolution, object]:
    """Solve the payment model with a binary MILP baseline.

    Pairwise terms x_i*x_j are linearised with an auxiliary binary variable y:
      y <= x_i
      y <= x_j
      y >= x_i + x_j - 1
    """

    problem.validate()
    invoice_count = len(problem.invoices)
    dependency_count = len(problem.dependencies)
    variable_count = invoice_count + dependency_count

    invoice_index = {
        invoice.invoice_id: index for index, invoice in enumerate(problem.invoices)
    }

    c = np.zeros(variable_count, dtype=float)
    constant_loss = 0.0

    for index, invoice in enumerate(problem.invoices):
        constant_loss += invoice.deferral_loss_gbp
        c[index] = -float(invoice.deferral_loss_gbp)

    for dep_index, dependency in enumerate(problem.dependencies):
        constant_loss += dependency.loss_if_not_all_paid_gbp
        c[invoice_count + dep_index] = -float(dependency.loss_if_not_all_paid_gbp)

    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    budget_row = np.zeros(variable_count, dtype=float)
    for index, invoice in enumerate(problem.invoices):
        budget_row[index] = float(invoice.amount_gbp)
    rows.append(budget_row)
    lower.append(-np.inf)
    upper.append(float(problem.available_budget_gbp))

    for dep_index, dependency in enumerate(problem.dependencies):
        left_id, right_id = dependency.invoice_ids
        left = invoice_index[left_id]
        right = invoice_index[right_id]
        y = invoice_count + dep_index

        row = np.zeros(variable_count, dtype=float)
        row[y] = 1.0
        row[left] = -1.0
        rows.append(row)
        lower.append(-np.inf)
        upper.append(0.0)

        row = np.zeros(variable_count, dtype=float)
        row[y] = 1.0
        row[right] = -1.0
        rows.append(row)
        lower.append(-np.inf)
        upper.append(0.0)

        row = np.zeros(variable_count, dtype=float)
        row[left] = 1.0
        row[right] = 1.0
        row[y] = -1.0
        rows.append(row)
        lower.append(-np.inf)
        upper.append(1.0)

    constraints = LinearConstraint(
        np.vstack(rows),
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
    )
    bounds = Bounds(np.zeros(variable_count), np.ones(variable_count))
    integrality = np.ones(variable_count, dtype=int)

    result = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"MILP failed: {result.message}")

    rounded = np.rint(result.x[:invoice_count]).astype(int)
    decision = {
        invoice.invoice_id: int(rounded[index])
        for index, invoice in enumerate(problem.invoices)
    }

    solution = PaymentSolution(
        decision=decision,
        spend_gbp=problem.spend_gbp(decision),
        business_loss_gbp=problem.business_loss_gbp(decision),
    )

    # Cross-check the solver's linearised objective with the business scorer.
    milp_loss = constant_loss + float(np.dot(c, np.rint(result.x)))
    if not np.isclose(milp_loss, solution.business_loss_gbp):
        raise RuntimeError(
            "MILP objective and business-loss scorer disagree: "
            f"{milp_loss} vs {solution.business_loss_gbp}"
        )

    return solution, result


def benchmark_classical(problem: PaymentProblem) -> ClassicalBenchmark:
    start = perf_counter()
    exact = problem.solve_exact()
    exact_runtime_ms = (perf_counter() - start) * 1000.0

    start = perf_counter()
    milp_solution, result = solve_milp(problem)
    milp_runtime_ms = (perf_counter() - start) * 1000.0

    gap = getattr(result, "mip_gap", None)
    return ClassicalBenchmark(
        exact_solution=exact,
        milp_solution=milp_solution,
        exact_runtime_ms=exact_runtime_ms,
        milp_runtime_ms=milp_runtime_ms,
        milp_success=bool(result.success),
        milp_message=str(result.message),
        milp_gap=None if gap is None else float(gap),
    )
