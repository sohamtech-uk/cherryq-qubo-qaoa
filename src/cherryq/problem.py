from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Invoice:
    invoice_id: str
    supplier: str
    amount_gbp: int
    deferral_loss_gbp: int


@dataclass(frozen=True)
class PaymentDependency:
    invoice_ids: tuple[str, ...]
    loss_if_not_all_paid_gbp: int
    description: str


@dataclass(frozen=True)
class PaymentSolution:
    decision: dict[str, int]
    spend_gbp: int
    business_loss_gbp: int

    @property
    def paid_invoice_ids(self) -> tuple[str, ...]:
        return tuple(invoice_id for invoice_id, paid in self.decision.items() if paid)


@dataclass(frozen=True)
class QuboModel:
    variable_names: tuple[str, ...]
    linear: dict[str, float]
    quadratic: dict[tuple[str, str], float]
    offset: float
    constraint_weights: dict[str, int]
    constraint_rhs: int
    supplier_variable_names: tuple[str, ...]
    penalty_gbp: float

    def energy(self, values: Mapping[str, float] | Sequence[float]) -> float:
        if isinstance(values, Mapping):
            vector = values
        else:
            if len(values) != len(self.variable_names):
                raise ValueError(
                    f"Expected {len(self.variable_names)} values, got {len(values)}"
                )
            vector = dict(zip(self.variable_names, values, strict=True))

        total = self.offset
        total += sum(self.linear[name] * float(vector[name]) for name in self.variable_names)
        total += sum(
            coeff * float(vector[left]) * float(vector[right])
            for (left, right), coeff in self.quadratic.items()
        )
        return float(total)

    def constraint_residual(self, values: Mapping[str, float] | Sequence[float]) -> float:
        if isinstance(values, Mapping):
            vector = values
        else:
            vector = dict(zip(self.variable_names, values, strict=True))
        lhs = sum(
            self.constraint_weights[name] * float(vector[name])
            for name in self.variable_names
        )
        return float(lhs - self.constraint_rhs)

    def solve_exact(self) -> tuple[dict[str, int], float]:
        best_values: dict[str, int] | None = None
        best_energy = float("inf")
        for bits in product((0, 1), repeat=len(self.variable_names)):
            values = dict(zip(self.variable_names, bits, strict=True))
            energy = self.energy(values)
            if energy < best_energy:
                best_values = values
                best_energy = energy

        assert best_values is not None
        return best_values, best_energy


@dataclass(frozen=True)
class PaymentProblem:
    cleared_cash_gbp: int
    protected_cash_gbp: int
    budget_unit_gbp: int
    invoices: tuple[Invoice, ...]
    dependencies: tuple[PaymentDependency, ...]

    @property
    def available_budget_gbp(self) -> int:
        available = self.cleared_cash_gbp - self.protected_cash_gbp
        if available < 0:
            raise ValueError("Protected cash cannot exceed cleared cash")
        return available

    @property
    def invoice_ids(self) -> tuple[str, ...]:
        return tuple(invoice.invoice_id for invoice in self.invoices)

    @property
    def budget_units(self) -> int:
        if self.available_budget_gbp % self.budget_unit_gbp != 0:
            raise ValueError("Available budget must be divisible by budget_unit_gbp")
        return self.available_budget_gbp // self.budget_unit_gbp

    def validate(self) -> None:
        ids = self.invoice_ids
        if len(set(ids)) != len(ids):
            raise ValueError("Invoice IDs must be unique")
        if self.budget_unit_gbp <= 0:
            raise ValueError("budget_unit_gbp must be positive")
        if any(invoice.amount_gbp <= 0 for invoice in self.invoices):
            raise ValueError("Invoice amounts must be positive")
        if any(invoice.deferral_loss_gbp < 0 for invoice in self.invoices):
            raise ValueError("Deferral losses cannot be negative")
        if any(invoice.amount_gbp % self.budget_unit_gbp != 0 for invoice in self.invoices):
            raise ValueError("Every invoice amount must be divisible by budget_unit_gbp")
        known_ids = set(ids)
        for dependency in self.dependencies:
            if not set(dependency.invoice_ids).issubset(known_ids):
                raise ValueError("Dependency references an unknown invoice")
            if len(dependency.invoice_ids) != 2:
                raise ValueError(
                    "The Day 1 QUBO prototype currently supports pairwise dependencies only"
                )
            if dependency.loss_if_not_all_paid_gbp < 0:
                raise ValueError("Dependency loss cannot be negative")

    def spend_gbp(self, decision: Mapping[str, int]) -> int:
        return sum(
            invoice.amount_gbp * int(decision.get(invoice.invoice_id, 0))
            for invoice in self.invoices
        )

    def is_business_feasible(self, decision: Mapping[str, int]) -> bool:
        return self.spend_gbp(decision) <= self.available_budget_gbp

    def business_loss_gbp(self, decision: Mapping[str, int]) -> int:
        loss = sum(
            invoice.deferral_loss_gbp * (1 - int(decision.get(invoice.invoice_id, 0)))
            for invoice in self.invoices
        )
        for dependency in self.dependencies:
            all_paid = 1
            for invoice_id in dependency.invoice_ids:
                all_paid *= int(decision.get(invoice_id, 0))
            loss += dependency.loss_if_not_all_paid_gbp * (1 - all_paid)
        return loss

    def solve_exact(self) -> PaymentSolution:
        best: PaymentSolution | None = None
        for bits in product((0, 1), repeat=len(self.invoices)):
            decision = dict(zip(self.invoice_ids, bits, strict=True))
            if not self.is_business_feasible(decision):
                continue
            solution = PaymentSolution(
                decision=decision,
                spend_gbp=self.spend_gbp(decision),
                business_loss_gbp=self.business_loss_gbp(decision),
            )
            if best is None or (
                solution.business_loss_gbp,
                -solution.spend_gbp,
                solution.paid_invoice_ids,
            ) < (
                best.business_loss_gbp,
                -best.spend_gbp,
                best.paid_invoice_ids,
            ):
                best = solution

        if best is None:
            raise RuntimeError("No feasible payment plan found")
        return best

    def solve_greedy(self) -> PaymentSolution:
        """A deliberately simple classical rule used as a weak product baseline.

        It ranks invoices by standalone deferral-loss-per-pound and ignores pairwise
        dependencies. This is not intended to be the main classical benchmark.
        """

        ranked = sorted(
            self.invoices,
            key=lambda invoice: (
                invoice.deferral_loss_gbp / invoice.amount_gbp,
                invoice.deferral_loss_gbp,
            ),
            reverse=True,
        )
        remaining = self.available_budget_gbp
        decision = {invoice.invoice_id: 0 for invoice in self.invoices}
        for invoice in ranked:
            if invoice.amount_gbp <= remaining:
                decision[invoice.invoice_id] = 1
                remaining -= invoice.amount_gbp

        return PaymentSolution(
            decision=decision,
            spend_gbp=self.spend_gbp(decision),
            business_loss_gbp=self.business_loss_gbp(decision),
        )

    def build_qubo(self, penalty_gbp: float = 2000.0) -> QuboModel:
        """Build the 5-invoice budget QUBO with binary slack variables.

        The budget is converted into integer units. The inequality

            sum_i amount_i * x_i <= budget

        becomes an equality by adding binary slack representing unused budget.
        For the example, the budget is six £500 units, so three slack bits
        (1, 2, and 4 units) are sufficient.
        """

        self.validate()
        if penalty_gbp <= 0:
            raise ValueError("penalty_gbp must be positive")

        invoice_weights = {
            invoice.invoice_id: invoice.amount_gbp // self.budget_unit_gbp
            for invoice in self.invoices
        }
        max_slack = self.budget_units
        slack_bit_count = max(1, max_slack.bit_length())
        slack_names = tuple(f"s{index}" for index in range(slack_bit_count))
        slack_weights = {name: 1 << index for index, name in enumerate(slack_names)}

        variable_names = self.invoice_ids + slack_names
        constraint_weights = invoice_weights | slack_weights

        linear = {name: 0.0 for name in variable_names}
        quadratic: dict[tuple[str, str], float] = {}
        offset = 0.0

        # Business loss: standalone cost of deferring each invoice.
        for invoice in self.invoices:
            offset += invoice.deferral_loss_gbp
            linear[invoice.invoice_id] -= invoice.deferral_loss_gbp

        # Business loss: pairwise dependency cost disappears only when both are paid.
        for dependency in self.dependencies:
            left, right = dependency.invoice_ids
            if variable_names.index(left) > variable_names.index(right):
                left, right = right, left
            offset += dependency.loss_if_not_all_paid_gbp
            key = (left, right)
            quadratic[key] = quadratic.get(key, 0.0) - dependency.loss_if_not_all_paid_gbp

        # Penalty P * (sum_i a_i z_i - B)^2, reduced using z_i^2 = z_i.
        budget_units = self.budget_units
        offset += penalty_gbp * (budget_units**2)
        for name in variable_names:
            weight = constraint_weights[name]
            linear[name] += penalty_gbp * (
                weight**2 - 2 * budget_units * weight
            )

        for i, left in enumerate(variable_names):
            for right in variable_names[i + 1 :]:
                key = (left, right)
                quadratic[key] = quadratic.get(key, 0.0) + (
                    2
                    * penalty_gbp
                    * constraint_weights[left]
                    * constraint_weights[right]
                )

        return QuboModel(
            variable_names=variable_names,
            linear=linear,
            quadratic=quadratic,
            offset=offset,
            constraint_weights=constraint_weights,
            constraint_rhs=budget_units,
            supplier_variable_names=self.invoice_ids,
            penalty_gbp=float(penalty_gbp),
        )

    def encode_with_slack(self, decision: Mapping[str, int], qubo: QuboModel) -> dict[str, int]:
        if not self.is_business_feasible(decision):
            raise ValueError("Cannot add budget slack to an infeasible payment decision")

        values = {name: int(decision.get(name, 0)) for name in self.invoice_ids}
        spent_units = self.spend_gbp(decision) // self.budget_unit_gbp
        slack_units = self.budget_units - spent_units

        for name in qubo.variable_names:
            if name.startswith("s"):
                bit_index = int(name[1:])
                values[name] = (slack_units >> bit_index) & 1
        return values

    def supplier_decision_from_values(
        self, values: Mapping[str, int] | Sequence[int], variable_names: Iterable[str]
    ) -> dict[str, int]:
        if isinstance(values, Mapping):
            mapping = values
        else:
            mapping = dict(zip(variable_names, values, strict=True))
        return {invoice_id: int(mapping[invoice_id]) for invoice_id in self.invoice_ids}


def build_example_problem() -> PaymentProblem:
    """Return the mentor-reviewed five-invoice CherryQ Day 1 scenario."""

    return PaymentProblem(
        cleared_cash_gbp=8000,
        protected_cash_gbp=5000,
        budget_unit_gbp=500,
        invoices=(
            Invoice("A", "Materials supplier", 1500, 100),
            Invoice("B", "Delivery partner", 1000, 100),
            Invoice("C", "Equipment hire", 1000, 350),
            Invoice("D", "IT support", 500, 100),
            Invoice("E", "Packaging supplier", 1500, 250),
        ),
        dependencies=(
            PaymentDependency(
                invoice_ids=("A", "B"),
                loss_if_not_all_paid_gbp=900,
                description="Both payments are required to keep a customer order moving",
            ),
        ),
    )
