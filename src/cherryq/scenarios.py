from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from .forecast import FEATURE_NAMES, predict_payment_probability
from .problem import PaymentProblem


@dataclass(frozen=True)
class CustomerReceivable:
    receipt_id: str
    customer: str
    amount_gbp: int
    features: tuple[float, ...]

    def validate(self) -> None:
        if self.amount_gbp <= 0:
            raise ValueError("Receivable amount must be positive")
        if len(self.features) != len(FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(FEATURE_NAMES)} forecast features, got {len(self.features)}"
            )


@dataclass(frozen=True)
class PredictedReceipt:
    receipt_id: str
    customer: str
    amount_gbp: int
    probability_before_payment_run: float

    def validate(self) -> None:
        if self.amount_gbp <= 0:
            raise ValueError("Predicted receipt amount must be positive")
        if not 0.0 <= self.probability_before_payment_run <= 1.0:
            raise ValueError("Receipt probability must be between 0 and 1")


@dataclass(frozen=True)
class CashScenario:
    scenario_id: str
    probability: float
    arrived_receipt_ids: tuple[str, ...]
    missed_receipt_ids: tuple[str, ...]
    extra_cash_gbp: int

    def to_payment_problem(self, base_problem: PaymentProblem) -> PaymentProblem:
        return replace(
            base_problem,
            cleared_cash_gbp=base_problem.cleared_cash_gbp + self.extra_cash_gbp,
        )


def predict_receivables(
    model: GradientBoostingClassifier,
    receivables: Sequence[CustomerReceivable],
) -> tuple[PredictedReceipt, ...]:
    if not receivables:
        return ()

    for receivable in receivables:
        receivable.validate()

    features = np.asarray([receivable.features for receivable in receivables], dtype=float)
    probabilities = predict_payment_probability(model, features)

    return tuple(
        PredictedReceipt(
            receipt_id=receivable.receipt_id,
            customer=receivable.customer,
            amount_gbp=receivable.amount_gbp,
            probability_before_payment_run=float(probability),
        )
        for receivable, probability in zip(receivables, probabilities, strict=True)
    )


def build_cash_scenarios(
    predicted_receipts: Sequence[PredictedReceipt],
    max_receipts: int = 8,
) -> tuple[CashScenario, ...]:
    """Enumerate discrete cash-arrival scenarios.

    Each customer receipt is either fully settled before the payment run or not.
    The prototype assumes receipt events are independent when calculating scenario
    probabilities. The optimiser never spends a fractional "expected" receipt.
    """

    receipts = tuple(predicted_receipts)
    if len(receipts) > max_receipts:
        raise ValueError(
            f"Scenario enumeration is capped at {max_receipts} receipts, got {len(receipts)}"
        )
    for receipt in receipts:
        receipt.validate()

    if not receipts:
        return (
            CashScenario(
                scenario_id="current-cash",
                probability=1.0,
                arrived_receipt_ids=(),
                missed_receipt_ids=(),
                extra_cash_gbp=0,
            ),
        )

    scenarios: list[CashScenario] = []
    for bits in product((0, 1), repeat=len(receipts)):
        probability = 1.0
        arrived: list[str] = []
        missed: list[str] = []
        extra_cash = 0

        for receipt, arrived_bit in zip(receipts, bits, strict=True):
            p = receipt.probability_before_payment_run
            if arrived_bit:
                probability *= p
                arrived.append(receipt.receipt_id)
                extra_cash += receipt.amount_gbp
            else:
                probability *= 1.0 - p
                missed.append(receipt.receipt_id)

        scenario_id = "current-cash" if not arrived else "arrive-" + "-".join(arrived)
        scenarios.append(
            CashScenario(
                scenario_id=scenario_id,
                probability=float(probability),
                arrived_receipt_ids=tuple(arrived),
                missed_receipt_ids=tuple(missed),
                extra_cash_gbp=extra_cash,
            )
        )

    total_probability = sum(scenario.probability for scenario in scenarios)
    if total_probability <= 0:
        raise RuntimeError("Cash scenario probabilities sum to zero")

    normalized = [
        replace(scenario, probability=scenario.probability / total_probability)
        for scenario in scenarios
    ]
    return tuple(
        sorted(
            normalized,
            key=lambda scenario: (
                -scenario.probability,
                scenario.extra_cash_gbp,
                scenario.scenario_id,
            ),
        )
    )


def build_demo_receivables() -> tuple[CustomerReceivable, ...]:
    """Receivables used only for the hackathon demo.

    Feature order matches FEATURE_NAMES:
    amount, payment terms, days until due, average days late over previous five,
    90-day on-time rate, previous invoice count.
    """

    return (
        CustomerReceivable(
            receipt_id="R1",
            customer="Customer North",
            amount_gbp=2000,
            features=(2000.0, 30.0, 3.0, 2.0, 0.85, 18.0),
        ),
        CustomerReceivable(
            receipt_id="R2",
            customer="Customer West",
            amount_gbp=1000,
            features=(1000.0, 30.0, -5.0, 14.0, 0.45, 10.0),
        ),
    )
