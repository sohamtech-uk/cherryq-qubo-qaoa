import pytest

from cherryq.forecast import generate_synthetic_payment_history, train_payment_classifier
from cherryq.problem import build_example_problem
from cherryq.scenarios import (
    PredictedReceipt,
    build_cash_scenarios,
    build_demo_receivables,
    predict_receivables,
)


def test_predicted_receipts_feed_discrete_cash_scenarios():
    dataset = generate_synthetic_payment_history(n_samples=400, seed=42)
    forecast = train_payment_classifier(dataset, seed=42)

    receipts = predict_receivables(forecast.model, build_demo_receivables())
    scenarios = build_cash_scenarios(receipts)

    assert len(receipts) == 2
    assert len(scenarios) == 4
    assert sum(scenario.probability for scenario in scenarios) == pytest.approx(1.0)
    assert all(
        0.0 <= receipt.probability_before_payment_run <= 1.0
        for receipt in receipts
    )

    current = next(s for s in scenarios if not s.arrived_receipt_ids)
    assert current.extra_cash_gbp == 0
    assert current.to_payment_problem(build_example_problem()).available_budget_gbp == 3000


def test_scenario_uses_full_receipts_not_fractional_expected_cash():
    receipts = (
        PredictedReceipt("R1", "A customer", 2000, 0.6),
        PredictedReceipt("R2", "Another customer", 1000, 0.25),
    )
    scenarios = build_cash_scenarios(receipts)

    extra_cash_values = {scenario.extra_cash_gbp for scenario in scenarios}
    assert extra_cash_values == {0, 1000, 2000, 3000}
    assert 1200 not in extra_cash_values
