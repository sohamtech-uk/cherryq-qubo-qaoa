from cherryq.pipeline import run_demo_pipeline


def test_end_to_end_pipeline_without_quantum_is_explainable_and_cash_safe():
    report = run_demo_pipeline(include_quantum=False, seed=42)

    assert len(report.predicted_receipts) == 2
    assert len(report.scenarios) == 4
    assert len(report.benchmarks) == 4
    assert all(benchmark.classical.solutions_agree for benchmark in report.benchmarks)
    assert all(not benchmark.qaoa_runs for benchmark in report.benchmarks)

    current = report.recommendation.current_plan
    assert current.scenario_id == "current-cash"
    assert current.available_budget_gbp == 3000
    assert current.paid_invoice_ids == ("A", "B", "D")
    assert current.spend_gbp <= current.available_budget_gbp
    assert "cleared cash only" in current.condition.lower()

    for plan in report.recommendation.conditional_plans:
        assert "actually settled" in plan.condition
        assert plan.spend_gbp <= plan.available_budget_gbp

    assert report.recommendation.all_classical_benchmarks_agree
    assert "never treats uncleared expected receipts as spendable cash" in (
        report.recommendation.decision_policy
    )
