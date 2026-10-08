from cherryq.classical import benchmark_classical
from cherryq.problem import build_example_problem
from cherryq.scenarios import PredictedReceipt, build_cash_scenarios


def test_milp_matches_exact_for_every_cash_scenario():
    base = build_example_problem()
    scenarios = build_cash_scenarios(
        (
            PredictedReceipt("R1", "Customer North", 2000, 0.65),
            PredictedReceipt("R2", "Customer West", 1000, 0.35),
        )
    )

    for scenario in scenarios:
        benchmark = benchmark_classical(scenario.to_payment_problem(base))
        assert benchmark.solutions_agree
        assert benchmark.milp_success
        assert benchmark.exact_runtime_ms >= 0.0
        assert benchmark.milp_runtime_ms >= 0.0


def test_current_cash_classical_reference_remains_abd():
    benchmark = benchmark_classical(build_example_problem())

    assert benchmark.exact_solution.paid_invoice_ids == ("A", "B", "D")
    assert benchmark.milp_solution.paid_invoice_ids == ("A", "B", "D")
    assert benchmark.exact_solution.business_loss_gbp == 600
