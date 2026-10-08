from itertools import product

import pytest

from cherryq.problem import build_example_problem


def test_exact_classical_optimum_is_abd():
    problem = build_example_problem()
    result = problem.solve_exact()

    assert result.paid_invoice_ids == ("A", "B", "D")
    assert result.spend_gbp == 3000
    assert result.business_loss_gbp == 600


def test_greedy_baseline_misses_dependency_value():
    problem = build_example_problem()
    result = problem.solve_greedy()

    assert result.paid_invoice_ids == ("C", "D", "E")
    assert result.spend_gbp == 3000
    assert result.business_loss_gbp == 1100


def test_qubo_exact_optimum_matches_business_optimum():
    problem = build_example_problem()
    qubo = problem.build_qubo(penalty_gbp=2000.0)

    values, energy = qubo.solve_exact()
    decision = problem.supplier_decision_from_values(values, qubo.variable_names)

    assert tuple(k for k, v in decision.items() if v) == ("A", "B", "D")
    assert energy == pytest.approx(600.0)
    assert qubo.constraint_residual(values) == pytest.approx(0.0)


def test_feasible_business_loss_equals_qubo_energy_with_correct_slack():
    problem = build_example_problem()
    qubo = problem.build_qubo(penalty_gbp=2000.0)

    for bits in product((0, 1), repeat=len(problem.invoice_ids)):
        decision = dict(zip(problem.invoice_ids, bits, strict=True))
        if not problem.is_business_feasible(decision):
            continue
        values = problem.encode_with_slack(decision, qubo)
        assert qubo.constraint_residual(values) == pytest.approx(0.0)
        assert qubo.energy(values) == pytest.approx(problem.business_loss_gbp(decision))


def test_penalty_keeps_global_qubo_optimum_feasible():
    problem = build_example_problem()
    qubo = problem.build_qubo(penalty_gbp=2000.0)

    values, _ = qubo.solve_exact()
    decision = problem.supplier_decision_from_values(values, qubo.variable_names)

    assert problem.is_business_feasible(decision)
    assert qubo.constraint_residual(values) == pytest.approx(0.0)
