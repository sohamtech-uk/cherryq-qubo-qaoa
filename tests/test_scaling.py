from cherryq.classical import benchmark_classical
from cherryq.problem import build_ten_invoice_problem
from cherryq.scaling import benchmark_problem


def test_ten_invoice_exact_and_milp_optimum_agree():
    problem = build_ten_invoice_problem()
    benchmark = benchmark_classical(problem)

    assert len(problem.invoices) == 10
    assert problem.available_budget_gbp == 5000
    assert benchmark.solutions_agree
    assert benchmark.exact_solution.paid_invoice_ids == ("A", "B", "F", "G", "H")
    assert benchmark.exact_solution.spend_gbp == 5000
    assert benchmark.exact_solution.business_loss_gbp == 950


def test_ten_invoice_greedy_misses_dependency_value():
    problem = build_ten_invoice_problem()
    greedy = problem.solve_greedy()

    assert greedy.paid_invoice_ids == ("C", "D", "F", "G", "H", "I", "J")
    assert greedy.spend_gbp == 5000
    assert greedy.business_loss_gbp == 1350


def test_ten_invoice_scaling_resources_are_explicit():
    result = benchmark_problem(build_ten_invoice_problem(), label="10-invoice")

    assert result.business_decision_space == 1024
    assert result.exact_paid_invoice_ids == ("A", "B", "F", "G", "H")
    assert result.greedy_gap_gbp == 400
    assert result.selected_penalty_gbp == 250
    assert result.qubo_variable_count == 14
    assert result.slack_qubit_count == 4
    assert result.qubo_state_space == 16384
    assert result.qubo_quadratic_terms == 91
    assert result.p1_logical_cx_estimate == 182


def test_ten_invoice_categories_cover_business_roles():
    categories = {invoice.category for invoice in build_ten_invoice_problem().invoices}

    assert {
        "Revenue-critical",
        "Fulfilment-critical",
        "Business continuity",
        "Penalty-sensitive",
        "Deferrable",
    }.issubset(categories)
