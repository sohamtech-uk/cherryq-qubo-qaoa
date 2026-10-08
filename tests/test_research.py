import pytest

from cherryq.pipeline import run_demo_pipeline
from cherryq.research import (
    aggregate_runs,
    choose_smallest_safe_penalty,
    diagnose_penalty,
    diagnose_penalty_grid,
    run_repeated_qaoa,
    warm_start_diagnostic,
)


def test_penalty_diagnostic_detects_under_penalised_current_cash_qubo():
    pipeline = run_demo_pipeline(include_quantum=False, seed=42)
    current = next(
        scenario
        for scenario in pipeline.scenarios
        if scenario.scenario_id == "current-cash"
    )
    problem = current.to_payment_problem(pipeline.base_problem)

    weak = diagnose_penalty(problem, current.scenario_id, 100.0)
    strong = diagnose_penalty(problem, current.scenario_id, 500.0)

    assert not weak.safe
    assert weak.feasibility_margin < 0
    assert strong.safe
    assert strong.feasibility_margin > 0


def test_penalty_grid_selects_smallest_safe_value_for_all_scenarios():
    pipeline = run_demo_pipeline(include_quantum=False, seed=42)
    diagnostics = diagnose_penalty_grid(
        pipeline.base_problem,
        pipeline.scenarios,
        (100.0, 500.0, 2000.0),
    )

    selected = choose_smallest_safe_penalty(
        diagnostics,
        (scenario.scenario_id for scenario in pipeline.scenarios),
    )

    assert selected == 500.0


def test_warm_start_diagnostic_exposes_initial_state_quality():
    pipeline = run_demo_pipeline(include_quantum=False, seed=42)
    current = next(
        scenario
        for scenario in pipeline.scenarios
        if scenario.scenario_id == "current-cash"
    )
    problem = current.to_payment_problem(pipeline.base_problem)

    diagnostic = warm_start_diagnostic(
        problem,
        scenario_id=current.scenario_id,
        penalty_gbp=500.0,
        epsilon=0.25,
        seed=42,
        relaxation_multistart=8,
    )

    assert 0.0 <= diagnostic.relaxation_fractionality <= 0.5
    assert 0.0 <= diagnostic.initial_exact_qubo_probability <= 1.0
    assert 0.0 <= diagnostic.initial_qubo_feasible_probability <= 1.0


def test_repeated_seed_aggregate_compares_mixer_variants():
    pipeline = run_demo_pipeline(include_quantum=False, seed=42)
    current = next(
        scenario
        for scenario in pipeline.scenarios
        if scenario.scenario_id == "current-cash"
    )
    problem = current.to_payment_problem(pipeline.base_problem)

    runs = run_repeated_qaoa(
        problem=problem,
        penalty_gbp=500.0,
        methods=("warm-start-x", "warm-start"),
        p_values=(1,),
        seeds=(11, 29),
        maxiter=6,
        shots=64,
        epsilon=0.25,
        relaxation_multistart=4,
        warm_start_parameter_jitter=0.01,
    )
    aggregates = aggregate_runs(
        runs,
        exact_business_loss_gbp=problem.solve_exact().business_loss_gbp,
    )

    assert len(runs) == 4
    assert {item.method for item in aggregates} == {
        "warm-start-x",
        "warm-start",
    }
    assert all(item.runs == 2 for item in aggregates)
    assert all(0.0 <= item.sampled_optimum_rate <= 1.0 for item in aggregates)
