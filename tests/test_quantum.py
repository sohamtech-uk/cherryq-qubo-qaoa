from itertools import product

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")
from qiskit.quantum_info import Statevector  # noqa: E402

from cherryq.problem import build_example_problem  # noqa: E402
from cherryq.quantum import (  # noqa: E402
    qubo_to_ising,
    run_qaoa,
    solve_continuous_relaxation,
)


def test_ising_mapping_matches_qubo_on_every_basis_state():
    problem = build_example_problem()
    qubo = problem.build_qubo(penalty_gbp=2000.0)
    ising = qubo_to_ising(qubo)

    for bits in product((0, 1), repeat=len(qubo.variable_names)):
        label = "".join(str(bit) for bit in reversed(bits))
        state = Statevector.from_label(label)
        hamiltonian_energy = float(
            np.real(state.expectation_value(ising.operator)) + ising.offset
        )
        assert hamiltonian_energy == pytest.approx(qubo.energy(bits), abs=1e-8)


def test_continuous_relaxation_is_a_valid_reportable_classical_baseline():
    problem = build_example_problem()
    qubo = problem.build_qubo(penalty_gbp=2000.0)

    relaxation = solve_continuous_relaxation(qubo, multistart=16, seed=42)

    assert relaxation.values.shape == (len(qubo.variable_names),)
    assert np.all(relaxation.values >= 0.0)
    assert np.all(relaxation.values <= 1.0)
    assert set(relaxation.rounded_values.tolist()).issubset({0, 1})
    assert relaxation.rounded_energy == pytest.approx(
        qubo.energy(relaxation.rounded_values), abs=1e-8
    )


@pytest.mark.parametrize(
    ("method", "p"),
    [
        ("standard", 1),
        ("warm-start-x", 1),
        ("warm-start", 1),
        ("standard", 2),
        ("warm-start-x", 2),
        ("warm-start", 2),
    ],
)
def test_qaoa_paths_smoke_on_statevector(method: str, p: int):
    problem = build_example_problem()
    qubo = problem.build_qubo(penalty_gbp=2000.0)

    result = run_qaoa(
        problem,
        qubo,
        method=method,
        p=p,
        maxiter=6,
        shots=128,
        seed=42,
        relaxation_multistart=8,
        warm_start_parameter_jitter=0.01,
    )

    assert result.method == method
    assert result.p == p
    assert result.penalty_gbp == 2000.0
    assert result.num_qubits == len(qubo.variable_names)
    assert result.circuit_depth > 0
    assert result.two_qubit_gate_count > 0
    assert result.total_runtime_ms >= result.optimizer_runtime_ms
    assert np.isfinite(result.optimized_expected_qubo_energy)
    assert np.isfinite(result.best_sampled_qubo_energy)
    assert 0.0 <= result.qubo_feasible_probability <= 1.0
    assert 0.0 <= result.business_feasible_probability <= 1.0
    assert 0.0 <= result.optimal_qubo_probability <= 1.0
    assert 0.0 <= result.optimal_business_plan_probability <= 1.0
