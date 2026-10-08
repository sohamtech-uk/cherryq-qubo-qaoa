from __future__ import annotations

import pytest

from cherryq.hardware.iqm_backend import (
    Q20,
    Q50,
    connect_iqm_backend,
    get_iqm_device_config,
)
from cherryq.hardware.q20 import analyse_cherryq_counts, build_bell_circuit
from cherryq.problem import build_example_problem


def _values_to_qiskit_label(values: dict[str, int], names: tuple[str, ...]) -> str:
    logical_bits = "".join(str(values[name]) for name in names)
    return logical_bits[::-1]


def test_iqm_device_runtime_identifiers() -> None:
    assert get_iqm_device_config("q20") == Q20
    assert Q20.cortex_env_var == "Q20_CORTEX_URL"
    assert Q20.quantum_computer == "radiance20"

    assert get_iqm_device_config("Q50") == Q50
    assert Q50.cortex_env_var == "Q50_CORTEX_URL"
    assert Q50.quantum_computer == "q50"


def test_connect_q20_requires_cortex_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("Q20_CORTEX_URL", raising=False)
    with pytest.raises(RuntimeError, match="Q20_CORTEX_URL"):
        connect_iqm_backend("q20")


def test_bell_circuit_has_measurements() -> None:
    circuit = build_bell_circuit()
    assert circuit.num_qubits == 2
    assert circuit.num_clbits == 2
    assert circuit.count_ops()["measure"] == 2


def test_hardware_counts_decode_exact_cherryq_plan() -> None:
    problem = build_example_problem()
    qubo = problem.build_qubo(penalty_gbp=500)
    exact_values, _ = qubo.solve_exact()
    exact_label = _values_to_qiskit_label(exact_values, qubo.variable_names)

    report = analyse_cherryq_counts({exact_label: 1000}, qubo, problem)

    assert report["optimal_qubo_probability"] == pytest.approx(1.0)
    assert report["optimal_business_plan_probability"] == pytest.approx(1.0)
    assert report["business_feasible_probability"] == pytest.approx(1.0)
    assert report["best_sampled_paid_invoice_ids"] == ("A", "B", "D")
    assert report["best_sampled_business_loss_gbp"] == 600
