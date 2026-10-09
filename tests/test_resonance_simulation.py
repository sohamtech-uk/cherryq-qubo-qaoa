from types import SimpleNamespace

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from cherryq.hardware.resonance_simulation import (
    BETA, GAMMA, METRICS, check_facade_target, frozen_circuits,
    require_mock_target, state_masks, summarise_counts,
)


@pytest.mark.parametrize("url,alias", [
    ("https://resonance.iqm.tech", "garnet"),
    ("https://resonance.iqm.tech", "q20"),
    ("https://resonance.iqm.tech", None),
    ("https://resonance.iqm.tech", "garnet:mock:timeslot"),
    ("https://resonance.iqm.tech/garnet", "garnet:mock"),
    ("https://other.example", "garnet:mock"),
])
def test_physical_or_ambiguous_targets_are_rejected(url, alias):
    with pytest.raises(ValueError):
        require_mock_target(url, alias)


def test_resolved_real_computer_is_rejected_before_execution():
    backend = SimpleNamespace(client=SimpleNamespace(_iqm_server_client=SimpleNamespace(
        root_url="https://resonance.iqm.tech", quantum_computer="garnet")))
    with pytest.raises(ValueError):
        check_facade_target(backend)


@pytest.fixture(scope="module")
def prepared():
    return frozen_circuits()


def test_frozen_parameters_schedules_and_measured_invoice_mapping(prepared):
    problem, qubo, _, circuits = prepared
    assert (GAMMA, BETA) == (-0.7439674535854706, 0.3004519607226279)
    assert qubo.variable_names == (*problem.invoice_ids, "s0", "s1", "s2", "s3")
    states = []
    for order, circuit in circuits.items():
        plain = circuit.remove_final_measurements(inplace=False)
        assert plain.depth() == (78 if order == "original" else 42)
        assert plain.count_ops()["cx"] == 182
        assert [float(i.operation.params[0]) for i in plain.data if i.operation.name == "rx"] == [2 * BETA] * 14
        assert {circuit.find_bit(i.clbits[0]).index: circuit.find_bit(i.qubits[0]).index
                for i in circuit.data if i.operation.name == "measure"} == dict(enumerate(range(14)))
        states.append(Statevector.from_instruction(plain).data)
    assert np.allclose(states[0], states[1], atol=1e-12, rtol=0)


def test_statevector_metrics_agree_with_repository_count_interpreter(prepared):
    problem, qubo, _, _ = prepared
    masks, _, _ = state_masks(qubo, problem)
    # Optimum invoices with correct/incorrect slack, empty plan, all invoices.
    indices = (227, 227 + 1024, 0, 1023)
    counts = {format(i, "014b"): count for i, count in zip(indices, (700, 200, 90, 10), strict=True)}
    analysis = summarise_counts(counts, qubo, problem)
    for key in METRICS:
        exact = sum(counts[format(i, "014b")] * masks[key][i] for i in indices) / 1000
        assert analysis[key] == pytest.approx(exact)
    assert analysis[METRICS[0]] == pytest.approx(0.9)
    assert analysis[METRICS[3]] == pytest.approx(0.7)
    assert analysis["best_sampled_business_loss_gbp"] == 950
