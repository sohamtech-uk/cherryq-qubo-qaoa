from itertools import combinations

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparsePauliOp, Statevector

from cherryq.problem import build_ten_invoice_problem
from cherryq.quantum import build_warm_start_qaoa, qubo_to_ising, round_robin_pair_layers
from cherryq.hardware.q20_routing_audit import cost_basis_audit, validate_routed


@pytest.mark.parametrize("n", [1, 2, 3, 5, 14])
def test_round_robin_covers_pairs_once_with_no_qubit_conflict(n):
    layers = round_robin_pair_layers(n)
    pairs = [pair for layer in layers for pair in layer]
    assert sorted(pairs) == list(combinations(range(n), 2))
    for layer in layers:
        assert len({q for pair in layer for q in pair}) == 2 * len(layer)


@pytest.mark.parametrize("gamma", [-0.7439674535854706, 0.0, 0.31])
def test_emitted_ten_invoice_cost_on_all_16384_basis_states(gamma):
    operator = qubo_to_ising(build_ten_invoice_problem().build_qubo(250)).normalized_operator
    report = cost_basis_audit(operator, gamma)
    assert report["basis_states_verified"] == 16384
    assert report["max_phase_error_between_orders"] < 1e-12


@pytest.mark.parametrize("p,mixer", [(1, "x"), (2, "x"), (1, "matched"), (2, "matched")])
def test_full_warm_start_circuit_equivalence_and_depth(p, mixer):
    operator = qubo_to_ising(build_ten_invoice_problem().build_qubo(250)).normalized_operator
    states, circuits = [], []
    for order in ("original", "round-robin"):
        circuit, gammas, betas = build_warm_start_qaoa(
            operator, p, np.linspace(0.1, 0.9, 14), mixer=mixer, cost_order=order,
        )
        bound = circuit.assign_parameters(dict(zip(
            gammas + betas, [-0.7439674535854706] * p + [0.21] * p,
        )))
        circuits.append(bound)
        states.append(Statevector.from_instruction(bound))
    assert states[0].equiv(states[1], atol=1e-11)
    assert circuits[0].count_ops()["cx"] == circuits[1].count_ops()["cx"] == 182 * p
    if p == 1 and mixer == "x":
        assert [c.depth() for c in circuits] == [78, 42]


@pytest.mark.parametrize("label,coefficient", [("XZ", 1), ("ZZZ", 1), ("ZZ", 1j)])
def test_reordering_rejects_unsupported_hamiltonians(label, coefficient):
    operator = SparsePauliOp.from_list([(label, coefficient)])
    with pytest.raises(ValueError):
        build_warm_start_qaoa(operator, 1, [0.5] * len(label), cost_order="round-robin")


def test_unknown_order_rejected_and_default_preserved():
    operator = SparsePauliOp.from_list([("ZZ", 1)])
    default = build_warm_start_qaoa(operator, 1, [0.3, 0.7])[0]
    explicit = build_warm_start_qaoa(operator, 1, [0.3, 0.7], cost_order="original")[0]
    # Independent ParameterVectors have different identities; compare bound circuits.
    assert default.assign_parameters([0.2, 0.4]) == explicit.assign_parameters([0.2, 0.4])
    with pytest.raises(ValueError, match="cost_order"):
        build_warm_start_qaoa(operator, 1, [0.3, 0.7], cost_order="unknown")


def test_routed_validation_accounts_for_layout_and_rejects_wrong_readout():
    backend = GenericBackendV2(
        num_qubits=4, basis_gates=["rz", "sx", "x", "cx"],
        coupling_map=[[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]], seed=42,
    )
    logical = QuantumCircuit(3)
    logical.ry(0.47, 0)
    logical.ry(0.93, 1)
    logical.cx(0, 2)
    logical.rz(0.31, 2)
    logical.cx(2, 1)
    logical.measure_all()
    compiled = transpile(logical, backend, initial_layout=[3, 0, 1], seed_transpiler=42)
    result = validate_routed(logical, compiled, backend)
    assert result["max_amplitude_error_up_to_global_phase"] < 1e-10
    measurement_indices = [i for i, item in enumerate(compiled.data) if item.operation.name == "measure"]
    first, second = measurement_indices[:2]
    a, b = compiled.data[first], compiled.data[second]
    compiled.data[first] = a.replace(clbits=b.clbits)
    compiled.data[second] = b.replace(clbits=a.clbits)
    with pytest.raises(RuntimeError, match="Measurement mapping"):
        validate_routed(logical, compiled, backend)


@pytest.mark.parametrize("gate", ["cz", "cx"])
def test_iqm_single_direction_locus_accepts_only_symmetric_cz(gate):
    backend = GenericBackendV2(
        num_qubits=2, basis_gates=["rz", "sx", "x", gate],
        coupling_map=[[0, 1]], seed=42,
    )
    logical = QuantumCircuit(2)
    logical.h(0)
    logical.ry(0.47, 1)
    getattr(logical, gate)(0, 1)
    logical.measure_all()
    compiled = transpile(logical, backend, initial_layout=[0, 1], optimization_level=0)
    for i, item in enumerate(compiled.data):
        if item.operation.name == gate:
            compiled.data[i] = item.replace(qubits=tuple(reversed(item.qubits)))
    assert not backend.target.instruction_supported(operation_name=gate, qargs=(1, 0))
    if gate == "cz":
        result = validate_routed(logical, compiled, backend)
        assert result["symmetric_cz_reverse_locus_matches"] == 1
        assert result["max_amplitude_error_up_to_global_phase"] < 1e-10
    else:
        with pytest.raises(RuntimeError, match="Unsupported target operation cx"):
            validate_routed(logical, compiled, backend)


def test_symmetric_cz_still_rejects_missing_physical_coupling():
    backend = GenericBackendV2(
        num_qubits=3, basis_gates=["rz", "sx", "x", "cz"],
        coupling_map=[[0, 1], [1, 2]], seed=42,
    )
    logical = QuantumCircuit(3)
    logical.h(0)
    logical.cz(0, 1)
    logical.measure_all()
    compiled = transpile(logical, backend, initial_layout=[0, 1, 2], optimization_level=0)
    for i, item in enumerate(compiled.data):
        if item.operation.name == "cz":
            compiled.data[i] = item.replace(qubits=(compiled.qubits[0], compiled.qubits[2]))
    with pytest.raises(RuntimeError, match="Unsupported target operation cz"):
        validate_routed(logical, compiled, backend)
