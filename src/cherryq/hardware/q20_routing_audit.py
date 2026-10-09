"""Compare cost schedules without submitting any QPU job.

An ordinary backend connection is used only to read the current target. This
module never calls backend.run and deliberately emits no runnable manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.circuit.library import CZGate
from qiskit.quantum_info import Statevector

from ..quantum import _apply_cost_unitary, build_warm_start_qaoa, qubo_to_ising
from .iqm_backend import backend_summary, connect_iqm_backend, _two_qubit_gate_count
from .q20_ten import prepare_ten_invoice_qaoa


def circuit_metrics(circuit):
    unmeasured = circuit.remove_final_measurements(inplace=False)
    return {
        "num_qubits": circuit.num_qubits,
        "depth_with_measurements": circuit.depth(),
        "depth_without_measurements": unmeasured.depth(),
        "two_qubit_gates": _two_qubit_gate_count(circuit),
        "operation_counts": dict(circuit.count_ops()),
    }


def cost_basis_audit(operator, gamma):
    """Verify every column of each diagonal cost unitary against exp(-i gamma H).

    Propagate all basis labels and phases through the actual emitted CX/RZ gates
    simultaneously. No 2**n by 2**n unitary matrix is allocated.
    """
    n = operator.num_qubits
    labels = np.arange(1 << n, dtype=np.uint64)
    energies = np.zeros(len(labels))
    for pauli, coeff in zip(operator.paulis, operator.coeffs, strict=True):
        indices = [i for i, axis in enumerate(pauli.to_label()[::-1]) if axis == "Z"]
        if not indices:  # The builder omits only a global identity phase.
            continue
        spins = np.ones(len(labels))
        for i in indices:
            spins *= 1 - 2 * ((labels >> i) & 1).astype(float)
        energies += float(complex(coeff).real) * spins
    expected = np.exp(-1j * gamma * energies)
    results = {}
    phases = {}
    for order in ("original", "round-robin"):
        circuit = QuantumCircuit(n)
        _apply_cost_unitary(circuit, operator, gamma, order)
        values, angles = labels.copy(), np.zeros(len(labels))
        for instruction in circuit.data:
            qubits = [circuit.find_bit(q).index for q in instruction.qubits]
            if instruction.operation.name == "cx":
                control, target = qubits
                values ^= ((values >> control) & 1) << target
            elif instruction.operation.name == "rz":
                angles -= float(instruction.operation.params[0]) / 2 * (
                    1 - 2 * ((values >> qubits[0]) & 1).astype(float)
                )
            else:
                raise ValueError("Unexpected gate in diagonal cost audit")
        phases[order] = np.exp(1j * angles)
        error = float(np.max(np.abs(phases[order] - expected)))
        restored = bool(np.array_equal(values, labels))
        if not restored or error > 1e-10:
            raise RuntimeError(f"Cost unitary validation failed for {order}")
        results[order] = {"basis_labels_restored": restored, "max_phase_error_vs_ising": error}
    results["basis_states_verified"] = len(labels)
    results["max_phase_error_between_orders"] = float(np.max(np.abs(
        phases["original"] - phases["round-robin"]
    )))
    return results


def reordered_from_preparation(original, operator, preparation):
    """Use precisely the existing circuit's initial RY angles and frozen parameters."""
    n = original.num_qubits
    initial = list(original.data[:n])
    if len(initial) != n or any(
        item.operation.name != "ry" or original.find_bit(item.qubits[0]).index != i
        for i, item in enumerate(initial)
    ):
        raise ValueError("Expected one initial RY per logical qubit")
    probabilities = [np.sin(float(item.operation.params[0]) / 2) ** 2 for item in initial]
    candidate, gammas, betas = build_warm_start_qaoa(
        operator, p=1, warm_start_probabilities=probabilities,
        epsilon=0.25, mixer="x", cost_order="round-robin",
    )
    # Avoid even round-trip trigonometric changes to the initial amplitudes.
    for i in range(n):
        candidate.data[i] = initial[i].replace(qubits=(candidate.qubits[i],))
    candidate = candidate.assign_parameters(dict(zip(
        gammas + betas, preparation["optimized_parameters"], strict=True,
    )))
    candidate.measure_all()
    return candidate


def state_error(expected, actual):
    overlap = np.vdot(expected, actual)
    phase = overlap / abs(overlap) if abs(overlap) else 1
    return float(np.max(np.abs(actual - phase * expected)))


def validate_routed(logical, compiled, backend):
    """Validate target support, measurement mapping and ideal routed amplitudes."""
    if compiled.num_qubits > 22:
        raise ValueError("Routed statevector exceeds the audit's 22-qubit memory bound")
    reversed_cz_loci = 0
    for item in compiled.data:
        if item.operation.name == "barrier":
            continue
        indices = tuple(compiled.find_bit(q).index for q in item.qubits)
        supported = backend.target.instruction_supported(
            operation_name=item.operation.name, qargs=indices,
        )
        # IQM lists CZ calibration loci in one direction. CZ is symmetric,
        # so the reverse locus denotes the same physical operation. Keep
        # directional gates and absent couplings subject to the exact check.
        if not supported and isinstance(item.operation, CZGate):
            supported = backend.target.instruction_supported(
                operation_name="cz", qargs=indices[::-1],
            )
            reversed_cz_loci += int(supported)
        if not supported:
            raise RuntimeError(f"Unsupported target operation {item.operation.name} on {indices}")
    layout = compiled.layout.final_index_layout(filter_ancillas=True)
    if len(layout) != logical.num_qubits or len(set(layout)) != len(layout):
        raise RuntimeError("Invalid final logical-to-physical layout")
    observed = {
        compiled.find_bit(item.clbits[0]).index: compiled.find_bit(item.qubits[0]).index
        for item in compiled.data if item.operation.name == "measure"
    }
    if observed != dict(enumerate(layout)):
        raise RuntimeError("Measurement mapping does not preserve the invoice bit order")
    ideal = Statevector.from_instruction(logical.remove_final_measurements(inplace=False)).data
    logical_indices = np.arange(len(ideal), dtype=np.uint64)
    physical_indices = np.zeros(len(ideal), dtype=np.uint64)
    for source, target in enumerate(layout):
        physical_indices |= ((logical_indices >> source) & 1) << target
    expected = np.zeros(1 << compiled.num_qubits, dtype=complex)
    expected[physical_indices] = ideal
    actual = Statevector.from_instruction(compiled.remove_final_measurements(inplace=False)).data
    error = state_error(expected, actual)
    if error > 1e-9:
        raise RuntimeError(f"Routed statevector validation failed: {error}")
    return {"target_operations_valid": True, "measurement_mapping_valid": True,
            "symmetric_cz_reverse_locus_matches": reversed_cz_loci,
            "final_logical_to_physical": layout, "max_amplitude_error_up_to_global_phase": error}


def _freeze(circuit, path):
    with path.open("wb") as stream:
        qpy.dump(circuit, stream)
    return {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def audit(*, output_dir, logical_only=False, seeds=(42, 7, 123), maxiter=80,
          preparation_seed=42, relaxation_multistart=32):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least one unique routing seed")
    # Exclusive directory creation protects every previous preflight/result.
    output_dir.mkdir(parents=True, exist_ok=False)
    versions = {}
    for name in ("qiskit", "numpy", "scipy", "iqm-client", "qiskit-iqm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    backend = None if logical_only else connect_iqm_backend("q20")
    original, qubo, problem, preparation = prepare_ten_invoice_qaoa(
        seed=preparation_seed, maxiter=maxiter, relaxation_multistart=relaxation_multistart,
    )
    operator = qubo_to_ising(qubo).normalized_operator
    candidate = reordered_from_preparation(original, operator, preparation)
    circuits = {"original": original, "round-robin": candidate}
    basis = cost_basis_audit(operator, preparation["optimized_parameters"][0])
    states = {name: Statevector.from_instruction(c.remove_final_measurements(inplace=False)).data
              for name, c in circuits.items()}
    error = state_error(states["original"], states["round-robin"])
    if error > 1e-10:
        raise RuntimeError(f"Full logical circuit equivalence failed: {error}")
    root = Path(__file__).resolve().parents[3]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                            text=True, capture_output=True, check=False).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                           text=True, capture_output=True, check=False).stdout.strip()
    payload = {
        "experiment": "10-invoice-cost-order-routing-audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit, "source_worktree_dirty": bool(dirty),
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), root / "src/cherryq/quantum.py",
                                    root / "src/cherryq/hardware/q20_ten.py")},
        "versions": versions, "qpu_submitted": False,
        "hardware_decision": "NO-GO: physical results require review; no submission authorized by this audit",
        "physical_routing_verified": False,
        "scope": "logical-only" if logical_only else "current-Q20-target-transpilation-and-ideal-simulation",
        "historical_reference_reported_by_uploaded_audit": {"cz": 300, "depth": 283},
        "preparation": preparation, "preparation_note":
            "New preparation, optimized once and shared by both schedules; not a replay of the archived Q20 circuit.",
        "classical_optimum": {"paid_invoice_ids": problem.solve_exact().paid_invoice_ids,
                              "spend_gbp": problem.solve_exact().spend_gbp,
                              "business_loss_gbp": problem.solve_exact().business_loss_gbp},
        "basis_audit": basis, "max_logical_state_error": error,
        "logical": {}, "routing": [], "paired_deltas": [],
    }
    for name, circuit in circuits.items():
        payload["logical"][name] = {
            **circuit_metrics(circuit),
            "qpy": _freeze(circuit, output_dir / f"{name}-logical.qpy"),
        }
    report = output_dir / "routing-audit.json"
    def save():
        report.write_text(json.dumps(payload, indent=2, default=list) + "\n")
    save()
    if backend is not None:
        payload["backend"] = backend_summary(backend)
        # Capture target operation loci, not only a union coupling graph.
        payload["target_qargs"] = {
            name: [list(q) if q is not None else None for q in backend.target[name]]
            for name in backend.target.operation_names
        }
        payload["backend_calibration_set_id"] = str(getattr(backend, "calibration_set_id", None))
        save()
        for seed in seeds:
            paired = {}
            for order, logical in circuits.items():
                compiled = transpile(logical, backend=backend, optimization_level=3, seed_transpiler=seed)
                metrics = circuit_metrics(compiled)
                frozen = _freeze(compiled, output_dir / f"{order}-seed-{seed}-routed.qpy")
                try:
                    verification = validate_routed(logical, compiled, backend)
                except Exception as exc:
                    payload["failed_routing"] = {
                        "order": order, "seed_transpiler": seed, **metrics,
                        "error": f"{type(exc).__name__}: {exc}", "qpy": frozen,
                        "qubits": [repr(q) for q in compiled.qubits],
                        "layout": str(compiled.layout),
                    }
                    save()
                    raise
                paired[order] = metrics
                payload["routing"].append({
                    "order": order, "seed_transpiler": seed, "optimization_level": 3,
                    **metrics, "verification": verification,
                    "qpy": frozen,
                })
                save()
            payload["paired_deltas"].append({
                "seed_transpiler": seed,
                "candidate_minus_original_cz": paired["round-robin"]["operation_counts"].get("cz", 0)
                    - paired["original"]["operation_counts"].get("cz", 0),
                "candidate_minus_original_depth": paired["round-robin"]["depth_with_measurements"]
                    - paired["original"]["depth_with_measurements"],
            })
        payload["physical_routing_verified"] = True
        save()
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--logical-only", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--preparation-seed", type=int, default=42)
    parser.add_argument("--relaxation-multistart", type=int, default=32)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Routing seeds must be unique")
    payload = audit(output_dir=args.output_dir, logical_only=args.logical_only,
                    seeds=tuple(args.seeds), maxiter=args.maxiter,
                    preparation_seed=args.preparation_seed,
                    relaxation_multistart=args.relaxation_multistart)
    print(json.dumps(payload, indent=2, default=list))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
