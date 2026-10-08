from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from qiskit import QuantumCircuit, transpile


@dataclass(frozen=True)
class IQMDeviceConfig:
    key: str
    display_name: str
    cortex_env_var: str
    quantum_computer: str


Q20 = IQMDeviceConfig(
    key="q20",
    display_name="Aalto Q20",
    cortex_env_var="Q20_CORTEX_URL",
    quantum_computer="radiance20",
)

Q50 = IQMDeviceConfig(
    key="q50",
    display_name="VTT Q50",
    cortex_env_var="Q50_CORTEX_URL",
    quantum_computer="q50",
)

_DEVICES = {Q20.key: Q20, Q50.key: Q50}


def get_iqm_device_config(device: str) -> IQMDeviceConfig:
    key = device.strip().lower()
    try:
        return _DEVICES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown IQM device {device!r}; choose from {sorted(_DEVICES)}"
        ) from exc


def connect_iqm_backend(device: str = "q20"):
    """Connect to an IQM backend using the Cortex URL exported by CSC modules."""

    config = get_iqm_device_config(device)
    cortex_url = os.getenv(config.cortex_env_var)
    if not cortex_url:
        raise RuntimeError(
            f"{config.cortex_env_var} is not set. Load the CSC quantum module, "
            "export DEVICES for the target device, and source $RUN_SETUP first."
        )

    try:
        from iqm.qiskit_iqm import IQMProvider
    except ImportError as exc:
        raise RuntimeError(
            "IQM's Qiskit adapter is not available in this Python environment. "
            "On LUMI use the CSC quantum module; locally install the CherryQ "
            "IQM optional dependency."
        ) from exc

    provider = IQMProvider(
        cortex_url,
        quantum_computer=config.quantum_computer,
    )
    return provider.get_backend()


def _two_qubit_gate_count(circuit: QuantumCircuit) -> int:
    return sum(1 for instruction in circuit.data if len(instruction.qubits) == 2)


def transpile_for_backend(
    circuit: QuantumCircuit,
    backend,
    *,
    optimization_level: int = 1,
) -> tuple[QuantumCircuit, dict[str, Any]]:
    if optimization_level not in {0, 1, 2, 3}:
        raise ValueError("optimization_level must be one of 0, 1, 2, 3")

    compiled = transpile(
        circuit,
        backend=backend,
        optimization_level=optimization_level,
    )
    metrics = {
        "logical_qubits": circuit.num_qubits,
        "transpiled_qubits": compiled.num_qubits,
        "logical_depth": int(circuit.depth()),
        "transpiled_depth": int(compiled.depth()),
        "logical_two_qubit_gates": _two_qubit_gate_count(circuit),
        "transpiled_two_qubit_gates": _two_qubit_gate_count(compiled),
        "optimization_level": optimization_level,
    }
    return compiled, metrics


def backend_summary(backend) -> dict[str, Any]:
    coupling_map = getattr(backend, "coupling_map", None)
    if coupling_map is None:
        coupling_edges: list[list[int]] | None = None
    else:
        try:
            coupling_edges = [list(edge) for edge in coupling_map.get_edges()]
        except AttributeError:
            coupling_edges = None

    operation_names = getattr(backend, "operation_names", None)
    return {
        "backend_name": getattr(backend, "name", backend.__class__.__name__),
        "num_qubits": int(getattr(backend, "num_qubits", 0)),
        "operation_names": (
            sorted(str(name) for name in operation_names)
            if operation_names is not None
            else None
        ),
        "coupling_map": coupling_edges,
    }


def submit_and_collect(
    circuit: QuantumCircuit,
    backend,
    *,
    shots: int,
) -> dict[str, Any]:
    if shots < 1:
        raise ValueError("shots must be positive")

    job = backend.run(circuit, shots=shots)
    job_id_value = job.job_id()
    print(f"Submitted IQM job ID: {job_id_value}", flush=True)
    result = job.result()
    counts = result.get_counts()

    if isinstance(counts, list):
        if len(counts) != 1:
            raise RuntimeError(
                "CherryQ hardware runner expects one circuit per submitted job"
            )
        counts = counts[0]

    return {
        "job_id": str(job_id_value),
        "shots": shots,
        "counts": {str(key): int(value) for key, value in dict(counts).items()},
    }
