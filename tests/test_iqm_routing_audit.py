"""Optional offline integration with the real IQM transpiler plugin.

Run with the LUMI package versions as well as the core suite. No provider
connection or QPU/backend execution is needed.
"""
import pytest
from qiskit import QuantumCircuit

iqm = pytest.importorskip("iqm.qiskit_iqm")

from cherryq.hardware.q20_routing_audit import transpile_q20_for_audit, validate_routed


def test_iqm_grid_routing_preserves_target_amplitudes_and_readout(monkeypatch):
    backend = iqm.IQMFakeAdonis()
    def forbidden_run(*args, **kwargs):
        raise AssertionError("The routing audit must not execute a backend")
    monkeypatch.setattr(backend, "run", forbidden_run)
    logical = QuantumCircuit(3)
    logical.ry(0.4, 0)
    logical.ry(0.7, 1)
    logical.cx(0, 2)
    logical.rz(0.5, 2)
    logical.cx(2, 1)
    logical.measure_all()
    compiled = transpile_q20_for_audit(logical, backend, 42)
    result = validate_routed(logical, compiled, backend)
    assert result["target_operations_valid"]
    assert result["measurement_mapping_valid"]
    assert result["max_amplitude_error_up_to_global_phase"] < 1e-10


def test_grid_audit_rejects_resonator_architecture():
    with pytest.raises(ValueError, match="grid without resonators"):
        transpile_q20_for_audit(QuantumCircuit(1), iqm.IQMFakeDeneb(), 42)
