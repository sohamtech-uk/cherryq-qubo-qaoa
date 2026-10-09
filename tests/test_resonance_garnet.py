"""Offline execution-boundary and bit-order regression checks. Never contact IQM."""
from types import SimpleNamespace

import pytest

pytest.importorskip("iqm.qiskit_iqm")

from cherryq.hardware import resonance_garnet as experiment
from cherryq.hardware.resonance_simulation import frozen_circuits


def test_refuses_fake_and_facade_before_any_network_or_run():
    from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
    with pytest.raises(ValueError, match="ordinary IQMBackend"):
        experiment.require_real_garnet(IQMFakeGarnet())


def test_refuses_other_aliases_and_endpoints():
    from iqm.qiskit_iqm.iqm_provider import IQMBackend
    for url, alias in [(experiment.URL, "garnet:mock"), (experiment.URL, "emerald"),
                       ("https://q20.example", "garnet")]:
        backend = object.__new__(IQMBackend)
        backend.client = SimpleNamespace(_iqm_server_client=SimpleNamespace(root_url=url, quantum_computer=alias))
        with pytest.raises(ValueError, match="Only real Resonance"):
            experiment.require_real_garnet(backend)


def test_submit_once_and_refuse_rerun(tmp_path, monkeypatch):
    calls = []
    job = SimpleNamespace(job_id=lambda: "test-id")
    backend = SimpleNamespace(run=lambda circuits, **kw: (calls.append((circuits, kw)) or job))
    monkeypatch.setattr(experiment, "require_real_garnet", lambda b: None)
    monkeypatch.setenv("GITHUB_REF", experiment.BRANCH)
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    report = {"preflight": "NO-GO"}
    with pytest.raises(ValueError, match="Both circuits"):
        experiment.submit_once(backend, [1, 2], report, tmp_path)
    assert not calls
    report["preflight"] = "PASS"
    experiment.submit_once(backend, [1, 2], report, tmp_path)
    with pytest.raises(FileExistsError):
        experiment.submit_once(backend, [1, 2], report, tmp_path)
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    with pytest.raises(ValueError, match="retries"):
        experiment.submit_once(backend, [1, 2], report, tmp_path)
    assert calls == [([1, 2], {"shots": 1000})]


def test_complete_offline_routing_and_serialized_measurement_mapping(tmp_path):
    from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
    from iqm.qiskit_iqm.iqm_provider import IQMBackend
    from cherryq.hardware.q20_routing_audit import transpile_q20_for_audit
    backend = IQMFakeGarnet()
    problem, qubo, _, logicals = frozen_circuits()
    compiled = []
    serialized = []
    for order, logical in logicals.items():
        native = transpile_q20_for_audit(logical, backend, 42)
        native.name = "test-" + order
        record = experiment.circuit_record(logical, native, backend, tmp_path, order)
        assert record["verification"]["measurement_mapping_valid"]
        compiled.append(native)
        serialized.append(IQMBackend.serialize_circuit(backend, native))
    request = SimpleNamespace(circuits=serialized, shots=1000)
    result = experiment.check_serialized_measurements(request, compiled, backend, qubo, problem)
    assert result["optimal_full_qubo_bitstring"] == "00000011100011"
    assert result["sdk_result_bit_order_round_trip_valid"]
