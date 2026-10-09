"""Read and verify the existing real Garnet job. No submission API is called."""
import hashlib
import dataclasses
import json
import logging
import os
from pathlib import Path
import traceback
import warnings
from uuid import UUID

from qiskit import qpy
from iqm.qiskit_iqm import IQMProvider
from cherryq.hardware.resonance_garnet import require_real_garnet
from cherryq.hardware.resonance_simulation import METRICS, frozen_circuits, summarise_counts

JOB_ID = "01a12051-58d8-7226-87ed-795caff11d3d"


def canonical(circuit):
    if dataclasses.is_dataclass(circuit):
        circuit = dataclasses.asdict(circuit)
    elif hasattr(circuit, "model_dump"):
        circuit = circuit.model_dump(mode="json")
    return json.dumps(circuit, sort_keys=True, separators=(",", ":"), default=str)


def main():
    os.environ.pop("IQM_CLIENT_DEBUG", None)
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    root = Path("results/real-garnet")
    report = json.loads((root / "garnet-experiment.json").read_text())
    assert report["iqm_job_id"] == JOB_ID and report["preflight"] == "PASS"
    try:
        provider = IQMProvider("https://resonance.iqm.tech", quantum_computer="garnet")
        backend = provider.get_backend(calibration_set_id=UUID(report["backend"]["calibration_set_id"]))
        require_real_garnet(backend)
        job = backend.retrieve_job(JOB_ID)
        result = job.result(timeout=60, cancel_after_timeout=False)
        assert result.success and len(result.results) == 2
        payload_circuits, parameters = job._iqm_job.payload()
        assert len(payload_circuits) == 2 and parameters.shots == 1000
        problem, qubo, _, _ = frozen_circuits()
        payload_checks = []
        for index, order in enumerate(("original", "round-robin")):
            record = report["circuits"][order]
            path = root / record["compiled_qpy"]["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["compiled_qpy"]["sha256"]
            with path.open("rb") as f:
                compiled = qpy.load(f)[0]
            expected = backend.serialize_circuit(compiled)
            actual = payload_circuits[index]
            same = canonical(expected) == canonical(actual)
            safe_identity = {"schedule": order, "payload_type": type(actual).__name__,
                             "submitted_name": compiled.name,
                             "result_header_name": result.results[index].header["name"],
                             "entire_payload_equal_to_archived_compiled_circuit": same,
                             "expected_sha256": hashlib.sha256(canonical(expected).encode()).hexdigest(),
                             "retrieved_sha256": hashlib.sha256(canonical(actual).encode()).hexdigest()}
            print("CHERRYQ_PAYLOAD_IDENTITY " + json.dumps(safe_identity), flush=True)
            if not same:
                raise ValueError("Payload identity is not verified; counts will not be attributed")
            payload_checks.append(safe_identity)
            counts = {str(k): int(v) for k, v in result.get_counts(index).items()}
            assert sum(counts.values()) == 1000
            summary = summarise_counts(counts, qubo, problem)
            summary["optimal_plan_shots"] = round(summary[METRICS[0]] * 1000)
            summary["metric_shot_counts"] = {m: round(summary[m] * 1000) for m in METRICS}
            record["counts"] = counts
            record["hardware_metrics"] = summary
            record["execution_calibration_set_id"] = str(getattr(result.results[index], "calibration_set_id", None))
            print("CHERRYQ_HARDWARE_RESULT " + json.dumps({"schedule": order, **summary}), flush=True)
        report["result_success"] = True
        report["recovered_read_only"] = True
        report["retrieval_workflow_run_id"] = os.getenv("GITHUB_RUN_ID")
        report["result_identity_verification"] = payload_checks
        report["initial_analysis_failure"] = report.pop("failure")
        report["result_name_explanation"] = "IQM Client 34.0.2 CircuitJob.payload converts circuits to dictionaries with model_dump; IQMJob uses circuit_0/circuit_1 fallback names for non-Circuit objects. Full retrieved circuit payloads were checked against archived QPY serialization, including instruction order, loci, angles, measurement keys, names and metadata."
        (root / "garnet-experiment.json").write_text(json.dumps(report, indent=2) + "\n")
        print("CHERRYQ_REPORT_JSON " + json.dumps(report), flush=True)
    except Exception as exc:
        failure = {"type": type(exc).__name__, "safe_trace": [
            {"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
            for f in traceback.extract_tb(exc.__traceback__)]}
        print("CHERRYQ_RETRIEVAL_FAILURE " + json.dumps(failure), flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
