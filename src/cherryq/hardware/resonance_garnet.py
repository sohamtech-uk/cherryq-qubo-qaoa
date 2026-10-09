"""One explicitly authorised Garnet batch, gated by a no-submit preflight.

No facade construction, alternative device, retry, or second submission path.
The existing simulator module is imported only for its pure circuit/scoring helpers.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone

import numpy as np
from qiskit import qpy
from qiskit.quantum_info import Statevector

from ..classical import benchmark_classical
from .q20 import _label_to_values
from .q20_routing_audit import cost_basis_audit, state_error, transpile_q20_for_audit, validate_routed
from .resonance_simulation import BETA, GAMMA, METRICS, frozen_circuits, state_masks, summarise_counts

URL = "https://resonance.iqm.tech"
ALIAS = "garnet"
SHOTS = 1000
SEED = 42
AUTHORISATION = "cherryq-garnet-20261009-one-batch-2x1000"
BRANCH = "refs/heads/experiment/resonance-garnet-20261009"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def freeze(circuit, path):
    with path.open("wb") as stream:
        qpy.dump(circuit, stream)
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def require_real_garnet(backend):
    from iqm.qiskit_iqm.iqm_provider import IQMBackend, IQMFacadeBackend
    if type(backend) is not IQMBackend or isinstance(backend, IQMFacadeBackend):
        raise ValueError("An ordinary IQMBackend is required")
    resolved = backend.client._iqm_server_client
    if resolved.root_url != URL or resolved.quantum_computer != ALIAS:
        raise ValueError("Only real Resonance garnet is authorised")
    if "facade" in backend.name.lower() or "mock" in backend.name.lower():
        raise ValueError("Mock or facade backend is forbidden")
    if backend.num_qubits < 14 or backend.num_qubits > 22:
        raise ValueError("Unexpected Garnet qubit capacity")


def circuit_record(logical, compiled, backend, output, order):
    verification = validate_routed(logical, compiled, backend)
    counts = dict(compiled.count_ops())
    one_qubit = sum(len(i.qubits) == 1 and i.operation.name not in
                    {"barrier", "measure", "delay", "reset"} for i in compiled.data)
    return {
        "schedule": order, "circuit_name": compiled.name,
        "routing_seed": SEED,
        "physical_depth": compiled.remove_final_measurements(inplace=False).depth(),
        "physical_depth_with_measurements": compiled.depth(),
        "cz_count": counts.get("cz", 0), "single_qubit_gate_count": one_qubit,
        "operation_counts": counts, "allocated_physical_qubits": compiled.num_qubits,
        "active_physical_qubits": sorted({compiled.find_bit(q).index for i in compiled.data
                                         if i.operation.name != "barrier" for q in i.qubits}),
        "final_logical_to_physical": verification["final_logical_to_physical"],
        "final_logical_to_qubit_name": [backend.index_to_qubit_name(i) for i in
                                         verification["final_logical_to_physical"]],
        "verification": verification,
        "logical_qpy": freeze(logical, output / f"{order}-logical.qpy"),
        "compiled_qpy": freeze(compiled, output / f"{order}-compiled.qpy"),
    }


def check_serialized_measurements(request, compiled, backend, qubo, problem):
    from iqm.qiskit_iqm.iqm_job import IQMJob
    from iqm.qiskit_iqm.qiskit_to_iqm import MeasurementKey
    if len(request.circuits) != 2 or request.shots != SHOTS:
        raise ValueError("Request must contain two circuits with 1000 shots each")
    optimum, _ = qubo.solve_exact()
    label = "".join(str(optimum[n]) for n in reversed(qubo.variable_names))
    for serial, circuit in zip(request.circuits, compiled, strict=True):
        if serial.name != circuit.name:
            raise ValueError("Serialized circuit order or identity changed")
        measures = [i for i in serial.instructions if i.name == "measure"]
        if len(measures) != 14:
            raise ValueError("Expected exactly 14 serialized measurements")
        layout = circuit.layout.final_index_layout(filter_ancillas=True)
        seen, synthetic = set(), {}
        for i in measures:
            key = MeasurementKey.from_string(i.args["key"])
            if key.creg_len != 14 or key.creg_idx != 0 or key.clbit_idx in seen:
                raise ValueError("Invalid serialized classical register")
            seen.add(key.clbit_idx)
            if tuple(i.locus) != (backend.index_to_qubit_name(layout[key.clbit_idx]),):
                raise ValueError("Serialized measurement locus disagrees with final layout")
            synthetic[i.args["key"]] = [[optimum[qubo.variable_names[key.clbit_idx]]]]
        if seen != set(range(14)):
            raise ValueError("Serialized measurement coverage failed")
        if IQMJob._iqm_format_measurement_results(synthetic, 1) != [label]:
            raise ValueError("SDK result bit ordering is inconsistent")
    decoded = _label_to_values(label, qubo.variable_names)
    if decoded != optimum or problem.supplier_decision_from_values(decoded, qubo.variable_names) != problem.solve_exact().decision:
        raise ValueError("Business-bit interpretation failed")
    return {"serialized_measurement_mapping_valid": True,
            "sdk_result_bit_order_round_trip_valid": True,
            "optimal_full_qubo_bitstring": label,
            "bit_order_left_to_right": list(reversed(qubo.variable_names)),
            "optimal_plan_probability_marginalizes_slack": True}


def submit_once(backend, compiled, report, output):
    require_real_garnet(backend)
    if report.get("preflight") != "PASS" or len(compiled) != 2:
        raise ValueError("Both circuits must pass before submission")
    if os.environ.get("GITHUB_RUN_ATTEMPT") != "1":
        raise ValueError("Workflow retries cannot submit hardware jobs")
    if os.environ.get("GITHUB_REF") != BRANCH:
        raise ValueError("Unexpected branch for this authorisation")
    # Exclusive marker is created before the only call; ambiguous network failure
    # is never retried. The workflow also prohibits rerun attempts and new pushes.
    with (output / "submission-attempted.json").open("x") as f:
        json.dump({"authorization": AUTHORISATION, "shots_per_circuit": SHOTS,
                   "circuits": 2, "submission_attempted": True}, f)
    report["submission_attempted"] = True
    write_json(output / "garnet-experiment.json", report)
    job = backend.run([compiled[0], compiled[1]], shots=1000)
    report["iqm_job_id"] = job.job_id()
    report["physical_qpu_submitted"] = True
    report["submitted_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "garnet-experiment.json", report)
    print("CHERRYQ_GARNET_JOB " + job.job_id(), flush=True)
    return job


def run(output, authorization):
    if authorization != AUTHORISATION:
        raise ValueError("This experiment requires its specific one-batch authorisation")
    output.mkdir(parents=True, exist_ok=False)
    os.environ.pop("IQM_CLIENT_DEBUG", None)
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    report = {"experiment": AUTHORISATION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "preflight": "IN_PROGRESS", "physical_qpu_submitted": False,
              "submission_attempted": False, "iqm_job_id": None,
              "source_commit": os.getenv("GITHUB_SHA"), "github_run_id": os.getenv("GITHUB_RUN_ID"),
              "gamma": GAMMA, "beta": BETA, "p": 1, "logical_qubits": 14,
              "shots_per_circuit": SHOTS, "max_real_batch_jobs": 1,
              "transpilation": {"seed_transpiler": SEED, "optimization_level": 3,
                                "scheduling_method": "default"},
              "versions": {p: importlib.metadata.version(p) for p in
                           ("iqm-client", "qiskit", "numpy", "scipy")}, "circuits": {}}
    stage = "connect-real-garnet"
    try:
        if not os.getenv("IQM_TOKEN"):
            raise ValueError("IQM_TOKEN is unavailable in the configured environment")
        from iqm.qiskit_iqm import IQMProvider
        provider = IQMProvider("https://resonance.iqm.tech", quantum_computer="garnet")
        backend = provider.get_backend()
        require_real_garnet(backend)
        qubits = list(backend.architecture.qubits)
        report["backend"] = {"name": backend.name, "requested_alias": ALIAS,
                             "resolved_alias": backend.client._iqm_server_client.quantum_computer,
                             "class": type(backend).__name__, "real_hardware": True,
                             "facade": False, "mock": False,
                             "usable_qubit_count": len(qubits), "usable_qubits": qubits,
                             "calibration_set_id": str(backend.architecture.calibration_set_id)}
        print("CHERRYQ_BACKEND " + json.dumps(report["backend"]), flush=True)
        stage = "frozen-logical-circuits"
        problem, qubo, operator, logicals = frozen_circuits()
        ground = benchmark_classical(problem).as_dict()
        if not ground["solutions_agree"] or ground["exact_spend_gbp"] != 5000 or ground["exact_business_loss_gbp"] != 950 or tuple(ground["exact_paid"]) != ("A", "B", "F", "G", "H"):
            raise ValueError("Ground truth differs from the verified checkpoint")
        report["ground_truth"] = ground
        report["variable_names"] = list(qubo.variable_names)
        masks, _, _ = state_masks(qubo, problem)
        states = {k: Statevector.from_instruction(v.remove_final_measurements(inplace=False)).data
                  for k, v in logicals.items()}
        report["logical_state_equivalence_error"] = state_error(states["original"], states["round-robin"])
        if report["logical_state_equivalence_error"] > 1e-10:
            raise ValueError("Logical circuits are not equivalent")
        report["cost_basis_audit"] = cost_basis_audit(operator, GAMMA)
        compiled = []
        for order, logical in logicals.items():
            stage = f"transpile-and-validate-{order}"
            logical.name = f"cherryq-10-{order}"
            native = transpile_q20_for_audit(logical, backend, SEED)
            native.name = logical.name
            record = circuit_record(logical, native, backend, output, order)
            record["ideal_metrics"] = {k: float(np.sum(np.abs(states[order][m]) ** 2))
                                       for k, m in masks.items()}
            report["circuits"][order] = record
            compiled.append(native)
            write_json(output / "garnet-experiment.json", report)
            print("CHERRYQ_CIRCUIT " + json.dumps(record), flush=True)
        stage = "create-run-request-no-submit"
        request = backend.create_run_request([compiled[0], compiled[1]], shots=1000)
        report["measurement_interpretation"] = check_serialized_measurements(request, compiled, backend, qubo, problem)
        if str(request.calibration_set_id) != report["backend"]["calibration_set_id"]:
            raise ValueError("Run request calibration does not match compilation")
        report["request_validation"] = {"passed": True, "circuit_count": 2, "shots_per_circuit": 1000,
                                        "calibration_set_id": str(request.calibration_set_id)}
        report["preflight"] = "PASS"
        write_json(output / "preflight.json", report)
        write_json(output / "garnet-experiment.json", report)
        print("CHERRYQ_PREFLIGHT PASS; two circuits; 1000 shots each", flush=True)
        stage = "single-submission"
        job = submit_once(backend, compiled, report, output)
        stage = "wait-for-results"
        from qiskit.providers import JobStatus
        started = time.monotonic()
        while True:
            status = job.status()
            report["job_status"] = status.name
            write_json(output / "garnet-experiment.json", report)
            print("CHERRYQ_JOB_STATUS " + status.name, flush=True)
            if status in (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED):
                break
            if time.monotonic() - started > 5400:
                raise TimeoutError("Completion wait expired; retrieve this job, never resubmit")
            time.sleep(20)
        if status != JobStatus.DONE:
            raise RuntimeError("Submitted hardware job did not complete successfully")
        result = job.result(timeout=60, cancel_after_timeout=False)
        if not result.success or len(result.results) != 2:
            raise ValueError("Expected exactly two successful circuit results")
        stage = "analyse-results"
        for index, order in enumerate(logicals):
            if result.results[index].header["name"] != compiled[index].name:
                raise ValueError("Result circuit identity disagrees with batch order")
            counts = {str(k): int(v) for k, v in result.get_counts(index).items()}
            if sum(counts.values()) != 1000:
                raise ValueError("Unexpected shot total")
            summary = summarise_counts(counts, qubo, problem)
            summary["optimal_plan_shots"] = round(summary[METRICS[0]] * 1000)
            summary["metric_shot_counts"] = {m: round(summary[m] * 1000) for m in METRICS}
            report["circuits"][order]["counts"] = counts
            report["circuits"][order]["hardware_metrics"] = summary
            report["circuits"][order]["execution_calibration_set_id"] = str(getattr(result.results[index], "calibration_set_id", None))
            print("CHERRYQ_HARDWARE_RESULT " + json.dumps({"schedule": order, **summary}), flush=True)
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["result_success"] = True
        write_json(output / "garnet-experiment.json", report)
        return 0
    except Exception as exc:
        if not report["submission_attempted"]:
            report["preflight"] = "NO-GO"
        report["failure"] = {"stage": stage, "exception_type": type(exc).__name__,
                             "message": "Details suppressed to prevent credentials or server secrets appearing in output.",
                             "safe_trace": [{"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                                            for f in traceback.extract_tb(exc.__traceback__)]}
        write_json(output / "garnet-experiment.json", report)
        print("CHERRYQ_FAILURE " + json.dumps(report["failure"]), flush=True)
        return 1
    finally:
        # Only a deliberately constructed, nonsecret report is printed. No SDK
        # object, request, environment, HTTP response or raw Result is dumped.
        print("CHERRYQ_REPORT_JSON " + json.dumps(report, default=str), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args()
    sys.exit(run(args.output_dir, args.authorization))


if __name__ == "__main__":
    main()
