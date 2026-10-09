"""Frozen CherryQ comparison. The only remote execution target is garnet:mock.

No physical backend selection or QPU execution path is exposed by this module.
IQM's stock facade validates on the remote mock and simulates locally with Aer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter

import numpy as np
from qiskit import qpy
from qiskit.quantum_info import Statevector

from ..classical import benchmark_classical
from ..problem import build_ten_invoice_problem
from ..quantum import build_warm_start_qaoa, qubo_to_ising, solve_continuous_relaxation
from .q20 import _label_to_values, analyse_cherryq_counts
from .q20_routing_audit import (
    circuit_metrics, cost_basis_audit, state_error,
    transpile_q20_for_audit, validate_routed,
)

RESONANCE_URL = "https://resonance.iqm.tech"
MOCK_ALIAS = "garnet:mock"
FACADE_NAME = "facade_garnet"
GAMMA = -0.7439674535854706
BETA = 0.3004519607226279
WARM_START = (1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0)
METRICS = (
    "optimal_business_plan_probability", "business_feasible_probability",
    "qubo_feasible_probability", "optimal_qubo_probability",
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=list) + "\n")


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_mock_target(url, alias):
    # Exact allowlist: no real computer, default computer, alternative URL or timeslot.
    if url != RESONANCE_URL or alias != MOCK_ALIAS:
        raise ValueError("Only the explicitly allowlisted Resonance garnet:mock is permitted")


def frozen_circuits():
    problem = build_ten_invoice_problem()
    qubo = problem.build_qubo(penalty_gbp=250.0)
    relaxation = solve_continuous_relaxation(qubo, multistart=32, seed=42)
    if not np.allclose(relaxation.values, WARM_START, atol=1e-12, rtol=0):
        raise RuntimeError("Warm-start preparation differs from the verified checkpoint")
    operator = qubo_to_ising(qubo).normalized_operator
    circuits = {}
    for order in ("original", "round-robin"):
        circuit, gammas, betas = build_warm_start_qaoa(
            operator, p=1, warm_start_probabilities=relaxation.values,
            epsilon=0.25, mixer="x", cost_order=order,
        )
        circuit = circuit.assign_parameters(dict(zip(gammas + betas, (GAMMA, BETA), strict=True)))
        circuit.measure_all()
        circuits[order] = circuit
    return problem, qubo, operator, circuits


def state_masks(qubo, problem):
    """Use the repository's little-endian measurement interpretation and scorers."""
    exact_values, energy = qubo.solve_exact()
    exact = problem.solve_exact()
    masks = {name: [] for name in METRICS}
    optimal_count = 0
    for index in range(1 << len(qubo.variable_names)):
        values = _label_to_values(format(index, f"0{len(qubo.variable_names)}b"), qubo.variable_names)
        decision = problem.supplier_decision_from_values(values, qubo.variable_names)
        masks[METRICS[0]].append(decision == exact.decision)
        masks[METRICS[1]].append(problem.is_business_feasible(decision))
        masks[METRICS[2]].append(abs(qubo.constraint_residual(values)) < 1e-9)
        masks[METRICS[3]].append(values == exact_values)
        optimal_count += int(abs(qubo.energy(values) - energy) < 1e-9)
    if optimal_count != 1:
        raise RuntimeError("Expected the checkpoint's unique optimal QUBO state")
    return {k: np.asarray(v, dtype=bool) for k, v in masks.items()}, exact_values, energy


def wilson(successes, shots):
    z = 1.959963984540054
    p = successes / shots
    denominator = 1 + z * z / shots
    centre = (p + z * z / (2 * shots)) / denominator
    half = z * np.sqrt(p * (1 - p) / shots + z * z / (4 * shots * shots)) / denominator
    return [float(max(0, centre - half)), float(min(1, centre + half))]


def summarise_counts(counts, qubo, problem):
    result = analyse_cherryq_counts(counts, qubo, problem)
    result["binomial_wilson_95_intervals"] = {
        name: wilson(round(result[name] * result["shots"]), result["shots"])
        for name in METRICS
    }
    return result


def inventory_and_facade(output_dir):
    """Read only alias metadata, then connect only the authorized mock facade."""
    import requests
    from iqm.qiskit_iqm import IQMProvider
    from iqm.qiskit_iqm.iqm_provider import IQMFacadeBackend

    require_mock_target(RESONANCE_URL, MOCK_ALIAS)
    token = os.environ.get("IQM_TOKEN")
    if not token:
        raise RuntimeError("IQM_TOKEN is not configured; use a secure environment, never a CLI argument")
    os.environ.pop("IQM_CLIENT_DEBUG", None)
    response = requests.get(
        RESONANCE_URL + "/api/v1/quantum-computers",
        headers={"Authorization": "Bearer " + token}, timeout=30, allow_redirects=False,
    )
    if response.status_code != 200:
        raise RuntimeError("Authenticated mock inventory failed; response body intentionally omitted")
    aliases = sorted(item["alias"] for item in response.json()["quantum_computers"])
    mocks = [alias for alias in aliases if alias.endswith(":mock")]
    inventory = {"authenticated_api_inventory_verified": True, "mock_aliases": mocks,
                 "garnet_mock_available": MOCK_ALIAS in mocks, "physical_qpu_submitted": False}
    write_json(output_dir / "mock-inventory.json", inventory)
    if MOCK_ALIAS not in mocks:
        raise RuntimeError("garnet:mock is unavailable; refusing any physical or alternate target")
    provider = IQMProvider("https://resonance.iqm.tech", quantum_computer="garnet:mock")
    backend = provider.get_backend("facade_garnet")
    if not isinstance(backend, IQMFacadeBackend):
        raise RuntimeError("Expected IQMFacadeBackend")
    check_facade_target(backend)
    if backend.num_qubits != 20:
        raise RuntimeError("Expected 20-qubit Garnet mock")
    # IQMFacadeBackend itself verifies compatible static architectures.
    inventory["facade_static_architecture_compatibility_verified"] = True
    inventory["mock_num_qubits"] = backend.num_qubits
    write_json(output_dir / "mock-inventory.json", inventory)
    return backend


def check_facade_target(backend):
    # IQM 34/35 exposes the resolved computer on its server client. Fail closed
    # if a future version changes that interface; never infer from backend.name.
    client = backend.client._iqm_server_client
    require_mock_target(client.root_url, client.quantum_computer)


def export_csv(path, rows):
    columns = ["stage", "schedule", "routing_seed", "simulator_seed", "repeat", "shots", *METRICS,
               "best_sampled_business_loss_gbp", "logical_depth", "transpiled_depth", "physical_cz_count"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_comparison(output_dir, *, facade=False, local_compile=False,
                   shots=(1000, 5000, 10000), routing_seeds=(42, 7, 123),
                   ideal_seeds=(42, 7, 123), repeats=1):
    if not shots or min(shots) < 1000 or max(shots) > 20000 or repeats < 1:
        raise ValueError("Use 1000–20000 shots and at least one repeat")
    output_dir.mkdir(parents=True, exist_ok=False)
    problem, qubo, operator, circuits = frozen_circuits()
    classical = benchmark_classical(problem).as_dict()
    if (not classical["solutions_agree"] or tuple(classical["exact_paid"]) != ("A", "B", "F", "G", "H")
            or classical["exact_spend_gbp"] != 5000 or classical["exact_business_loss_gbp"] != 950):
        raise RuntimeError("Exact/MILP ground truth changed")
    masks, optimum_values, optimum_energy = state_masks(qubo, problem)
    states = {order: Statevector.from_instruction(c.remove_final_measurements(inplace=False)).data
              for order, c in circuits.items()}
    error = state_error(states["original"], states["round-robin"])
    if error > 1e-10:
        raise RuntimeError("Logical schedules are not equivalent")
    versions = {}
    for package in ("qiskit", "qiskit-aer", "iqm-client", "numpy", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    source_root = Path(__file__).resolve().parents[3]
    source_files = [Path(__file__), source_root / "src/cherryq/quantum.py", source_root / "src/cherryq/problem.py",
                    source_root / "src/cherryq/hardware/q20.py"]
    report = {
        "experiment": "CherryQ frozen 10-invoice IQM Resonance comparison",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "versions": versions,
        "source_sha256": {str(p.relative_to(source_root)): checksum(p) for p in source_files},
        "physical_qpu_submitted": False, "qpu_submitted": False,
        "gamma": GAMMA, "beta": BETA, "p": 1, "logical_qubits": 14,
        "penalty_gbp": 250.0, "variable_names": qubo.variable_names,
        "warm_start_probabilities_before_clipping": WARM_START, "epsilon": 0.25,
        "preparation": "Reproduced verified relaxation (seed 42, 32 starts), frozen gamma/beta; no QAOA reoptimization",
        "ground_truth": classical, "optimal_qubo_values": optimum_values, "optimal_qubo_energy": optimum_energy,
        "basis_audit": cost_basis_audit(operator, GAMMA), "max_logical_state_error": error,
        "interpretation": "Rightmost Qiskit bit is A; first 10 logical bits are invoices, next 4 are slack",
        "business_plan_probability_definition": "Invoice optimum, marginalized over all slack assignments",
        "business_feasibility_definition": "Invoice spend <= £5000, regardless of slack",
        "qubo_feasibility_definition": "Budget equality including encoded slack, residual zero",
        "optimal_qubo_probability_definition": "Unique exact minimum-energy 14-bit state",
        "ideal": {}, "routing": [], "samples": [],
        "noisy_simulation": {"status": "not_run", "target": MOCK_ALIAS, "facade": FACADE_NAME,
            "label": "Representative IQM Garnet noisy simulation; not an exact Aalto Q20 calibration model",
            "aalto_q20_architecture_or_calibration_equivalence_verified": False,
            "seed_simulator_supported": False,
            "seed_note": "IQM Client 34.0.2 stock fake-backend run forwards shots only. No seeded noisy claim is made.",
            "noise_limitation": "Stock model applies gate-duration T1/T2, depolarizing and readout errors; no idle-time scheduling noise. Depth alone is not a causal timing-noise test."},
        "aalto_q20": {"status": "future_not_run", "physical_execution_authorized": False},
        "quantum_advantage_claimed": False,
    }
    rows = [{"stage": "Exact/MILP", "best_sampled_business_loss_gbp": 950}]
    for order, circuit in circuits.items():
        path = output_dir / f"{order}-frozen.qpy"
        with path.open("wb") as stream:
            qpy.dump(circuit, stream)
        probabilities = np.abs(states[order]) ** 2
        metrics = {name: float(probabilities[mask].sum()) for name, mask in masks.items()}
        report["ideal"][order] = {**metrics, **circuit_metrics(circuit), "qpy_sha256": checksum(path)}
        rows.append({"stage": "Ideal statevector", "schedule": order,
                     "logical_depth": circuit_metrics(circuit)["depth_without_measurements"], **metrics})
        for nshots in shots:
            for seed in ideal_seeds:
                draws = np.random.default_rng(seed).multinomial(nshots, probabilities)
                counts = {format(i, "014b"): int(n) for i, n in enumerate(draws) if n}
                result = summarise_counts(counts, qubo, problem)
                sample = {"stage": "Ideal statevector sampling", "schedule": order,
                          "simulator_seed": seed, **result, "counts": counts}
                report["samples"].append(sample)
                rows.append(sample)
    write_json(output_dir / "comparison.json", report)
    export_csv(output_dir / "comparison.csv", rows)
    if facade or local_compile:
        if facade:
            backend = inventory_and_facade(output_dir)
            report["noisy_simulation"]["status"] = "running"
            fake_backend = backend._fake_backend
        else:
            from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
            backend = fake_backend = IQMFakeGarnet()
        # Store nonsecret, exact built-in noise model parameters for reproducibility.
        from dataclasses import asdict
        def serializable(value):
            if isinstance(value, dict):
                return {("|".join(k) if isinstance(k, tuple) else k): serializable(v) for k, v in value.items()}
            return value
        write_json(output_dir / "iqm-garnet-error-profile.json", serializable(asdict(fake_backend.error_profile)))
        for seed in routing_seeds:
            for order, circuit in circuits.items():
                compiled = transpile_q20_for_audit(circuit, backend, seed)
                validation = validate_routed(circuit, compiled, backend)
                physical = circuit_metrics(compiled)
                path = output_dir / f"{order}-garnet-route-{seed}.qpy"
                with path.open("wb") as stream:
                    qpy.dump(compiled, stream)
                routing = {"schedule": order, "routing_seed": seed, **physical, **validation,
                           "qpy_sha256": checksum(path),
                           "scope": "authenticated mock target" if facade else "offline IQMFakeGarnet target; not account-verified"}
                report["routing"].append(routing)
                write_json(output_dir / "comparison.json", report)
                if not facade:
                    continue
                for nshots in shots:
                    for repeat in range(repeats):
                        check_facade_target(backend)
                        started = perf_counter()
                        # Never pass a misleading simulator seed to the stock facade.
                        job = backend.run(compiled, shots=nshots, use_timeslot=False)
                        result = job.result()
                        counts = {str(k): int(v) for k, v in result.get_counts().items()}
                        if sum(counts.values()) != nshots:
                            raise RuntimeError("Unexpected shot count; no silent postselection")
                        analysis = summarise_counts(counts, qubo, problem)
                        sample = {"stage": "IQM Resonance noisy facade", "schedule": order,
                                  "routing_seed": seed, "simulator_seed": None, "repeat": repeat,
                                  "logical_depth": circuit_metrics(circuit)["depth_without_measurements"],
                                  "transpiled_depth": physical["depth_without_measurements"],
                                  "physical_cz_count": physical["operation_counts"].get("cz", 0),
                                  "elapsed_seconds": perf_counter() - started,
                                  "local_aer_job_id": job.job_id(), **analysis, "counts": counts}
                        report["samples"].append(sample)
                        rows.append(sample)
                        write_json(output_dir / "comparison.json", report)
                        export_csv(output_dir / "comparison.csv", rows)
                        print(json.dumps({k: sample[k] for k in ("stage", "schedule", "routing_seed", "shots", *METRICS)}), flush=True)
        if facade:
            report["noisy_simulation"]["status"] = "completed"
    if not facade:
        report["noisy_simulation"]["blocker"] = "No API access token configured; authenticated mock API inventory and facade execution pending"
    rows.extend([{"stage": "IQM Resonance noisy facade — pending"}] if not facade else [])
    rows.append({"stage": "Aalto Q20 — future, not run"})
    write_json(output_dir / "comparison.json", report)
    export_csv(output_dir / "comparison.csv", rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--facade", action="store_true", help="Execute only authenticated garnet:mock + facade_garnet")
    parser.add_argument("--local-compile", action="store_true", help="Compile offline for Garnet; no remote jobs or noisy samples")
    parser.add_argument("--shots", nargs="+", type=int, default=[1000, 5000, 10000])
    parser.add_argument("--routing-seeds", nargs="+", type=int, default=[42, 7, 123])
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    try:
        run_comparison(args.output_dir, facade=args.facade, local_compile=args.local_compile,
                       shots=tuple(args.shots), routing_seeds=tuple(args.routing_seeds), repeats=args.repeats)
    except Exception as exc:
        # SDK/network exceptions can contain sensitive request context: do not echo.
        print(f"Comparison stopped ({type(exc).__name__}); inspect saved nonsecret checkpoint. No physical QPU path exists.")
        return 1
    print(f"Saved comparison to {args.output_dir}. Physical QPU submissions: 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
