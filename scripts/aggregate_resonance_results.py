"""Validate and combine the six authenticated mock-facade result shards."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from cherryq.hardware.resonance_simulation import (
    GAMMA, BETA, METRICS, export_csv, frozen_circuits, summarise_counts, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.input.glob("resonance-*-route-*/comparison.json"))
    if len(paths) != 6:
        raise RuntimeError("Expected all six schedule/routing result shards")
    reports = [json.loads(p.read_text()) for p in paths]
    problem, qubo, _, _ = frozen_circuits()
    result = reports[0].copy()
    result["samples"] = [s for s in result["samples"] if s["stage"] == "Ideal statevector sampling"]
    result["routing"] = []
    result["source_shards"] = []
    result["requested_noisy_grid"] = {"schedules": ["original", "round-robin"],
        "routing_seeds": [42, 7, 123], "shots": [1000, 5000, 10000], "repeats": 1}
    seen = set()
    for path, report in zip(paths, reports, strict=True):
        assert (report["gamma"], report["beta"], report["p"]) == (GAMMA, BETA, 1)
        assert report["logical_qubits"] == 14
        assert not report["physical_qpu_submitted"]
        assert report["noisy_simulation"]["status"] == "completed"
        assert report["ground_truth"]["solutions_agree"]
        assert report["ground_truth"]["exact_business_loss_gbp"] == 950
        assert report["ground_truth"]["exact_spend_gbp"] == 5000
        assert report["ground_truth"]["milp_gap"] == 0
        inventory = json.loads((path.parent / "mock-inventory.json").read_text())
        architecture = json.loads((path.parent / "mock-architecture.json").read_text())
        assert inventory["garnet_mock_available"] and inventory["mock_num_qubits"] == 20
        assert inventory["facade_static_architecture_compatibility_verified"]
        assert architecture["metadata_adapter"]["noise_profile_modified"] is False
        result["source_shards"].append({"artifact": path.parent.name,
            "created_at_utc": report["created_at_utc"], "versions": report["versions"],
            "source_sha256": report["source_sha256"], "mock_inventory": inventory,
            "architecture_audit": architecture})
        for order in ("original", "round-robin"):
            for metric in METRICS:
                assert abs(report["ideal"][order][metric] - result["ideal"][order][metric]) < 1e-12
        for route in report["routing"]:
            assert route["target_operations_valid"] and route["measurement_mapping_valid"]
            assert route["max_amplitude_error_up_to_global_phase"] < 1e-10
            assert route["two_qubit_gates"] == route["operation_counts"]["cz"]
            result["routing"].append(route)
        for sample in report["samples"]:
            if sample["stage"] != "IQM Resonance noisy facade":
                continue
            key = (sample["schedule"], sample["routing_seed"], sample["shots"])
            assert key not in seen
            seen.add(key)
            assert sum(sample["counts"].values()) == sample["shots"]
            verified = summarise_counts(sample["counts"], qubo, problem)
            for metric in (*METRICS, "best_sampled_business_loss_gbp"):
                assert abs(verified[metric] - sample[metric]) < 1e-12
            result["samples"].append(sample)
    expected = {(o, s, n) for o in ("original", "round-robin")
                for s in (42, 7, 123) for n in (1000, 5000, 10000)}
    assert seen == expected
    groups = defaultdict(list)
    for sample in result["samples"]:
        if sample["stage"] == "IQM Resonance noisy facade":
            groups[(sample["schedule"], sample["shots"])].append(sample)
    aggregates = []
    for (order, nshots), samples in sorted(groups.items()):
        counts = Counter()
        for sample in samples:
            counts.update(sample["counts"])
        metrics = summarise_counts(dict(counts), qubo, problem)
        aggregates.append({"schedule": order, "shots_per_routing_seed": nshots,
            "total_shots": metrics["shots"], "routing_seeds": sorted(s["routing_seed"] for s in samples),
            **{m: metrics[m] for m in (*METRICS, "best_sampled_business_loss_gbp")}})
    result["aggregate_by_schedule_and_shots"] = aggregates
    result["aggregation_note"] = "Descriptive pooling across three fixed routing cases; per-run Wilson intervals remain in samples. No pooled iid-binomial interval is claimed."
    result["source_workflow_run"] = "https://github.com/sohamtech-uk/cherryq-qubo-qaoa/actions/runs/37916961742"
    result["noisy_total_shots"] = sum(s["shots"] for s in result["samples"] if s["stage"] == "IQM Resonance noisy facade")
    assert result["noisy_total_shots"] == 96000
    rows = [{"stage": "Exact/MILP", "best_sampled_business_loss_gbp": 950}]
    for order, ideal in result["ideal"].items():
        rows.append({"stage": "Ideal statevector", "schedule": order,
                     "logical_depth": ideal["depth_without_measurements"], **{m: ideal[m] for m in METRICS}})
    rows += result["samples"]
    rows.append({"stage": "Aalto Q20 — future, not run"})
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "CherryQ_Resonance_Comparison.json", result)
    export_csv(args.output / "CherryQ_Resonance_Comparison.csv", rows)
    with (args.output / "CherryQ_Resonance_Aggregates.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    def probabilities(row):
        return " | ".join(f"{100 * row[m]:.4f}%" for m in METRICS)
    text = ["# CherryQ: frozen QAOA comparison", "",
        "Exact/MILP → Ideal QAOA → IQM Resonance noisy simulation → Aalto Q20 (future)", "",
        "Exact enumeration and MILP agree: **A+B+F+G+H**, **£5,000 spend**, **£950 minimum business loss**; MILP gap zero.",
        "Gamma = -0.7439674535854706; beta = 0.3004519607226279; p=1; 14 logical qubits. No parameter reoptimization.", "",
        "## Main comparison", "",
        "Noisy rows pool three routing seeds at 10,000 shots each (30,000 per schedule). These are descriptive averages across fixed routing cases.", "",
        "| Stage / schedule | P(A+B+F+G+H) | Business feasible | QUBO feasible | Optimal QUBO state |",
        "|---|---:|---:|---:|---:|",
        "| Exact/MILP | Deterministic optimum | Yes | Exact encoding | Exact optimum |"]
    for order in ("original", "round-robin"):
        text.append(f"| Ideal / {order} | {probabilities(result['ideal'][order])} |")
    for row in aggregates:
        if row["shots_per_routing_seed"] == 10000:
            text.append(f"| IQM noisy / {row['schedule']} | {probabilities(row)} |")
    text += ["| Aalto Q20 | Future; not run | — | — | — |", "",
        "## All noisy samples", "",
        "| Schedule | Routing seed | Shots | P(optimal invoices) | Business feasible | QUBO feasible | Optimal QUBO | Best loss | Depth | CZ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    noisy = sorted((s for s in result["samples"] if s["stage"] == "IQM Resonance noisy facade"),
                   key=lambda s: (s["shots"], s["routing_seed"], s["schedule"]))
    for sample in noisy:
        text.append(f"| {sample['schedule']} | {sample['routing_seed']} | {sample['shots']:,} | "
                    f"{probabilities(sample)} | £{sample['best_sampled_business_loss_gbp']:,.0f} | "
                    f"{sample['transpiled_depth']} | {sample['physical_cz_count']} |")
    text += ["", "## Routing and schedule interpretation", "",
        "Logical depth falls from 78 to 42; both logical circuits have 182 CX gates and the same ideal state. Physical depths exclude final measurements.", "",
        "| Routing seed | Original depth | Reordered depth | Original CZ | Reordered CZ |",
        "|---:|---:|---:|---:|---:|"]
    for seed in (42, 7, 123):
        routes = {r["schedule"]: r for r in result["routing"] if r["routing_seed"] == seed}
        a, b = routes["original"], routes["round-robin"]
        text.append(f"| {seed} | {a['depth_without_measurements']} | {b['depth_without_measurements']} | "
                    f"{a['two_qubit_gates']} | {b['two_qubit_gates']} |")
    pairs = {r["schedule"]: r for r in aggregates if r["shots_per_routing_seed"] == 10000}
    delta = 100 * (pairs["round-robin"][METRICS[0]] - pairs["original"][METRICS[0]])
    text += ["", f"At 10,000 shots per routing case, the observed change in optimal-invoice probability is **{delta:+.4f} percentage points** (reordered minus original).",
        "This is a comparison of complete routed circuits. Changes in CZ count, placement and gate order accompany the depth change. The stock IQM model includes gate-duration relaxation/dephasing, depolarization and readout error, but no automatic idle-time scheduling noise; the experiment does not isolate a causal depth benefit.", "",
        "## Model, account and reproducibility", "",
        "Authenticated inventory: emerald:mock, garnet:mock and sirius:mock. The runner uses only garnet:mock with facade_garnet; IQM's remote mock results are discarded and replaced by its local Aer noisy results.",
        "The stock facade rejected a DUT-label difference: model M153_W0_P06_Z99 versus mock M194_W0_P08_Z99. All 20 qubit names, the empty resonator list and all 30 connections matched exactly. A checked label adapter used IQM's IQMFakeBackend constructor with the unchanged official Garnet error profile. The stock IQMFacadeBackend execution and compatibility check remained active; topology changes are rejected. Both architecture snapshots and the adapter audit are included.",
        "**Representative IQM Garnet noisy simulation, not an exact Aalto Q20 calibration model.** Current mock-device calibration equivalence is also not claimed.",
        "18 noisy runs, 96,000 total shots. Routing seeds: 42, 7, 123. IQM Client 34.0.2 forwards shots but not seed_simulator, so noisy simulator seeds are unspecified. Ideal sampling uses seeds 42, 7, 123. No misleading seeded-noise claim is made.",
        "The warm-start relaxation already encodes the classical optimum, with epsilon 0.25. Exact/MILP remains ground truth. This experiment makes no quantum-advantage claim.",
        "Measurement interpretation is unchanged: rightmost Qiskit bit is A, followed by B–J, then four slack bits. Optimal-invoice probability marginalizes over slack; business feasibility ignores slack; QUBO feasibility requires the budget equality including slack. The optimal QUBO probability refers to the unique full 14-bit optimum.",
        "All 18 raw count dictionaries were rescored using the existing repository interpreter. Every shot total, metric and optimum was checked; all six routed circuits passed native-target, final-measurement and ideal-amplitude checks. Per-run Wilson 95% intervals are in JSON and describe shot uncertainty only, not calibration/model uncertainty.",
        "No physical QPU job was submitted. Real Garnet and Aalto Q20 remain unrun.", "",
        "## Files and provenance", "",
        "- CherryQ_Resonance_Comparison.json: combined raw counts, metrics, uncertainty, circuit validation, source hashes, runtime versions and account/model audit.",
        "- CherryQ_Resonance_Comparison.csv: exact, ideal, all sampled runs and future-hardware status.",
        "- CherryQ_Resonance_Aggregates.csv: descriptive pooling by schedule and shot budget.",
        "- CherryQ_Resonance_Raw_Results.zip: original Actions artifacts, including QPY circuits and the exact built-in noise profile.", "",
        f"[Source Actions run]({result['source_workflow_run']}) — simulator commit `beeff437c04c3c2329c42b11fbb505eeda829fd8`.",
        "[IQM documented facade workflow](https://docs.iqm.tech/iqm-client/user_guide_qiskit.html#running-a-quantum-circuit-on-a-facade-backend)", ""]
    (args.output / "CherryQ_Resonance_Report.md").write_text("\n".join(text))
    print(json.dumps({"noisy_runs": len(seen), "noisy_total_shots": result["noisy_total_shots"],
                      "aggregates": aggregates, "routing": result["routing"]}, indent=2))


if __name__ == "__main__":
    main()
