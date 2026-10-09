"""Generate the saved hardware comparison from nonsecret, verified result files."""
import argparse
import csv
import json
from pathlib import Path

from cherryq.hardware.resonance_simulation import METRICS, summarise_counts, frozen_circuits
from cherryq.hardware.q20 import _label_to_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    hardware = json.loads(args.hardware.read_text())
    prior = json.loads(args.reference.read_text())
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    rows = [{"stage": "Exact/MILP", "schedule": "deterministic", "shots": None,
             "best_sampled_business_loss_gbp": 950}]
    for schedule in ("original", "round-robin"):
        rows.append({"stage": "Ideal QAOA (exact statevector)", "schedule": schedule,
                     "shots": None, **{k: prior["ideal"][schedule][k] for k in METRICS},
                     "best_sampled_business_loss_gbp": None})
    for aggregate in prior["aggregate_by_schedule_and_shots"]:
        if aggregate["shots_per_routing_seed"] == 10000:
            rows.append({"stage": "garnet:mock noisy facade (pooled)",
                         "schedule": aggregate["schedule"], "shots": aggregate["total_shots"],
                         **{k: aggregate[k] for k in METRICS},
                         "best_sampled_business_loss_gbp": aggregate["best_sampled_business_loss_gbp"]})
    complete = hardware.get("result_success", False)
    count_rows = []
    if complete:
        problem, qubo, _, _ = frozen_circuits()
        for schedule, record in hardware["circuits"].items():
            counts = record["counts"]
            summary = summarise_counts(counts, qubo, problem)
            assert summary["shots"] == 1000
            for k in METRICS:
                assert abs(summary[k] - record["hardware_metrics"][k]) < 1e-12
                assert abs(record["ideal_metrics"][k] - prior["ideal"][schedule][k]) < 1e-12
            assert summary["best_sampled_business_loss_gbp"] == record["hardware_metrics"]["best_sampled_business_loss_gbp"]
            rows.append({"stage": "Real IQM Resonance Garnet", "schedule": schedule, "shots": 1000,
                         **{k: summary[k] for k in METRICS},
                         "optimal_plan_shots": round(summary[METRICS[0]] * 1000),
                         "best_sampled_business_loss_gbp": summary["best_sampled_business_loss_gbp"],
                         "physical_depth": record["physical_depth"], "cz_count": record["cz_count"],
                         "single_qubit_gate_count": record["single_qubit_gate_count"],
                         "iqm_job_id": hardware["iqm_job_id"]})
            for label, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])):
                values = _label_to_values(label, qubo.variable_names)
                decision = problem.supplier_decision_from_values(values, qubo.variable_names)
                feasible = problem.is_business_feasible(decision)
                paid = [k for k, v in decision.items() if v]
                count_rows.append({"schedule": schedule, "bitstring": label, "count": count,
                                   "probability": count / 1000, "paid_invoice_ids": "+".join(paid),
                                   "business_feasible": feasible,
                                   "business_loss_gbp": problem.business_loss_gbp(decision) if feasible else None,
                                   "qubo_feasible": abs(qubo.constraint_residual(values)) < 1e-9})
            hardware["circuits"][schedule]["top_counts_decoded"] = [r for r in count_rows if r["schedule"] == schedule][:10]
    combined = {"hardware_experiment": hardware, "reference_comparison": prior,
                "comparison_rows": rows, "quantum_advantage_demonstrated": False,
                "second_hardware_job_submitted": False,
                "independent_count_rescoring_passed": complete}
    (out / "CherryQ_Real_Garnet_Comparison.json").write_text(json.dumps(combined, indent=2) + "\n")
    fields = ["stage", "schedule", "shots", *METRICS, "optimal_plan_shots",
              "best_sampled_business_loss_gbp", "physical_depth", "cz_count",
              "single_qubit_gate_count", "iqm_job_id"]
    with (out / "CherryQ_Real_Garnet_Comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if count_rows:
        with (out / "CherryQ_Real_Garnet_Counts.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(count_rows[0]))
            writer.writeheader()
            writer.writerows(count_rows)
    lines = ["# CherryQ: real IQM Resonance Garnet comparison", "",
             "Exact/MILP → Ideal QAOA → garnet:mock noisy facade → real Garnet", "",
             "Ground truth: **A+B+F+G+H**, £5,000 spend, **£950 minimum business loss**. Exact enumeration and MILP agree.",
             f"Frozen parameters: gamma = {hardware['gamma']}, beta = {hardware['beta']}; p=1; 14 logical qubits.", ""]
    if complete:
        lines += [f"**One real hardware batch completed, with two circuits and 1,000 shots per circuit.** IQM job ID: `{hardware['iqm_job_id']}`.",
                  "This used an ordinary IQMBackend on real Resonance `garnet`. No facade was attached to hardware. No Aalto Q20/LUMI job or second hardware job was submitted.", ""]
    else:
        lines += [f"**Hardware outcome: {hardware.get('preflight')}; results not available.**",
                  f"Submission attempted: {hardware.get('submission_attempted')}; IQM job ID: {hardware.get('iqm_job_id') or 'none'}.",
                  f"Failure stage: {hardware.get('failure', {}).get('stage', 'none')}.", ""]
    lines += ["## Comparison", "",
              "| Stage / schedule | Shots | P(optimal invoices) | Business feasible | QUBO feasible | Optimal QUBO state | Best sampled loss |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["stage"] == "Exact/MILP":
            lines.append("| Exact/MILP | Deterministic | Returns optimum | Yes | Exact encoding | Exact optimum | £950 |")
            continue
        probs = " | ".join(f"{row[k]:.4%}" for k in METRICS)
        loss = row.get("best_sampled_business_loss_gbp")
        lines.append(f"| {row['stage']} / {row['schedule']} | {row['shots'] or 'Exact probabilities'} | {probs} | {'£'+str(loss) if loss is not None else '—'} |")
    lines += ["", "The noisy reference pools three routing seeds (7, 42, 123), 10,000 shots each: 30,000 shots per schedule. Hardware uses routing seed 42 and the current hardware calibration. The representative facade noise model is not an exact model of this real calibration.", ""]
    if hardware.get("backend"):
        b = hardware["backend"]
        lines += ["## Real-backend preflight", "",
                  f"Backend name: `{b['name']}`; resolved alias: `{b['resolved_alias']}`; usable qubits: {b['usable_qubit_count']}. Calibration-set ID: `{b['calibration_set_id']}`.",
                  f"Preflight status: **{hardware['preflight']}**. Checks cover current native target support, statevector equivalence up to global phase, final measurement mapping, serialized measurement keys, SDK bit-order round trip, and the no-submit `create_run_request` for both circuits.", "",
                  "| Schedule | Depth, excluding measurement | Depth, including measurement | CZ | Single-qubit gates | Final logical→physical indices |",
                  "|---|---:|---:|---:|---:|---|"]
        for order, r in hardware["circuits"].items():
            lines.append(f"| {order} | {r['physical_depth']} | {r['physical_depth_with_measurements']} | {r['cz_count']} | {r['single_qubit_gate_count']} | `{r['final_logical_to_physical']}` |")
        lines += ["", "Mappings are zero-based physical indices, in logical order A, B, C, D, E, F, G, H, I, J, s0, s1, s2, s3. Named physical qubits and amplitude errors are preserved in JSON.", ""]
    if complete:
        lines += ["## Hardware outcomes and shot uncertainty", ""]
        for order, r in hardware["circuits"].items():
            m = r["hardware_metrics"]
            lo, hi = m["binomial_wilson_95_intervals"][METRICS[0]]
            lines.append(f"- **{order}: {m['optimal_plan_shots']}/1,000 optimal-plan shots**, P(optimum) {m[METRICS[0]]:.2%}; 95% Wilson interval {lo:.2%}–{hi:.2%}. Best feasible sampled loss: £{m['best_sampled_business_loss_gbp']}.")
        lines += ["", "The intervals cover binomial shot uncertainty only. A single batch does not establish a reliable schedule improvement or isolate the effect of depth; placement, routing, CZ count and calibration also matter.", ""]
        for order, r in hardware["circuits"].items():
            lines += [f"### Top counts: {order}", "",
                      "| Qiskit bitstring | Count | Paid invoices | Business feasible | Business loss |",
                      "|---|---:|---|---|---:|"]
            for row in r["top_counts_decoded"]:
                loss = row["business_loss_gbp"]
                lines.append(f"| `{row['bitstring']}` | {row['count']} | {row['paid_invoice_ids'] or 'none'} | {'Yes' if row['business_feasible'] else 'No'} | {'£'+str(loss) if loss is not None else '—'} |")
            lines.append("")
    lines += ["## Interpretation", "",
              "The rightmost count bit is A, followed by B–J and four slack bits. Optimal-invoice probability sums over all slack assignments. Business feasibility means invoice spend ≤ £5,000 and ignores slack. QUBO feasibility requires the budget equality including slack. Optimal QUBO probability counts the unique full 14-bit optimum `00000011100011`.", "",
              "**This is a real IQM hardware experiment, but it does not demonstrate quantum advantage.** The warm-start relaxation already encodes the classical optimum before epsilon-0.25 clipping; exact/MILP is the ground truth. No runtime or scaling advantage over classical methods has been established.", "",
              f"[Hardware workflow](https://github.com/sohamtech-uk/cherryq-qubo-qaoa/actions/runs/{hardware.get('github_run_id')}) · source commit `{hardware.get('source_commit')}`.",
              f"[Completed simulator workflow]({prior['source_workflow_run']}).", "",
              "[IQM backend documentation](https://docs.iqm.tech/iqm-client/api/iqm.qiskit_iqm.iqm_provider.IQMBackend.html) · [IQM provider documentation](https://docs.iqm.tech/iqm-client/api/iqm.qiskit_iqm.iqm_provider.IQMProvider.html).", "",
              "JSON preserves the raw hardware counts, provenance, preflight metadata, metrics, intervals and reference aggregates. The comparison CSV stores probabilities as fractions; the counts CSV contains every observed hardware bitstring. No IQM token or full run request is included."]
    if not complete:
        lines = [line.replace("**This is a real IQM hardware experiment, but it does not demonstrate quantum advantage.**", "**No completed real-hardware result is claimed; no quantum advantage is demonstrated.**") for line in lines]
    (out / "CherryQ_Real_Garnet_Report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"complete": complete, "rows": len(rows), "count_rows": len(count_rows),
                      "files": [p.name for p in out.iterdir()]}))


if __name__ == "__main__":
    main()
