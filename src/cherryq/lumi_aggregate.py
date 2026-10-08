from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev


def aggregate(input_dir: Path) -> dict[str, object]:
    files = sorted(input_dir.glob("sweep-shard-*.json"))
    if not files:
        raise FileNotFoundError(f"No sweep-shard JSON files found in {input_dir}")

    rows = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["results"])

    groups: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(row["scenario_id"], row["method"], int(row["p"]))].append(row)

    summary = []
    for (scenario_id, method, p), group in sorted(groups.items()):
        optimal = [float(row["optimal_business_plan_probability"]) for row in group]
        feasible = [float(row["business_feasible_probability"]) for row in group]
        runtimes = [float(row["total_runtime_ms"]) for row in group]
        exact_loss = int(group[0]["exact_business_loss_gbp"])
        gaps = [
            int(row["best_sampled_business_loss_gbp"]) - exact_loss
            for row in group
            if row["best_sampled_business_loss_gbp"] is not None
        ]
        summary.append(
            {
                "scenario_id": scenario_id,
                "method": method,
                "p": p,
                "runs": len(group),
                "mean_optimal_plan_probability": fmean(optimal),
                "std_optimal_plan_probability": pstdev(optimal) if len(optimal) > 1 else 0.0,
                "mean_business_feasible_probability": fmean(feasible),
                "std_business_feasible_probability": pstdev(feasible) if len(feasible) > 1 else 0.0,
                "sampled_optimum_rate": sum(gap == 0 for gap in gaps) / len(group),
                "mean_best_gap_gbp": fmean(gaps) if gaps else None,
                "mean_total_runtime_ms": fmean(runtimes),
                "circuit_depth": int(group[0]["circuit_depth"]),
                "two_qubit_gate_count": int(group[0]["two_qubit_gate_count"]),
            }
        )

    return {"shards": len(files), "rows": len(rows), "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate CherryQ LUMI sweep shards")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = aggregate(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Aggregated {report['rows']} runs from {report['shards']} shards")
    for item in report["summary"]:
        print(
            f"{item['scenario_id']:18s} {item['method']:13s} p={item['p']}: "
            f"runs={item['runs']}, "
            f"P(opt)={item['mean_optimal_plan_probability']:.2%}, "
            f"feasible={item['mean_business_feasible_probability']:.2%}, "
            f"runtime={item['mean_total_runtime_ms']:.1f} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
