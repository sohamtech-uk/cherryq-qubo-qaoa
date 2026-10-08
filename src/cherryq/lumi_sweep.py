from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .pipeline import run_demo_pipeline
from .problem import PaymentProblem
from .quantum import run_qaoa


@dataclass(frozen=True)
class SweepTask:
    scenario_id: str
    scenario_probability: float
    method: str
    p: int
    seed: int


def build_task_matrix(
    scenario_ids: Sequence[str],
    scenario_probabilities: dict[str, float],
    methods: Sequence[str],
    p_values: Sequence[int],
    seeds: Sequence[int],
) -> tuple[SweepTask, ...]:
    tasks = []
    for scenario_id in scenario_ids:
        for method in methods:
            for p in p_values:
                for seed in seeds:
                    tasks.append(
                        SweepTask(
                            scenario_id=scenario_id,
                            scenario_probability=scenario_probabilities[scenario_id],
                            method=method,
                            p=int(p),
                            seed=int(seed),
                        )
                    )
    return tuple(tasks)


def shard_tasks(
    tasks: Sequence[SweepTask],
    shard_index: int,
    shard_count: int,
) -> tuple[SweepTask, ...]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    return tuple(task for index, task in enumerate(tasks) if index % shard_count == shard_index)


def _run_one(
    task: SweepTask,
    problem: PaymentProblem,
    penalty_gbp: float,
    maxiter: int,
    shots: int,
    epsilon: float,
    relaxation_multistart: int,
    warm_start_parameter_jitter: float,
) -> dict[str, object]:
    qubo = problem.build_qubo(penalty_gbp=penalty_gbp)
    result = run_qaoa(
        problem=problem,
        qubo=qubo,
        method=task.method,
        p=task.p,
        maxiter=maxiter,
        shots=shots,
        seed=task.seed,
        epsilon=epsilon,
        relaxation_multistart=relaxation_multistart,
        warm_start_parameter_jitter=warm_start_parameter_jitter,
    )
    return {
        **asdict(task),
        **result.as_dict(),
        "exact_business_loss_gbp": problem.solve_exact().business_loss_gbp,
    }


def run_sweep(
    output_dir: Path,
    scenario_ids: Sequence[str],
    methods: Sequence[str],
    p_values: Sequence[int],
    seeds: Sequence[int],
    penalty_gbp: float,
    maxiter: int,
    shots: int,
    workers: int,
    shard_index: int,
    shard_count: int,
    epsilon: float = 0.25,
    relaxation_multistart: int = 32,
    warm_start_parameter_jitter: float = 0.05,
) -> Path:
    pipeline = run_demo_pipeline(include_quantum=False, seed=42)
    scenario_by_id = {scenario.scenario_id: scenario for scenario in pipeline.scenarios}

    unknown = set(scenario_ids) - set(scenario_by_id)
    if unknown:
        raise ValueError(f"Unknown scenarios: {sorted(unknown)}")

    probabilities = {
        scenario_id: scenario_by_id[scenario_id].probability
        for scenario_id in scenario_ids
    }
    problems = {
        scenario_id: scenario_by_id[scenario_id].to_payment_problem(pipeline.base_problem)
        for scenario_id in scenario_ids
    }

    matrix = build_task_matrix(
        scenario_ids=scenario_ids,
        scenario_probabilities=probabilities,
        methods=methods,
        p_values=p_values,
        seeds=seeds,
    )
    shard = shard_tasks(matrix, shard_index=shard_index, shard_count=shard_count)
    if not shard:
        raise RuntimeError("This shard has no tasks")

    output_dir.mkdir(parents=True, exist_ok=True)

    def payload(task: SweepTask):
        return (
            task,
            problems[task.scenario_id],
            penalty_gbp,
            maxiter,
            shots,
            epsilon,
            relaxation_multistart,
            warm_start_parameter_jitter,
        )

    results: list[dict[str, object]] = []
    if workers == 1:
        for task in shard:
            results.append(_run_one(*payload(task)))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_one, *payload(task)) for task in shard]
            for future in futures:
                results.append(future.result())

    result_path = output_dir / f"sweep-shard-{shard_index:04d}-of-{shard_count:04d}.json"
    result_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                    "slurm_array_job_id": os.getenv("SLURM_ARRAY_JOB_ID"),
                    "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
                    "workers": workers,
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                    "task_count": len(shard),
                    "total_matrix_size": len(matrix),
                    "penalty_gbp": penalty_gbp,
                    "maxiter": maxiter,
                    "shots": shots,
                },
                "results": sorted(
                    results,
                    key=lambda item: (
                        item["scenario_id"],
                        item["method"],
                        item["p"],
                        item["seed"],
                    ),
                ),
            },
            indent=2,
            default=list,
        ),
        encoding="utf-8",
    )
    return result_path


def _csv(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _int_csv(raw: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _csv(raw))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a sharded CherryQ QAOA sweep on LUMI")
    parser.add_argument("--scenarios", default="current-cash,arrive-R1,arrive-R2,arrive-R1-R2")
    parser.add_argument("--methods", default="standard,warm-start-x,warm-start")
    parser.add_argument("--p", default="1,2")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--penalty", type=float, default=500.0)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument("--relaxation-multistart", type=int, default=32)
    parser.add_argument("--warm-start-jitter", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=max(1, int(os.getenv("SLURM_CPUS_PER_TASK", "1"))))
    parser.add_argument("--shard-index", type=int, default=int(os.getenv("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--shard-count", type=int, default=int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1")))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    seeds = tuple(range(args.seed_start, args.seed_start + args.seed_count))
    path = run_sweep(
        output_dir=args.output_dir,
        scenario_ids=_csv(args.scenarios),
        methods=_csv(args.methods),
        p_values=_int_csv(args.p),
        seeds=seeds,
        penalty_gbp=args.penalty,
        maxiter=args.maxiter,
        shots=args.shots,
        workers=args.workers,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        epsilon=args.epsilon,
        relaxation_multistart=args.relaxation_multistart,
        warm_start_parameter_jitter=args.warm_start_jitter,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
