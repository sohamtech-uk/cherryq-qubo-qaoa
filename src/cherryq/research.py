from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from statistics import fmean, pstdev
from typing import Iterable, Sequence

import numpy as np

from .pipeline import run_demo_pipeline
from .problem import PaymentProblem, QuboModel
from .quantum import QaoaRunResult, run_qaoa, solve_continuous_relaxation
from .scenarios import CashScenario


@dataclass(frozen=True)
class PenaltyDiagnostic:
    scenario_id: str
    penalty_gbp: float
    best_feasible_energy: float
    best_infeasible_energy: float
    feasibility_margin: float
    global_optimum_is_feasible: bool
    feasible_qubo_matches_business_optimum: bool
    best_feasible_paid_invoice_ids: tuple[str, ...]
    business_optimal_paid_invoice_ids: tuple[str, ...]

    @property
    def safe(self) -> bool:
        return (
            self.global_optimum_is_feasible
            and self.feasible_qubo_matches_business_optimum
            and self.feasibility_margin > 0.0
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "penalty_gbp": self.penalty_gbp,
            "best_feasible_energy": self.best_feasible_energy,
            "best_infeasible_energy": self.best_infeasible_energy,
            "feasibility_margin": self.feasibility_margin,
            "global_optimum_is_feasible": self.global_optimum_is_feasible,
            "feasible_qubo_matches_business_optimum": (
                self.feasible_qubo_matches_business_optimum
            ),
            "safe": self.safe,
            "best_feasible_paid_invoice_ids": self.best_feasible_paid_invoice_ids,
            "business_optimal_paid_invoice_ids": (
                self.business_optimal_paid_invoice_ids
            ),
        }


@dataclass(frozen=True)
class WarmStartDiagnostic:
    scenario_id: str
    penalty_gbp: float
    epsilon: float
    seed: int
    relaxation_energy: float
    relaxation_fractionality: float
    rounded_energy: float
    rounded_qubo_feasible: bool
    rounded_business_loss_gbp: int | None
    rounded_business_gap_gbp: int | None
    initial_exact_qubo_probability: float
    initial_qubo_feasible_probability: float

    def as_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "penalty_gbp": self.penalty_gbp,
            "epsilon": self.epsilon,
            "seed": self.seed,
            "relaxation_energy": self.relaxation_energy,
            "relaxation_fractionality": self.relaxation_fractionality,
            "rounded_energy": self.rounded_energy,
            "rounded_qubo_feasible": self.rounded_qubo_feasible,
            "rounded_business_loss_gbp": self.rounded_business_loss_gbp,
            "rounded_business_gap_gbp": self.rounded_business_gap_gbp,
            "initial_exact_qubo_probability": self.initial_exact_qubo_probability,
            "initial_qubo_feasible_probability": self.initial_qubo_feasible_probability,
        }


@dataclass(frozen=True)
class QaoaAggregate:
    method: str
    p: int
    runs: int
    sampled_optimum_rate: float
    optimizer_success_rate: float
    mean_optimal_plan_probability: float
    std_optimal_plan_probability: float
    mean_business_feasible_probability: float
    std_business_feasible_probability: float
    mean_best_gap_gbp: float | None
    mean_total_runtime_ms: float
    mean_preprocessing_runtime_ms: float
    mean_optimizer_runtime_ms: float
    mean_optimizer_calls: float
    circuit_depth: int
    two_qubit_gate_count: int

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class ResearchReport:
    target_scenario_id: str
    selected_penalty_gbp: float
    penalty_diagnostics: tuple[PenaltyDiagnostic, ...]
    warm_start_diagnostic: WarmStartDiagnostic
    aggregates: tuple[QaoaAggregate, ...]
    raw_runs: tuple[QaoaRunResult, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "target_scenario_id": self.target_scenario_id,
            "selected_penalty_gbp": self.selected_penalty_gbp,
            "penalty_diagnostics": [
                diagnostic.as_dict() for diagnostic in self.penalty_diagnostics
            ],
            "warm_start_diagnostic": self.warm_start_diagnostic.as_dict(),
            "aggregates": [aggregate.as_dict() for aggregate in self.aggregates],
            "raw_runs": [run.as_dict() for run in self.raw_runs],
        }


def diagnose_penalty(
    problem: PaymentProblem,
    scenario_id: str,
    penalty_gbp: float,
) -> PenaltyDiagnostic:
    qubo = problem.build_qubo(penalty_gbp=penalty_gbp)
    exact_business = problem.solve_exact()

    best_feasible_energy = float("inf")
    best_feasible_values: dict[str, int] | None = None
    best_infeasible_energy = float("inf")

    for bits in product((0, 1), repeat=len(qubo.variable_names)):
        values = dict(zip(qubo.variable_names, bits, strict=True))
        energy = qubo.energy(values)
        if abs(qubo.constraint_residual(values)) < 1e-9:
            if energy < best_feasible_energy:
                best_feasible_energy = energy
                best_feasible_values = values
        elif energy < best_infeasible_energy:
            best_infeasible_energy = energy

    if best_feasible_values is None:
        raise RuntimeError("No equality-feasible QUBO state found")

    feasible_decision = problem.supplier_decision_from_values(
        best_feasible_values,
        qubo.variable_names,
    )
    feasible_loss = problem.business_loss_gbp(feasible_decision)

    margin = best_infeasible_energy - best_feasible_energy
    global_is_feasible = best_feasible_energy < best_infeasible_energy - 1e-9

    return PenaltyDiagnostic(
        scenario_id=scenario_id,
        penalty_gbp=float(penalty_gbp),
        best_feasible_energy=float(best_feasible_energy),
        best_infeasible_energy=float(best_infeasible_energy),
        feasibility_margin=float(margin),
        global_optimum_is_feasible=bool(global_is_feasible),
        feasible_qubo_matches_business_optimum=(
            feasible_loss == exact_business.business_loss_gbp
        ),
        best_feasible_paid_invoice_ids=tuple(
            invoice_id
            for invoice_id in problem.invoice_ids
            if feasible_decision[invoice_id]
        ),
        business_optimal_paid_invoice_ids=exact_business.paid_invoice_ids,
    )


def diagnose_penalty_grid(
    base_problem: PaymentProblem,
    scenarios: Sequence[CashScenario],
    penalties_gbp: Sequence[float],
) -> tuple[PenaltyDiagnostic, ...]:
    diagnostics: list[PenaltyDiagnostic] = []
    for scenario in scenarios:
        problem = scenario.to_payment_problem(base_problem)
        for penalty in penalties_gbp:
            diagnostics.append(
                diagnose_penalty(problem, scenario.scenario_id, penalty)
            )
    return tuple(diagnostics)


def choose_smallest_safe_penalty(
    diagnostics: Sequence[PenaltyDiagnostic],
    scenario_ids: Iterable[str],
) -> float:
    scenario_id_set = set(scenario_ids)
    penalties = sorted({item.penalty_gbp for item in diagnostics})
    for penalty in penalties:
        by_scenario = {
            item.scenario_id: item
            for item in diagnostics
            if item.penalty_gbp == penalty
            and item.scenario_id in scenario_id_set
        }
        if scenario_id_set == set(by_scenario) and all(
            item.safe for item in by_scenario.values()
        ):
            return float(penalty)
    raise RuntimeError(
        "No penalty in the supplied grid strictly separates feasible and infeasible "
        "QUBO optima for every scenario"
    )


def warm_start_diagnostic(
    problem: PaymentProblem,
    scenario_id: str,
    penalty_gbp: float,
    epsilon: float = 0.25,
    seed: int = 42,
    relaxation_multistart: int = 64,
) -> WarmStartDiagnostic:
    qubo = problem.build_qubo(penalty_gbp=penalty_gbp)
    relaxation = solve_continuous_relaxation(
        qubo,
        multistart=relaxation_multistart,
        seed=seed,
    )
    clipped = np.clip(relaxation.values, epsilon, 1.0 - epsilon)

    exact_qubo_values, _ = qubo.solve_exact()
    exact_qubo_probability = 1.0
    for name, probability in zip(qubo.variable_names, clipped, strict=True):
        exact_qubo_probability *= (
            probability if exact_qubo_values[name] else 1.0 - probability
        )

    feasible_probability = 0.0
    for bits in product((0, 1), repeat=len(qubo.variable_names)):
        values = dict(zip(qubo.variable_names, bits, strict=True))
        if abs(qubo.constraint_residual(values)) >= 1e-9:
            continue
        probability = 1.0
        for bit, p in zip(bits, clipped, strict=True):
            probability *= p if bit else 1.0 - p
        feasible_probability += probability

    rounded_mapping = dict(
        zip(qubo.variable_names, relaxation.rounded_values.tolist(), strict=True)
    )
    rounded_qubo_feasible = abs(qubo.constraint_residual(rounded_mapping)) < 1e-9
    rounded_supplier = problem.supplier_decision_from_values(
        rounded_mapping,
        qubo.variable_names,
    )
    rounded_business_loss: int | None = None
    rounded_business_gap: int | None = None
    if problem.is_business_feasible(rounded_supplier):
        rounded_business_loss = problem.business_loss_gbp(rounded_supplier)
        rounded_business_gap = (
            rounded_business_loss - problem.solve_exact().business_loss_gbp
        )

    fractionality = float(
        np.mean(np.minimum(relaxation.values, 1.0 - relaxation.values))
    )

    return WarmStartDiagnostic(
        scenario_id=scenario_id,
        penalty_gbp=float(penalty_gbp),
        epsilon=float(epsilon),
        seed=seed,
        relaxation_energy=float(relaxation.energy),
        relaxation_fractionality=fractionality,
        rounded_energy=float(relaxation.rounded_energy),
        rounded_qubo_feasible=rounded_qubo_feasible,
        rounded_business_loss_gbp=rounded_business_loss,
        rounded_business_gap_gbp=rounded_business_gap,
        initial_exact_qubo_probability=float(exact_qubo_probability),
        initial_qubo_feasible_probability=float(feasible_probability),
    )


def aggregate_runs(
    runs: Sequence[QaoaRunResult],
    exact_business_loss_gbp: int,
) -> tuple[QaoaAggregate, ...]:
    groups: dict[tuple[str, int], list[QaoaRunResult]] = {}
    for run in runs:
        groups.setdefault((run.method, run.p), []).append(run)

    aggregates: list[QaoaAggregate] = []
    for (method, p), group in sorted(groups.items()):
        optimal_probs = [run.optimal_business_plan_probability for run in group]
        feasible_probs = [run.business_feasible_probability for run in group]
        gaps = [
            run.best_sampled_business_loss_gbp - exact_business_loss_gbp
            for run in group
            if run.best_sampled_business_loss_gbp is not None
        ]
        sampled_optimum_count = sum(gap == 0 for gap in gaps)
        aggregates.append(
            QaoaAggregate(
                method=method,
                p=p,
                runs=len(group),
                sampled_optimum_rate=sampled_optimum_count / len(group),
                optimizer_success_rate=(
                    sum(run.optimizer_success for run in group) / len(group)
                ),
                mean_optimal_plan_probability=float(fmean(optimal_probs)),
                std_optimal_plan_probability=float(
                    pstdev(optimal_probs) if len(optimal_probs) > 1 else 0.0
                ),
                mean_business_feasible_probability=float(fmean(feasible_probs)),
                std_business_feasible_probability=float(
                    pstdev(feasible_probs) if len(feasible_probs) > 1 else 0.0
                ),
                mean_best_gap_gbp=(
                    None if not gaps else float(fmean(gaps))
                ),
                mean_total_runtime_ms=float(
                    fmean(run.total_runtime_ms for run in group)
                ),
                mean_preprocessing_runtime_ms=float(
                    fmean(run.classical_preprocessing_runtime_ms for run in group)
                ),
                mean_optimizer_runtime_ms=float(
                    fmean(run.optimizer_runtime_ms for run in group)
                ),
                mean_optimizer_calls=float(
                    fmean(run.optimizer_calls for run in group)
                ),
                circuit_depth=group[0].circuit_depth,
                two_qubit_gate_count=group[0].two_qubit_gate_count,
            )
        )
    return tuple(aggregates)


def run_repeated_qaoa(
    problem: PaymentProblem,
    penalty_gbp: float,
    methods: Sequence[str],
    p_values: Sequence[int],
    seeds: Sequence[int],
    maxiter: int,
    shots: int,
    epsilon: float,
    relaxation_multistart: int,
    warm_start_parameter_jitter: float,
) -> tuple[QaoaRunResult, ...]:
    qubo = problem.build_qubo(penalty_gbp=penalty_gbp)
    runs: list[QaoaRunResult] = []
    for p in p_values:
        for method in methods:
            for seed in seeds:
                runs.append(
                    run_qaoa(
                        problem=problem,
                        qubo=qubo,
                        method=method,
                        p=p,
                        maxiter=maxiter,
                        shots=shots,
                        seed=seed,
                        epsilon=epsilon,
                        relaxation_multistart=relaxation_multistart,
                        warm_start_parameter_jitter=warm_start_parameter_jitter,
                    )
                )
    return tuple(runs)


def run_research(
    scenario_id: str = "current-cash",
    penalties_gbp: Sequence[float] = (100, 250, 500, 1000, 2000, 4000),
    methods: Sequence[str] = ("standard", "warm-start-x", "warm-start"),
    p_values: Sequence[int] = (1, 2),
    seeds: Sequence[int] = (11, 29, 47, 71, 97),
    maxiter: int = 80,
    shots: int = 2048,
    epsilon: float = 0.25,
    relaxation_multistart: int = 32,
    warm_start_parameter_jitter: float = 0.05,
) -> ResearchReport:
    pipeline = run_demo_pipeline(include_quantum=False, seed=42)
    scenario_by_id = {scenario.scenario_id: scenario for scenario in pipeline.scenarios}
    if scenario_id not in scenario_by_id:
        raise ValueError(
            f"Unknown scenario {scenario_id!r}; choose from {sorted(scenario_by_id)}"
        )

    diagnostics = diagnose_penalty_grid(
        pipeline.base_problem,
        pipeline.scenarios,
        penalties_gbp,
    )
    selected_penalty = choose_smallest_safe_penalty(
        diagnostics,
        scenario_by_id.keys(),
    )

    scenario = scenario_by_id[scenario_id]
    problem = scenario.to_payment_problem(pipeline.base_problem)
    warm_diag = warm_start_diagnostic(
        problem=problem,
        scenario_id=scenario_id,
        penalty_gbp=selected_penalty,
        epsilon=epsilon,
        seed=42,
        relaxation_multistart=relaxation_multistart,
    )
    runs = run_repeated_qaoa(
        problem=problem,
        penalty_gbp=selected_penalty,
        methods=methods,
        p_values=p_values,
        seeds=seeds,
        maxiter=maxiter,
        shots=shots,
        epsilon=epsilon,
        relaxation_multistart=relaxation_multistart,
        warm_start_parameter_jitter=warm_start_parameter_jitter,
    )
    aggregates = aggregate_runs(
        runs,
        exact_business_loss_gbp=problem.solve_exact().business_loss_gbp,
    )

    return ResearchReport(
        target_scenario_id=scenario_id,
        selected_penalty_gbp=selected_penalty,
        penalty_diagnostics=diagnostics,
        warm_start_diagnostic=warm_diag,
        aggregates=aggregates,
        raw_runs=runs,
    )


def _parse_csv_numbers(raw: str, converter):
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value")
    return tuple(converter(value) for value in values)


def _print_report(report: ResearchReport) -> None:
    print("CherryQ QAOA research")
    print("====================")
    print(f"Target scenario: {report.target_scenario_id}")
    print(f"Selected safe penalty from grid: £{report.selected_penalty_gbp:,.0f}")

    print("\nPenalty safety by scenario")
    print("--------------------------")
    for item in report.penalty_diagnostics:
        status = "SAFE" if item.safe else "UNSAFE/TIE"
        print(
            f"{item.scenario_id:18s} P={item.penalty_gbp:7.0f} "
            f"margin={item.feasibility_margin:8.1f} {status}"
        )

    warm = report.warm_start_diagnostic
    print("\nWarm-start diagnostic")
    print("---------------------")
    print(
        f"epsilon={warm.epsilon:.2f}; fractionality={warm.relaxation_fractionality:.4f}; "
        f"rounded business gap={warm.rounded_business_gap_gbp}; "
        f"initial exact-QUBO probability={warm.initial_exact_qubo_probability:.2%}; "
        f"initial QUBO-feasible probability={warm.initial_qubo_feasible_probability:.2%}"
    )

    print("\nRepeated-seed QAOA")
    print("------------------")
    for item in report.aggregates:
        gap = (
            "n/a"
            if item.mean_best_gap_gbp is None
            else f"£{item.mean_best_gap_gbp:.1f}"
        )
        print(
            f"{item.method:13s} p={item.p}: "
            f"optimum sampled={item.sampled_optimum_rate:.0%}, "
            f"mean P(optimal plan)={item.mean_optimal_plan_probability:.1%} "
            f"+/- {item.std_optimal_plan_probability:.1%}, "
            f"mean feasible={item.mean_business_feasible_probability:.1%}, "
            f"mean best gap={gap}, "
            f"runtime={item.mean_total_runtime_ms:.1f} ms, "
            f"depth={item.circuit_depth}, 2q={item.two_qubit_gate_count}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run CherryQ penalty, mixer and repeated-seed QAOA experiments"
    )
    parser.add_argument("--scenario", default="current-cash")
    parser.add_argument(
        "--penalties",
        default="100,250,500,1000,2000,4000",
        help="Comma-separated QUBO penalty values in GBP-equivalent score units",
    )
    parser.add_argument(
        "--methods",
        default="standard,warm-start-x,warm-start",
        help="Comma-separated methods",
    )
    parser.add_argument("--p", default="1,2", help="Comma-separated QAOA depths")
    parser.add_argument(
        "--seeds",
        default="11,29,47,71,97",
        help="Comma-separated repeated-run seeds",
    )
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument("--relaxation-multistart", type=int, default=32)
    parser.add_argument("--warm-start-jitter", type=float, default=0.05)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for a machine-readable experiment report",
    )
    args = parser.parse_args()

    report = run_research(
        scenario_id=args.scenario,
        penalties_gbp=_parse_csv_numbers(args.penalties, float),
        methods=tuple(
            item.strip() for item in args.methods.split(",") if item.strip()
        ),
        p_values=_parse_csv_numbers(args.p, int),
        seeds=_parse_csv_numbers(args.seeds, int),
        maxiter=args.maxiter,
        shots=args.shots,
        epsilon=args.epsilon,
        relaxation_multistart=args.relaxation_multistart,
        warm_start_parameter_jitter=args.warm_start_jitter,
    )
    _print_report(report)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report.as_dict(), indent=2, default=list),
            encoding="utf-8",
        )
        print(f"\nWrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
