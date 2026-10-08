from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Mapping

from qiskit import QuantumCircuit

if TYPE_CHECKING:
    from ..problem import PaymentProblem, QuboModel
from .iqm_backend import (
    backend_summary,
    connect_iqm_backend,
    submit_and_collect,
    transpile_for_backend,
)


def build_bell_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    return circuit


def _normalise_label(label: str) -> str:
    return label.replace(" ", "")


def _label_to_values(label: str, variable_names: tuple[str, ...]) -> dict[str, int]:
    compact = _normalise_label(label)
    if len(compact) != len(variable_names):
        raise ValueError(
            f"Expected {len(variable_names)} measured bits, got {len(compact)} in {label!r}"
        )
    bits = [int(bit) for bit in compact[::-1]]
    return dict(zip(variable_names, bits, strict=True))


def analyse_cherryq_counts(
    counts: Mapping[str, int],
    qubo: QuboModel,
    problem: PaymentProblem,
) -> dict[str, object]:
    total_shots = sum(int(value) for value in counts.values())
    if total_shots < 1:
        raise ValueError("counts must contain at least one shot")

    exact_qubo_values, _ = qubo.solve_exact()
    exact_business = problem.solve_exact()
    exact_qubo_vector = tuple(exact_qubo_values[name] for name in qubo.variable_names)
    exact_business_vector = tuple(
        exact_business.decision[name] for name in problem.invoice_ids
    )

    qubo_feasible_probability = 0.0
    business_feasible_probability = 0.0
    optimal_qubo_probability = 0.0
    optimal_business_plan_probability = 0.0
    best_business_loss: int | None = None
    best_paid: tuple[str, ...] = ()

    for label, count in counts.items():
        probability = int(count) / total_shots
        values = _label_to_values(label, qubo.variable_names)
        qubo_vector = tuple(values[name] for name in qubo.variable_names)
        supplier_decision = problem.supplier_decision_from_values(
            values,
            qubo.variable_names,
        )
        supplier_vector = tuple(
            supplier_decision[name] for name in problem.invoice_ids
        )

        if abs(qubo.constraint_residual(values)) < 1e-9:
            qubo_feasible_probability += probability
        if problem.is_business_feasible(supplier_decision):
            business_feasible_probability += probability
            loss = problem.business_loss_gbp(supplier_decision)
            if best_business_loss is None or loss < best_business_loss:
                best_business_loss = loss
                best_paid = tuple(
                    invoice_id
                    for invoice_id in problem.invoice_ids
                    if supplier_decision[invoice_id]
                )
        if qubo_vector == exact_qubo_vector:
            optimal_qubo_probability += probability
        if supplier_vector == exact_business_vector:
            optimal_business_plan_probability += probability

    top_counts = sorted(
        ((str(label), int(count)) for label, count in counts.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:10]

    return {
        "shots": total_shots,
        "best_sampled_business_loss_gbp": best_business_loss,
        "best_sampled_paid_invoice_ids": best_paid,
        "qubo_feasible_probability": qubo_feasible_probability,
        "business_feasible_probability": business_feasible_probability,
        "optimal_qubo_probability": optimal_qubo_probability,
        "optimal_business_plan_probability": optimal_business_plan_probability,
        "exact_business_paid_invoice_ids": exact_business.paid_invoice_ids,
        "exact_business_loss_gbp": exact_business.business_loss_gbp,
        "top_counts": top_counts,
    }


def prepare_current_cash_qaoa(
    *,
    penalty_gbp: float = 500.0,
    p: int = 1,
    seed: int = 42,
    maxiter: int = 80,
    epsilon: float = 0.25,
    relaxation_multistart: int = 32,
    warm_start_parameter_jitter: float = 0.0,
) -> tuple[QuantumCircuit, QuboModel, PaymentProblem, dict[str, object]]:
    """Optimise CherryQ classically/statevector, then freeze one QAOA circuit for QPU use."""

    if p < 1:
        raise ValueError("p must be at least 1")
    if maxiter < 1:
        raise ValueError("maxiter must be positive")

    # Keep the QPU runtime independent of the forecasting/scikit-learn stack.
    # The current-cash scenario is exactly the base five-invoice problem.
    import numpy as np
    from qiskit.primitives import StatevectorEstimator
    from scipy.optimize import minimize

    from ..problem import build_example_problem
    from ..quantum import (
        build_warm_start_qaoa,
        qubo_to_ising,
        solve_continuous_relaxation,
    )

    problem = build_example_problem()
    qubo = problem.build_qubo(penalty_gbp=penalty_gbp)

    started = perf_counter()
    ising = qubo_to_ising(qubo)
    cost_operator = ising.normalized_operator

    relaxation_started = perf_counter()
    relaxation = solve_continuous_relaxation(
        qubo,
        multistart=relaxation_multistart,
        seed=seed,
    )
    relaxation_runtime_ms = (perf_counter() - relaxation_started) * 1000.0

    circuit, gammas, betas = build_warm_start_qaoa(
        cost_operator,
        p=p,
        warm_start_probabilities=relaxation.values,
        epsilon=epsilon,
        mixer="x",
    )
    parameter_order = gammas + betas

    rng = np.random.default_rng(seed)
    initial_params = np.concatenate([np.zeros(p), np.full(p, np.pi / 4.0)])
    if warm_start_parameter_jitter:
        initial_params = initial_params + rng.normal(
            0.0,
            warm_start_parameter_jitter,
            size=len(initial_params),
        )

    estimator = StatevectorEstimator(seed=seed)
    optimizer_calls = 0

    def cost_function(params: np.ndarray) -> float:
        nonlocal optimizer_calls
        optimizer_calls += 1
        bound = circuit.assign_parameters(
            dict(zip(parameter_order, params, strict=True))
        )
        pub_result = estimator.run([(bound, cost_operator)]).result()[0]
        return float(np.asarray(pub_result.data.evs).real.item())

    optimizer_started = perf_counter()
    result = minimize(
        cost_function,
        initial_params,
        method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": 0.5},
    )
    optimizer_runtime_ms = (perf_counter() - optimizer_started) * 1000.0

    bound = circuit.assign_parameters(
        dict(zip(parameter_order, result.x, strict=True))
    )
    bound.measure_all()

    metadata = {
        "scenario_id": "current-cash",
        "method": "warm-start-x",
        "p": p,
        "seed": seed,
        "penalty_gbp": penalty_gbp,
        "num_qubits": len(qubo.variable_names),
        "variable_names": qubo.variable_names,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_calls": optimizer_calls,
        "optimized_parameters": [float(value) for value in result.x],
        "optimized_expected_qubo_energy": float(
            result.fun * ising.scale + ising.offset
        ),
        "relaxation_runtime_ms": relaxation_runtime_ms,
        "optimizer_runtime_ms": optimizer_runtime_ms,
        "preparation_runtime_ms": (perf_counter() - started) * 1000.0,
        "relaxation_fractionality": float(
            np.mean(np.minimum(relaxation.values, 1.0 - relaxation.values))
        ),
    }
    return bound, qubo, problem, metadata


def run_bell(
    *,
    shots: int,
    optimization_level: int,
) -> dict[str, object]:
    backend = connect_iqm_backend("q20")
    circuit = build_bell_circuit()
    compiled, transpile_metrics = transpile_for_backend(
        circuit,
        backend,
        optimization_level=optimization_level,
    )
    execution = submit_and_collect(compiled, backend, shots=shots)
    return {
        "experiment": "bell",
        "device": "q20",
        "backend": backend_summary(backend),
        "transpilation": transpile_metrics,
        "execution": execution,
    }


def run_cherryq(
    *,
    shots: int,
    optimization_level: int,
    maxiter: int,
    seed: int,
    penalty_gbp: float,
    relaxation_multistart: int,
) -> dict[str, object]:
    circuit, qubo, problem, preparation = prepare_current_cash_qaoa(
        penalty_gbp=penalty_gbp,
        p=1,
        seed=seed,
        maxiter=maxiter,
        relaxation_multistart=relaxation_multistart,
    )

    backend = connect_iqm_backend("q20")
    if int(getattr(backend, "num_qubits", 0)) < circuit.num_qubits:
        raise RuntimeError(
            f"Q20 backend reports {getattr(backend, 'num_qubits', 0)} qubits, "
            f"but CherryQ requires {circuit.num_qubits}"
        )

    compiled, transpile_metrics = transpile_for_backend(
        circuit,
        backend,
        optimization_level=optimization_level,
    )
    execution = submit_and_collect(compiled, backend, shots=shots)
    analysis = analyse_cherryq_counts(execution["counts"], qubo, problem)

    return {
        "experiment": "cherryq-current-cash",
        "device": "q20",
        "backend": backend_summary(backend),
        "preparation": preparation,
        "transpilation": transpile_metrics,
        "execution": execution,
        "analysis": analysis,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run CherryQ validation circuits on Aalto Q20"
    )
    parser.add_argument(
        "--mode",
        choices=("bell", "cherryq"),
        default="bell",
        help="Run a Bell-pair sanity test or the frozen CherryQ p=1 circuit",
    )
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--penalty", type=float, default=500.0)
    parser.add_argument("--relaxation-multistart", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.mode == "bell":
        payload = run_bell(
            shots=args.shots,
            optimization_level=args.optimization_level,
        )
    else:
        payload = run_cherryq(
            shots=args.shots,
            optimization_level=args.optimization_level,
            maxiter=args.maxiter,
            seed=args.seed,
            penalty_gbp=args.penalty,
            relaxation_multistart=args.relaxation_multistart,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, default=list),
        encoding="utf-8",
    )

    execution = payload["execution"]
    print(f"Saved result to {args.output}")
    print(f"IQM job ID: {execution['job_id']}")
    if args.mode == "cherryq":
        analysis = payload["analysis"]
        print(
            "CherryQ: "
            f"P(optimal business plan)={analysis['optimal_business_plan_probability']:.2%}, "
            f"P(business feasible)={analysis['business_feasible_probability']:.2%}, "
            f"best loss=£{analysis['best_sampled_business_loss_gbp']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
