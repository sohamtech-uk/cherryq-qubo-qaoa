from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from qiskit import qpy
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize

from ..problem import PaymentProblem, QuboModel, build_ten_invoice_problem
from ..quantum import build_warm_start_qaoa, qubo_to_ising, solve_continuous_relaxation
from .iqm_backend import backend_summary, connect_iqm_backend, submit_and_collect, transpile_for_backend
from .q20 import analyse_cherryq_counts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_ten_invoice_qaoa(
    *,
    penalty_gbp: float = 250.0,
    p: int = 1,
    seed: int = 42,
    maxiter: int = 80,
    epsilon: float = 0.25,
    relaxation_multistart: int = 32,
) -> tuple[Any, QuboModel, PaymentProblem, dict[str, object]]:
    """Optimise the 10-invoice warm-start-X circuit and freeze its parameters."""

    problem = build_ten_invoice_problem()
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
    initial_params = np.concatenate([np.zeros(p), np.full(p, np.pi / 4.0)])

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
        "problem": "10-invoice",
        "method": "warm-start-x",
        "p": p,
        "seed": seed,
        "penalty_gbp": penalty_gbp,
        "num_qubits": len(qubo.variable_names),
        "variable_names": qubo.variable_names,
        "business_optimum": problem.solve_exact().paid_invoice_ids,
        "business_optimum_loss_gbp": problem.solve_exact().business_loss_gbp,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_calls": optimizer_calls,
        "optimized_parameters": [float(value) for value in result.x],
        "optimized_expected_qubo_energy": float(
            result.fun * ising.scale + ising.offset
        ),
        "logical_depth": int(bound.depth()),
        "logical_operation_counts": {
            str(name): int(count) for name, count in bound.count_ops().items()
        },
        "relaxation_runtime_ms": relaxation_runtime_ms,
        "optimizer_runtime_ms": optimizer_runtime_ms,
        "preparation_runtime_ms": (perf_counter() - started) * 1000.0,
        "relaxation_fractionality": float(
            np.mean(np.minimum(relaxation.values, 1.0 - relaxation.values))
        ),
    }
    return bound, qubo, problem, metadata


def preflight(
    *,
    output_dir: Path,
    optimization_level: int,
    seed_transpiler: int | None,
    maxiter: int,
    seed: int,
    penalty_gbp: float,
    relaxation_multistart: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    circuit, qubo, problem, preparation = prepare_ten_invoice_qaoa(
        penalty_gbp=penalty_gbp,
        p=1,
        seed=seed,
        maxiter=maxiter,
        relaxation_multistart=relaxation_multistart,
    )

    backend = connect_iqm_backend("q20")
    backend_qubits = int(getattr(backend, "num_qubits", 0))
    if backend_qubits < circuit.num_qubits:
        raise RuntimeError(
            f"Q20 backend reports {backend_qubits} qubits, "
            f"but the 10-invoice QUBO requires {circuit.num_qubits}"
        )

    # Keep the common helper for metrics, but allow a seed-controlled transpile
    # here so the hardware layout can be reviewed before any submission.
    from qiskit import transpile

    compiled = transpile(
        circuit,
        backend=backend,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )

    def two_qubit_count(qc) -> int:
        return sum(
            1
            for instruction in qc.data
            if len(instruction.qubits) == 2
            and instruction.operation.name != "barrier"
            and not bool(getattr(instruction.operation, "_directive", False))
        )

    transpilation = {
        "logical_qubits": circuit.num_qubits,
        "transpiled_qubits": compiled.num_qubits,
        "logical_depth": int(circuit.depth()),
        "transpiled_depth": int(compiled.depth()),
        "logical_two_qubit_gates": two_qubit_count(circuit),
        "transpiled_two_qubit_gates": two_qubit_count(compiled),
        "logical_operation_counts": {
            str(name): int(count) for name, count in circuit.count_ops().items()
        },
        "transpiled_operation_counts": {
            str(name): int(count) for name, count in compiled.count_ops().items()
        },
        "optimization_level": optimization_level,
        "seed_transpiler": seed_transpiler,
    }

    qpy_path = output_dir / "ten-invoice-q20-frozen.qpy"
    with qpy_path.open("wb") as handle:
        qpy.dump(compiled, handle)

    manifest = {
        "experiment": "cherryq-10-invoice-preflight",
        "device": "q20",
        "backend": backend_summary(backend),
        "preparation": preparation,
        "transpilation": transpilation,
        "qpy_path": str(qpy_path),
        "qpy_sha256": _sha256(qpy_path),
        "qpu_submitted": False,
    }
    manifest_path = output_dir / "ten-invoice-q20-preflight.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=list),
        encoding="utf-8",
    )
    return manifest


def run_frozen(
    *,
    manifest_path: Path,
    shots: int,
    output_path: Path,
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qpy_path = Path(manifest["qpy_path"])
    expected_sha = manifest["qpy_sha256"]
    actual_sha = _sha256(qpy_path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"Frozen QPY checksum mismatch: expected {expected_sha}, got {actual_sha}"
        )

    with qpy_path.open("rb") as handle:
        circuits = qpy.load(handle)
    if len(circuits) != 1:
        raise RuntimeError(f"Expected one frozen circuit, got {len(circuits)}")
    circuit = circuits[0]

    problem = build_ten_invoice_problem()
    penalty_gbp = float(manifest["preparation"]["penalty_gbp"])
    qubo = problem.build_qubo(penalty_gbp=penalty_gbp)

    backend = connect_iqm_backend("q20")
    execution = submit_and_collect(circuit, backend, shots=shots)
    analysis = analyse_cherryq_counts(execution["counts"], qubo, problem)

    payload = {
        "experiment": "cherryq-10-invoice-q20",
        "device": "q20",
        "backend": backend_summary(backend),
        "frozen_manifest": str(manifest_path),
        "frozen_qpy_sha256": expected_sha,
        "transpilation": manifest["transpilation"],
        "preparation": manifest["preparation"],
        "execution": execution,
        "analysis": analysis,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, default=list),
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preflight or run the frozen CherryQ 10-invoice Q20 circuit"
    )
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    parser.add_argument("--output-dir", type=Path, default=Path("q20-ten-preflight"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--seed-transpiler", type=int, default=42)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--penalty", type=float, default=250.0)
    parser.add_argument("--relaxation-multistart", type=int, default=32)
    args = parser.parse_args()

    if args.mode == "preflight":
        payload = preflight(
            output_dir=args.output_dir,
            optimization_level=args.optimization_level,
            seed_transpiler=args.seed_transpiler,
            maxiter=args.maxiter,
            seed=args.seed,
            penalty_gbp=args.penalty,
            relaxation_multistart=args.relaxation_multistart,
        )
        print(json.dumps(payload, indent=2, default=list))
        return 0

    if args.manifest is None or args.output is None:
        parser.error("--mode run requires --manifest and --output")

    payload = run_frozen(
        manifest_path=args.manifest,
        shots=args.shots,
        output_path=args.output,
    )
    print(f"Saved result to {args.output}")
    print(f"IQM job ID: {payload['execution']['job_id']}")
    print(
        "10-invoice CherryQ: "
        f"P(optimal business plan)="
        f"{payload['analysis']['optimal_business_plan_probability']:.2%}, "
        f"P(business feasible)="
        f"{payload['analysis']['business_feasible_probability']:.2%}, "
        f"best loss=£{payload['analysis']['best_sampled_business_loss_gbp']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
