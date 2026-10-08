from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import qaoa_ansatz
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector

from .problem import PaymentProblem, QuboModel


@dataclass(frozen=True)
class IsingModel:
    operator: SparsePauliOp
    offset: float
    scale: float

    @property
    def normalized_operator(self) -> SparsePauliOp:
        return self.operator / self.scale


@dataclass(frozen=True)
class RelaxationResult:
    values: np.ndarray
    energy: float
    rounded_values: np.ndarray
    rounded_energy: float


@dataclass(frozen=True)
class QaoaRunResult:
    method: str
    p: int
    seed: int
    penalty_gbp: float
    num_qubits: int
    circuit_depth: int
    two_qubit_gate_count: int
    optimizer_success: bool
    optimizer_message: str
    optimizer_calls: int
    classical_preprocessing_runtime_ms: float
    optimizer_runtime_ms: float
    total_runtime_ms: float
    optimized_expected_qubo_energy: float
    optimized_parameters: tuple[float, ...]
    best_sampled_qubo_energy: float
    best_sampled_business_loss_gbp: int | None
    best_sampled_paid_invoice_ids: tuple[str, ...]
    qubo_feasible_probability: float
    business_feasible_probability: float
    optimal_qubo_probability: float
    optimal_business_plan_probability: float
    top_samples: tuple[tuple[str, int, float], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "p": self.p,
            "seed": self.seed,
            "penalty_gbp": self.penalty_gbp,
            "num_qubits": self.num_qubits,
            "circuit_depth": self.circuit_depth,
            "two_qubit_gate_count": self.two_qubit_gate_count,
            "optimizer_success": self.optimizer_success,
            "optimizer_message": self.optimizer_message,
            "optimizer_calls": self.optimizer_calls,
            "classical_preprocessing_runtime_ms": self.classical_preprocessing_runtime_ms,
            "optimizer_runtime_ms": self.optimizer_runtime_ms,
            "total_runtime_ms": self.total_runtime_ms,
            "optimized_expected_qubo_energy": self.optimized_expected_qubo_energy,
            "optimized_parameters": self.optimized_parameters,
            "best_sampled_qubo_energy": self.best_sampled_qubo_energy,
            "best_sampled_business_loss_gbp": self.best_sampled_business_loss_gbp,
            "best_sampled_paid_invoice_ids": self.best_sampled_paid_invoice_ids,
            "qubo_feasible_probability": self.qubo_feasible_probability,
            "business_feasible_probability": self.business_feasible_probability,
            "optimal_qubo_probability": self.optimal_qubo_probability,
            "optimal_business_plan_probability": self.optimal_business_plan_probability,
            "top_samples": self.top_samples,
        }


def qubo_to_ising(qubo: QuboModel) -> IsingModel:
    """Map E(x) to H(Z) using x_i = (1 - Z_i) / 2."""

    index = {name: i for i, name in enumerate(qubo.variable_names)}
    z_coeff = {name: 0.0 for name in qubo.variable_names}
    zz_coeff: dict[tuple[str, str], float] = {}
    offset = float(qubo.offset)

    for name, coeff in qubo.linear.items():
        offset += coeff / 2.0
        z_coeff[name] -= coeff / 2.0

    for (left, right), coeff in qubo.quadratic.items():
        offset += coeff / 4.0
        z_coeff[left] -= coeff / 4.0
        z_coeff[right] -= coeff / 4.0
        zz_coeff[(left, right)] = zz_coeff.get((left, right), 0.0) + coeff / 4.0

    terms: list[tuple[str, list[int], complex]] = []
    for name, coeff in z_coeff.items():
        if abs(coeff) > 1e-12:
            terms.append(("Z", [index[name]], coeff))
    for (left, right), coeff in zz_coeff.items():
        if abs(coeff) > 1e-12:
            terms.append(("ZZ", [index[left], index[right]], coeff))

    if not terms:
        operator = SparsePauliOp.from_list(
            [("I" * len(qubo.variable_names), 0.0)]
        )
    else:
        operator = SparsePauliOp.from_sparse_list(
            terms,
            num_qubits=len(qubo.variable_names),
        ).simplify()

    max_abs_coeff = max((abs(complex(c)) for c in operator.coeffs), default=1.0)
    scale = float(max(max_abs_coeff, 1.0))
    return IsingModel(operator=operator, offset=offset, scale=scale)


def solve_continuous_relaxation(
    qubo: QuboModel,
    multistart: int = 64,
    seed: int = 42,
) -> RelaxationResult:
    """Solve the QUBO relaxation over [0, 1]^n using multi-start L-BFGS-B."""

    if multistart < 1:
        raise ValueError("multistart must be at least 1")

    rng = np.random.default_rng(seed)
    bounds = [(0.0, 1.0)] * len(qubo.variable_names)
    best = None

    for start_index in range(multistart):
        if start_index == 0:
            x0 = np.full(len(qubo.variable_names), 0.5)
        else:
            x0 = rng.uniform(0.0, 1.0, len(qubo.variable_names))
        result = minimize(
            lambda values: qubo.energy(values),
            x0,
            method="L-BFGS-B",
            bounds=bounds,
        )
        if best is None or result.fun < best.fun:
            best = result

    assert best is not None
    rounded = np.rint(best.x).astype(int)
    return RelaxationResult(
        values=np.asarray(best.x, dtype=float),
        energy=float(best.fun),
        rounded_values=rounded,
        rounded_energy=float(qubo.energy(rounded)),
    )


def _apply_cost_unitary(
    circuit: QuantumCircuit,
    cost_operator: SparsePauliOp,
    gamma,
) -> None:
    """Apply exp(-i gamma H_C) for a diagonal Z/ZZ Ising Hamiltonian."""

    for pauli_term, coeff in zip(cost_operator.paulis, cost_operator.coeffs, strict=True):
        indices = [
            index
            for index, label in enumerate(pauli_term.to_label()[::-1])
            if label == "Z"
        ]
        real_coeff = float(complex(coeff).real)
        if len(indices) == 1:
            circuit.rz(2 * gamma * real_coeff, indices[0])
        elif len(indices) == 2:
            circuit.cx(indices[0], indices[1])
            circuit.rz(2 * gamma * real_coeff, indices[1])
            circuit.cx(indices[0], indices[1])
        elif len(indices) == 0:
            continue
        else:
            raise ValueError("CherryQ expects only Z and ZZ cost terms")


def build_warm_start_qaoa(
    cost_operator: SparsePauliOp,
    p: int,
    warm_start_probabilities: Sequence[float],
    epsilon: float = 0.25,
    mixer: str = "matched",
) -> tuple[QuantumCircuit, list, list]:
    """Build a warm-start circuit with either a matched or ordinary X mixer.

    The "warm-start-x" experiment intentionally keeps the same warm-start
    initial state but replaces the problem-matched mixer with the ordinary X
    mixer. That lets us isolate whether any improvement comes from the initial
    state alone or from the matched mixer.
    """

    if p < 1:
        raise ValueError("p must be at least 1")
    if not 0.0 < epsilon <= 0.5:
        raise ValueError("epsilon must be in (0, 0.5]")
    if mixer not in {"matched", "x"}:
        raise ValueError("mixer must be 'matched' or 'x'")

    clipped = np.clip(
        np.asarray(warm_start_probabilities, dtype=float),
        epsilon,
        1 - epsilon,
    )
    thetas = 2 * np.arcsin(np.sqrt(clipped))

    gammas = ParameterVector("gamma", p)
    betas = ParameterVector("beta", p)
    circuit = QuantumCircuit(cost_operator.num_qubits)

    for qubit, theta in enumerate(thetas):
        circuit.ry(float(theta), qubit)

    for layer in range(p):
        _apply_cost_unitary(circuit, cost_operator, gammas[layer])
        for qubit, theta in enumerate(thetas):
            if mixer == "matched":
                # Egger et al. warm-start mixer, matching IBM's tutorial form.
                circuit.ry(float(theta), qubit)
                circuit.rz(-2 * betas[layer], qubit)
                circuit.ry(-float(theta), qubit)
            else:
                # Ordinary X mixer applied to the same warm-start initial state.
                circuit.rx(2 * betas[layer], qubit)

    return circuit, list(gammas), list(betas)


def _make_cost_function(
    circuit: QuantumCircuit,
    parameter_order: Sequence,
    cost_operator: SparsePauliOp,
    estimator: StatevectorEstimator,
    history: list[float],
) -> Callable[[np.ndarray], float]:
    def cost_function(params: np.ndarray) -> float:
        bound = circuit.assign_parameters(
            dict(zip(parameter_order, params, strict=True))
        )
        result = estimator.run([(bound, cost_operator)]).result()[0]
        energy = float(np.asarray(result.data.evs).real.item())
        history.append(energy)
        return energy

    return cost_function


def _label_to_values(label: str, variable_names: Sequence[str]) -> dict[str, int]:
    bits = [int(bit) for bit in label.replace(" ", "")[::-1]]
    return dict(zip(variable_names, bits, strict=True))


def _state_metrics(
    state: Statevector,
    qubo: QuboModel,
    problem: PaymentProblem,
    shots: int,
    seed: int,
) -> dict[str, object]:
    probabilities = state.probabilities_dict()
    labels = np.array(list(probabilities.keys()), dtype=object)
    probs = np.array([float(probabilities[label]) for label in labels], dtype=float)
    probs = probs / probs.sum()

    exact_qubo_values, _ = qubo.solve_exact()
    exact_business = problem.solve_exact()

    qubo_feasible_probability = 0.0
    business_feasible_probability = 0.0
    optimal_qubo_probability = 0.0
    optimal_business_plan_probability = 0.0

    exact_qubo_vector = tuple(exact_qubo_values[name] for name in qubo.variable_names)
    exact_business_decision = tuple(
        exact_business.decision[name] for name in problem.invoice_ids
    )

    for label, probability in zip(labels, probs, strict=True):
        values = _label_to_values(str(label), qubo.variable_names)
        vector = tuple(values[name] for name in qubo.variable_names)
        supplier_decision = problem.supplier_decision_from_values(
            values, qubo.variable_names
        )
        supplier_vector = tuple(supplier_decision[name] for name in problem.invoice_ids)

        if abs(qubo.constraint_residual(values)) < 1e-9:
            qubo_feasible_probability += probability
        if problem.is_business_feasible(supplier_decision):
            business_feasible_probability += probability
        if vector == exact_qubo_vector:
            optimal_qubo_probability += probability
        if supplier_vector == exact_business_decision:
            optimal_business_plan_probability += probability

    rng = np.random.default_rng(seed)
    sampled_labels = rng.choice(labels, size=shots, p=probs)
    counts = Counter(str(label) for label in sampled_labels)

    best_qubo_energy = float("inf")
    best_business_loss: int | None = None
    best_paid: tuple[str, ...] = ()

    for label in counts:
        values = _label_to_values(label, qubo.variable_names)
        qubo_energy = qubo.energy(values)
        if qubo_energy < best_qubo_energy:
            best_qubo_energy = qubo_energy

        supplier_decision = problem.supplier_decision_from_values(
            values, qubo.variable_names
        )
        if problem.is_business_feasible(supplier_decision):
            business_loss = problem.business_loss_gbp(supplier_decision)
            if best_business_loss is None or business_loss < best_business_loss:
                best_business_loss = business_loss
                best_paid = tuple(
                    invoice_id
                    for invoice_id in problem.invoice_ids
                    if supplier_decision[invoice_id]
                )

    top_samples = tuple(
        (label, count, qubo.energy(_label_to_values(label, qubo.variable_names)))
        for label, count in counts.most_common(8)
    )

    return {
        "best_sampled_qubo_energy": float(best_qubo_energy),
        "best_sampled_business_loss_gbp": best_business_loss,
        "best_sampled_paid_invoice_ids": best_paid,
        "qubo_feasible_probability": float(qubo_feasible_probability),
        "business_feasible_probability": float(business_feasible_probability),
        "optimal_qubo_probability": float(optimal_qubo_probability),
        "optimal_business_plan_probability": float(optimal_business_plan_probability),
        "top_samples": top_samples,
    }


def _circuit_two_qubit_gate_count(circuit: QuantumCircuit) -> int:
    return sum(1 for instruction in circuit.data if len(instruction.qubits) == 2)


def run_qaoa(
    problem: PaymentProblem,
    qubo: QuboModel,
    method: str,
    p: int,
    maxiter: int = 150,
    shots: int = 4096,
    seed: int = 42,
    epsilon: float = 0.25,
    relaxation_multistart: int = 64,
    warm_start_parameter_jitter: float = 0.0,
) -> QaoaRunResult:
    """Run standard, warm-start-X, or matched warm-start QAOA.

    method="warm-start-x" uses the warm-start initial state with an ordinary X
    mixer. method="warm-start" uses the same initial state and the matched
    warm-start mixer. This three-way comparison isolates the mixer contribution.
    """

    allowed_methods = {"standard", "warm-start-x", "warm-start"}
    if method not in allowed_methods:
        raise ValueError(f"method must be one of {sorted(allowed_methods)}")
    if p < 1:
        raise ValueError("p must be at least 1")
    if maxiter < 1 or shots < 1:
        raise ValueError("maxiter and shots must be positive")
    if warm_start_parameter_jitter < 0:
        raise ValueError("warm_start_parameter_jitter cannot be negative")

    total_started = perf_counter()
    ising = qubo_to_ising(qubo)
    cost_operator = ising.normalized_operator
    estimator = StatevectorEstimator(seed=seed)
    history: list[float] = []
    preprocessing_runtime_ms = 0.0
    rng = np.random.default_rng(seed)

    if method == "standard":
        circuit = qaoa_ansatz(cost_operator, reps=p, flatten=True)
        parameter_order = list(circuit.parameters)
        initial_params = rng.uniform(0.0, np.pi, len(parameter_order))
    else:
        preprocessing_started = perf_counter()
        relaxation = solve_continuous_relaxation(
            qubo, multistart=relaxation_multistart, seed=seed
        )
        preprocessing_runtime_ms = (
            perf_counter() - preprocessing_started
        ) * 1000.0

        mixer = "matched" if method == "warm-start" else "x"
        circuit, gammas, betas = build_warm_start_qaoa(
            cost_operator,
            p=p,
            warm_start_probabilities=relaxation.values,
            epsilon=epsilon,
            mixer=mixer,
        )
        parameter_order = gammas + betas
        initial_params = np.concatenate(
            [np.zeros(p), np.full(p, np.pi / 4.0)]
        )
        if warm_start_parameter_jitter:
            initial_params = initial_params + rng.normal(
                0.0,
                warm_start_parameter_jitter,
                size=len(initial_params),
            )

    circuit_depth = int(circuit.depth())
    two_qubit_gate_count = _circuit_two_qubit_gate_count(circuit)

    optimizer_started = perf_counter()
    result = minimize(
        _make_cost_function(
            circuit,
            parameter_order,
            cost_operator,
            estimator,
            history,
        ),
        initial_params,
        method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": 0.5},
    )
    optimizer_runtime_ms = (perf_counter() - optimizer_started) * 1000.0

    bound = circuit.assign_parameters(
        dict(zip(parameter_order, result.x, strict=True))
    )
    state = Statevector.from_instruction(bound)
    metrics = _state_metrics(state, qubo, problem, shots=shots, seed=seed)

    original_expected_energy = float(result.fun * ising.scale + ising.offset)
    total_runtime_ms = (perf_counter() - total_started) * 1000.0

    return QaoaRunResult(
        method=method,
        p=p,
        seed=seed,
        penalty_gbp=qubo.penalty_gbp,
        num_qubits=len(qubo.variable_names),
        circuit_depth=circuit_depth,
        two_qubit_gate_count=two_qubit_gate_count,
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        optimizer_calls=len(history),
        classical_preprocessing_runtime_ms=float(preprocessing_runtime_ms),
        optimizer_runtime_ms=float(optimizer_runtime_ms),
        total_runtime_ms=float(total_runtime_ms),
        optimized_expected_qubo_energy=original_expected_energy,
        optimized_parameters=tuple(float(value) for value in result.x),
        best_sampled_qubo_energy=float(metrics["best_sampled_qubo_energy"]),
        best_sampled_business_loss_gbp=metrics["best_sampled_business_loss_gbp"],
        best_sampled_paid_invoice_ids=metrics["best_sampled_paid_invoice_ids"],
        qubo_feasible_probability=float(metrics["qubo_feasible_probability"]),
        business_feasible_probability=float(metrics["business_feasible_probability"]),
        optimal_qubo_probability=float(metrics["optimal_qubo_probability"]),
        optimal_business_plan_probability=float(
            metrics["optimal_business_plan_probability"]
        ),
        top_samples=metrics["top_samples"],
    )
