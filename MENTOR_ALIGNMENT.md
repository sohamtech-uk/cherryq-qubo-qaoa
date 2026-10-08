# Mentor feedback alignment

## Joonas - differentiation and AI role

CherryQ is not using QAOA just to rank invoices. The differentiator being tested is a supplier-payment decision with both uncertain incoming cash and pairwise payment dependencies. The AI component is a predictive model that estimates whether an outstanding customer invoice will be paid before the next payment run. It informs cash scenarios but does not approve supplier payments.

Historical data remains classical. Only the small binary optimisation instance is mapped to a quantum circuit.

## Meeri - how QAOA is tailored

Qiskit provides the implementation framework and standard QAOA baseline. CherryQ adds a finance-specific QUBO containing operational payment dependencies, empirically tunes the QUBO penalty, rescales the Ising Hamiltonian for numerical conditioning, and compares standard QAOA with two warm-start variants at shallow depths `p=1` and `p=2`.

We now have a mixer ablation rather than assuming the matched warm-start mixer is better. Across five seeds in the current statevector benchmark, standard QAOA at p=1 assigned about 2.9% mean probability to the exact business plan, warm-start with the matched mixer about 20.0%, and warm-start with an ordinary X mixer about 23.4%. The matched mixer therefore has **not** demonstrated an additional benefit in our current experiment. The evidence points mainly to the classical warm-start initialisation.

Warm-start QAOA is not claimed as a new algorithm. The classical relaxation is reported separately, and no quantum advantage is claimed.

## Amir - customer, decision and quantum advantage

Initial target user: an owner or finance manager of a small wholesaler that must make supplier payments before all customer receipts have arrived.

Decision: which supplier invoices to pay in the next payment run, and which to defer, while protecting payroll, tax and essential reserves.

No quantum advantage is claimed. The toy instance is exactly solvable classically and is used to verify the QUBO and quantum implementation. Larger experiments will compare quantum results with strong classical alternatives.

## Lu - concrete QUBO

The repository implements the five-invoice, £3,000 supplier-payment budget scenario from the mentor response. Invoice variables are binary, the budget is represented in £500 units with binary slack, and the objective minimises standalone deferral losses plus a £900 A/B dependency loss. An exact solver verifies A + B + D as the optimum.

## Petri - customer validation

The current scenario remains a hypothesis. Before treating the dependency values as product requirements, the team should interview at least one SME user and one accountant/bookkeeper. If real users do not recognise the dependency pattern, the formulation should change rather than keeping complexity solely for the quantum experiment.
