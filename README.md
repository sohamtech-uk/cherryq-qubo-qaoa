# CherryQ - Quantum-AI Treasury Optimiser

CherryQ is a hackathon prototype for one specific small-business treasury decision:

> Which supplier invoices should we pay in the next payment run, and which should we defer, while protecting payroll, tax and minimum reserves?

The end-to-end development pipeline is:

**AI payment probability -> cash-arrival scenarios -> QUBO per scenario -> exact/MILP classical benchmark -> QAOA research -> explainable CherryQ recommendation**

This is a public hackathon engineering prototype. It does **not** claim quantum advantage, and its forecasting metrics currently use synthetic data only. Private Cherry Money product code, credentials, customer data, and integration logic are intentionally excluded.

## Predictive AI and cash scenarios

The first AI model is a `GradientBoostingClassifier`. Its target is whether an outstanding customer invoice will be paid before the next payment run. Prototype features are invoice amount, payment terms, days until due, historical average days late, 90-day on-time rate and prior invoice count.

CherryQ does not multiply an invoice by its payment probability and treat that expected value as cash. Forecast receipts become discrete settled/not-settled scenarios. Only the `current-cash` plan is actionable immediately; conditional plans are activated only after named receipts actually settle.

## Five-invoice optimisation model

Base cash position:

- cleared cash: £8,000
- protected payroll/tax/minimum reserve: £5,000
- supplier-payment budget before new receipts: £3,000

| Invoice | Supplier | Amount | Standalone loss if deferred |
| --- | --- | ---: | ---: |
| A | Materials supplier | £1,500 | £100 |
| B | Delivery partner | £1,000 | £100 |
| C | Equipment hire | £1,000 | £350 |
| D | IT support | £500 | £100 |
| E | Packaging supplier | £1,500 | £250 |

A and B form a dependency: if both are not paid, the model includes a £900 operational loss because the customer order cannot move forward.

The exact optimum for the current £3,000 budget is **A + B + D**, spending £3,000 with £600 remaining modelled loss.

## QUBO and classical benchmarks

For the current-cash scenario:

```text
3xA + 2xB + 2xC + xD + 3xE <= 6

L(x) = 100(1-xA) + 100(1-xB) + 350(1-xC)
     + 100(1-xD) + 250(1-xE) + 900(1-xA*xB)

Q(x,s) = L(x) + P(3xA + 2xB + 2xC + xD + 3xE + s - 6)^2
```

The implementation now treats `P` as an experimental parameter rather than assuming a fixed value is automatically good.

Two classical references are used:

- exact enumeration for the small hackathon instance
- binary MILP using SciPy/HiGHS, with the A/B dependency linearised through an auxiliary binary variable

MILP and exact enumeration must agree before a recommendation is treated as trustworthy.

## Penalty tuning

The research command enumerates every QUBO state and calculates:

- best equality-feasible QUBO energy
- best infeasible QUBO energy
- **feasibility margin = best infeasible energy - best feasible energy**
- whether the best feasible QUBO solution matches the original business optimum

A penalty is labelled **SAFE** only when the margin is strictly positive for the scenario. The experiment chooses the smallest tested penalty that is safe across all generated cash scenarios.

This matters because an unnecessarily large penalty can dominate the financial objective and distort QAOA's energy landscape, while a penalty that is too small can make an over-budget state globally attractive.

## Mixer analysis: what is actually tailored?

CherryQ now runs three variants:

1. **standard** - ordinary QAOA from Qiskit's `qaoa_ansatz`
2. **warm-start-x** - continuous-relaxation warm-start initial state, but an ordinary X mixer
3. **warm-start** - the same warm-start initial state plus the matched warm-start mixer

The second variant is an ablation experiment. Comparing `warm-start-x` with `warm-start` helps isolate whether any measured improvement comes from classical initialisation or from the matched mixer itself.

Both warm-start variants use the same continuous QUBO relaxation. The research report also records:

- relaxation fractionality
- rounded relaxation gap to the exact business optimum
- probability assigned by the clipped warm-start initial state to the exact QUBO optimum
- probability that the warm-start initial state assigns to QUBO-feasible states

The classical relaxation remains explicitly reported as classical preprocessing.

## Repeated seeded runs

One favourable run is not treated as evidence. The research harness repeats each method/depth across multiple seeds and aggregates:

- rate at which the finite-shot sample contains the exact business optimum
- mean and standard deviation of exact-plan probability
- mean and standard deviation of business-feasible probability
- mean best sampled gap to the exact optimum
- optimiser success rate and calls
- end-to-end runtime
- classical warm-start preprocessing runtime
- circuit depth and two-qubit gate count

Warm-start runs can optionally apply a small seed-controlled perturbation around the canonical initial QAOA parameters. This probes robustness near the warm-start solution without changing the underlying QUBO.

## Explainable recommendation

The customer-facing recommendation still uses the verified exact/MILP optimum. QAOA remains a benchmark until repeated experiments show a reproducible benefit.

The output separates:

- **Actionable now** - only cleared cash, protected reserves untouched
- **Conditional plans** - only after forecast receipts have actually settled
- **Why** - deferral costs, payment dependencies and budget effects
- **Benchmark evidence** - exact/MILP agreement and separate QAOA results
- **Safety** - human approval before payment execution

## Run locally

```bash
# run these commands from the repository root
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

Run the product-style pipeline without quantum:

```bash
python -m cherryq.demo --skip-quantum
```

Run the all-scenario QAOA demo:

```bash
python -m cherryq.demo --maxiter 80 --shots 2048
```

Run the mentor-facing penalty/mixer/repeated-seed experiment:

```bash
python -m cherryq.research \
  --scenario current-cash \
  --penalties 100,250,500,1000,2000,4000 \
  --methods standard,warm-start-x,warm-start \
  --p 1,2 \
  --seeds 11,29,47,71,97 \
  --maxiter 80 \
  --shots 2048 \
  --json-out research-report.json
```

For a fast CI-style research smoke test:

```bash
python -m cherryq.research \
  --scenario current-cash \
  --penalties 100,500,2000 \
  --p 1 \
  --seeds 11,29 \
  --maxiter 8 \
  --shots 256 \
  --relaxation-multistart 8
```

Optional IBM Quantum Runtime dependency:

```bash
python -m pip install -e ".[hardware]"
```

## What is validated in code

- exact business optimum and MILP agreement
- QUBO/Ising basis-state equivalence
- scenario-specific QUBO rebuilding
- explicit detection of unsafe/tied penalty values
- standard, warm-start-X and matched warm-start circuit paths
- p=1 and p=2 execution
- repeated-seed aggregate metrics
- circuit depth/two-qubit gate reporting
- AI probabilities feed discrete cash scenarios rather than fractional expected cash
- recommendation never spends forecast receipts before settlement
- automated CI smoke-tests the full research harness

## What still requires evidence

1. Validate payment dependencies and deferral costs with real SME/accounting users.
2. Replace synthetic forecasting history with anonymised, permissioned data.
3. Run the full repeated-seed experiment at mentor-agreed optimiser budgets.
4. Add noisy-simulator and hardware results only after simulator behaviour is understood.
5. Scale problem families and benchmark against strong classical optimisation.
6. Treat any quantum-performance claim as unsupported until it survives those comparisons.

## Implementation references

- IBM Quantum QAOA tutorial: https://qiskit.qotlabs.org/docs/tutorials/quantum-approximate-optimization-algorithm
- IBM Quantum warm-start QAOA tutorial: https://qiskit.qotlabs.org/docs/tutorials/warm-start-qaoa
- Qiskit `qaoa_ansatz`: https://qiskit.qotlabs.org/docs/api/qiskit/qiskit.circuit.library.qaoa_ansatz
- SciPy `milp`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html
