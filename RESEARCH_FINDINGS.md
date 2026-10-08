# CherryQ QAOA research findings

Date: 7 October 2026

Source run: GitHub Actions `37612088195`

These results are from a noiseless statevector simulation of the five-invoice hackathon model. They are useful for algorithm design and mentor discussion, but they are not evidence of quantum advantage or hardware performance.

## Experiment configuration

- target scenario: `current-cash`
- five supplier-payment variables plus binary budget slack
- penalty grid: £100, £250, £500, £1,000, £2,000, £4,000
- methods: standard QAOA, warm-start initial state + ordinary X mixer, warm-start initial state + matched warm-start mixer
- depths: `p=1`, `p=2`
- seeds: 11, 29, 47, 71, 97
- COBYLA maximum iterations: 80
- finite-shot candidate sample: 2,048
- warm-start relaxation multi-starts: 32
- warm-start epsilon: 0.25
- seed-controlled parameter jitter: 0.05

## 1. Penalty tuning

The QUBO penalty is not treated as an arbitrary constant. We enumerate the QUBO state space and compare the best equality-feasible state with the best infeasible state.

For the current-cash scenario:

| Penalty | Feasibility margin | Result |
| ---: | ---: | --- |
| £100 | -£150 | Unsafe |
| £250 | £0 | Tie - not accepted |
| **£500** | **£250** | **Safe** |
| £1,000 | £750 | Safe |
| £2,000 | £1,750 | Safe |
| £4,000 | £3,750 | Safe |

£500 was the **smallest tested penalty that was strictly safe across every generated cash scenario**.

This is preferable to simply choosing a very large penalty, because an unnecessarily large penalty can dominate the business objective and alter the QAOA energy landscape.

## 2. Warm-start diagnostic

With the selected £500 penalty and 32 relaxation multi-starts:

- relaxation fractionality: **0.0000**
- rounded relaxation business gap: **£0**
- probability assigned by the clipped warm-start initial state to the exact QUBO optimum: **10.01%**
- probability assigned by that initial state to QUBO-feasible states: **16.96%**

The classical relaxation therefore provided a strong initial state on this small instance. That benefit must be reported as classical preprocessing, not as quantum performance.

## 3. Repeated-seed results

| Method | Depth | Mean P(exact business plan) | Mean business-feasible probability | Mean runtime | Circuit depth | 2-qubit gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard QAOA | p=1 | 2.9% +/- 0.1% | 79.1% | 441 ms | 16 | 28 |
| Standard QAOA | p=2 | 1.8% +/- 0.5% | 71.2% | 620 ms | 26 | 56 |
| Warm-start + matched mixer | p=1 | 20.0% +/- <0.1% | 81.3% | 639 ms | 44 | 56 |
| Warm-start + matched mixer | p=2 | 19.4% +/- 1.2% | 81.7% | 1,243 ms | 72 | 112 |
| **Warm-start + X mixer** | **p=1** | **23.4% +/- <0.1%** | **82.0%** | **618 ms** | **42** | **56** |
| Warm-start + X mixer | p=2 | 21.4% +/- 3.0% | 83.0% | 1,155 ms | 68 | 112 |

Every configuration sampled the exact business optimum at least once in every run with 2,048 samples, so "did we see the optimum at all?" is too weak a discriminator for this small instance. Probability of the exact plan is more informative here.

## 4. What the mixer ablation tells us

The experiment deliberately compares two warm-start variants using the same type of classical initialisation:

1. warm-start initial state + ordinary X mixer
2. warm-start initial state + matched warm-start mixer

On this test, the matched mixer **did not improve** the probability of the exact business plan. The warm-start + X mixer produced the highest mean exact-plan probability at both p=1 and p=2.

Therefore we should **not** tell mentors that the custom matched mixer is currently our differentiating quantum contribution.

The evidence instead suggests that most of the measured gain over standard QAOA is coming from the **warm-start initialisation / classical relaxation**, while the mixer choice still needs investigation.

## 5. Depth result

For this instance, increasing from p=1 to p=2:

- approximately doubled two-qubit gate count for the warm-start circuits
- increased runtime substantially
- did not improve exact-plan probability

So **p=1 is currently the better depth for the prototype**.

This is especially relevant for near-term hardware, where additional circuit depth is a cost rather than a free improvement.

## 6. What can we credibly say to Meeri?

A concise evidence-based answer is:

> We use Qiskit rather than claiming a new QAOA algorithm. We tailor the experiment through a finance-specific dependency-aware QUBO, empirical penalty tuning, a classical warm start, and a mixer ablation. In five repeated simulator runs, warm-start QAOA increased the probability of the exact payment plan substantially over standard QAOA, but the matched warm-start mixer did not beat an ordinary X mixer. So our current evidence points to the warm-start initialisation, not a novel mixer, as the useful adaptation. We will not claim a mixer or quantum advantage until that survives larger, noisy and classical comparisons.

## 7. Limitations and next experiments

- This is a tiny five-invoice model that is easy to solve classically.
- Results are from a noiseless statevector simulator.
- Forecasting inputs are synthetic.
- The warm-start relaxation is unusually strong on this instance.
- Five seeds are enough for a hackathon engineering check, not a research claim.
- Candidate sampling at 2,048 shots is generous for an eight-qubit toy instance.
- Hardware transpilation depth and hardware noise have not yet been measured.

Next work should test problem families where the relaxation is not integral, add noisy simulation / hardware measurements, and compare against strong classical solvers at larger sizes.
