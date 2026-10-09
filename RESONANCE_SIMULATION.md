# CherryQ frozen IQM Resonance comparison

This runner only permits `https://resonance.iqm.tech` with `garnet:mock` and
`facade_garnet`. It does not select real Garnet, Aalto Q20, a default computer,
or a timeslot. No physical QPU execution is authorized.

## Verified checkpoint

- Gamma: -0.7439674535854706; beta: 0.3004519607226279; p=1.
- Exact and MILP: A+B+F+G+H, £5,000 spend, £950 business loss.
- 14 logical qubits; penalty £250; warm-start-X, epsilon 0.25.
- The repository relaxation (seed 42, 32 starts) reproduces
  `[1,1,0,0,0,1,1,1,0,0,0,0,0,0]`. It already encodes the classical optimum.
  The QAOA parameters are frozen; no QAOA optimizer is rerun.
- Logical depths without measurements: original 78, round-robin 42;
  both have 182 CX gates.

The older `evidence/cost-order-20261009` QPY files use slightly different
locally optimized parameters. This runner does not mistake them for the
later LUMI checkpoint. It rebuilds the same ansatz/preparation with the exact
parameters in `evidence/lumi-cost-order-22664032/summary.json` and verifies
both cost layers over all 16,384 basis states.

## Runtime and commands

Validated offline with the LUMI-compatible runtime: Python 3.12, Qiskit 2.1.2,
IQM Client 34.0.2, Qiskit Aer 0.17.2, NumPy 2.4.6, SciPy 1.17.1.
This is an existing runtime; the repository's current general install requires
Qiskit >=2.5.2. Use `PYTHONPATH=src` with the existing compatible environment.

Ideal references, 1,000/5,000/10,000-shot samples at three seeds, and six
offline Garnet compilation/measurement/statevector checks:

```bash
PYTHONPATH=src python -m cherryq.hardware.resonance_simulation \
  --local-compile --output-dir results/resonance-preflight
```

Authenticated mock/facade execution, after provisioning `IQM_TOKEN` securely
in the execution environment:

```bash
PYTHONPATH=src python -m cherryq.hardware.resonance_simulation \
  --facade --output-dir results/resonance-facade
```

Do not put the token in arguments, source, notebooks, Git, or result files.
Output directories must be new. The remote command first reads the account's
mock alias inventory, requires `garnet:mock`, and lets IQM verify static
architecture compatibility before execution. It rechecks the resolved alias
before every submission. There is no fallback to physical hardware.

Default grid: both schedules × transpiler seeds 42/7/123 × shots
1,000/5,000/10,000 = 18 noisy runs, 96,000 total shots. `--repeats N` enables
independent repeated executions; these are not described as seeded runs.
IQM Client 34.0.2's stock local fake backend forwards only `shots` to Aer and
ignores `seed_simulator`, so this runner makes no noisy-seed reproducibility
claim. Do not confuse routing seeds with simulator seeds. Review this behavior
if upgrading the IQM SDK.

The facade submits to a hardware-free mock, discards its random bit results,
then returns local IQM/Aer noisy simulation results. It records the latter.
Noisy runs cannot be claimed complete merely because an offline compile passed.

### Current Garnet mock label compatibility

Authenticated inspection on 2026-10-09 found `emerald:mock`, `garnet:mock`, and
`sirius:mock`. Garnet mock reports DUT `M194_W0_P08_Z99`; IQM's bundled Garnet
noise model reports `M153_W0_P06_Z99`. All 20 qubit names, the empty resonator
list, and all 30 connections match exactly. IQM Client 34.0.2's stock facade
compares the complete architecture object and therefore rejects this label
difference (the inspected 36.0.0 implementation does the same).

The runner accepts only this kind of label-only mismatch. It constructs IQM's
`IQMFakeBackend` with the returned mock architecture and the **unchanged**
official Garnet error profile, temporarily registers that model for
`facade_garnet`, and calls the requested `IQMProvider(...,
quantum_computer="garnet:mock").get_backend("facade_garnet")`. The original
registry entry is restored immediately. IQM's facade compatibility check and
mock execution remain active. Any change to qubits, resonators or connectivity
is rejected. Both labels and the adapter audit are saved with each run.
This compatibility adapter does not establish current-device or Aalto
calibration equivalence, and does not modify the gate-level noise model.

## GitHub Actions execution

`resonance-mock.yml` uses the existing `IQM_TOKEN` environment secret in
**Preview**. The token is injected only into the account-check and simulation
steps; it is never copied out or stored in artifacts. Production is unused.
An inventory-only job must verify `garnet:mock` and facade architecture before
the six schedule/routing-seed jobs can start. Each runs 1,000, 5,000 and 10,000
shots and uploads nonsecret JSON/CSV/QPY checkpoints even after a failure.
The workflow installs `requirements-resonance.txt` in an isolated Python 3.12
runtime and uses `PYTHONPATH=src` instead of the general package dependencies.
The `--schedules` filter divides the noisy grid; both ideal schedules are still
checked for equivalence in every job. Routing seeds are not simulator seeds.
The workflow runs on pushes affecting its runner/configuration on
`feature/resonance-mock-comparison`, and also supports manual dispatch where
GitHub makes it available. It does not deploy the CherryQ application.

## Metrics and interpretation

Qiskit's rightmost measured bit is A, then B through J, then s0 through s3.
The existing repository scorer is reused for every sampled result.

- P(A+B+F+G+H): invoice optimum, marginalized over all slack assignments.
- Business feasibility: invoice spend <= £5,000, regardless of slack.
- QUBO feasibility: equality constraint including slack has zero residual.
- Optimal QUBO state: unique minimum-energy complete 14-bit assignment.
- Best sampled loss: minimum business loss among business-feasible samples.
- Depth excludes final measurements; logical 2Q gates are CX, physical 2Q are CZ.

JSON stores raw counts, per-run metrics, shot uncertainty (Wilson 95% intervals),
source/circuit hashes, versions, layouts, validation and the built-in error
profile. CSV contains comparison rows. Finite-shot intervals do not represent
uncertainty in the noise model or estimate real hardware calibration uncertainty.

## Interpretation boundary

Label results **representative IQM Garnet noisy simulation**. Neither
architecture nor calibration equivalence to Aalto Q20 has been established.
The stock model applies T1/T2 noise for gate durations, depolarizing errors and
readout errors; it does not automatically add idle-time scheduling noise.
Consequently this compares complete routed gate sequences, gate counts and
qubit assignments. It does not isolate a causal benefit from wall-clock depth.
Keep Exact/MILP -> Ideal QAOA -> Resonance noisy facade -> Aalto Q20 (future)
as separate stages. No quantum advantage is claimed.

Official workflow: https://docs.iqm.tech/iqm-client/user_guide_qiskit.html#running-a-quantum-circuit-on-a-facade-backend
