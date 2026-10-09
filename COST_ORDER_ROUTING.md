# Cost-order routing checkpoint

The 10-invoice cost-order candidate is now implemented in the Qiskit circuit
builder as `cost_order="round-robin"`. The default remains `original`.
The five-invoice QPU results have not been rerun or modified.

## Verified locally

- Exact enumeration and MILP agree: **A+B+F+G+H**, £5,000 spend, £950 loss.
- Unmeasured logical depth: **78 → 42**. With final measurements: **79 → 43**.
- Both schedules have **182 CX** gates (91 ZZ terms, 14 logical qubits).
- The actual emitted cost gates are checked on every one of **16,384 basis
  states**, including restoration of every basis label and agreement with the
  Ising phase. Tests include the uploaded audit's gamma and two other values.
- Full Qiskit statevectors agree for p=1 and p=2, with X and matched mixers.
- Routing validation is tested on a synthetic target with a nontrivial layout;
  that test is not evidence of actual Q20 routing.

The uploaded audit's historical Q20 reference remains **300 CZ, depth 283**.
No newer Q20 metrics have been measured by this checkpoint. **NO-GO remains.**

## Comparison protocol

The audit optimizes a new warm-start-X p=1 preparation exactly once, then
reuses its initial RY angles and frozen gamma/beta for both schedules. This is
a paired experiment, not a replay of the earlier archived Q20 preparation.
Optimized parameters can differ slightly across dependency versions.

The audit reads the actual Q20 backend target and compiles both schedules with
optimization level 3 at seeds 42, 7, and 123. It records native gate counts,
measured and unmeasured depths, target operation loci, final layouts, package
versions, source hashes, and QPY hashes. It verifies target gate support,
invoice measurement order, and the ideal routed output state including idle
ancillas and final qubit permutation.

There is no QPU submission call in this command, and its output is deliberately
not a `q20_ten --mode run` manifest. A fresh output directory is required so an
older audit cannot be overwritten. Partial results remain marked unverified if
compilation or validation fails. Successful routing still requires resource and
hardware-quality review; this audit never changes NO-GO into GO automatically.
No claim of speedup, advantage, or improved noisy-hardware quality follows from
logical or ideal-state equivalence.

## Run on LUMI

From the existing checkout, after updating the branch with a fast-forward pull:

```bash
cd /users/srinarum/cherryq-qubo-qaoa
git switch feature/10-invoice-scaling
git pull --ff-only origin feature/10-invoice-scaling
sbatch lumi/q20_cost_order_audit.sbatch
```

This reserves CPU time for compilation and ideal simulation only. It does not
submit to Q20. It uses the existing project, event reservation and CSC module.
The result is written to a new directory:

```text
/scratch/project_462001763/cherryq/q20-ten/cost-order-audit-<Slurm-job-ID>/routing-audit.json
```

Review the paired CZ/depth deltas across all three seeds, rather than comparing
only a selected best seed with the historical reference. A smaller logical
depth may produce a larger routed circuit. Current calibration/noise performance
and the earlier NO-GO review are still relevant even if routing improves.

## Reproduce without Q20 access

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m cherryq.hardware.q20_routing_audit \
  --logical-only --output-dir /tmp/cherryq-cost-order-local
```

The logical-only report explicitly sets `physical_routing_verified=false` and
`qpu_submitted=false`. It is not a hardware result.
