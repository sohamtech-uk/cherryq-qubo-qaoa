# LUMI Q20 cost-order audit — Slurm 22664032

**All six routing and ideal-state checks passed. No QPU job was submitted.**

The compilation-only job completed in 58 seconds with exit code 0:0, using
source commit `f3757251e674ef9e19be123d92cb8dfad8e0c044`. The current Q20
target exposed 19 physical qubits for the 14-logical-qubit problem.

## Installed-runtime finding and fix

The actual LUMI runtime uses Qiskit 2.1.2 and IQM Client 34.0.2. The diagnostic
job 22663749 printed only safe instruction, target and layout metadata.
All 257 two-qubit instructions failing direct target support were
`_SingletonCZGate` instances and passed `isinstance(..., CZGate)`.
Their module attribute was `None`. Of those instructions, 22 had supported
reverse loci and 235 had neither direction supported. The observed failing
CZ pair `(2, 0)` had direct=false and reverse=false.

The proposed semantic-name change was therefore not applied. It would not fix
this case. The audit retains CZ-only symmetry; reversed CX and absent CZ
couplings remain rejected.

We reproduced the default IQM scheduling failure offline with the same
Qiskit/IQM versions. The fix explicitly selects Qiskit's standard scheduling
stage with the live native target, avoiding the IQM resonator round trip and
measurement-phase dropping. The audit rejects resonator/MOVE architectures.
All target, measurement and ideal-state checks remain in force.

Relevant tests: **22 passed** with LUMI's Qiskit/IQM versions. Full suite:
**57 passed** with those versions; **55 passed**, with the optional IQM module
skipped, in the normal development environment. CI for the audited source
commit passed. The existing small-iteration COBYLA test warning remains.

## Paired routing results

All depths below exclude final measurements. Logical 2Q gates are CX;
physical 2Q gates are native CZ. Layout entry `i` gives the final physical
index of logical qubit `i`. Amplitude errors are for the prepared state,
up to one global phase, including idle physical ancillas.

| Schedule | Seed | Logical depth | Physical depth | Logical 2Q | Physical CZ | Reversed CZ | Final logical-to-physical layout | Max amplitude error | Target | Measurement |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| original | 42 | 78 | 323 | 182 | 298 | 0 | [14,17,16,11,6,13,12,3,7,8,10,5,4,9] | 8.442e-16 | PASS | PASS |
| round-robin | 42 | 42 | 299 | 182 | 320 | 0 | [17,18,16,14,11,13,8,10,9,3,4,7,15,12] | 9.115e-15 | PASS | PASS |
| original | 7 | 78 | 413 | 182 | 308 | 0 | [5,1,3,10,4,17,9,14,11,16,7,8,12,13] | 6.651e-16 | PASS | PASS |
| round-robin | 7 | 42 | 402 | 182 | 329 | 0 | [9,8,5,16,14,17,3,12,11,15,18,7,13,10] | 1.341e-15 | PASS | PASS |
| original | 123 | 78 | 448 | 182 | 315 | 0 | [0,4,1,10,17,9,3,11,16,7,14,8,12,13] | 9.356e-16 | PASS | PASS |
| round-robin | 123 | 42 | 310 | 182 | 298 | 0 | [14,18,10,17,9,16,4,6,7,3,12,8,13,15] | 2.401e-15 | PASS | PASS |

Reordered-minus-original deltas by seed:

| Seed | CZ delta | Depth delta |
|---:|---:|---:|
| 42 | +22 | -24 |
| 7 | +21 | -11 |
| 123 | -17 | -138 |

The reordered schedule reduced depth in every pair but increased CZ count in
two of three pairs. Across the three seeds, average CZ count rose from 307 to
315.67, while average depth fell from 394.67 to 337.

Among circuits with the minimum observed 298 CZ gates, reordered seed 123 has
depth 310 versus original seed 42's 323: about a 4.0% depth reduction. It also
has more native R gates (815 versus 763), so this is not a uniform resource
improvement. Reordered seed 42 has the lowest depth, 299, but uses 320 CZ gates.

## Physical-run recommendation

**NO-GO for a 1,000-shot physical run at this checkpoint.**

The routing blocker is resolved. This recommendation now rests on the missing
calibrated/noisy-performance review, rather than unverified routing. The
reordering does not robustly reduce entangling-gate count, and the best count
remains 298 CZ. Circuit depth alone does not establish lower hardware error.
The report's calibration-set field is `"None"`; its target topology is
captured, but calibrated errors and durations were not assessed.

Shortlist reordered seed 123 for that review, alongside original seed 42.
A future physical-run approval must identify the frozen circuit and 1,000-shot
budget. No QPU run is authorized or performed by this audit or recommendation.
No quantum-advantage claim follows from these results.

## Preserved problem and logical evidence

- Exact/MILP checkpoint: **A+B+F+G+H**, **£5,000 spend**, **£950 minimum loss**.
- **14 logical qubits**, **182 CX** in both schedules.
- Logical depth **78 → 42** (79 → 43 including final measurements).
- Both emitted cost circuits verified over all **16,384 basis states**.
- Maximum phase difference between cost schedules: **5.274435786004485e-15**.
- Frozen parameters shared across schedules:
  gamma = -0.7439674535854706, beta = 0.3004519607226279.
- No Bell or five-invoice physical experiment was rerun.

## Evidence locations

The original complete report, two logical QPY files and six routed QPY files,
with source and circuit hashes, remain on LUMI:

```text
/scratch/project_462001763/cherryq/q20-ten/cost-order-audit-22664032/
```

The complete report is `routing-audit.json`. The LUMI checkout retains
`cherryq-q20-cost-audit-22664032.out` and `.err`. Failed audit evidence
remains in the separate 22663680 directory.

`summary.json` is a selected-field transcription of the completed report
observed through the LUMI shell; it is not a replacement for the original report.
