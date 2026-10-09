# Local validation evidence, 9 October 2026

Source implementation: `02feb6addb62f9a3419c8f319fd0fe90825953a3`.
The report was produced with a clean source worktree, Qiskit 2.5.2, NumPy 2.3.5,
and SciPy 1.17.0. It records source SHA-256 and both frozen circuit SHA-256 values.

Commands completed locally:

- Full pytest suite: **52 passed** (7.94 s). One existing COBYLA warning adjusts
  the intentionally tiny smoke test's maximum evaluations to the minimum.
- The supplied dependency-free cost audit was reproduced.
- `cherryq.scaling` reconfirmed the ten-invoice exact/MILP optimum at £950 loss.
  Its five-invoice portion was classical only; no hardware was rerun.
- `cherryq.hardware.q20_routing_audit --logical-only`: all 16,384 basis labels
  restored; maximum phase error between schedules 4.47e-15; maximum full-state
  amplitude error after global-phase alignment 9.93e-17.
- Slurm wrapper passed `bash -n`; source changes passed `git diff --check`.

Logical depths exclude measurement: 78 and 42; including measurement: 79 and
43. Both have 182 CX gates. The new optimizer's frozen angles are in the report;
this run does not reproduce the archived preparation's exact angles.

**Physical routing is unverified. NO-GO remains. No QPU job was submitted.**

The QPY files are logical circuits, not physically routed or approved circuits.
Use the routing audit wrapper documented in `COST_ORDER_ROUTING.md` to perform
the actual Q20 target comparison. QPY loading requires a compatible Qiskit/QPY
version; LUMI's wrapper creates its own shared preparation in that environment.
