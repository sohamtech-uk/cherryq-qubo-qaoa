# Running CherryQ on LUMI

The hackathon CSC project shown in MyCSC is:

- Project number: `462001763`
- Unix/Slurm account: `project_462001763`

The allocation visible in MyCSC is large enough for meaningful experiments. Treat it as a shared hackathon allocation unless the organisers explicitly say otherwise.

## Best use of LUMI for CherryQ

The most useful immediate target is **LUMI-C CPU compute** for the QAOA research matrix:

- repeated seeds
- standard vs warm-start-X vs matched warm-start
- p=1/p=2
- all cash scenarios
- future penalty and larger-problem sweeps

Do **not** assume LUMI-G will accelerate the current Qiskit path. LUMI-G uses AMD MI250x/ROCm GPUs, while the standard Qiskit Aer GPU packages target NVIDIA CUDA. Use the GPU allocation later for ROCm-native AI workloads such as PyTorch cash-flow/payment-probability models.

## 1. Put the repository on LUMI

From a LUMI login node:

```bash
cd /projappl/project_462001763
git clone git@github.com:sohamtech-uk/cherryq-qubo-qaoa.git
cd cherryq-qubo-qaoa
```

If GitHub SSH is not configured on LUMI, transfer the repository from your laptop instead. Do not put access tokens in the repository.

## 2. Build a LUMI-C container

LUMI recommends containerized Python environments instead of large pip/Conda environments directly on Lustre.

From the repository root:

```bash
chmod +x lumi/build_lumi_c_container.sh
lumi/build_lumi_c_container.sh
```

This builds:

```text
/projappl/project_462001763/cherryq/cherryq-lumi-c.sif
```

with `cotainr --system=lumi-c`.

## 3. Submit the pilot array

Submit from the repository root so `SLURM_SUBMIT_DIR` resolves to the checkout:

```bash
sbatch lumi/cpu_qaoa_array.sbatch
```

Default pilot allocation per array task:

- account: `project_462001763`
- partition: `small`
- 16 CPU cores
- 32 GiB RAM
- 30 minute walltime
- 40 shards, with at most 10 running concurrently

The matrix is:

```text
4 cash scenarios
x 3 QAOA variants
x 2 depths
x 100 seeds
= 2,400 QAOA runs
```

The Python harness uses one worker process per allocated CPU core. BLAS/OpenMP threading is pinned to one thread per process to avoid oversubscription.

Results are written to:

```text
/scratch/project_462001763/cherryq/results/<array-job-id>/
```

## 4. Monitor utilisation

```bash
squeue -u "$USER"
sacct -j <job-id> --format=JobID,State,Elapsed,AllocCPUS,MaxRSS
seff <job-id>
sinfo -s
scontrol show partition small
```

Start small, inspect utilisation, then increase the matrix only if the jobs are actually using the requested resources efficiently.

## 5. Aggregate the shards

After the array has finished:

```bash
sbatch lumi/aggregate_results.sbatch <array-job-id>
```

The aggregate report is written to:

```text
/scratch/project_462001763/cherryq/results/<array-job-id>/aggregate.json
```

It reports mean and standard deviation of optimal-plan probability, feasibility, sampled gap, runtime and circuit complexity by scenario/method/depth.

## 6. When to move beyond the small partition

The current five-invoice model is still small. LUMI is designed for scale-out workloads, and the `small` partition is the correct place for this pilot.

The next scientifically useful step is to create larger generated invoice problem families where:

- the continuous relaxation is not integral
- invoice count increases
- dependency graph density varies
- budget tightness varies
- qubit count grows

Only move to the full-node `standard` partition after an individual workload can use a 128-core LUMI-C node efficiently.

## 7. GPU plan

LUMI-G should be used for workloads that are actually ROCm-capable.

For CherryQ, the sensible GPU track is:

1. keep the gradient-boosted classifier as the classical baseline
2. add a PyTorch payment-probability model
3. run it in the LUMI AI Factory `lumi-multitorch` container on `small-g`
4. compare calibration and temporal holdout performance
5. scale to larger synthetic or permissioned transaction data

Do not spend GPU hours on the current Qiskit simulator simply to say we used a GPU.

## Resource discipline

The MyCSC quota is substantial, but LUMI is shared infrastructure. The aim is useful experiments, not quota consumption. Use the smallest efficient partition, measure utilisation, and scale only when the workload justifies it.
