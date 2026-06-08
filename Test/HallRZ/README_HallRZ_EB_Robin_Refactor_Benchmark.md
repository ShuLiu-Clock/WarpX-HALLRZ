# HallRZ EB Robin Refactor Benchmark

This is the Stage-0 benchmark for the HallRZ EB Robin refactor plan.  Its
purpose is to freeze the current PIC behavior and timing before further solver
rework.

## Baseline State

Initial baseline record, captured on 2026-06-05:

```text
target tree: GPU
WarpX: /home/shuliu/work/cuda_warpx_202604/WarpX
environment: /home/shuliu/work/gpu_warpx2026/bin/activate
commit: fb17f2e422ca3d960ea4e4f3e5705f6903bc52de
```

The baseline is a dirty worktree, not a pristine upstream commit.  Dirty files
present before starting Stage 0 benchmark work included:

```text
00_init.sh
01_build.sh
AGENTS.md
Python/pywarpx/extensions/WarpXParticleContainer.py
Python/pywarpx/hallrz.py
Python/pywarpx/picmi.py
Source/EmbeddedBoundary/WarpXInitEB.cpp
Source/Evolve/WarpXEvolve.cpp
Source/FieldSolver/ElectrostaticSolvers/HallRZPoissonSolver.H
Source/FieldSolver/ElectrostaticSolvers/HallRZPoissonSolver.cpp
Source/Python/WarpX.cpp
Source/Python/pyWarpX.cpp
Test/HallRZ/README_HallRZ_Python_EB_Neumann.md
Test/HallRZ/picmi_hall_rz_core_smoke.py
```

Do not compare later stages against a different tree without recording the new
baseline explicitly.

## Benchmark Driver

```text
Test/HallRZ/picmi_hall_rz_pic_benchmark.py
```

The driver reuses `picmi_hall_rz_core_smoke.py` for geometry, particles, HallRZ
parameters, and Python EB Neumann data.  It adds:

- default `max_steps = 1000`;
- chunked wall-time timing;
- JSON summary output;
- finite checks for `rho_fp`, `phi_fp`, `Er_fp`, and `Ez_fp`;
- particle-count checks for electrons and protons.

The benchmark checks are intentionally observable PIC-level checks.  They do
not replace HallRZ C++ diagnostics such as MLMG residual, EB flux, particle
NaN/Inf, and charge diagnostics printed by the solver.

## GPU Baseline Command

Run in the GPU tree with the GPU environment activated:

```bash
source /home/shuliu/work/gpu_warpx2026/bin/activate
cd /home/shuliu/work/cuda_warpx_202604/WarpX
python3 Test/HallRZ/picmi_hall_rz_pic_benchmark.py \
  --max-steps 1000 \
  --chunk-steps 50 \
  --summary-json Test/HallRZ/hallrz_stage0_gpu_baseline.json \
  > Test/HallRZ/hallrz_stage0_gpu_baseline.log 2>&1
```

The command uses Python EB Neumann arrays by default.  For the scalar C++ EB
Neumann path, add:

```bash
--scalar-boundaries
```

## Recorded GPU Baseline

2026-06-05, GPU tree, 1 MPI rank, Python EB Neumann arrays:

```text
summary: Test/HallRZ/hallrz_stage0_gpu_baseline.json
log:     Test/HallRZ/hallrz_stage0_gpu_baseline.log
steps:   1000
total_step_wall_time_s: 193.8262595169872
avg_step_wall_time_s:   0.1938262595169872
chunk_wall_time_s:      min=9.41557318699779, p50=9.705075969497557, p95=9.913781455496792, max=10.09750188900216
particles:              electrons=1800, protons=1800
checks.pass:            true
```

Final field checks:

```text
rho_fp: finite=true, nan=0, inf=0, l2=1.0191540820196311e-10, max_abs=2.6268995028890336e-10
phi_fp: finite=true, nan=0, inf=0, l2=0.17949536958085804, max_abs=0.2615878917009116
Er_fp:  finite=true, nan=0, inf=0, l2=0.3829273544985057, max_abs=5.146809525514579
Ez_fp:  finite=true, nan=0, inf=0, l2=1.9872186074245926, max_abs=9.788142020226944
```

Final HallRZ diagnostics reported `N_NaN=0`, `N_Inf=0`,
`N_particle_NaN=0`, `N_particle_Inf=0`, no out-of-domain particles, no
inside-covered-EB particles, and no recorded warnings.

## CPU MPI Baseline Command

CPU MPI validation is a required stage gate, with no more than four CPU cores
total.  The CPU tree uses the same benchmark semantics, but keeps HallRZ
parameters on its existing `pywarpx.warpx.hall_rz_*` ParmParse path:

```bash
source /home/shuliu/work/cpu_warpx2026/bin/activate
cd /home/shuliu/work/warpx_202604/WarpX
OMP_NUM_THREADS=1 mpirun -np 4 -bind-to none python3 \
  Test/HallRZ/picmi_hall_rz_pic_benchmark.py \
  --max-steps 1000 \
  --chunk-steps 50 \
  --summary-json Test/HallRZ/hallrz_stage0_cpu_np4_baseline.json \
  > Test/HallRZ/hallrz_stage0_cpu_np4_baseline.log 2>&1
```

`--warpx-max-grid-size 32` is part of the CPU MPI benchmark setup.  It gives
`np=4` enough cell boxes and avoids the WarpX high-priority performance warning
for idle MPI ranks.  It does not change the physical domain, physical boundary
data, EB boundary data, or HallRZ geometry.

## Recorded CPU MPI Baseline

2026-06-05, CPU tree, `np=4`, `OMP_NUM_THREADS=1`, Python EB Neumann arrays:

```text
summary: /home/shuliu/work/warpx_202604/WarpX/Test/HallRZ/hallrz_stage0_cpu_np4_baseline.json
log:     /home/shuliu/work/warpx_202604/WarpX/Test/HallRZ/hallrz_stage0_cpu_np4_baseline.log
steps:   1000
total_step_wall_time_s: 103.77750101601123
avg_step_wall_time_s:   0.10377750101601123
chunk_wall_time_s:      min=4.96324965399981, p50=5.190010416499717, p95=5.333419901104935, max=5.3434430490015075
particles:              electrons=1800, protons=1800
checks.pass:            true
```

Final field checks:

```text
rho_fp: finite=true, nan=0, inf=0, l2=1.0191965679539766e-10, max_abs=2.672453342393127e-10
phi_fp: finite=true, nan=0, inf=0, l2=0.17949499128549903, max_abs=0.26158853983579483
Er_fp:  finite=true, nan=0, inf=0, l2=0.3829279956652793, max_abs=5.147513115993152
Ez_fp:  finite=true, nan=0, inf=0, l2=1.987220635798978, max_abs=9.788140705269118
```

Final HallRZ diagnostics reported `N_NaN=0`, `N_Inf=0`,
`N_particle_NaN=0`, `N_particle_Inf=0`, no out-of-domain particles, no
inside-covered-EB particles, and no recorded warnings.

## Acceptance Fields

The JSON summary contains:

```text
init_wall_time_s
total_step_wall_time_s
avg_step_wall_time_s
chunk_wall_time_s.{min,max,mean,p50,p95}
fields.<rho_fp|phi_fp|Er_fp|Ez_fp>.{finite,nan_count,inf_count,min,max,max_abs,l2}
particles.<electrons|protons>
checks.{fields_finite,particles_positive,pass}
```

Stage gate acceptance requires:

- `checks.pass == true`;
- no solver-reported NaN/Inf in stdout;
- HallRZ particle diagnostics report no particle NaN/Inf;
- no unexpected particle-loss or charge jump in diagnostics;
- Stage 1 and later total wall time does not regress by more than 10% against
  this Stage-0 baseline unless explicitly accepted.

## Smoke Variant

For fast local feedback, run:

```bash
python3 Test/HallRZ/picmi_hall_rz_pic_benchmark.py \
  --max-steps 1 \
  --chunk-steps 1 \
  --summary-json hallrz_stage0_smoke.json
```

This smoke only validates the benchmark harness and basic HallRZ path.  It is
not a stage gate.
