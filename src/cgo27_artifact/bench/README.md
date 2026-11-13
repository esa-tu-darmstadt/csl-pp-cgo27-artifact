# Benchmark drivers

The drivers that produced the figure and the tables of the paper's
evaluation. One subdirectory per experiment; each documents itself.

| Subdirectory      | Result   | What it measures                                                |
| ----------------- | -------- | --------------------------------------------------------------- |
| `suspend_resume/` | Figure 7 | `suspend`/resume latency vs. await-chain depth                  |
| `end2end/`        | Table 1  | Cycles, tasks, code/globals size, LOC across the SDK benchmarks |
| `profile/`        | Table 2  | Share of the runtime spent in the coroutine runtime             |

`common.py` holds the paths they share. The drivers write below
`$CGO27_AE_OUT` (default `./cgo27-ae-output`), one directory per experiment
with `results/` and `work/`. They default the SDK to `$CSL_SDK_DIR` (falling
back to `~/cs_sdk-2.10.0`) and `csl-pp` to the one installed with the
artifact; both are overridable per run.

`suspend_resume` and `end2end` take `--targets sim wse`. `wse` runs on a real
CS-3 through `wafer.py`, which is shared between them: every configuration is
compiled first, then all runs execute in one job, through the appliance SDK
or, with `--cmaddr`, directly against a system this host reaches. The
benchmarks' own runner scripts drive the system with `--cmaddr` exactly as
they drive the simulator.

`cgo27-ae` (`../cli.py`) runs the drivers and the scripts that render their
results; see the README of the artifact.
