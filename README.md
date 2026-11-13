# Artifact of "Structured Coroutines for the Wafer-Scale Engine"

This is the artifact of the paper

> Tim Noack and Andreas Koch. Structured Coroutines for the Wafer-Scale Engine. In Proceedings of the 2027 IEEE/ACM International Symposium on Code Generation and Optimization (CGO '27), Salt Lake City, UT, USA, 2027.

The paper extends the Cerebras Software Language (CSL) with async functions and tasks, and evaluates two lowerings of them on the Wafer-Scale Engine (WSE): a stackless one and a stackful one.

## The compiler

The language extensions are part of csl++, an open-source language frontend for CSL, and its compiler csl-pp. The artifact does not contain a copy of the compiler: it installs the released package `csl-pp` from PyPI as it is, the same package that anyone uses to write programs with these extensions. The artifact itself only adds the benchmark programs, the drivers that measure them, and the data recorded for the paper.

- Documentation, with a tutorial and the reference of the language extensions: https://esa-tu-darmstadt.github.io/csl-pp/
- Source code: https://github.com/esa-tu-darmstadt/csl-pp
- The package: https://pypi.org/project/csl-pp/
- Editor support: the VS Code extension "Language Server for CSL": https://marketplace.visualstudio.com/items?itemName=esa-tu-darmstadt.csl-lsp

## What the artifact reproduces

The artifact reproduces the three results of the evaluation:

| Result | Content | Experiment |
| --- | --- | --- |
| Figure 7 | Suspend and resume latency as a function of await depth and suspension points | `suspend_resume` |
| Table 1 | Runtime, local tasks, instruction and data memory, and lines of code of seven SDK benchmarks: the vanilla CSL version and the versions with coroutines, stackless and stackful | `end2end` |
| Table 2 | Share of the runtime spent in the components of the stackful coroutine runtime | `profile` |

Every reviewer runs the experiments on the cycle-accurate simulator of the Cerebras SDK, which needs no Cerebras hardware. Reviewers with access to a CS-3 run them on the CS-3 as well, where the paper took its hardware measurements:

| Result | On the simulator, all reviewers | On a CS-3, in addition |
| --- | --- | --- |
| Figure 7 | the latencies on the WSE-2 and WSE-3 simulators | the latencies on a CS-3, which the figure shows |
| Table 1 | the rows "wse-2 sim" and "wse-3 sim" | the rows "wse-3 wafer" |
| Table 2 | the whole table | nothing: the table needs the simulator's instruction trace |

The paper states how closely the simulator matches the hardware: within 2 cycles for Figure 7 and 1 % in geometric mean for Table 1.

The evaluation has four steps:

1. Install the artifact.
2. Run the experiments on the simulator.
3. With access to a CS-3: run the experiments on the CS-3.
4. Generate the figure and the tables and compare them with the paper.

## Requirements

All reviewers need:

- Linux on x86-64.
- The Cerebras SDK, version 2.10.0, unpacked into a directory that contains `cslc`, `cs_python` and the SDK image `sdk-cbcore-2.10.0-*.sif`.
- Apptainer or Singularity, which the SDK uses to run its tools.
- Python 3.10 or newer with pip and venv.
- `patch`.
- About 20 GB of free disk space and 8 cores; the simulator experiments then take about 10 minutes.

Reviewers who also run on a CS-3 need in addition:

- Access to a CS-3 cluster through the Cerebras appliance SDK, or to a CS-3 system that this host reaches directly.
- Python 3.11 instead of a newer version if the CS-3 is reserved through the appliance SDK, because its Python packages exist for Python 3.11 only. Hosts that can reach a CS-3 cluster usually have Python 3.11 already, because the appliance SDK needs it there as well.

The Python package of the artifact installs everything else from PyPI, including the compiler csl-pp and, for the CS-3, the appliance SDK.

## Step 1: installation

Create a virtual environment. Reviewers who will also run on a CS-3 create it with Python 3.11 (`python3.11 -m venv ae-venv`), all others with any Python 3.10 or newer:

```bash
python3 -m venv ae-venv
. ae-venv/bin/activate
```

Install the artifact into it. Reviewers who will also run on a CS-3 install it with the extra `hardware`, which adds the appliance SDK 2.10.0 (the PyPI package `cerebras-sdk`):

```bash
pip install ./csl-pp-cgo27-artifact                # the directory of this repository
pip install "./csl-pp-cgo27-artifact[hardware]"    # instead, to also run on a CS-3
```

Then name the Cerebras SDK and check the installation:

```bash
export CSL_SDK_DIR=/path/to/cerebras-sdk-2.10.0
cgo27-ae check
```

`cgo27-ae check` reports every requirement it finds, and then whether the installation is ready to run on the simulator and on a CS-3.

pip installs the compiler csl-pp in the version v2.0.0 from PyPI. The archived version of the artifact on Zenodo also holds that wheel of csl-pp and the source of the same csl-pp release. To install the compiler from the archive instead of PyPI, point pip at the directory with the downloaded wheel:

```bash
pip install --find-links /path/to/downloaded/files ./csl-pp-cgo27-artifact
```

The other Python packages (numpy, matplotlib, pyelftools, tqdm) come from PyPI either way.

## Step 2: the simulator

This step is needed with or without access to a CS-3: only the simulator produces Table 2 and the simulator results of Table 1 and Figure 7, so step 3 does not replace it. All results go to `./cgo27-ae-output`, or to the directory given with `--out`. A quick run of a few configurations of every experiment first checks that everything works:

```bash
cgo27-ae simulate --quick
```

The full run then repeats all simulator experiments of the paper (`end2end`, `suspend_resume` and `profile`). It skips the configurations that the quick run has already measured.

```bash
cgo27-ae simulate
```

| Run | Time with 8 parallel jobs | Disk |
| --- | --- | --- |
| `cgo27-ae simulate --quick` | about 1 minute | 0.5 GB |
| `cgo27-ae simulate` | about 7 minutes | 16 GB |

Single experiments can be run alone, for example `cgo27-ae simulate end2end`, and `-j` sets the number of parallel compiles and simulations. A run that is interrupted continues where it stopped: configurations already in the results are skipped.

The stackless versions of GEMM and GEMV fail to compile, as the paper reports (Table 1, "Stackless GEMM and GEMV trigger a vendor-compiler bug"). The run lists them as expected failures and ends successfully. Any other failure is printed with its log, and the run then ends with exit code 1.

## Step 3: the CS-3

This step needs access to a CS-3. Without it, continue with step 4. It repeats the hardware measurements of Figure 7 and Table 1, in addition to the simulator results of step 2:

```bash
cgo27-ae hardware                    # reserves a CS-3 through the appliance SDK
cgo27-ae hardware --cmaddr IP:PORT   # instead, for a system that this host reaches directly
```

The artifact compiles every configuration for the full wafer on this host first, and then runs all of them on the system in one job. Without `--cmaddr`, the system is reserved with `SdkLauncher` of the appliance SDK, which the installation with the extra `hardware` provides: each configuration is staged into the job and run on the appliance. With `--cmaddr`, the runners run on this host against the given address, and the appliance SDK is not needed.

As for the paper, each configuration of Table 1 runs three times and each of Figure 7 once. `--reps` changes that, and `--quick` runs a few configurations only. Cycle counts on hardware vary slightly from run to run.

## Step 4: the report

```bash
cgo27-ae report
```

`cgo27-ae report` generates the figure and the tables in `cgo27-ae-output/report/` from all results so far, and compares them with the data recorded for the paper (`src/cgo27_artifact/paper/`). It prints the comparison and writes it to `report/comparison.txt`. It can be run at any time, for example after the quick run of step 2, and shows the platforms that have been measured:

- `table1.tex` is Table 1, with the simulator rows from step 2 and the "wse-3 wafer" rows from step 3.
- `figure7/<platform>/paper_suspend_resume.pdf` is Figure 7 for each measured platform, and `numbers.tex` holds the latencies the text cites. The figure in the paper is the platform `wse3-wafer`, which step 3 measures.
- `table2.tex` is Table 2.

Expected agreement with the recorded data:

| Quantity | Expected |
| --- | --- |
| Local tasks and lines of code | identical |
| Cycles, instruction and data memory of a configuration | within 5 % |
| Geometric means of Table 1 | within 5 % |
| Latencies of Figure 7 | within 3 cycles |
| Shares of Table 2 | within 1 percentage point |


## Contents

| Path | Content |
| --- | --- |
| `src/cgo27_artifact/cli.py` | The command-line tool `cgo27-ae` |
| `src/cgo27_artifact/bench/` | The benchmark drivers of the paper, one directory per experiment, and `wafer.py` for runs on a CS-3 |
| `src/cgo27_artifact/benchmarks/coro-benchmarks/` | The seven benchmarks written with coroutines, and the libraries they use, as measured for the paper |
| `src/cgo27_artifact/benchmarks/vanilla-patches/` | The timestamps added to the vanilla benchmarks, so that both versions measure the same interval |
| `src/cgo27_artifact/vendor/sdk-examples/` | The examples of the Cerebras SDK 2.10.0, unmodified, from github.com/Cerebras/sdk-examples |
| `src/cgo27_artifact/vendor/simtracer/` | The trace tool that attributes the simulator's instruction trace to functions (Table 2) |
| `src/cgo27_artifact/paper/` | The data recorded for the paper, including the CS-3 measurements |

Each driver directory has a README that describes its method in detail.

## License

The artifact is under the Apache License v2.0 with LLVM Exceptions (`LICENSE.TXT`). `NOTICE` lists the third-party code it contains.
