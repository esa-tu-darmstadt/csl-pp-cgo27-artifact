# Coroutine Runtime Profile

Attributes every dispatched instruction of a benchmark run to the fiber
runtime (the runtime of the stackful backend, configuration `fiber`), the
task allocator, an `@await_parallel` wrapper, the system modules, or user
code. The benchmark set is the one used by `../end2end`. Table 2 of the paper
shows the result.

The result says where a coroutine program's cycles go inside the code the
lowering emits. It is not a measurement of what coroutines cost: a CSL program
expressing the same asynchrony by hand pays for task dispatch, task-id
management and activation as well, and nothing here is measured against such a
baseline.

## Scope

Fiber backend only. The stackless backend has no separable runtime: its
generated code is distributed through the state machines the transform emits,
so there is no symbol to attribute cycles to. `csl-pp` rejects
`--coro-instrument` for `--coro-mode=trampoline`.

Simulator only. Attribution is derived from the simulator's CTF instruction
trace, which has no hardware equivalent.

## Relationship to the end2end sweep

Builds are compiled with `--coro-instrument=await-parallel`. This gives each
`@await_parallel` site an ordinary named function before backend optimization;
it is deliberately neither `inline` (which would let the CSL frontend erase
its debug scope) nor `noinline`. LLVM normally inlines that wrapper and emits
a `DW_TAG_inlined_subroutine` range for it. Runtime functions are ordinary as
well and are attributed through the same DWARF ranges.

The wrapper changes the input to backend optimization even when LLVM inlines
it away, so zero timing impact is verified rather than assumed. The profile
and end2end sweeps write to separate result files; `results/runs.csv` records
the profiled cycle count and `summarize.py` reports it beside the corresponding
plain value.

## Method

Per-benchmark configuration — source layout, build commands, and the matching
vanilla variant — is reused from `../end2end/bench.py`. Only the work tree and
the result files are separate. Instrumentation flags reach `csl-pp` through
`$CSL_FLAGS`, which it splices into its own argument vector; neither the
integration `cslc` wrapper nor the end2end driver requires modification.

Instructions are attributed by instruction pointer against each PE's ELF,
using the trace tool in `../../vendor/simtracer`. Physical calls and fiber
resumes still come from the symbol table; optional DWARF expansion adds the
LLVM-inlined function chain without inventing physical calls or returns. Flat
counts go to the innermost function, while the nested cycle view retains the
whole coroutine-generated suffix of the logical stack.

That suffix is deliberately unbroken. If an `@await_parallel` wrapper invokes
a synchronous user callee, the user frame ends the suffix, including when the
callee is itself inlined. Its work is therefore excluded from the wrapper. If
that callee calls an inlined runtime function such as `suspend`, only the
runtime function below the user frame is charged, not the wrapper above it.

## Aggregation

Percentages are computed per PE and then aggregated across PEs (median,
minimum, maximum, mean). A figure pooled over all PEs would be dominated by
the longest-running PE, and in most of these benchmarks the coroutine work is
not distributed uniformly across the rectangle.

A PE is included only if it dispatched at least one coroutine-runtime
instruction. Runs activate PEs that execute system-module code exclusively,
such as memcpy relays; including them would bias every percentage downward.

## Usage

`cgo27-ae simulate profile` and `cgo27-ae report` run the three scripts below.
Run directly, with the Python the artifact is installed in; the driver
invokes the SDK's container wrappers itself for compilation and simulation:

```bash
python3 bench.py --archs wse3
python3 summarize.py
python3 table.py --arch wse3
```

Results go to `<out>/profile/results/` and build directories to
`<out>/profile/work/`, where `<out>` is `$CGO27_AE_OUT` (see `../common.py`).

`bench.py` skips `(benchmark, arch)` pairs already present in the CSV;
`--force` re-runs all of them. Traces occupy roughly 150 MB per configuration
and are deleted once consumed. `--keep-traces` retains them for inspection
with `../../vendor/simtracer/simtrace.py`.

## Outputs

```
results/data.csv     raw counts: (benchmark, arch, tile, function, instructions)
results/runs.csv     per run: instrumented cycles, PE count, event count
results/summary.csv  per category and per runtime function:
                     pct_median, pct_min, pct_max, pct_mean
```

`data.csv` holds the measurement and remains uncategorised. The classification
is applied by `summarize.py::classify`, so the categories can be revised
without repeating the sweep.
