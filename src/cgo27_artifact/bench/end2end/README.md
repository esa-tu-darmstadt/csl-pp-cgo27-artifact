# End-to-End Benchmarks

For each SDK benchmark written with coroutines
(`../../benchmarks/coro-benchmarks/<name>/`) and its vanilla device-controlled
baseline, compile and run under three configurations — vanilla cslc, our
stackful back-end (configuration `fiber`), our stackless back-end
(configuration `trampoline`) — on wse2 and wse3. The vanilla baselines are the
SDK's examples (`../../vendor/sdk-examples/`) with the timestamps of
`../../benchmarks/vanilla-patches/`, prepared in the work directory.

Results go to `<out>/end2end/results/` and build directories to
`<out>/end2end/work/`, where `<out>` is `$CGO27_AE_OUT` (see `../common.py`);
`results/` and `work/` below refer to these.

Two targets: `sim` runs the SDK's fabric simulator, `wse` runs a real CS-3.
Hardware reuses the benchmark's own `commands_wse3.sh` unchanged apart from
two edits to our temp copy — the compile is retargeted to the full wafer
(`--fabric-dims=762,1172`), and the runner is taken out of the script and
run on the system with `--cmaddr` — so the same runner script drives both
targets. See `../wafer.py` for how the system is obtained.

Per (benchmark, config, arch) we record:

- **cycles_send** — wall-clock cycles parsed from the runner's stdout.
  The only column that differs between `sim` and `wse`; everything below is
  a property of the compile and is measured from the local ELFs either way
- **tasks (local/control/data)** — max-per-kind across PEs, from the
  `count-tasks` pass running inside `csl-pp` (stderr; the driver passes
  `--count-tasks`)
- **prog_size** / **globals_size** — max `.text` / sum of allocated
  non-code sections across all PE ELFs (via `pyelftools`)
- **loc** — transitive `@import_module` / `@set_tile_code` closure from
  the entry layout, excluding system imports, `benchmark-libs/`, and the
  shared `blas.csl` utility. The license header at the top of the SDK's
  files is not counted, so the vanilla programs count as the SDK ships them

## Compile times

`bench.py --timing` does not run anything. It compiles every configuration
`--reps` times, strictly one at a time, and records into
`results/compile_times.csv` the wall-clock time of the whole `cslc` line of
the benchmark's `commands_wseN.sh` (`total_s`) and of the real cslc runs
inside it (`cslc_s`, `cslc_runs`, and each run separately, in order, in
`cslc_run_s`). A timing shim stands in for cslc: on `PATH` for vanilla, and as
csl-pp's `--cslc` for ours. The stackful back-end runs cslc twice (the first
round measures the fiber stack sizes), so its `cslc_s` covers both rounds;
`total_s - cslc_s` is the time csl-pp itself takes.
`target=wse` compiles for the full wafer, as the hardware sweep does, but
reserves nothing. `tabulate.py` adds the minimum over the repetitions to the
table: csl-pp's time in seconds and cslc relative to vanilla.

```bash
python3 bench.py --timing --targets sim wse --reps 3
```

## Scripts

- `bench.py` — sweeps the matrix, writes `results/data.csv`. Idempotent
  (existing rows skipped unless `--force`). Use `-j N` for parallelism.
- `tabulate.py` — renders `results/data.csv` as `results/table.tex` (a
  full-width `table*` with vanilla absolute and SF/SL as ×ratios, plus a
  per-arch geometric-mean summary row). With `results/compile_times.csv`
  present, the table gains the csl-pp and cslc time columns.
- `plot.py` — renders `results/scatter.pdf`, the LOC-vs-cycles trade-off.

## Vanilla overrides

For benchmarks that ship both a host-RPC variant and a device-controlled
variant, we patch our temp copy of `commands_wse{2,3}.sh` to point at
the device variant so the comparison is device-vs-device. The SDK
sources are not modified. See `VANILLA_DEVICE_VARIANT` in `bench.py`.

## Typical run

`cgo27-ae simulate end2end`, `cgo27-ae hardware end2end` and
`cgo27-ae report` run this driver and `tabulate.py`. Run directly:

```bash
python3 bench.py -j 8        # ~10 min wall time with 8 cores
python3 tabulate.py
python3 plot.py
```

On hardware — wse3 only, as the paper's measurements were taken on a CS-3.
Every configuration is compiled first; then the runs execute one after
another in a single job, and `--reps` takes several samples per
configuration because hardware cycle counts vary between runs:

```bash
python3 bench.py --targets wse --archs wse3 --reps 3
```
