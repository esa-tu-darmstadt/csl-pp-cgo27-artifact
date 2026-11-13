# Suspend / Resume Latency Microbenchmark

Measures the cycle latency of `suspend` and the matching resume on both
backends — stackful (configuration `fiber`) and stackless (configuration
`trampoline`) — as a function of await-chain depth `d` **and** states per
coroutine `K`. Runs on WSE-2 and WSE-3 via the SDK simulator, and on a real
CS-3 (`--targets wse`, WSE-3 only). Figure 7 of the paper shows the CS-3
results.

Results go to `<out>/suspend_resume/results/` and build directories to
`<out>/suspend_resume/work/`, where `<out>` is `$CGO27_AE_OUT` (see
`../common.py`); `results/` and `work/` below refer to these.

## What is measured

For each `(d, K, backend, arch)` combination the bench renders a single PE
program with:

- A top-level `async task f_main` that awaits a chain of `d` nested
  `async fn` levels.
- Every function on the chain (intermediates *and* leaf) has `K` `suspend`
  points. The leaf's last suspend is the **timed** one; the `K-1` preceding
  suspends are self-activated dummies (`@activate(@current_task()); suspend;`)
  that force the lowered state machine to have `K+1` arms instead of
  collapsing to a trivial dispatcher.
- A `witness` task that fires *between* the suspend and the resume, used to
  split the round trip into two halves.

Three timestamp-counter samples bracket the measurement:

| marker | location                                    |
| ------ | ------------------------------------------- |
| `M0`   | inside leaf, just before the timed `suspend` |
| `M1`   | inside `witness`                            |
| `M2`   | inside leaf, just after resume              |

Per run:

- `suspend_cycles = M1 - M0` (leaf yield → witness)
- `resume_cycles  = M2 - M1` (witness → leaf resumed)

The markers are copied back to the host after the run. Everything happens on
one PE and therefore one clock, so unlike the multi-PE `end2end` benchmarks
no reference-clock correction is needed.

### Subtracting the observer

The intervals above do not contain only the primitive being measured. They
also contain the marker reads themselves, the task dispatch to the witness,
and the witness's prologue and epilogue. At these magnitudes that is not a
rounding error: on WSE-3 a raw `M1 - M0` for the stackless backend is 14
cycles, of which 12 are the measurement rig.

Two things are done about it.

**The marker is made cheap.** A marker is one `@get_config` of the timestamp
counter's low half into a scalar — *not* the SDK's `<time>.get_timestamp()`,
which delivers the full 48 bits as three config reads plus three array
stores. Measured back to back, the SDK helper costs 10 cycles and a
single-word read costs 2. The counter is zeroed at `enable_tsc()` and the
longest interval in the sweep is under 400 cycles, so the low 16 bits are
sufficient; `bench.py` still differences them modulo `2**16`.

**The remainder is measured and subtracted.** `template/null_pe.csl` is a
null configuration: the same markers, the same `witness` task, the same
activation chain, but no coroutine anywhere — its producer leaves by plain
task return rather than by suspending. Whatever it reports is apparatus.
`bench.py` runs it once per `(arch, target)` and records it as a
`backend=null` row at `depth=0, k=0`; `plot.py` subtracts it from every other
row. It is re-measured per target rather than assumed, because there is no
reason for task dispatch to cost the same on a simulator and on silicon.

On the SDK 2.10 simulator the harness costs 6 / 7 cycles (suspend / resume
leg) on WSE-2 and 4 / 7 on WSE-3.

Subtracting a constant leaves curve **slopes** untouched, so `α(K)` is the
same either way, and the crossover depth is invariant too (the constant
cancels in its algebra). What the correction does change is every absolute
cycle count and every intercept `β`.

The raw markers stay in `results/data.csv`, so the correction can be revised
without re-running the sweep.

> Earlier revisions took the markers from `simprt` output in `sim.log`,
> which the cycle-accurate simulator annotates with a cycle count. That only
> exists under the simulator — there is no such log on hardware — so the
> measurement moved to the timestamp counter, which both targets have. The
> constant overhead differs slightly from the `simprt` version, so absolute
> numbers are not comparable with results recorded before that change.

## Findings (SDK 2.10 simulator, harness subtracted)

- **Stackful suspend**: constant in `d`, essentially constant in `K`
  (WSE-3 `20-22`, WSE-2 `20-27` cycles).
- **Stackful resume**: constant in both (WSE-3 `33`, WSE-2 `46` cycles).
- **Stackless suspend**: constant in both, and *tiny* — `2` cycles on WSE-3,
  `1` on WSE-2. The run-to-completion task model means there are no live
  frames to release on yield regardless of nesting, so a stackless suspend is
  barely distinguishable from a bare hardware task exit. This number is at
  the resolution floor and only became visible once the observer was
  subtracted; before that it read as `14` / `15`.
- **Stackless resume**: linear in `d`, sub-linear in `K`. Linear fits of
  `R(d) = α(K)·d + β(K)` on WSE-3:

  | K | α [cyc/level] | β [cyc] |
  |---|---------------|---------|
  | 1 | 11            | 12      |
  | 2 | 20            | 11      |
  | 4 | 32            | 11      |
  | 8 | 44            | 12      |

  α grows sub-linearly because cslc lowers each `switch` to a balanced
  compare-and-branch tree (the WSE has no indirect-jump primitive), so
  reaching the deepest arm of a `K+1`-wide switch costs
  `⌈log₂(K+1)⌉` taken conditional branches × 5-cycle pipeline bubble each.
  β stays nearly constant because leaf and intermediates share the same
  per-frame dispatch cost in the symmetric design — the K-dependent leaf
  contribution gets absorbed into α alongside the intermediates.

The stackless round trip crosses the stackful one at `d = 4` on WSE-3 and
`d = 5` on WSE-2 (at `K = 1`). That crossover is unaffected by the harness
subtraction — the constant cancels — so it is the one headline number that
did not move when the correction landed.

## Usage

`cgo27-ae simulate suspend_resume`, `cgo27-ae hardware suspend_resume` and
`cgo27-ae report` run this driver and `plot.py`; csl-pp is the one installed with the artifact. Run directly:

```bash
# Drive the full sweep on the simulator (80 configs by default, 16 parallel
# workers). Writes results/data.csv.
python3 bench.py

# The same sweep on a real CS-3. Hardware is wse3-only and runs serially in
# a single job; --reps takes several samples per config because hardware
# timings vary.
python3 bench.py --targets wse --archs wse3 --reps 1

# Custom sweep:
python3 bench.py --depths 1 2 4 --ks 1 8 \
                 --backends fiber trampoline --archs wse3 \
                 --jobs 8

# Regenerate figures and numbers for all three platforms
python3 plot.py

# Or regenerate only the real WSE-3 hardware results
python3 plot.py --platforms wse3-wafer
```

Hardware sweeps that reserve a CS-3 must be driven by the Python
environment of the appliance SDK, the only one that can reserve a system;
`--cmaddr` runs on a system that this host reaches directly instead (see
`../wafer.py`):

```bash
python bench.py --targets wse --archs wse3                    # in the appliance SDK's environment
python3 bench.py --targets wse --archs wse3 --cmaddr IP:PORT
```

`bench.py` builds every configuration *before* obtaining the system, so the
job covers the runs alone. On the CS-3 a run costs ~25 s almost
independently of the program — that is appliance connection setup, not ELF
load — against ~7 s to compile one configuration off the wafer.

`bench.py` is idempotent: it skips `(d, k, backend, arch, target, rep)` rows
already in the CSV. Pass `--force` to re-run everything. `--keep-work` preserves the
per-config build dirs under `work/` (useful for inspecting ELFs or
sim.log).

## Outputs

```
results/data.csv                       combined raw measurements: (d, k,
                                       backend, arch, target, M0, M1, M2, ...)
                                       plus one backend=null baseline row per
                                       (arch, target)
results/<platform>/numbers.tex             paper constants
results/<platform>/paper_suspend_resume.*  single-column paper figure
results/<platform>/opt1_split.*            suspend/resume split figure
results/<platform>/opt2_resume.*           resume-only figure
results/<platform>/opt3_total.*            round-trip-only figure
results/<platform>/supp_alpha_beta_*.*     α(K), β(K) linear-fit figures
```

Here `<platform>` is one of `wse2-sim`, `wse3-sim`, or `wse3-wafer`; every
folder contains only measurements from that exact architecture and target.

Every cycle count in `numbers.tex` has the harness subtracted; the
harness constants themselves are emitted alongside as
`\nWseThreeHarnessSuspend{}` / `\nWseThreeHarnessResume{}` so the paper can
state what was removed.

The applicable platform-specific `numbers.tex` files are regenerated on each
`plot.py` run. For WSE-3 hardware results, the paper should `\input{}`
`results/wse3-wafer/numbers.tex`; simulator numbers live separately and cannot
overwrite it. The prose references macros such as
`\nWseThreeStackfulSuspend{}` and `\nWseThreeAlphaKOne{}`. LaTeX
`\newcommand` names contain no digits — arch and K indices are spelled out
(`WseTwo` / `WseThree`, `One` / `Two` / `Four` / `Eight`). The macro names are
target-neutral because each input file contains exactly one platform. The
first measured K whose fitted crossover is below depth one is emitted as
`\nWseThreeCrossoverKBelowDepthOne{}`.

Each platform writes its own directory because the series are keyed by arch
and backend alone; mixing simulator and wafer rows would silently let one
overwrite the other.
