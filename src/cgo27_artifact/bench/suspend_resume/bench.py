#!/usr/bin/env python3
"""Driver for the suspend/resume latency microbenchmark.

Sweeps (depth, k, backend, arch, target); for each tuple renders pe.csl from
the template, compiles via csl-pp, runs it, and appends the three timestamp
markers to results/data.csv.

Each (arch, target) also gets one `backend=null` row from template/null_pe.csl:
the identical observer harness with no coroutine in it. plot.py subtracts that
row from every other row, so what the paper reports is the cost of a suspend
and of a resume rather than the cost of watching one happen.

`target=sim` runs the SDK's cycle-precise simulator. `target=wse` compiles the
same source for the full WSE-3 wafer and runs the *same* runner on a CS-3
with --cmaddr: all configurations are compiled first, then run one after
another in a single job (see ../wafer.py). Hardware is wse3-only: the paper's
measurements were taken on a CS-3.

Because this sweep is many short runs, it is the case that benefits most from
one job: scheduling a job costs minutes, each run costs seconds.

The driver is idempotent: it skips any row already in the CSV. Delete
results/data.csv (or pass --force) to re-run.
"""

import argparse
import csv
import os
import re
import shutil
import string
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "template"

sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402
import wafer  # noqa: E402

RESULTS = common.results_dir("suspend_resume")
WORK = common.work_dir("suspend_resume")

# Overridable on the command line.
DEFAULT_SDK = common.DEFAULT_SDK
DEFAULT_CSL_PP = common.default_csl_pp()

CSV_FIELDS = ["depth", "k", "backend", "arch", "target", "rep",
              "M0", "M1", "M2",
              "suspend_cycles", "resume_cycles", "round_trip"]

# The null configuration is recorded as an ordinary row under this backend
# name, at depth 0 / k 0. It measures the harness -- markers, task dispatch,
# witness prologue and epilogue -- with no coroutine in the program at all;
# plot.py subtracts it from every real row. It is swept per (arch, target)
# because there is no reason for dispatch to cost the same on a simulator and
# on silicon. See template/null_pe.csl.
NULL_BACKEND = "null"


def find_sif(sdk: Path) -> Path:
    """The SDK's cbcore .sif. Its filename carries the SDK build stamp, so it
    is globbed rather than spelled out."""
    sifs = sorted(sdk.glob("*.sif"))
    if not sifs:
        raise RuntimeError(f"no .sif found in SDK directory {sdk}")
    return sifs[0]


def _leaf_body(k_suspends: int) -> str:
    """K-1 self-activated dummies followed by 1 timed suspend.

    Per-state side effects are unique (`acc += <i+1>`) so cslc can't merge
    adjacent states. The timed suspend samples the cycle counter into mark0
    before yielding and into mark2 after resuming.
    """
    lines = []
    for i in range(k_suspends - 1):
        lines.append(f"    @activate(@current_task()); suspend; acc += {i + 1};")
    if lines:
        lines.append("")
    lines.append("    // === MEASURED SUSPEND (timed, the K-th and last) ===")
    lines.append("    @activate(WITNESS);")
    lines.append("    mark0 = @get_config(TSC_BASE);")
    lines.append("    suspend;")
    lines.append("    mark2 = @get_config(TSC_BASE);")
    lines.append(f"    acc += {100 + k_suspends};")
    return "\n".join(lines)


def _level_fn(level: int, callee: str, k_suspends: int) -> str:
    """Generate one intermediate level: K-1 dummies then @await(callee)."""
    lines = [f"noinline async fn level_{level}() void {{"]
    lines.append(f"    var acc_{level}: u16 = sink_{level};")
    for i in range(k_suspends - 1):
        lines.append(
            f"    @activate(@current_task()); suspend; "
            f"acc_{level} += {i + 1};"
        )
    lines.append(f"    @await({callee});")
    lines.append(f"    sink_{level} = acc_{level};")
    lines.append("}")
    return "\n".join(lines)


def render_pe(depth: int, k_suspends: int) -> str:
    """Render pe.csl source for the given chain depth and uniform K.

    depth=1 → f_main awaits leaf() directly (no intermediate level).
    depth=N → f_main awaits level_{N-1} → ... → level_1 → leaf.

    k_suspends = K applies to EVERY function on the chain:
      - leaf gets K-1 self-activated dummy suspends followed by 1 timed suspend.
      - each level_k gets K-1 self-activated dummy suspends followed by 1 await.

    K=1 means no dummies (base case, single suspend / single await per function).
    """
    if k_suspends < 1:
        raise ValueError("k_suspends must be >= 1")

    tmpl = (TEMPLATE / "pe.csl.tmpl").read_text()

    sink_decls = ["export var sink_leaf: u16 = 0;"]
    for k in range(1, depth):
        sink_decls.append(f"export var sink_{k}: u16 = 0;")

    if depth == 1:
        levels = "// (no intermediate levels — f_main awaits leaf directly)"
        top_call = "leaf()"
    else:
        fns = []
        for k in range(1, depth):
            callee = "leaf()" if k == 1 else f"level_{k - 1}()"
            fns.append(_level_fn(k, callee, k_suspends))
        levels = "\n\n".join(fns)
        top_call = f"level_{depth - 1}()"

    return string.Template(tmpl).substitute(
        DEPTH=str(depth), K=str(k_suspends),
        SINK_DECLS="\n".join(sink_decls),
        LEAF_BODY=_leaf_body(k_suspends),
        LEVELS=levels,
        TOP_CALL=top_call,
    )


def compile_config(depth: int, k_suspends: int, backend: str, arch: str,
                   csl_pp: Path, sdk: Path, target: str = "sim",
                   rep: int = 0) -> tuple[Path, float]:
    """Build one configuration. Returns its work dir and the compile seconds.

    `backend == NULL_BACKEND` builds the null configuration instead of a
    rendered chain; depth and k are ignored (and recorded as 0).

    Nothing here touches the wafer, which is the point: hardware sweeps
    compile every configuration up front and only then take a reservation, so
    the exclusive hold covers the runs alone.
    """
    cfg_dir = WORK / f"d{depth}_k{k_suspends}_{backend}_{arch}_{target}_r{rep}"
    if cfg_dir.exists():
        shutil.rmtree(cfg_dir)
    src_dir = cfg_dir / "src"
    src_dir.mkdir(parents=True)

    (src_dir / "layout.csl").write_text((TEMPLATE / "layout.csl").read_text())
    if backend == NULL_BACKEND:
        pe_src = (TEMPLATE / "null_pe.csl").read_text()
    else:
        pe_src = render_pe(depth, k_suspends)
    (src_dir / "pe.csl").write_text(pe_src)
    shutil.copy(TEMPLATE / "run.py", cfg_dir / "run.py")

    # The simulator only instantiates the fabric it is told to; hardware
    # always has the whole wafer, and the routing has to be laid out for it.
    # The PE rectangle is 1x1 at offset (4,1) either way.
    fabric_dims = wafer.WSE3_FABRIC_DIMS if target == "wse" else "8,3"
    csl_pp_cmd = [
        str(csl_pp),
        "./src/layout.csl",
        f"--arch={arch}",
        f"--fabric-dims={fabric_dims}",
        "--fabric-offsets=4,1",
        "-o", "out",
        "--memcpy",
        "--channels=1",
        "--num-threads=1",
        f"--coro-mode={backend if backend != NULL_BACKEND else 'fiber'}",
        f"--cslc={sdk / 'cslc'}",
        f"--system-module-sif={find_sif(sdk)}",
    ]
    started = time.monotonic()
    subprocess.run(csl_pp_cmd, cwd=cfg_dir, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return cfg_dir, time.monotonic() - started


# A run takes seconds; one that takes this long has hung and would otherwise
# hold the system indefinitely.
HARDWARE_RUN_TIMEOUT = 600


def run_config(cfg_dir: Path, sdk: Path, keep_work: bool,
               target: str = "sim",
               system=None) -> tuple[tuple[int, int, int], float]:
    """Run an already-compiled configuration. Returns markers and run seconds.

    The seconds cover the whole `run.py` invocation -- pushing the ELF to the
    target, both launches and the d2h -- because that, not the compile, is
    what a held wafer is actually paying for.
    """
    started = time.monotonic()
    if target == "wse":
        if system is None:
            raise RuntimeError("target=wse needs a system (see ../wafer.py)")
        stdout = system.run(cfg_dir, "cs_python ./run.py --name out",
                            HARDWARE_RUN_TIMEOUT)
    else:
        run_cmd = [str(sdk / "cs_python"), "./run.py", "--name", "out"]
        proc = subprocess.run(run_cmd, cwd=cfg_dir, check=True,
                              env=os.environ.copy(), stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        stdout = proc.stdout.decode("utf-8", errors="replace")
    elapsed = time.monotonic() - started

    markers = parse_markers(stdout)

    if not keep_work:
        shutil.rmtree(cfg_dir)
    return markers, elapsed


# The runner prints the three timestamp-counter samples it read back.
MARKER_RE = re.compile(r"^M([012])\s*=\s*(\d+)\s*$", re.MULTILINE)

# Markers are the low 16 bits of the counter, so consecutive ones are
# subtracted modulo 2**16. Any real interval here is a few hundred cycles;
# anything past this bound means the run went wrong, not that it was slow.
MARKER_WRAP = 1 << 16
MAX_PLAUSIBLE_INTERVAL = 1 << 14


def parse_markers(out: str) -> tuple[int, int, int]:
    """Recover (M0, M1, M2) from the runner's stdout."""
    found = {int(m.group(1)): int(m.group(2)) for m in MARKER_RE.finditer(out)}
    missing = [f"M{i}" for i in (0, 1, 2) if i not in found]
    if missing:
        raise RuntimeError(f"markers missing from runner output: {missing}")
    return found[0], found[1], found[2]


def interval(start: int, end: int) -> int:
    """Cycles from marker `start` to marker `end`, wrap-corrected."""
    d = (end - start) % MARKER_WRAP
    if d == 0 or d > MAX_PLAUSIBLE_INTERVAL:
        raise RuntimeError(
            f"implausible marker interval {start} -> {end} ({d} cycles)")
    return d


def load_existing(csv_path: Path) -> set[tuple[int, int, str, str]]:
    if not csv_path.exists():
        return set()
    with csv_path.open() as f:
        rd = csv.DictReader(f)
        missing = [c for c in ("target", "M0")
                   if c not in (rd.fieldnames or [])]
        if missing:
            raise SystemExit(
                f"{csv_path} is from an older schema (missing {missing}) - run "
                "with --force to regenerate (this discards existing rows)."
            )
        return {(int(r["depth"]), int(r["k"]), r["backend"], r["arch"],
                 r["target"], int(r["rep"]))
                for r in rd}


def append_row(csv_path: Path, row: dict):
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+",
                    default=[1, 2, 4, 6, 8])
    ap.add_argument("--ks", type=int, nargs="+",
                    default=[1, 2, 4, 8],
                    help="suspends per function (uniform across the chain)")
    ap.add_argument("--backends", nargs="+",
                    choices=["fiber", "trampoline"],
                    default=["fiber", "trampoline"])
    ap.add_argument("--archs", nargs="+",
                    choices=["wse2", "wse3"],
                    default=["wse2", "wse3"])
    ap.add_argument("--csl-pp", type=Path, default=DEFAULT_CSL_PP)
    ap.add_argument("--sdk", type=Path, default=DEFAULT_SDK)
    ap.add_argument("--force", action="store_true",
                    help="re-run even configs already in the CSV")
    ap.add_argument("--keep-work", action="store_true",
                    help="keep per-config build dirs under work/")
    ap.add_argument("--targets", nargs="+", choices=["sim", "wse"],
                    default=["sim"],
                    help="sim = SDK fabric simulator; wse = a real CS-3 "
                         "(wse3 only)")
    ap.add_argument("--cmaddr",
                    help="address (IP:port) of a CS-3 that this host can "
                         "reach. Without it, --targets wse reserves one "
                         "through the appliance SDK.")
    ap.add_argument("--reps", type=int, default=1,
                    help="runs per configuration. Simulation is deterministic "
                         "so 1 is enough; hardware timings vary, so use "
                         "several and take the median.")
    ap.add_argument("--jobs", "-j", type=int, default=16,
                    help="parallel worker threads, simulator only (default: 16)")
    args = ap.parse_args()

    if not args.csl_pp.exists():
        print(f"csl-pp not found at {args.csl_pp} — build it first", file=sys.stderr)
        sys.exit(1)
    if not (args.sdk / "cslc").exists():
        print(f"SDK not found at {args.sdk}", file=sys.stderr)
        sys.exit(1)

    if "wse" in args.targets:
        wafer.check_client(args.cmaddr)

    RESULTS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / "data.csv"
    done = set() if args.force else load_existing(csv_path)
    if args.force and csv_path.exists():
        csv_path.unlink()

    # Baselines first: cheap, and having them in the CSV is what makes every
    # other row interpretable.
    all_configs = [
        (0, 0, NULL_BACKEND, arch, target, rep)
        for arch in args.archs
        for target in args.targets
        if not (target == "wse" and arch != "wse3")
        for rep in range(args.reps if target == "wse" else 1)
    ]
    all_configs += [
        (d, k, be, arch, target, rep)
        for d in args.depths
        for k in args.ks
        for be in args.backends
        for arch in args.archs
        for target in args.targets
        # The cluster is a CS-3; there is no wse2 hardware to run on.
        if not (target == "wse" and arch != "wse3")
        for rep in range(args.reps if target == "wse" else 1)
    ]
    total = len(all_configs)
    todo = [c for c in all_configs if c not in done]
    skipped = total - len(todo)
    if skipped:
        print(f"skipping {skipped} configs already in {csv_path}")
    print(f"running {len(todo)} configs")

    csv_lock = threading.Lock()
    print_lock = threading.Lock()
    counter = {"done": 0}
    failures: list[tuple[tuple, str]] = []
    built: dict[tuple, Path] = {}
    compile_secs: list[float] = []
    run_secs: dict[str, list[float]] = {"sim": [], "wse": []}

    def as_failure(e):
        if isinstance(e, subprocess.CalledProcessError):
            msg = (e.stderr or b"").decode("utf-8", errors="replace")
            return ("subprocess", msg[-2000:])
        return ("error", str(e))

    def build(cfg):
        d, k, be, arch, target, rep = cfg
        try:
            cfg_dir, secs = compile_config(
                d, k, be, arch, args.csl_pp, args.sdk,
                target=target, rep=rep,
            )
        except Exception as e:  # noqa: BLE001 - reported, not swallowed
            return cfg, None, as_failure(e)
        return cfg, (cfg_dir, secs), None

    def worker(cfg, system):
        """Run one already-built configuration."""
        target = cfg[4]
        try:
            markers, secs = run_config(
                built[cfg], args.sdk, args.keep_work,
                target=target, system=system,
            )
        except Exception as e:  # noqa: BLE001 - reported, not swallowed
            return cfg, None, as_failure(e)
        run_secs[target].append(secs)
        return cfg, (markers, secs), None

    def report(cfg, result, err):
        d, k, be, arch, target, rep = cfg
        with print_lock:
            counter["done"] += 1
            idx = counter["done"]
            what = "baseline  " if be == NULL_BACKEND else f"d={d:<2} k={k:<2}"
            tag = (f"[{idx:>3}/{len(todo)}] {what} "
                   f"{be:<10} {arch} {target}")
            if target == "wse" and args.reps > 1:
                tag += f" r{rep}"
            if err is not None:
                kind, msg = err
                print(f"{tag}  FAIL ({kind})", file=sys.stderr)
                if kind == "subprocess":
                    print(msg, file=sys.stderr)
                else:
                    print(f"  {msg}", file=sys.stderr, flush=True)
                failures.append((cfg, kind))
                return
            (m0, m1, m2), secs = result
            suspend = interval(m0, m1)
            resume = interval(m1, m2)
            row = {
                "depth": d, "k": k, "backend": be, "arch": arch,
                "target": target, "rep": rep,
                "M0": m0, "M1": m1, "M2": m2,
                "suspend_cycles": suspend,
                "resume_cycles": resume,
                "round_trip": suspend + resume,
            }
            with csv_lock:
                append_row(csv_path, row)
            print(f"{tag}  S={row['suspend_cycles']:>4}  "
                  f"R={row['resume_cycles']:>4}  "
                  f"total={row['round_trip']:>4}  "
                  f"[{secs:5.1f}s]", flush=True)

    # Phase 1: build everything, in parallel, with no wafer held. A 1x1
    # program costs the same ~3s to compile for the full wafer as for the
    # simulator's 8x3 fabric, so this is cheap -- but it used to happen
    # *inside* the reservation, which meant paying for it in exclusive wafer
    # time.
    build_started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for fut in as_completed([pool.submit(build, cfg) for cfg in todo]):
            cfg, result, err = fut.result()
            if err is not None:
                report(cfg, None, err)
                continue
            cfg_dir, secs = result
            built[cfg] = cfg_dir
            compile_secs.append(secs)
    if compile_secs:
        print(f"built {len(compile_secs)} configs in "
              f"{time.monotonic() - build_started:.1f}s wall "
              f"({sum(compile_secs) / len(compile_secs):.1f}s each, "
              f"{sum(compile_secs):.1f}s cpu)", flush=True)

    # Phase 2: run. One job covers every hardware run, and now covers nothing
    # else.
    sim_todo = [c for c in todo if c[4] == "sim" and c in built]
    wse_todo = [c for c in todo if c[4] == "wse" and c in built]

    with (wafer.system(args.sdk, args.cmaddr, WORK / "appliance") if wse_todo
          else nullcontext(None)) as system:
        if wse_todo:
            print(f"running on {system.describe()}", flush=True)
            hold_started = time.monotonic()

        if sim_todo:
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                futs = {pool.submit(worker, cfg, None): cfg for cfg in sim_todo}
                for fut in as_completed(futs):
                    report(*fut.result())

        # A single wafer cannot be shared, so hardware runs are serial.
        for cfg in wse_todo:
            report(*worker(cfg, system))

        if wse_todo:
            print(f"wafer held for {time.monotonic() - hold_started:.1f}s "
                  f"across {len(wse_todo)} runs", flush=True)

    for target, secs in run_secs.items():
        if secs:
            print(f"{target} runs: {len(secs)}, "
                  f"{sum(secs) / len(secs):.1f}s mean, "
                  f"{min(secs):.1f}s min, {max(secs):.1f}s max, "
                  f"{sum(secs):.1f}s total")

    if failures:
        print(f"\n{len(failures)} configurations failed:", file=sys.stderr)
        for cfg, why in failures:
            print(f"  {cfg}: {why}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
