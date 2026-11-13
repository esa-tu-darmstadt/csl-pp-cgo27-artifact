#!/usr/bin/env python3
"""Summarize results/data.csv: where do a coroutine PE's instructions go?

Reads the raw per-(benchmark, arch, tile, function) counts and reports each
category's share of a PE's dispatched instructions, aggregated across the PEs
running the program: median, min, max and mean of the per-PE percentage.

Percentages are per-PE and then aggregated, rather than pooled across PEs,
because a pooled figure is dominated by whichever PE ran longest. The spread
is the interesting part: in a benchmark where only some PEs do coroutine work,
the median and the max say different and both-useful things.

Writes results/summary.csv and prints a text report.
"""

import argparse
import csv
import statistics
from collections import defaultdict
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

RESULTS = common.results_dir("profile")

# Category order is report order: coroutine code first, then everything else.
CORO_CATEGORIES = ["fiber-runtime", "task-alloc", "await-parallel",
                   "fiber-infra"]
CATEGORIES = CORO_CATEGORIES + ["system", "user"]

SUMMARY_FIELDS = ["benchmark", "arch", "scope", "name", "pes",
                  "instructions", "pct_median", "pct_min", "pct_max",
                  "pct_mean"]


def classify(fn: str) -> str:
    """Bucket an ELF function name.

    `trampoline.X` entries are the linker's 16-byte jump thunks; they belong
    to whatever they dispatch to, so they are classified by their target.
    """
    if fn.startswith("trampoline."):
        return classify(fn[len("trampoline."):])
    if fn.startswith("__fiber_await_parallel_"):
        return "await-parallel"
    if fn.startswith("__fiber_runtime__"):
        return "fiber-runtime"
    if fn.startswith("__alloc_runtime__"):
        return "task-alloc"
    if fn.startswith("__fiber_"):  # __fiber_reset_*, __fiber_stack_*
        return "fiber-infra"
    if fn.startswith("sys_mod__") or fn.startswith("memcpy") or fn == ".teardown":
        return "system"
    return "user"


def load(csv_path: Path):
    """(benchmark, arch) -> {tile: {function: instructions}}."""
    by_cfg = defaultdict(lambda: defaultdict(dict))
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            by_cfg[(r["benchmark"], r["arch"])][int(r["tile"])][r["function"]] \
                = int(r["instructions"])
    return by_cfg


def coro_pes(tiles: dict) -> list[int]:
    """PEs that ran the coroutine program.

    A wafer run lights up PEs that only ever execute system-module code --
    memcpy relays and the like. They are not running the program under study,
    and averaging them in would dilute every percentage toward zero. A PE
    counts if it dispatched at least one coroutine-runtime instruction.
    """
    return [t for t, fns in tiles.items()
            if any(classify(fn) in CORO_CATEGORIES for fn in fns)]


def stats(values: list[float]) -> tuple[float, float, float, float]:
    return (statistics.median(values), min(values), max(values),
            statistics.fmean(values))


def summarize(by_cfg):
    rows = []
    for (bench, arch), tiles in sorted(by_cfg.items()):
        pes = coro_pes(tiles)
        if not pes:
            continue
        # Per-PE percentage for each category, then aggregate over PEs.
        cat_pct = defaultdict(list)
        fn_pct = defaultdict(list)
        cat_tot = defaultdict(int)
        fn_tot = defaultdict(int)
        for t in pes:
            fns = tiles[t]
            total = sum(fns.values())
            if not total:
                continue
            per_cat = defaultdict(int)
            for fn, n in fns.items():
                per_cat[classify(fn)] += n
                fn_tot[fn] += n
            for cat in CATEGORIES:
                cat_pct[cat].append(100.0 * per_cat.get(cat, 0) / total)
                cat_tot[cat] += per_cat.get(cat, 0)
            for fn, n in fns.items():
                fn_pct[fn].append(100.0 * n / total)
            # A PE that never entered a given function still counts as 0%.
            for fn in fn_tot:
                if fn not in fns:
                    fn_pct[fn].append(0.0)

        for cat in CATEGORIES:
            med, lo, hi, mean = stats(cat_pct[cat])
            rows.append({"benchmark": bench, "arch": arch, "scope": "category",
                         "name": cat, "pes": len(pes),
                         "instructions": cat_tot[cat],
                         "pct_median": round(med, 2), "pct_min": round(lo, 2),
                         "pct_max": round(hi, 2), "pct_mean": round(mean, 2)})
        # "coro" = everything the coroutine lowering costs, all categories.
        coro = [sum(p) for p in zip(*(cat_pct[c] for c in CORO_CATEGORIES))]
        med, lo, hi, mean = stats(coro)
        rows.append({"benchmark": bench, "arch": arch, "scope": "category",
                     "name": "coro-total", "pes": len(pes),
                     "instructions": sum(cat_tot[c] for c in CORO_CATEGORIES),
                     "pct_median": round(med, 2), "pct_min": round(lo, 2),
                     "pct_max": round(hi, 2), "pct_mean": round(mean, 2)})

        for fn, pcts in fn_pct.items():
            if classify(fn) not in CORO_CATEGORIES:
                continue
            med, lo, hi, mean = stats(pcts)
            rows.append({"benchmark": bench, "arch": arch, "scope": "function",
                         "name": fn, "pes": len(pes),
                         "instructions": fn_tot[fn],
                         "pct_median": round(med, 2), "pct_min": round(lo, 2),
                         "pct_max": round(hi, 2), "pct_mean": round(mean, 2)})
    return rows


def call_attribution(calls_path: Path, arch: str):
    """Evidence that the reported categories match how the code is reached.

    Instruction attribution is exclusive -- each instruction is charged to the
    function containing it, never also to a caller -- so the categories cannot
    double-count. What it cannot show on its own is whether a symbol is
    reached from more than one phase: `suspend` is called both by a user
    coroutine and, when a spawning caller has to wait, by `caller_join`. This
    reports the call sites so the grouping can be checked rather than assumed.
    """
    if not calls_path.exists():
        return
    by = defaultdict(lambda: defaultdict(int))
    with calls_path.open() as f:
        for r in csv.DictReader(f):
            if r["arch"] == arch:
                by[r["callee"]][r["caller"]] += int(r["entries"])

    runtime = {c: d for c, d in by.items()
               if classify(c) in CORO_CATEGORIES}
    if not runtime:
        return
    print(f"\n--- {arch}: how the runtime is reached (entry counts, all "
          f"benchmarks) ---\n")
    print(f"{'callee':<38}{'entries':>9}  dominant call site")
    for callee in sorted(runtime, key=lambda c: -sum(runtime[c].values())):
        sites = runtime[callee]
        total = sum(sites.values())
        top, n = max(sites.items(), key=lambda kv: kv[1])
        print(f"{callee:<38}{total:>9,}  {100 * n / total:5.1f}% {top}")


def report(rows, runs_path: Path):
    cfgs = sorted({(r["benchmark"], r["arch"]) for r in rows})
    archs = sorted({a for _b, a in cfgs})

    for arch in archs:
        print(f"\n=== {arch}: coroutine runtime as % of a PE's dispatched "
              f"instructions ===\n")
        hdr = (f"{'benchmark':<36}{'PEs':>5}{'median':>9}{'min':>8}"
               f"{'max':>8}{'mean':>8}")
        print(hdr)
        print("-" * len(hdr))
        medians = []
        for bench, a in cfgs:
            if a != arch:
                continue
            row = next(r for r in rows if r["benchmark"] == bench
                       and r["arch"] == a and r["name"] == "coro-total")
            medians.append(row["pct_median"])
            print(f"{bench:<36}{row['pes']:>5}{row['pct_median']:>8.1f}%"
                  f"{row['pct_min']:>7.1f}%{row['pct_max']:>7.1f}%"
                  f"{row['pct_mean']:>7.1f}%")
        if medians:
            print("-" * len(hdr))
            print(f"{'median across benchmarks':<36}{'':>5}"
                  f"{statistics.median(medians):>8.1f}%")

        print(f"\n--- {arch}: breakdown by category (median per-PE %) ---\n")
        cats = CATEGORIES
        print(f"{'benchmark':<36}" + "".join(f"{c:>15}" for c in cats))
        for bench, a in cfgs:
            if a != arch:
                continue
            cells = []
            for c in cats:
                row = next(r for r in rows if r["benchmark"] == bench
                           and r["arch"] == a and r["name"] == c)
                cells.append(f"{row['pct_median']:>14.1f}%")
            print(f"{bench:<36}" + "".join(cells))

        print(f"\n--- {arch}: top runtime functions (median per-PE %) ---\n")
        fn_meds = defaultdict(list)
        for r in rows:
            if r["arch"] == arch and r["scope"] == "function":
                fn_meds[r["name"]].append(r["pct_median"])
        ranked = sorted(fn_meds.items(),
                        key=lambda kv: statistics.median(kv[1]), reverse=True)
        print(f"{'function':<44}{'median':>9}{'min':>8}{'max':>8}")
        for fn, meds in ranked[:14]:
            print(f"{fn:<44}{statistics.median(meds):>8.1f}%"
                  f"{min(meds):>7.1f}%{max(meds):>7.1f}%")

        call_attribution(RESULTS / "calls.csv", arch)

    if runs_path.exists():
        print("\n--- profiling cost (cycles) ---\n")
        with runs_path.open() as f:
            runs = list(csv.DictReader(f))
        e2e_csv = HERE.parent / "end2end" / "results" / "data.csv"
        base = {}
        if e2e_csv.exists():
            with e2e_csv.open() as f:
                for r in csv.DictReader(f):
                    if (r["config"] == "fiber" and r["target"] == "sim"
                            and r["cycles_send"]):
                        base[(r["benchmark"], r["arch"])] = int(r["cycles_send"])
        print(f"{'benchmark':<36}{'arch':>6}{'plain':>10}{'profile':>10}"
              f"{'inflation':>11}")
        for r in sorted(runs, key=lambda r: (r["arch"], r["benchmark"])):
            key = (r["benchmark"], r["arch"])
            got = int(r["cycles_instrumented"]) if r["cycles_instrumented"] else 0
            ref = base.get(key)
            infl = f"{got / ref:.3f}x" if ref and got else "n/a"
            print(f"{r['benchmark']:<36}{r['arch']:>6}"
                  f"{ref if ref else 'n/a':>10}{got:>10}{infl:>11}")
        print("\nProfiled/plain cycle ratios verify whether introducing the "
              "named\n@await_parallel debug scope affected optimization.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=RESULTS / "data.csv")
    ap.add_argument("--out", type=Path, default=RESULTS / "summary.csv")
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"no data at {args.csv} — run bench.py first")

    rows = summarize(load(args.csv))
    if not rows:
        raise SystemExit(f"{args.csv} has no PE that ran coroutine code")

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)

    report(rows, RESULTS / "runs.csv")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
