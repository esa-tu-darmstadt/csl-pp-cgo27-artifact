#!/usr/bin/env python3
"""Build the paper's coroutine-runtime breakdown table from results/.

Reads the flamegraph produced by bench.py (results/nested.csv, keyed by the
full call path, with results/tiles.csv giving each PE's executing cycles) and
emits one row per coroutine function, nested under the function that called
it. A parent row is the accumulated figure for that call path; its children
plus a dedicated `self` row sum to it exactly, so the table reads as a
flamegraph turned on its side.

Every figure is the share of a PE's executing cycles, taken per PE and then
aggregated across the PEs that ran coroutine code, rather than pooled: a
pooled figure is dominated by whichever PE ran longest.

The table says where a coroutine program's cycles go inside the code the
lowering emits -- the fiber runtime and the per-site `@await_parallel` code.
It deliberately does not say what coroutines *cost*: a hand-written CSL
program expressing the same asynchrony pays for task dispatch, task-id
management and activation as well, and nothing here is measured against such
a baseline.

The numbers describe a *profiled* build. `--coro-instrument=await-parallel`
gives each site an ordinary function-level debug scope, then the backend is
free to inline it along with the runtime. The profiler recovers both through
DWARF.

Writes results/table.tex, containing the tabularx environment alone.
"""

import argparse
import csv
import re
import statistics
from collections import defaultdict
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

RESULTS = common.results_dir("profile")

PATH_SEP = ">"
TABLE_ARRAY_STRETCH = "0.85"
HEADER_SHADE = "blue!8"
SUMMARY_SHADE = "blue!4"
RULE_TO_ROW_SPACE = "0.65ex"
ROW_TO_RULE_SPACE = "0.4ex"

# Short column headings, in the order the paper lists the benchmarks.
# Column headings and their order, matching ../end2end/results/table.tex so
# the two tables can be read side by side.
BENCH_LABELS = {
    "7pt-stencil-spmv": "7pt-stencil",
    "bicgstab": "BiCGStab",
    "conjugate-gradient": "CG",
    "gemv-collectives_2d": "GEMV",
    "gemm-collectives_2d": "GEMM",
    "power-method": "Power",
    "preconditioned-conjugate-gradient": "PCG",
}

# Symbols that differ only by call site or by the coroutine they belong to are
# one row: the paper asks what `@await_parallel` costs, not what site 0 costs.
MERGE_RULES = (
    (re.compile(r"^__fiber_await_parallel_\d+$"), "@await_parallel"),
    (re.compile(r"^__fiber_reset_.*$"), "reset_frame"),
    (re.compile(r"^__fiber_stack_.*$"), "stack_setup"),
)
STRIP_PREFIXES = ("__fiber_runtime__", "__alloc_runtime__")


def display(fn: str) -> str:
    for pattern, label in MERGE_RULES:
        if pattern.match(fn):
            return label
    for prefix in STRIP_PREFIXES:
        if fn.startswith(prefix):
            return fn[len(prefix):]
    return fn


def load_paths(nested_csv: Path, tiles_csv: Path, arch: str):
    """(benchmark) -> {tile: ({path: cycles}, task_cycles)} for one arch."""
    cycles = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    with nested_csv.open() as f:
        for r in csv.DictReader(f):
            if r["arch"] != arch:
                continue
            path = tuple(display(p) for p in r["path"].split(PATH_SEP))
            # Merging sites can make a path repeat a name; collapse the repeat
            # rather than showing a function nested inside itself.
            merged = tuple(p for i, p in enumerate(path)
                           if i == 0 or p != path[i - 1])
            cycles[r["benchmark"]][int(r["tile"])][merged] += int(r["cycles"])

    totals = defaultdict(dict)
    with tiles_csv.open() as f:
        for r in csv.DictReader(f):
            if r["arch"] == arch:
                totals[r["benchmark"]][int(r["tile"])] = int(r["task_cycles"])

    out = {}
    for bench, tiles in cycles.items():
        out[bench] = {t: (paths, totals[bench][t])
                      for t, paths in tiles.items()
                      if totals.get(bench, {}).get(t)}
    return out


def accumulated(paths: dict, prefix: tuple) -> int:
    """Cycles charged to `prefix` and everything nested below it."""
    return sum(n for path, n in paths.items()
               if path[:len(prefix)] == prefix)


def median_pct(by_tile: dict, prefix: tuple) -> float | None:
    """Median across PEs of the share of executing cycles under `prefix`.

    Returns None when no PE ever executed the path, which the table prints as
    a dash: the benchmark does not exercise that part of the runtime at all,
    which is different from exercising it for a negligible number of cycles.
    """
    shares = []
    seen = False
    for paths, task_cycles in by_tile.values():
        n = accumulated(paths, prefix)
        if n:
            seen = True
        shares.append(100.0 * n / task_cycles)
    if not seen:
        return None
    return statistics.median(shares)


def median_self_pct(by_tile: dict, path: tuple) -> float | None:
    """Median share charged directly to `path`, excluding its callees."""
    shares = [100.0 * paths.get(path, 0) / task_cycles
              for paths, task_cycles in by_tile.values()]
    if not any(shares):
        return None
    return statistics.median(shares)


def ordered_rows(by_tile: dict, keep) -> list[tuple[int, tuple, bool]]:
    """(depth, path, is_self) rows, children under parents, largest first.

    `keep(prefix)` decides whether a path is worth a row. A parent is kept
    whenever any descendant is, so pruning never orphans a row. A node gets
    a dedicated `self` row when it has both surviving children and cycles of
    its own; without children, its accumulated figure already is self time.
    """
    totals: dict = defaultdict(int)
    for paths, _ in by_tile.values():
        for path, n in paths.items():
            for i in range(len(path)):
                totals[path[:i + 1]] += n

    rows: list[tuple[int, tuple, bool]] = []

    def worth_showing(prefix: tuple) -> bool:
        return any(keep(p) for p in totals if p[:len(prefix)] == prefix)

    def emit(prefix: tuple, depth: int):
        kids = sorted((p for p in totals if len(p) == depth + 1
                       and p[:depth] == prefix and worth_showing(p)),
                      key=lambda p: -totals[p])
        for kid in kids:
            rows.append((depth, kid, False))
            before = len(rows)
            emit(kid, depth + 1)
            # A node with surviving children also has cycles of its own. Show
            # those exclusive cycles explicitly after its child call paths.
            own = sum(paths.get(kid, 0) for paths, _ in by_tile.values())
            if own and len(rows) > before:
                rows.append((depth + 1, kid, True))

    emit((), 0)
    return rows


def fmt(value: float | None) -> str:
    """Format one cycle share."""
    if value is None:
        return "---"
    return rf"{value:.1f}\%"


def tex(name: str) -> str:
    """Typeset a symbol name: underscores are literal, not subscripts."""
    return r"\texttt{" + name.replace("_", r"\_") + "}"


def sized(body: str, size: str) -> str:
    """Wrap a label in a size command, scoped so it cannot leak into the row."""
    return body if not size else "{" + size + " " + body + "}"


def shade_band(shade: str, height: str) -> str:
    """Colored replacement for booktabs' normally white rule spacing."""
    return "\n".join((
        rf"\arrayrulecolor{{{shade}}}",
        rf"\specialrule{{{height}}}{{0pt}}{{0pt}}",
        r"\arrayrulecolor{black}",
    ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="wse3", choices=["wse2", "wse3"])
    ap.add_argument("--nested", type=Path, default=RESULTS / "nested.csv")
    ap.add_argument("--tiles", type=Path, default=RESULTS / "tiles.csv")
    ap.add_argument("--out", type=Path, default=RESULTS / "table.tex")
    ap.add_argument("--indent", type=float, default=0.6,
                    help="indent per nesting level, in em (default 0.6); the "
                         "label column is as wide as the deepest row, so this "
                         "sets how much width the tree costs")
    ap.add_argument("--group-space", default="",
                    help="vertical space between top-level groups, e.g. 2pt; "
                         "empty (the default) packs the rows")
    ap.add_argument("--name-size", default=r"\smaller",
                    help=r"size command applied to function names; "
                         r"\smaller (the default) steps down one size from "
                         r"the table body and needs the relsize package, "
                         r"\small and \footnotesize are absolute, "
                         r"empty leaves them at body size")
    ap.add_argument("--min-pct", type=float, default=0.0,
                    help="drop a call path no benchmark spends this much in")
    args = ap.parse_args()

    for path in (args.nested, args.tiles):
        if not path.exists():
            raise SystemExit(f"no data at {path} — run bench.py first")

    data = load_paths(args.nested, args.tiles, args.arch)
    if not data:
        raise SystemExit(f"no {args.arch} data in {args.nested}")

    benches = [b for b in BENCH_LABELS if b in data]
    benches += [b for b in sorted(data) if b not in BENCH_LABELS]

    # Row structure comes from whichever benchmark exercises the most of the
    # runtime, so a benchmark that never calls a function still gets a dash in
    # that row rather than the row disappearing.
    widest = max(data.values(), key=lambda t: len({p for paths, _ in t.values()
                                                   for p in paths}))

    def significant(prefix: tuple) -> bool:
        """True if some benchmark spends at least --min-pct under `prefix`."""
        return any((median_pct(data[b], prefix) or 0.0) >= args.min_pct
                   for b in benches)

    rows = ordered_rows(widest, significant)

    total = {b: median_pct(data[b], ()) for b in benches}

    lines = []
    # A LaTeX comment, so the file explains its own notation without adding
    # anything to the rendered table.
    lines.append(r"% Requires booktabs, relsize, tabularx, and xcolor with table support.")
    lines.append("% Share of a PE's executing cycles spent in coroutine "
                 "runtime code.")
    lines.append("% Component rows are accumulated (the function and everything "
                 "it calls);")
    lines.append("% indented `self` rows contain the function's own cycles. "
                 "Median over the PEs")
    lines.append("% running coroutine code, taken per row, so a parent may "
                 "differ from the sum")
    lines.append("% of its children plus its own share by a fraction of a "
                 "point.")
    # Keep the tall, deeply nested table compact enough for a paper column.
    # Scope arraystretch so including this fragment cannot affect later tables.
    lines.append(r"\begingroup")
    lines.append(rf"\renewcommand{{\arraystretch}}{{{TABLE_ARRAY_STRETCH}}}")
    # booktabs normally leaves uncolored space on either side of its rules.
    # Remove it locally so shaded rows meet the rules above and below them.
    lines.append(r"\setlength{\aboverulesep}{0pt}")
    lines.append(r"\setlength{\belowrulesep}{0pt}")
    lines.append(r"\begin{tabularx}{\textwidth}{X" +
                 "r" * (len(benches) + 1) + "}")
    lines.append(r"\toprule")
    lines.append(shade_band(HEADER_SHADE, RULE_TO_ROW_SPACE))
    header = " & ".join(BENCH_LABELS.get(b, b) for b in benches)
    lines.append(rf"\rowcolor{{{HEADER_SHADE}}}")
    lines.append(rf"Component & {header} & Median \\")
    lines.append(shade_band(HEADER_SHADE, ROW_TO_RULE_SPACE))
    lines.append(
        rf"\specialrule{{\lightrulewidth}}{{0pt}}{{{RULE_TO_ROW_SPACE}}}")

    def row(label, values, bold=False):
        cells = " & ".join(fmt(v) for v in values)
        known = [v for v in values if v is not None]
        med = statistics.median(known) if known else None
        name = rf"\textbf{{{label}}}" if bold else label
        lines.append(rf"{name} & {cells} & {fmt(med)} \\")

    for index, (depth, path, is_self) in enumerate(rows):
        if depth == 0 and index:
            # Separate top-level runtime functions, but keep each function's
            # nested callees visually attached to their parent.
            lines.append(
                r"\specialrule{\lightrulewidth}{0pt}{0pt}")
            if args.group_space:
                lines.append(rf"\addlinespace[{args.group_space}]")
        # The indent stays at body size so nesting levels line up regardless
        # of how the names themselves are set.
        if is_self:
            name = r"\textit{self}"
            values = [median_self_pct(data[b], path) for b in benches]
        else:
            name = tex(display(path[-1]))
            values = [median_pct(data[b], path) for b in benches]
        label = (rf"\hspace{{{depth * args.indent:g}em}}" if depth else "") + sized(
            name, args.name_size)
        row(label, values)

    # Close the breakdown with its accumulated runtime share. Keeping the
    # total at the bottom makes it read like the result of the rows above.
    lines.append(
        rf"\specialrule{{\lightrulewidth}}{{{ROW_TO_RULE_SPACE}}}{{0pt}}")
    lines.append(shade_band(SUMMARY_SHADE, RULE_TO_ROW_SPACE))
    lines.append(rf"\rowcolor{{{SUMMARY_SHADE}}}")
    row("Accumulated", [total[b] for b in benches], bold=True)

    lines.append(shade_band(SUMMARY_SHADE, ROW_TO_RULE_SPACE))
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\endgroup")

    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
