#!/usr/bin/env python3
"""Render results/data.csv as a LaTeX table for the paper.

Usage:
    python3 tabulate.py [--in results/data.csv] [--out results/table.tex]
"""

import argparse
import csv
from collections import defaultdict
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

RESULTS = common.results_dir("end2end")
DEFAULT_IN = RESULTS / "data.csv"
DEFAULT_OUT = RESULTS / "table.tex"
DEFAULT_OUT_NUMBERS = RESULTS / "numbers.tex"
DEFAULT_TIMING_IN = RESULTS / "compile_times.csv"

LOCAL_TASK_BUDGET = {"wse2": 18, "wse3": 21}
MEMCPY_CONTROL_TASKS = 5
MEMCPY_DATA_TASKS = {"wse2": 0, "wse3": 1}
ARCH_LATEX = {"wse2": "WseTwo", "wse3": "WseThree"}
TABLE_ARRAY_STRETCH = "0.85"
HEADER_SHADE = "blue!8"
SUMMARY_SHADE = "blue!4"
RULE_TO_ROW_SPACE = "0.65ex"
ROW_TO_RULE_SPACE = "0.4ex"

# The table's platform columns. A platform is an (arch, target) pair rather
# than just an arch: the same architecture is measured both under the
# simulator and on silicon, and the two must never be pooled. There is no
# wse-2 hardware row because the cluster is a CS-3.
PLATFORMS = {
    "wse2-sim": {"arch": "wse2", "target": "sim",
                 "arch_label": "wse-2", "target_label": "sim",
                 "latex": "WseTwoSim"},
    "wse3-sim": {"arch": "wse3", "target": "sim",
                 "arch_label": "wse-3", "target_label": "sim",
                 "latex": "WseThreeSim"},
    "wse3-wafer": {"arch": "wse3", "target": "wse",
                   "arch_label": "wse-3", "target_label": "wafer",
                   "latex": "WseThreeWafer"},
}
DEFAULT_PLATFORMS = ["wse2-sim", "wse3-sim", "wse3-wafer"]


def platform_arch(platform: str) -> str:
    return PLATFORMS[platform]["arch"]

# Pretty short names used in the table's leftmost column.
BENCH_SHORT = {
    "7pt-stencil-spmv": "7pt-stencil",
    "bicgstab": "BiCGStab",
    "conjugate-gradient": "CG",
    "gemm-collectives_2d": "GEMM",
    "gemv-collectives_2d": "GEMV",
    "power-method": "Power",
    "preconditioned-conjugate-gradient": "PCG",
}

# Order of benchmarks in the table (matches bench.py).
BENCH_ORDER = [
    "7pt-stencil-spmv",
    "bicgstab",
    "conjugate-gradient",
    "gemv-collectives_2d",
    "gemm-collectives_2d",
    "power-method",
    "preconditioned-conjugate-gradient",
]


def load_csv(path: Path) -> dict:
    """data[(bench, config, platform)] = {field: value}.

    Repetitions are collapsed by taking the *minimum* cycle count. Simulation
    is deterministic and yields one row. On hardware the measured window can
    only be inflated, never shortened, by a delayed command delivery, so the
    contamination is one-sided and the minimum is the best estimate of the
    undisturbed execution -- a median would happily report an inflated sample.
    Every other column is a property of the compile, identical across
    repetitions of the same configuration.
    """
    by_platform = {
        (v["arch"], v["target"]): k for k, v in PLATFORMS.items()
    }
    reps = defaultdict(list)
    with path.open() as f:
        for row in csv.DictReader(f):
            platform = by_platform.get((row["arch"], row["target"]))
            if platform is None:
                continue
            reps[(row["benchmark"], row["config"], platform)].append(row)

    out = {}
    for key, rows in reps.items():
        first = rows[0]
        out[key] = {
            "cycles_send": min(int(r["cycles_send"]) for r in rows),
            "tasks_local": int(first["tasks_local"]),
            "tasks_total": int(first["tasks_total"]),
            "prog_size": int(first["prog_size"]),
            "globals_size": int(first["globals_size"]),
            "loc": int(first["loc"]),
        }
    return out


def load_timing(path: Path) -> dict:
    """timing[(bench, config, platform)] = {"transpile", "cslc"} in seconds.

    `transpile` is csl-pp's own time (the whole compile minus its cslc runs),
    `cslc` the time of all cslc runs. Repetitions are collapsed per field by
    the minimum: interference from other load on the host only adds time.
    """
    by_platform = {
        (v["arch"], v["target"]): k for k, v in PLATFORMS.items()
    }
    reps = defaultdict(list)
    with path.open() as f:
        for row in csv.DictReader(f):
            platform = by_platform.get((row["arch"], row["target"]))
            if platform is None:
                continue
            reps[(row["benchmark"], row["config"], platform)].append(row)
    return {
        key: {
            "transpile": min(
                float(r["total_s"]) - float(r["cslc_s"]) for r in rows),
            "cslc": min(float(r["cslc_s"]) for r in rows),
        }
        for key, rows in reps.items()
    }


def fmt_seconds(x: float) -> str:
    return f"{x:.1f}"


def fmt_int(n: int) -> str:
    """Group thousands with a thin space (LaTeX-friendly), e.g. 13\\,281."""
    s = f"{n:,}"
    return s.replace(",", r"\,")


def fmt_ratio(num: int, den: int) -> str:
    if den == 0:
        return "---"
    r = num / den
    return rf"${r:.2f}\times$"


def shade_band(shade: str, height: str) -> str:
    """Colored replacement for booktabs' normally white rule spacing."""
    return "\n".join((
        rf"\arrayrulecolor{{{shade}}}",
        rf"\specialrule{{{height}}}{{0pt}}{{0pt}}",
        r"\arrayrulecolor{black}",
    ))


def render(data: dict, platforms: list[str], timing: dict | None = None) -> str:
    """Render the LaTeX. Metric groups: cycles | tasks | prog | globals | LOC,
    plus transpile | cslc when `timing` is given.
    Inside each group: vanilla abs, fiber ratio, trampoline ratio. LOC has
    just vanilla + ours-ratio since fiber/trampoline LOC are identical.
    Transpilation exists only for ours and is given in absolute seconds."""

    # Layout:
    #   l l l | rrr | rrr | rrr | rrr | rr
    #   bench arch target | V F× T× | V F× T× | V F× T× | V F× T× | V O×
    metric_groups = [
        ("Runtime [cycles]", 3, ["Van.", r"SL$\times$", r"SF$\times$"]),
        ("Local tasks", 2, ["Van.", "SL/SF"]),
        ("Instruction memory [B]", 3,
         ["Van.", r"SL$\times$", r"SF$\times$"]),
        ("Data memory [B]", 3, ["Van.", r"SL$\times$", r"SF$\times$"]),
        ("LOC", 2, ["Van.", r"Ours $\times$"]),
    ]
    if timing is not None:
        metric_groups += [
            ("Transpile [s]", 2, ["SL", "SF"]),
            ("cslc [s]", 3, ["Van.", r"SL$\times$", r"SF$\times$"]),
        ]
    col_spec = "lll" + "".join("|" + "r" * g[1] for g in metric_groups)

    lines = [r"% Requires booktabs, multirow, and xcolor with table support."]
    lines.append(r"\begingroup")
    lines.append(rf"\renewcommand{{\arraystretch}}{{{TABLE_ARRAY_STRETCH}}}")
    # booktabs normally leaves uncolored space on either side of its rules.
    # Remove it locally so shaded rows meet the rules above and below them.
    lines.append(r"\setlength{\aboverulesep}{0pt}")
    lines.append(r"\setlength{\belowrulesep}{0pt}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append(shade_band(HEADER_SHADE, RULE_TO_ROW_SPACE))

    # Multicolumn header row 1: group labels.
    header1 = ["", r"\multicolumn{2}{c|}{Platform}"]
    for label, ncols, _ in metric_groups:
        header1.append(rf"\multicolumn{{{ncols}}}{{c|}}{{{label}}}")
    # Trim the trailing | on the last group.
    header1[-1] = (
        header1[-1].replace("|}{", "}{", 1)
        if header1[-1].endswith("}")
        else header1[-1]
    )
    lines.append(rf"\rowcolor{{{HEADER_SHADE}}}")
    lines.append(" & ".join(header1) + r" \\")

    # Header row 2: sub-labels per group. Keep it directly adjacent to the
    # first row so the background remains one continuous block; booktabs'
    # cmidrule spacing otherwise leaves a white band through the colored head.
    header2 = ["Benchmark", "Arch.", "Target"]
    for _, _, sublabels in metric_groups:
        header2.extend(sublabels)
    lines.append(rf"\rowcolor{{{HEADER_SHADE}}}")
    lines.append(" & ".join(header2) + r" \\")
    lines.append(shade_band(HEADER_SHADE, ROW_TO_RULE_SPACE))
    lines.append(
        rf"\specialrule{{\lightrulewidth}}{{0pt}}{{{RULE_TO_ROW_SPACE}}}")

    # Body rows.
    for i, name in enumerate(BENCH_ORDER):
        pretty = BENCH_SHORT.get(name, name)
        # Skip benchmarks missing from CSV entirely.
        present = [
            pf
            for pf in platforms
            if any(
                (name, c, pf) in data
                for c in ("vanilla", "fiber", "trampoline")
            )
        ]
        if not present:
            continue
        for j, platform in enumerate(present):
            row_cells = []
            if j == 0:
                row_cells.append(rf"\multirow{{{len(present)}}}{{*}}{{{pretty}}}")
            else:
                row_cells.append("")

            # Split the platform into architecture and target. Merge adjacent
            # rows for the same architecture so wse-3 is printed only once
            # for its simulator and wafer measurements.
            arch = platform_arch(platform)
            if j and platform_arch(present[j - 1]) == arch:
                row_cells.append("")
            else:
                span = 1
                while (j + span < len(present)
                       and platform_arch(present[j + span]) == arch):
                    span += 1
                arch_label = PLATFORMS[platform]["arch_label"]
                row_cells.append(
                    rf"\multirow{{{span}}}{{*}}{{{arch_label}}}"
                    if span > 1 else arch_label
                )
            row_cells.append(PLATFORMS[platform]["target_label"])

            van = data.get((name, "vanilla", platform))
            fib = data.get((name, "fiber", platform))
            tram = data.get((name, "trampoline", platform))

            def cell_abs(d, field):
                return fmt_int(d[field]) if d else "---"

            def cell_ratio(num_dict, den_dict, field):
                if num_dict is None or den_dict is None:
                    return "---"
                return fmt_ratio(num_dict[field], den_dict[field])

            # Column order is SL then SF. SF = fiber, SL = trampoline.
            # SF comes second so the "better" (fiber) numbers sit on the right.
            # cycles
            row_cells.append(cell_abs(van, "cycles_send"))
            row_cells.append(cell_ratio(tram, van, "cycles_send"))
            row_cells.append(cell_ratio(fib, van, "cycles_send"))
            # tasks — local only (control/data are runtime-fixed; see caption).
            # SF and SL always produce the same task count; collapse to one column.
            row_cells.append(cell_abs(van, "tasks_local"))
            ours_tasks = tram if tram else fib
            if fib and tram:
                assert fib["tasks_local"] == tram["tasks_local"], (
                    f"SF/SL tasks_local differ for {name}/{platform}: "
                    f"fiber={fib['tasks_local']} trampoline={tram['tasks_local']}"
                )
            row_cells.append(cell_abs(ours_tasks, "tasks_local"))
            # prog
            row_cells.append(cell_abs(van, "prog_size"))
            row_cells.append(cell_ratio(tram, van, "prog_size"))
            row_cells.append(cell_ratio(fib, van, "prog_size"))
            # globals
            row_cells.append(cell_abs(van, "globals_size"))
            row_cells.append(cell_ratio(tram, van, "globals_size"))
            row_cells.append(cell_ratio(fib, van, "globals_size"))
            # LOC — vanilla absolute, ours ratio (fiber LOC = trampoline LOC).
            ours = fib if fib else tram
            row_cells.append(cell_abs(van, "loc"))
            row_cells.append(cell_ratio(ours, van, "loc"))
            if timing is not None:
                t_van = timing.get((name, "vanilla", platform))
                t_tram = timing.get((name, "trampoline", platform))
                t_fib = timing.get((name, "fiber", platform))
                for t in (t_tram, t_fib):
                    row_cells.append(
                        fmt_seconds(t["transpile"]) if t else "---")
                row_cells.append(fmt_seconds(t_van["cslc"]) if t_van else "---")
                for t in (t_tram, t_fib):
                    row_cells.append(
                        fmt_ratio(t["cslc"], t_van["cslc"])
                        if t and t_van else "---")

            lines.append(" & ".join(row_cells) + r" \\")
            if (j + 1 < len(present)
                    and platform_arch(present[j + 1]) != arch):
                # Separate architectures within a benchmark without drawing
                # through either the benchmark name or the metric columns.
                lines.append(r"\cline{2-3}")
        if i < len(BENCH_ORDER) - 1:
            lines.append(r"\specialrule{\lightrulewidth}{0pt}{0pt}")

    # Geometric mean of ratios per metric, reported per arch.
    lines.append(
        rf"\specialrule{{\lightrulewidth}}{{{ROW_TO_RULE_SPACE}}}{{0pt}}")
    lines.append(shade_band(SUMMARY_SHADE, RULE_TO_ROW_SPACE))

    def geomean_ratio(field: str, cfg: str, platforms_to_use: list[str]) -> str:
        prod = 1.0
        n = 0
        for name in BENCH_ORDER:
            for platform in platforms_to_use:
                v = data.get((name, "vanilla", platform))
                o = data.get((name, cfg, platform))
                if v and o and v[field] > 0:
                    prod *= o[field] / v[field]
                    n += 1
        if n == 0:
            return "---"
        gm = prod ** (1.0 / n)
        return rf"${gm:.2f}\times$"

    for platform in platforms:
        if not any(
            (name, c, platform) in data
            for name in BENCH_ORDER
            for c in ("vanilla", "fiber", "trampoline")
        ):
            continue
        summary = [
            r"\textit{Geo.\ mean}",
            PLATFORMS[platform]["arch_label"],
            PLATFORMS[platform]["target_label"],
        ]
        for field in ("cycles_send", "tasks_local", "prog_size", "globals_size"):
            summary.append("")  # vanilla absolute slot — no aggregate
            if field == "tasks_local":
                # SF/SL share a single column; one ratio slot. Use fiber, which
                # covers every benchmark (trampoline lacks GEMV and GEMM).
                summary.append(geomean_ratio(field, "fiber", [platform]))
                continue
            summary.append(geomean_ratio(field, "trampoline", [platform]))
            summary.append(geomean_ratio(field, "fiber", [platform]))
        # LOC group: vanilla slot empty, ours ratio (fiber LOC = trampoline LOC).
        summary.append("")
        summary.append(geomean_ratio("loc", "fiber", [platform]))
        if timing is not None:
            # Transpilation and vanilla cslc are absolute — no aggregate.
            summary.extend(["", "", ""])
            for cfg in ("trampoline", "fiber"):
                gm = _geomean_timing_ratio(timing, cfg, platform)
                summary.append(rf"${gm:.2f}\times$" if gm else "---")
        lines.append(rf"\rowcolor{{{SUMMARY_SHADE}}}")
        lines.append(" & ".join(summary) + r" \\")
    lines.append(shade_band(SUMMARY_SHADE, ROW_TO_RULE_SPACE))
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\endgroup")
    return "\n".join(lines) + "\n"


def _geomean_ratio_value(
    data: dict, field: str, cfg: str, platforms_to_use: list[str]
) -> float | None:
    prod = 1.0
    n = 0
    for name in BENCH_ORDER:
        for platform in platforms_to_use:
            v = data.get((name, "vanilla", platform))
            o = data.get((name, cfg, platform))
            if v and o and v[field] > 0:
                prod *= o[field] / v[field]
                n += 1
    if n == 0:
        return None
    return prod ** (1.0 / n)


def _geomean_timing_ratio(timing: dict, cfg: str, platform: str) -> float | None:
    """Geometric mean over the benchmarks of cfg's cslc time over vanilla's."""
    prod = 1.0
    n = 0
    for name in BENCH_ORDER:
        v = timing.get((name, "vanilla", platform))
        o = timing.get((name, cfg, platform))
        if v and o and v["cslc"] > 0:
            prod *= o["cslc"] / v["cslc"]
            n += 1
    return prod ** (1.0 / n) if n else None


def render_timing_numbers(timing: dict, platforms: list[str]) -> str:
    """\\newcommand definitions for the compile-time numbers: per back-end
    and platform the largest and the geometric-mean transpilation time over
    the benchmarks, in seconds, and the geometric-mean cslc ratio."""
    out = ["", "% Compile times: transpilation (csl-pp alone) in seconds, and",
           "% the geometric-mean cslc time relative to vanilla."]
    config_lat = {"fiber": "SF", "trampoline": "SL"}
    for platform in platforms:
        alat = PLATFORMS[platform]["latex"]
        for cfg, clat in config_lat.items():
            ts = [timing[(n, cfg, platform)]["transpile"] for n in BENCH_ORDER
                  if (n, cfg, platform) in timing]
            if ts:
                gm = 1.0
                for t in ts:
                    gm *= t
                gm **= 1.0 / len(ts)
                out.append(rf"\newcommand{{\nTranspileMax{clat}{alat}}}"
                           rf"{{{fmt_seconds(max(ts))}}}")
                out.append(rf"\newcommand{{\nTranspileGeoMean{clat}{alat}}}"
                           rf"{{{fmt_seconds(gm)}}}")
            gm = _geomean_timing_ratio(timing, cfg, platform)
            if gm is not None:
                out.append(rf"\newcommand{{\nGeoMeanCslc{clat}{alat}}}"
                           rf"{{{gm:.2f}}}")
    return "\n".join(out) + "\n"


def render_numbers(data: dict, platforms: list[str]) -> str:
    """Emit \\newcommand definitions for the numbers cited in the paper's
    end-to-end section. Bare-decimal/-integer values; the source wraps them in
    $...\\times$ or appends \\,\\% as needed (matches suspend_resume style)."""

    def fmt_ratio(x: float) -> str:
        return f"{x:.2f}"

    def pct_growth(x: float) -> int:
        return round((x - 1.0) * 100)

    out = [
        "% AUTO-GENERATED by scripts/paper-bench/end2end/tabulate.py",
        "% Do not edit by hand — regenerate via `python3 tabulate.py`.",
        "",
        rf"\newcommand{{\nBenchCount}}{{{len(BENCH_ORDER)}}}",
        "",
        "% Memcpy/SDK runtime task slots (constant; not affected by our extension).",
        rf"\newcommand{{\nMemcpyControlTasks}}{{{MEMCPY_CONTROL_TASKS}}}",
    ]
    for arch, alat in ARCH_LATEX.items():
        out.append(
            rf"\newcommand{{\nMemcpyDataTasks{alat}}}{{{MEMCPY_DATA_TASKS[arch]}}}"
        )
    out.append("")

    # Per-(metric, config, arch) geomeans, plus integer-percent growth.
    metric_lat = {
        "cycles_send": "Cycles",
        "prog_size": "Prog",
        "globals_size": "DataMem",
    }
    config_lat = {"fiber": "SF", "trampoline": "SL"}
    out.append("% Geometric-mean ratios vs vanilla, per (metric, backend, arch).")
    for field, mlat in metric_lat.items():
        for cfg, clat in config_lat.items():
            for platform in platforms:
                alat = PLATFORMS[platform]["latex"]
                gm = _geomean_ratio_value(data, field, cfg, [platform])
                if gm is None:
                    continue
                out.append(
                    rf"\newcommand{{\nGeoMean{mlat}{clat}{alat}}}{{{fmt_ratio(gm)}}}"
                )
                out.append(
                    rf"\newcommand{{\nGeoMean{mlat}Growth{clat}Pct{alat}}}"
                    rf"{{{pct_growth(gm)}}}"
                )
    out.append("")

    # SL-over-SF prog growth in percentage points, per arch.
    out.append("% Stackless overhead over stackful for instruction memory, "
               "in percentage points.")
    for platform in platforms:
        alat = PLATFORMS[platform]["latex"]
        sl = _geomean_ratio_value(data, "prog_size", "trampoline", [platform])
        sf = _geomean_ratio_value(data, "prog_size", "fiber", [platform])
        if sl is not None and sf is not None:
            out.append(
                rf"\newcommand{{\nGeoMeanProgSLOverSFPct{alat}}}"
                rf"{{{round((sl - sf) * 100)}}}"
            )
    out.append("")

    # Worst-case data-memory geomean growth across (backend, arch).
    gms_globals = [
        _geomean_ratio_value(data, "globals_size", cfg, [platform])
        for cfg in ("fiber", "trampoline")
        for platform in platforms
    ]
    gms_globals = [g for g in gms_globals if g is not None]
    if gms_globals:
        out.append(
            rf"\newcommand{{\nMaxGeoMeanDataMemGrowthPct}}"
            rf"{{{pct_growth(max(gms_globals))}}}"
        )
        out.append("")

    # Local tasks: per-arch geomean. SF==SL per benchmark; use fiber so the
    # geomean covers every benchmark (trampoline lacks GEMV and GEMM).
    out.append("% Local-task table consumption (matches the table cell), "
               "geomean over all benchmarks.")
    for platform in platforms:
        alat = PLATFORMS[platform]["latex"]
        gm = _geomean_ratio_value(data, "tasks_local", "fiber", [platform])
        if gm is not None:
            out.append(
                rf"\newcommand{{\nGeoMeanLocalTasks{alat}}}{{{fmt_ratio(gm)}}}"
            )
            out.append(
                rf"\newcommand{{\nGeoMeanLocalTasksReductionPct{alat}}}"
                rf"{{{-pct_growth(gm)}}}"
            )

    # Min/max local-task counts across the suite.
    vanilla_tasks = [
        data[(n, "vanilla", a)]["tasks_local"]
        for n in BENCH_ORDER
        for a in platforms
        if (n, "vanilla", a) in data
    ]
    ours_tasks: list[int] = []
    for n in BENCH_ORDER:
        for a in platforms:
            for cfg in ("fiber", "trampoline"):
                if (n, cfg, a) in data:
                    ours_tasks.append(data[(n, cfg, a)]["tasks_local"])
                    break
    if vanilla_tasks:
        out.append(rf"\newcommand{{\nLocalTasksVanillaLow}}{{{min(vanilla_tasks)}}}")
        out.append(rf"\newcommand{{\nLocalTasksVanillaHigh}}{{{max(vanilla_tasks)}}}")
    if ours_tasks:
        out.append(rf"\newcommand{{\nLocalTasksOursLow}}{{{min(ours_tasks)}}}")
        out.append(rf"\newcommand{{\nLocalTasksOursHigh}}{{{max(ours_tasks)}}}")

    # Hardware budget (constants).
    for arch, alat in ARCH_LATEX.items():
        out.append(
            rf"\newcommand{{\nLocalTaskBudget{alat}}}{{{LOCAL_TASK_BUDGET[arch]}}}"
        )

    # Mean occupancy as % of the per-arch budget, rounded to the nearest 10.
    def _round_nearest_ten(x: float) -> int:
        return int(round(x / 10.0) * 10)

    for platform in platforms:
        alat = PLATFORMS[platform]["latex"]
        budget = LOCAL_TASK_BUDGET[platform_arch(platform)]
        vs = [
            data[(n, "vanilla", platform)]["tasks_local"]
            for n in BENCH_ORDER
            if (n, "vanilla", platform) in data
        ]
        os_ = []
        for n in BENCH_ORDER:
            for cfg in ("fiber", "trampoline"):
                if (n, cfg, platform) in data:
                    os_.append(data[(n, cfg, platform)]["tasks_local"])
                    break
        if vs:
            pct = _round_nearest_ten(sum(vs) / len(vs) / budget * 100)
            out.append(rf"\newcommand{{\nLocalTaskOccVanilla{alat}}}{{{pct}}}")
        if os_:
            pct = _round_nearest_ten(sum(os_) / len(os_) / budget * 100)
            out.append(rf"\newcommand{{\nLocalTaskOccOurs{alat}}}{{{pct}}}")

    out.append("")
    # LOC is identical for SF and SL and independent of the execution target.
    # Use fiber as canonical and one platform per architecture so adding both
    # simulator and wafer results does not give wse-3 extra weight.
    loc_platforms = []
    seen_archs = set()
    for platform in platforms:
        arch = platform_arch(platform)
        if arch not in seen_archs:
            seen_archs.add(arch)
            loc_platforms.append(platform)
    gm_loc = _geomean_ratio_value(data, "loc", "fiber", loc_platforms)
    if gm_loc is not None:
        out.append(rf"\newcommand{{\nGeoMeanLoc}}{{{fmt_ratio(gm_loc)}}}")
        out.append(
            rf"\newcommand{{\nGeoMeanLocReductionPct}}{{{-pct_growth(gm_loc)}}}"
        )

    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--out-numbers", type=Path, default=DEFAULT_OUT_NUMBERS)
    ap.add_argument("--timing-in", type=Path, default=DEFAULT_TIMING_IN)
    ap.add_argument(
        "--platforms",
        nargs="+",
        choices=list(PLATFORMS),
        default=DEFAULT_PLATFORMS,
        help="(arch, target) pairs to render, in column order",
    )
    args = ap.parse_args()

    data = load_csv(args.in_path)
    timing = load_timing(args.timing_in) if args.timing_in.exists() else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(data, args.platforms, timing))
    print(f"wrote {args.out}")
    numbers = render_numbers(data, args.platforms)
    if timing is not None:
        numbers += render_timing_numbers(timing, args.platforms)
    args.out_numbers.parent.mkdir(parents=True, exist_ok=True)
    args.out_numbers.write_text(numbers)
    print(f"wrote {args.out_numbers}")


if __name__ == "__main__":
    main()
