#!/usr/bin/env python3
"""Render platform-specific figures and paper constants from results/data.csv.

Produces three layout variants for comparison:
  - opt1_split   — 2x2 grid (suspend/resume × WSE-2/WSE-3), multi-K stackless
  - opt2_resume  — 1x2 grid, resume only, multi-K stackless
  - opt3_total   — 1x2 grid, round-trip only, multi-K stackless

For each stackless curve, a linear fit `R(d) = α·d + β` is performed
and printed; α(K) / β(K) per arch are summarized at the end.
"""

import argparse
import csv
import statistics
from collections import defaultdict
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

RESULTS = common.results_dir("suspend_resume")

# The raw CSV is shared because bench.py can sweep several platforms in one
# invocation. Paper-facing constants are split so selecting WSE-3 hardware
# cannot accidentally pick up WSE-3 simulator measurements.
PLATFORMS = {
    "wse2-sim": {"arch": "wse2", "target": "sim"},
    "wse3-sim": {"arch": "wse3", "target": "sim"},
    "wse3-wafer": {"arch": "wse3", "target": "wse"},
}

ALL_ARCHS = ["wse2", "wse3"]
ARCH_LABEL = {"wse2": "WSE-2", "wse3": "WSE-3"}
BACKEND_LABEL = {"fiber": "stackful", "trampoline": "stackless"}

# Stackless K values get a graded color ramp; stackful is a single dashed line.
K_COLORS = {1: "#fcbba1", 2: "#fc9272", 4: "#ef3b2c", 8: "#a50f15"}


# bench.py records the null configuration under this backend name.
NULL_BACKEND = "null"


def baselines(rows):
    """(arch, target) -> (suspend, resume) cost of the observer harness.

    The null configuration contains the same markers, the same witness task
    and the same activation chain as the benchmark, but no coroutine: its
    producer leaves by plain task return. Whatever it costs is measurement
    apparatus, and is subtracted from every real row. Medians over reps,
    because hardware repeats and the simulator does not.
    """
    acc = defaultdict(lambda: ([], []))
    for r in rows:
        if r["backend"] != NULL_BACKEND:
            continue
        s, rc = acc[(r["arch"], r["target"])]
        s.append(int(r["suspend_cycles"]))
        rc.append(int(r["resume_cycles"]))
    return {key: (round(statistics.median(s)), round(statistics.median(rc)))
            for key, (s, rc) in acc.items()}


def archs_in(data):
    """Archs actually present, in ALL_ARCHS order."""
    return [a for a in ALL_ARCHS if any(key[0] == a for key in data)]


def load(csv_path: Path, target: str, arch: str | None = None):
    """Read one platform's rows, with the harness cost removed.

    Filtering by target is not optional: the series are keyed by
    (arch, backend, kind) and (d, k), so simulator and wafer rows for the same
    arch collide, and whichever is read last silently wins.
    """
    with csv_path.open() as f:
        rows = [r for r in csv.DictReader(f)
                if r["target"] == target
                and (arch is None or r["arch"] == arch)]
    if not rows:
        raise SystemExit(
            f"{csv_path} has no rows for arch={arch!r}, target={target!r}")

    base = baselines(rows)
    if not base:
        raise SystemExit(
            f"{csv_path} has no {NULL_BACKEND!r} baseline rows for "
            f"target={target!r}. Re-run bench.py (with --force) so the observer "
            "harness can be subtracted; without it the reported cycles are "
            "inflated by the measurement itself."
        )

    by = defaultdict(dict)
    for r in rows:
        if r["backend"] == NULL_BACKEND:
            continue
        key = (r["arch"], r["target"])
        if key not in base:
            raise SystemExit(f"no baseline row for arch={key[0]} target={key[1]}")
        base_s, base_r = base[key]
        d, k = int(r["depth"]), int(r["k"])
        s = int(r["suspend_cycles"]) - base_s
        rc = int(r["resume_cycles"]) - base_r
        by[(r["arch"], r["backend"], "suspend")][(d, k)] = s
        by[(r["arch"], r["backend"], "resume")][(d, k)] = rc
        by[(r["arch"], r["backend"], "total")][(d, k)] = s + rc
    return by, base


def fit(depths, cycles):
    if len(depths) < 2:
        return None, None
    a, b = np.polyfit(depths, cycles, 1)
    return float(a), float(b)


def stackless_series(data, arch, kind, k):
    """Return (depths, cycles) lists for stackless at the given K."""
    raw = data[(arch, "trampoline", kind)]
    pts = sorted([(d, c) for (d, kk), c in raw.items() if kk == k])
    return [d for d, _ in pts], [c for _, c in pts]


def stackful_series(data, arch, kind):
    """Stackful is K-invariant; collapse to a single curve in d (average over K)."""
    raw = data[(arch, "fiber", kind)]
    by_d = defaultdict(list)
    for (d, _k), c in raw.items():
        by_d[d].append(c)
    ds = sorted(by_d.keys())
    return ds, [int(round(np.mean(by_d[d]))) for d in ds]


def plot_curves_in_axes(ax, data, arch, kind, ks, fit_records):
    """Helper: plot stackful flat line + stackless K-family on ax."""
    # Stackful
    ds_f, cs_f = stackful_series(data, arch, kind)
    if ds_f:
        ax.plot(ds_f, cs_f, color="C0", marker="^", linestyle="--",
                label="stackful")

    # Stackless, one curve per K
    for k in ks:
        ds, cs = stackless_series(data, arch, kind, k)
        if not ds:
            continue
        col = K_COLORS.get(k, "C3")
        ax.plot(ds, cs, color=col, marker="o", linestyle="-",
                label=f"stackless K={k}")
        if kind in ("resume", "total"):
            a, b = fit(ds, cs)
            if a is not None:
                fit_records.append((arch, kind, k, a, b))


def render_opt1(data, ks, out_prefix):
    """Option 1: 2 columns (suspend/resume) x one row per arch present."""
    archs = archs_in(data)
    fig, axes = plt.subplots(len(archs), 2, figsize=(10, 3.25 * len(archs)),
                             sharex=True, squeeze=False)
    fit_records = []
    for row, arch in enumerate(archs):
        for col, kind in enumerate(("suspend", "resume")):
            ax = axes[row][col]
            plot_curves_in_axes(ax, data, arch, kind, ks, fit_records)
            ax.set_title(f"{ARCH_LABEL[arch]} — {kind}")
            ax.grid(True, alpha=0.3)
            if row == 1:
                ax.set_xlabel("await-chain depth $d$")
            if col == 0:
                ax.set_ylabel("cycles")
    # one shared legend
    handles, labels = axes[0][1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Option 1: suspend / resume × arch (multi-K stackless)",
                 y=1.06)
    fig.tight_layout()
    save(fig, out_prefix)
    return fit_records


def render_opt2(data, ks, out_prefix):
    """Option 2: resume only, one panel per arch present."""
    archs = archs_in(data)
    fig, axes = plt.subplots(1, len(archs), figsize=(4.75 * len(archs), 3.6),
                             sharey=True, squeeze=False)
    axes = axes[0]
    fit_records = []
    for ax, arch in zip(axes, archs):
        plot_curves_in_axes(ax, data, arch, "resume", ks, fit_records)
        ax.set_title(ARCH_LABEL[arch])
        ax.set_xlabel("await-chain depth $d$")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("resume cycles")
    axes[-1].legend(loc="upper left", fontsize=8, framealpha=0.9)
    s9, s22 = stackless_const(data, "suspend"), stackful_const(data, "suspend")
    fig.suptitle(
        f"Option 2: resume vs $d$, multi-K stackless  "
        f"(suspend: ~{s9} cyc stackless, ~{s22} cyc stackful)",
        y=1.02,
    )
    fig.tight_layout()
    save(fig, out_prefix)
    return fit_records


def render_opt3(data, ks, out_prefix):
    """Option 3: round-trip (suspend+resume), one panel per arch present."""
    archs = archs_in(data)
    fig, axes = plt.subplots(1, len(archs), figsize=(4.75 * len(archs), 3.6),
                             sharey=True, squeeze=False)
    axes = axes[0]
    fit_records = []
    for ax, arch in zip(axes, archs):
        plot_curves_in_axes(ax, data, arch, "total", ks, fit_records)
        ax.set_title(ARCH_LABEL[arch])
        ax.set_xlabel("await-chain depth $d$")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("round-trip cycles (S+R)")
    axes[-1].legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.suptitle("Option 3: round-trip (suspend+resume) vs $d$, multi-K stackless",
                 y=1.02)
    fig.tight_layout()
    save(fig, out_prefix)
    return fit_records


def render_alpha_beta(fit_records, kind, out_prefix):
    """Supplementary: α(K), β(K) vs K, one panel per arch.

    Only rows with the requested kind ('resume' or 'total') are used.
    """
    archs = [a for a in ALL_ARCHS
             if any(r[0] == a and r[1] == kind for r in fit_records)]
    fig, axes = plt.subplots(1, len(archs), figsize=(4.75 * len(archs), 3.4),
                             squeeze=False)
    axes = axes[0]
    for ax, arch in zip(axes, archs):
        rows = [r for r in fit_records if r[0] == arch and r[1] == kind]
        if not rows:
            continue
        rows.sort(key=lambda r: r[2])
        ks = [r[2] for r in rows]
        alphas = [r[3] for r in rows]
        betas = [r[4] for r in rows]
        ax2 = ax.twinx()
        l1 = ax.plot(ks, alphas, "o-", color="#a50f15", label=r"$\alpha(K)$ slope")
        l2 = ax2.plot(ks, betas, "s--", color="#08519c", label=r"$\beta(K)$ intercept")
        ax.set_title(ARCH_LABEL[arch])
        ax.set_xlabel("states per function $K$")
        ax.set_ylabel(r"$\alpha$ (cycles per chain level)", color="#a50f15")
        ax2.set_ylabel(r"$\beta$ (intercept, cycles)", color="#08519c")
        ax.grid(True, alpha=0.3)
        ax.legend(l1 + l2, [h.get_label() for h in l1 + l2], loc="upper left",
                  fontsize=8)
    fig.suptitle(f"Supplementary: linear-fit coefficients of stackless {kind} "
                 f"$= \\alpha(K)\\cdot d + \\beta(K)$",
                 y=1.02)
    fig.tight_layout()
    save(fig, out_prefix)


def stackless_const(data, kind):
    vals = [v for (arch, be, k), series in data.items()
            if be == "trampoline" and k == kind
            for v in series.values()]
    return int(round(np.mean(vals))) if vals else 0


def stackful_const(data, kind):
    vals = [v for (arch, be, k), series in data.items()
            if be == "fiber" and k == kind
            for v in series.values()]
    return int(round(np.mean(vals))) if vals else 0


def save(fig, prefix: Path):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        out = prefix.with_suffix(suffix)
        fig.savefig(out, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


def print_fit_table(fit_records, kind):
    rows = [r for r in fit_records if r[1] == kind]
    if not rows:
        return
    print(f"\nLinear fits for stackless {kind} = α·d + β:")
    print(f"  {'arch':<5}  {'K':>2}  {'α':>8}  {'β':>8}")
    for arch, _kind, k, a, b in sorted(rows, key=lambda r: (r[0], r[2])):
        print(f"  {arch:<5}  {k:>2}  {a:>8.3f}  {b:>8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=RESULTS / "data.csv")
    ap.add_argument("--out-root", type=Path, default=RESULTS,
                    help="root for platform result folders (default: results)")
    ap.add_argument("--platforms", nargs="+", choices=list(PLATFORMS),
                    default=list(PLATFORMS),
                    help="platforms to render (default: all three)")
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"no data at {args.csv} — run bench.py first")

    for platform_name in args.platforms:
        platform = PLATFORMS[platform_name]
        arch = platform["arch"]
        target = platform["target"]
        outdir = args.out_root / platform_name

        print(f"\n=== {platform_name} ===")
        data, base = load(args.csv, target, arch)
        for (_arch, _target), (bs, br) in sorted(base.items()):
            print(f"harness baseline: suspend {bs} cyc, resume {br} cyc "
                  "(subtracted)")

        # K values are whatever appears in this platform's rows.
        ks_seen = sorted({k for series in data.values()
                          for (_d, k) in series.keys()})

        fits1 = render_opt1(data, ks_seen, outdir / "opt1_split")
        fits2 = render_opt2(data, ks_seen, outdir / "opt2_resume")
        fits3 = render_opt3(data, ks_seen, outdir / "opt3_total")

        render_alpha_beta(fits2, "resume",
                          outdir / "supp_alpha_beta_resume")
        render_alpha_beta(fits3, "total",
                          outdir / "supp_alpha_beta_total")
        render_paper(data, ks_seen, arch,
                     outdir / "paper_suspend_resume")
        write_numbers_tex(data, fits2, base, outdir / "numbers.tex",
                          platform_name)

        print_fit_table(fits2, "resume")
        print_fit_table(fits3, "total")


# LaTeX \newcommand names can only contain letters, no digits.
_K_WORD = {1: "One", 2: "Two", 3: "Three", 4: "Four", 6: "Six", 8: "Eight",
           12: "Twelve", 16: "Sixteen"}
_ARCH_WORD = {"wse2": "WseTwo", "wse3": "WseThree"}


def write_numbers_tex(data, fits_resume, base, path: Path, platform: str):
    """Emit \\newcommand definitions for every constant cited in the paper.

    The paper does `\\input{numbers.tex}` once; the prose then refers to
    `\\nStackfulSuspend{}` etc. and stays in sync with the CSV.
    """
    lines = ["% AUTO-GENERATED by scripts/paper-bench/suspend_resume/plot.py",
             "% Do not edit by hand — regenerate via `python3 plot.py`.",
             f"% Platform: {platform}",
             "%",
             "% Every cycle count below has the observer harness subtracted:",
             "% a null configuration with the identical markers, witness task",
             "% and activation chain, but no coroutine. Those constants are",
             "% emitted here too so the paper can state what was removed.",
             ""]

    def cmd(name: str, value):
        # Macro names describe the architecture and measurement, not where
        # the measurement ran. The paper selects one platform-specific file,
        # so simulator and wafer output intentionally share macro names.
        lines.append(f"\\newcommand{{\\n{name}}}{{{value}}}")

    for arch in archs_in(data):
        Arch = _ARCH_WORD[arch]  # "wse3" -> "WseThree"
        # What was subtracted, so the paper can quote it. `base` only holds
        # the target that was loaded, so this picks up whichever it is.
        harness = base.get((arch, "sim")) or base.get((arch, "wse"))
        if harness is not None:
            cmd(f"{Arch}HarnessSuspend", harness[0])
            cmd(f"{Arch}HarnessResume", harness[1])

        cmd(f"{Arch}StackfulSuspend",
            stackful_const_for_arch(data, arch, "suspend"))
        cmd(f"{Arch}StackfulResume",
            stackful_const_for_arch(data, arch, "resume"))
        cmd(f"{Arch}StackfulRoundTrip",
            stackful_const_for_arch(data, arch, "suspend")
            + stackful_const_for_arch(data, arch, "resume"))
        cmd(f"{Arch}StacklessSuspend",
            stackless_const_for_arch(data, arch, "suspend"))

        for arch_r, _kind, k, a, b in fits_resume:
            if arch_r != arch:
                continue
            w = _K_WORD.get(k, f"K{k}")
            cmd(f"{Arch}AlphaK{w}", int(round(a)))
            cmd(f"{Arch}BetaK{w}", int(round(b)))

        # Crossover depth at K=1: smallest d such that stackless RT >= stackful RT.
        susp_stless = stackless_const_for_arch(data, arch, "suspend")
        rt_stful = (stackful_const_for_arch(data, arch, "suspend")
                    + stackful_const_for_arch(data, arch, "resume"))
        a1 = next((a for ar, _k, kk, a, _b in fits_resume
                   if ar == arch and kk == 1), None)
        b1 = next((b for ar, _k, kk, _a, b in fits_resume
                   if ar == arch and kk == 1), None)
        if a1 is not None and b1 is not None and a1 > 0:
            d_cross = (rt_stful - susp_stless - b1) / a1
            cmd(f"{Arch}CrossoverDKOne", int(round(d_cross)))

        # Smallest measured K whose fitted round-trip crossover is below d=1.
        crossover_ks = []
        for arch_r, _kind, k, a, b in fits_resume:
            if arch_r != arch or a <= 0:
                continue
            d_cross = (rt_stful - susp_stless - b) / a
            if d_cross < 1:
                crossover_ks.append(k)
        if crossover_ks:
            cmd(f"{Arch}CrossoverKBelowDepthOne", min(crossover_ks))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def render_paper(data, ks, arch, out_prefix):
    """Single-column figure for the selected paper platform.

    Contains all four observables on one panel:
      - 4 stackless-resume curves (one per K)
      - stackful-resume (flat)
      - stackless-suspend (flat)
      - stackful-suspend (flat)

    Uses the smallest font sizes allowed by the ACM Primary Article Template
    (7pt for axis ticks / legend, 8pt for labels / title).
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "lines.linewidth": 0.9,
        "lines.markersize": 3.5,
    })

    fig, ax = plt.subplots(figsize=(3.74, 2.64))

    # ---- Stackless resume (4 curves) ----
    for k in ks:
        ds, cs = stackless_series(data, arch, "resume", k)
        if not ds:
            continue
        col = K_COLORS.get(k, "C3")
        ax.plot(ds, cs, color=col, marker="o", linestyle="-",
                label=f"stackless resume, K={k}")

    # ---- Stackful resume (single flat curve) ----
    ds_f, cs_f = stackful_series(data, arch, "resume")
    stful_resume_avg = stackful_const_for_arch(data, arch, "resume")
    ax.plot(ds_f, cs_f, color="#08519c", marker="^", linestyle="-",
            label=f"stackful resume (~{stful_resume_avg} cyc)")

    # ---- Suspend curves (collapse stackless across K — it's flat in K) ----
    susp_stless = stackless_const_for_arch(data, arch, "suspend")
    susp_stful = stackful_const_for_arch(data, arch, "suspend")
    xs_all = sorted({d for (d, _k) in
                     data[(arch, "trampoline", "resume")].keys()})
    ax.plot(xs_all, [susp_stless] * len(xs_all),
            color="#a50f15", marker="x", linestyle="--",
            label=f"stackless suspend (~{susp_stless} cyc)")
    ax.plot(xs_all, [susp_stful] * len(xs_all),
            color="#08519c", marker="x", linestyle="--",
            label=f"stackful suspend (~{susp_stful} cyc)")

    ax.set_xlabel("await-chain depth $d$")
    ax.set_ylabel("cycles")
    ax.set_title(f"{ARCH_LABEL[arch]}: suspend / resume vs $d$, by K")
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.legend(loc="upper left", framealpha=0.9, handlelength=2.2,
              borderpad=0.3, labelspacing=0.25)

    fig.tight_layout(pad=0.4)
    save(fig, out_prefix)
    plt.rcdefaults()


def stackless_const_for_arch(data, arch, kind):
    raw = data.get((arch, "trampoline", kind), {})
    return int(round(np.mean(list(raw.values())))) if raw else 0


def stackful_const_for_arch(data, arch, kind):
    raw = data.get((arch, "fiber", kind), {})
    return int(round(np.mean(list(raw.values())))) if raw else 0


if __name__ == "__main__":
    main()
