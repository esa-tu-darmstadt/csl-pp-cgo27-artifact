#!/usr/bin/env python3
"""Render results/data.csv as a single-column figure for the paper.

F2': LOC-reduction vs cycle-overhead scatter. One point per (benchmark, arch,
back-end). x = cycles_back-end / cycles_vanilla, y = loc_back-end / loc_vanilla.
The horizontal/vertical reference lines at 1.0 split the plane into "win/lose"
quadrants.

Usage:
    python3 plot.py [--in results/data.csv] [--out results/scatter.pdf]
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

RESULTS = common.results_dir("end2end")
DEFAULT_IN = RESULTS / "data.csv"
DEFAULT_OUT = RESULTS / "scatter.pdf"

BENCH_SHORT = {
    "7pt-stencil-spmv": "7pt",
    "bicgstab": "BiCGStab",
    "conjugate-gradient": "CG",
    "gemm-collectives_2d": "GEMM",
    "gemv-collectives_2d": "GEMV",
    "power-method": "Power",
    "preconditioned-conjugate-gradient": "PCG",
}


PLATFORM_ORDER = ["wse2-sim", "wse3-sim", "wse3-wafer"]
PLATFORM_LABEL = {
    "wse2-sim": "wse2 sim",
    "wse3-sim": "wse3 sim",
    "wse3-wafer": "wse3 wafer",
}


def load(path: Path) -> dict:
    """data[(bench, config, platform)] = {field: value}, sharing tabulate.py's
    platform axis and its median over repetitions."""
    from tabulate import load_csv

    return load_csv(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    data = load(args.in_path)

    # Single-column figure: ~3.3in wide is typical for IEEE/ACM two-column.
    fig, ax = plt.subplots(figsize=(3.3, 2.6))

    # Filled = simulator, hollow = the same arch on silicon.
    style = {
        ("fiber", "wse2-sim"): dict(marker="o", facecolor="C0", edgecolor="none"),
        ("fiber", "wse3-sim"): dict(marker="o", facecolor="none", edgecolor="C0",
                                    linewidth=1.2),
        ("fiber", "wse3-wafer"): dict(marker="^", facecolor="C0",
                                      edgecolor="none"),
        ("trampoline", "wse2-sim"): dict(marker="s", facecolor="C3",
                                         edgecolor="none"),
        ("trampoline", "wse3-sim"): dict(marker="s", facecolor="none",
                                         edgecolor="C3", linewidth=1.2),
        ("trampoline", "wse3-wafer"): dict(marker="v", facecolor="C3",
                                           edgecolor="none"),
    }

    benches = sorted({k[0] for k in data.keys()})
    platforms = [p for p in PLATFORM_ORDER if any(k[2] == p for k in data)]
    plotted = {key: False for key in style}

    for bench in benches:
        for cfg in ("fiber", "trampoline"):
            for arch in platforms:
                v = data.get((bench, "vanilla", arch))
                o = data.get((bench, cfg, arch))
                if not v or not o or v["cycles_send"] == 0 or v["loc"] == 0:
                    continue
                x = o["cycles_send"] / v["cycles_send"]
                y = o["loc"] / v["loc"]
                label_key = (cfg, arch)
                lbl = (f"{cfg} ({PLATFORM_LABEL[arch]})"
                       if not plotted[label_key] else None)
                plotted[label_key] = True
                ax.scatter([x], [y], label=lbl, s=36, zorder=3,
                           **style[label_key])
                # Bench label next to fiber+wse2 only, to avoid clutter.
                if cfg == "fiber" and arch == platforms[0]:
                    ax.annotate(BENCH_SHORT.get(bench, bench),
                                (x, y), xytext=(4, 3),
                                textcoords="offset points",
                                fontsize=7, zorder=4)

    ax.axvline(1.0, color="gray", linewidth=0.5, linestyle="--", zorder=1)
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--", zorder=1)
    ax.set_xlabel(r"cycles vs.\ vanilla ($\times$)")
    ax.set_ylabel(r"LOC vs.\ vanilla ($\times$)")
    ax.set_title("Trade-off: source-code reduction vs.\\ cycle overhead",
                 fontsize=9)
    ax.legend(fontsize=7, loc="best", frameon=False)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
