"""cgo27-ae: reproduce the evaluation of "Structured Coroutines for the
Wafer-Scale Engine" (CGO 2027).

  cgo27-ae check     checks the Cerebras SDK, Apptainer and csl-pp
  cgo27-ae simulate  runs the experiments on the SDK's simulator: Figure 7
                     and Table 1 on WSE-2 and WSE-3, and Table 2
  cgo27-ae hardware  runs Figure 7 and the "wse-3 wafer" rows of Table 1 on
                     a CS-3; needs the extra [hardware] or --cmaddr
  cgo27-ae report    regenerates Figure 7 and Tables 1 and 2 from the results
                     and compares them with the data recorded for the paper

All results go to the output directory (--out, default ./cgo27-ae-output).
"""

import argparse
import csv
import importlib.metadata
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
BENCH = PACKAGE / "bench"
PAPER = PACKAGE / "paper"
SDK_VERSION = "2.10.0"
EXPERIMENTS = ("end2end", "suspend_resume", "profile")
# Table 2 needs the simulator's instruction trace, which hardware lacks.
HARDWARE_EXPERIMENTS = ("end2end", "suspend_resume")

# What `run --quick` runs instead of the full sweeps: a few configurations of
# every experiment, enough to check that everything works.
QUICK = {
    "end2end": ["--archs", "wse3", "--benchmarks", "power-method", "gemv-collectives_2d"],
    "suspend_resume": ["--archs", "wse3", "--depths", "1", "4", "--ks", "1", "8"],
    "profile": ["--benchmarks", "power-method"],
}

# Expected agreement with the recorded data (see README.md, "Expected results").
TOLERANCE_PCT = 5.0          # Table 1: each configuration, and the geometric means
TOLERANCE_CYCLES_FIG7 = 3    # suspend and resume latencies of Figure 7
TOLERANCE_PP_TABLE2 = 1.0    # shares of Table 2, in percentage points


def _sdk(path):
    return Path(path or os.environ.get("CSL_SDK_DIR", Path.home() / "cs_sdk-2.10.0")).resolve()


def _env(out, sdk):
    env = os.environ.copy()
    env["CGO27_AE_OUT"] = str(out)
    env["CSL_SDK_DIR"] = str(sdk)
    return env


def _script(experiment, name):
    return BENCH / experiment / name


# ---------------------------------------------------------------- check ----

def check(args):
    sdk = _sdk(args.sdk)
    sys.path.insert(0, str(BENCH))
    import common  # noqa: E402

    problems = []

    def report(ok, what, detail=""):
        print(f"[{'ok' if ok else 'FAIL'}] {what}" + (f": {detail}" if detail else ""))
        if not ok:
            problems.append(what)

    report((sdk / "cslc").is_file(), "Cerebras SDK", str(sdk))
    images = sorted(sdk.glob("*.sif"))
    report(len(images) == 1 and SDK_VERSION in images[0].name,
           f"SDK image of version {SDK_VERSION}",
           images[0].name if len(images) == 1 else f"{len(images)} *.sif files in {sdk}")
    runtime = shutil.which("apptainer") or shutil.which("singularity")
    report(runtime is not None, "Apptainer or Singularity", runtime or "not on PATH")
    report(shutil.which("patch") is not None, "patch", shutil.which("patch") or "not on PATH")

    csl_pp = common.default_csl_pp()
    try:
        version = subprocess.run([str(csl_pp), "--version"], capture_output=True,
                                 text=True, check=True).stdout.splitlines()[0]
        hidden = subprocess.run([str(csl_pp), "--help-hidden"], capture_output=True,
                                text=True).stdout
        report(True, "csl-pp", f"{version} ({csl_pp})")
        report("trampoline" in hidden, "stackless lowering in csl-pp",
               "--coro-mode=trampoline")
    except (OSError, subprocess.CalledProcessError, IndexError) as e:
        report(False, "csl-pp", str(e))
    try:
        pinned = importlib.metadata.version("csl-pp")
        report(True, "csl-pp package", pinned)
    except importlib.metadata.PackageNotFoundError:
        report(False, "csl-pp package", "not installed in this environment")

    for module in ("elftools", "numpy", "matplotlib", "tqdm"):
        try:
            __import__(module)
            report(True, f"Python package {module}")
        except ImportError as e:
            report(False, f"Python package {module}", str(e))

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free / 2**30
    report(free >= 20, "free space in the output directory", f"{free:.0f} GiB at {out}")

    # Only runs on a CS-3 that is reserved through the appliance SDK need it.
    try:
        import cerebras.sdk.client  # noqa: F401
        appliance = True
        print("[ok] appliance SDK (cerebras-sdk)")
    except ImportError:
        appliance = False
        print("[--] appliance SDK (cerebras-sdk): not installed")

    print()
    if problems:
        print(f"Simulator (cgo27-ae simulate): not ready, {len(problems)} problem(s) "
              "above; see README.md, \"Requirements\".")
        print("CS-3 (cgo27-ae hardware):      not ready; it compiles on this host as well.")
        return 1
    print("Simulator (cgo27-ae simulate): ready.")
    if appliance:
        print("CS-3 (cgo27-ae hardware):      ready.")
    else:
        print("CS-3 (cgo27-ae hardware):      ready only with --cmaddr, for a system "
              "that this host reaches directly. To reserve a CS-3 through the "
              "appliance SDK, install the artifact with the extra [hardware] into "
              "a Python 3.11 environment; see README.md, \"Installation\".")
    return 0


# ------------------------------------------------------- simulate, hardware ----

def _run(args, target, allowed):
    out = Path(args.out).resolve()
    sdk = _sdk(args.sdk)
    env = _env(out, sdk)
    experiments = args.experiments or list(allowed)
    unknown = [e for e in experiments if e not in allowed]
    if unknown:
        print(f"unknown experiment(s) {', '.join(unknown)}; "
              f"choose from {', '.join(allowed)}", file=sys.stderr)
        return 2
    failed = []
    for experiment in experiments:
        command = [sys.executable, str(_script(experiment, "bench.py"))]
        if target == "sim":
            if experiment == "profile":
                command += ["--archs", "wse3", "-j", str(min(args.jobs, 4))]
            else:
                command += ["--targets", "sim", "-j", str(args.jobs)]
        else:
            reps = args.reps or (3 if experiment == "end2end" else 1)
            command += ["--targets", "wse", "--archs", "wse3", "--reps", str(reps),
                        "-j", str(args.jobs)]
            if args.cmaddr:
                command += ["--cmaddr", args.cmaddr]
        if args.csl_pp:
            command += ["--csl-pp", str(Path(args.csl_pp).resolve())]
        if args.quick:
            command += QUICK[experiment]
        print(f"\n=== {experiment}: {' '.join(command[1:])}", flush=True)
        if subprocess.run(command, env=env).returncode != 0:
            failed.append(experiment)
    if failed:
        print(f"\nfailed: {', '.join(failed)}; `cgo27-ae report` lists what "
              "is missing.")
        return 1
    return 0


def simulate(args):
    return _run(args, "sim", EXPERIMENTS)


def hardware(args):
    return _run(args, "wse", HARDWARE_EXPERIMENTS)


# --------------------------------------------------------------- report ----

def _rows(path):
    return list(csv.DictReader(open(path))) if Path(path).is_file() else []


def _write(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _table1(out, report_dir, lines):
    own = _rows(out / "end2end" / "results" / "data.csv")
    recorded = _rows(PAPER / "end2end" / "data.csv")
    if not own:
        lines.append("Table 1: no results (run `cgo27-ae simulate end2end`).")
        return
    # The table shows the platforms that were measured: the simulators after
    # `simulate`, the CS-3 after `hardware`.
    data = report_dir / "table1-data.csv"
    _write(data, own, list(own[0].keys()))
    platforms = sorted({f"{r['arch']}-{'wafer' if r['target'] == 'wse' else 'sim'}"
                        for r in own})
    subprocess.run([sys.executable, str(_script("end2end", "tabulate.py")),
                    "--in", str(data), "--out", str(report_dir / "table1.tex"),
                    "--out-numbers", str(report_dir / "table1-numbers.tex"),
                    "--timing-in", str(report_dir / "no-compile-times.csv"),
                    "--platforms", *platforms],
                   check=True, stdout=subprocess.DEVNULL)
    lines.append(f"Table 1 (table1.tex): {', '.join(platforms)}")

    def per_configuration(rows_):
        """The values that Table 1 shows: like tabulate.py, the minimum of the
        cycle counts over the repetitions, because a delayed command from the
        host can only lengthen the measured interval. The other columns are
        properties of the compiled program and the same in every repetition."""
        groups = {}
        for r in rows_:
            groups.setdefault((r["benchmark"], r["config"], r["arch"], r["target"]), []).append(r)
        return {k: {"cycles_send": min(float(r["cycles_send"]) for r in g),
                    **{c: float(g[0][c])
                       for c in ("tasks_local", "prog_size", "globals_size", "loc")}}
                for k, g in groups.items()}

    mine, paper = per_configuration(own), per_configuration(recorded)
    worst, outside = {}, []
    for key, values in sorted(mine.items()):
        if key not in paper:
            continue
        for column, value in values.items():
            ref = paper[key][column]
            dev = (value - ref) / ref * 100 if ref else 0.0
            if abs(dev) > abs(worst.get(column, (0.0, None))[0]):
                worst[column] = (dev, key)
            exact = column in ("tasks_local", "loc")
            if (exact and value != ref) or (not exact and abs(dev) > TOLERANCE_PCT):
                outside.append(f"    {'/'.join(key)} {column}: {value:g} (paper {ref:g}, {dev:+.1f} %)")
    missing = sorted(set(paper) - set(mine))
    compared = len(set(mine) & set(paper))
    lines.append(f"  {compared} configurations compared with the recorded data")
    for column, (dev, key) in sorted(worst.items()):
        lines.append(f"  largest deviation of {column}: {dev:+.1f} % ({'/'.join(key)})")
    lines.extend(["  outside the expected range:"] + outside if outside
                 else ["  all within the expected range"])
    own_targets = {k[3] for k in mine}
    missing = [k for k in missing if k[3] in own_targets]
    if missing:
        lines.append(f"  {len(missing)} configurations of the paper not measured")

    def means(path):
        found = {}
        for line in Path(path).read_text().splitlines():
            m = re.match(r"\\newcommand\{\\(nGeoMean\w+)\}\{([0-9.]+)\}", line)
            if m:
                found[m.group(1)] = float(m.group(2))
        return found

    paper_numbers = report_dir / "paper-table1-numbers.tex"
    subprocess.run([sys.executable, str(_script("end2end", "tabulate.py")),
                    "--in", str(PAPER / "end2end" / "data.csv"),
                    "--out", str(report_dir / "paper-table1.tex"),
                    "--out-numbers", str(paper_numbers),
                    "--timing-in", str(report_dir / "no-compile-times.csv"),
                    "--platforms", *platforms],
                   check=True, stdout=subprocess.DEVNULL)
    # The means are compared for the platforms on which every benchmark of the
    # paper was measured; a partial run (--quick) says nothing about them. The
    # percentages derived from the means are not compared separately.
    complete = []
    for arch, target, macro in (("wse2", "sim", "WseTwoSim"), ("wse3", "sim", "WseThreeSim"),
                                ("wse3", "wse", "WseThreeWafer")):
        wanted = {k for k in paper if k[2] == arch and k[3] == target}
        if wanted and wanted <= set(mine):
            complete.append(macro)
    mine_means, paper_means = means(report_dir / "table1-numbers.tex"), means(paper_numbers)
    off = [f"    {k}: {v:g} (paper {paper_means[k]:g})" for k, v in mine_means.items()
           if k in paper_means and "Pct" not in k and any(k.endswith(m) for m in complete)
           and abs(v - paper_means[k]) > TOLERANCE_PCT / 100 * abs(paper_means[k])]
    if not complete:
        lines.append("  geometric means not compared: no platform with all benchmarks measured")
    else:
        lines.extend([f"  geometric means ({', '.join(complete)}) outside {TOLERANCE_PCT:g} % of the paper:"]
                     + off if off else
                     [f"  geometric means ({', '.join(complete)}) within {TOLERANCE_PCT:g} % of the paper"])


def _figure7(out, report_dir, lines):
    own = _rows(out / "suspend_resume" / "results" / "data.csv")
    if not own:
        lines.append("Figure 7: no results (run `cgo27-ae simulate suspend_resume`).")
        return
    data = report_dir / "figure7-data.csv"
    _write(data, own, list(own[0].keys()))
    names = {("wse2", "sim"): "wse2-sim", ("wse3", "sim"): "wse3-sim",
             ("wse3", "wse"): "wse3-wafer"}
    platforms = sorted({names[(r["arch"], r["target"])] for r in own})
    for source, root in ((data, report_dir / "figure7"),
                         (PAPER / "suspend_resume" / "data.csv", report_dir / "paper-figure7")):
        subprocess.run([sys.executable, str(_script("suspend_resume", "plot.py")),
                        "--csv", str(source), "--out-root", str(root),
                        "--platforms", *platforms],
                       check=True, stdout=subprocess.DEVNULL)
    lines.append(f"Figure 7 (figure7/<platform>/): {', '.join(platforms)}")
    for platform in platforms:
        mine = _macros(report_dir / "figure7" / platform / "numbers.tex")
        paper = _macros(report_dir / "paper-figure7" / platform / "numbers.tex")
        off = [f"    {k}: {v:g} (paper {paper[k]:g})" for k, v in mine.items()
               if k in paper and abs(v - paper[k]) > TOLERANCE_CYCLES_FIG7]
        lines.append(f"  {platform}: " + ("all latencies within ±3 cycles of the recorded data"
                                          if not off else "outside ±3 cycles:"))
        lines.extend(off)


def _macros(path):
    found = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(-?[0-9.]+)\}", line)
        if m:
            found[m.group(1)] = float(m.group(2))
    return found


def _table2(out, report_dir, lines):
    results = out / "profile" / "results"
    if not (results / "data.csv").is_file():
        lines.append("Table 2: no results (run `cgo27-ae simulate profile`).")
        return
    summary = report_dir / "table2-summary.csv"
    subprocess.run([sys.executable, str(_script("profile", "summarize.py")),
                    "--csv", str(results / "data.csv"), "--out", str(summary)],
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, str(_script("profile", "table.py")), "--arch", "wse3",
                    "--nested", str(results / "nested.csv"),
                    "--tiles", str(results / "tiles.csv"),
                    "--out", str(report_dir / "table2.tex")],
                   check=True, stdout=subprocess.DEVNULL)
    lines.append("Table 2 (table2.tex)")
    key = lambda r: (r["benchmark"], r["arch"], r["scope"], r["name"])  # noqa: E731
    paper = {key(r): float(r["pct_median"]) for r in _rows(PAPER / "profile" / "summary.csv")}
    mine = {key(r): float(r["pct_median"]) for r in _rows(summary)}
    off = [f"    {'/'.join(k)}: {v:.2f} % (paper {paper[k]:.2f} %)"
           for k, v in sorted(mine.items())
           if k in paper and abs(v - paper[k]) > TOLERANCE_PP_TABLE2]
    lines.append(f"  {len(set(mine) & set(paper))} shares compared with the recorded data")
    lines.extend(["  outside ±1 percentage point:"] + off if off
                 else ["  all within ±1 percentage point"])


def report(args):
    out = Path(args.out).resolve()
    report_dir = out / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    env_out = os.environ.get("CGO27_AE_OUT")
    os.environ["CGO27_AE_OUT"] = str(out)
    lines = []
    try:
        _table1(out, report_dir, lines)
        _figure7(out, report_dir, lines)
        _table2(out, report_dir, lines)
    finally:
        if env_out is None:
            os.environ.pop("CGO27_AE_OUT", None)
        else:
            os.environ["CGO27_AE_OUT"] = env_out
    text = "\n".join(lines) + "\n"
    (report_dir / "comparison.txt").write_text(text)
    print(text + f"\nWritten to {report_dir}")
    return 0


# ----------------------------------------------------------------- main ----

def main(argv=None):
    parser = argparse.ArgumentParser(prog="cgo27-ae", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.environ.get("CGO27_AE_OUT", "cgo27-ae-output"),
                        help="output directory (default: ./cgo27-ae-output)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="check the environment")
    p.add_argument("--sdk", help="Cerebras SDK directory (default: $CSL_SDK_DIR)")
    p.set_defaults(func=check)

    p = sub.add_parser("simulate", help="run the experiments on the simulator")
    p.add_argument("experiments", nargs="*", metavar="experiment",
                   help=f"one of {', '.join(EXPERIMENTS)} (default: all)")
    p.add_argument("--sdk", help="Cerebras SDK directory (default: $CSL_SDK_DIR)")
    p.add_argument("--quick", action="store_true",
                   help="a few configurations of each experiment only")
    p.add_argument("-j", "--jobs", type=int, default=min(8, os.cpu_count() or 1),
                   help="parallel compiles and simulations (default: up to 8)")
    p.add_argument("--csl-pp", help="the csl-pp to use (default: the one installed "
                   "with the artifact, from PyPI)")
    p.set_defaults(func=simulate)

    p = sub.add_parser("hardware", help="run the experiments on a CS-3")
    p.add_argument("experiments", nargs="*", metavar="experiment",
                   help=f"one of {', '.join(HARDWARE_EXPERIMENTS)} (default: both)")
    p.add_argument("--sdk", help="Cerebras SDK directory (default: $CSL_SDK_DIR)")
    p.add_argument("--quick", action="store_true",
                   help="a few configurations of each experiment only")
    p.add_argument("--cmaddr", help="address (IP:port) of a CS-3 that this host "
                   "can reach; without it, one is reserved through the appliance SDK")
    p.add_argument("--reps", type=int,
                   help="runs per configuration (default: 3 for end2end, "
                   "1 for suspend_resume, as for the paper)")
    p.add_argument("-j", "--jobs", type=int, default=min(8, os.cpu_count() or 1),
                   help="parallel compiles (default: up to 8)")
    p.add_argument("--csl-pp", help="the csl-pp to use (default: the one installed "
                   "with the artifact, from PyPI)")
    p.set_defaults(func=hardware)

    p = sub.add_parser("report", help="regenerate the figure and tables and compare "
                       "them with the paper")
    p.set_defaults(func=report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
