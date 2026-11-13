#!/usr/bin/env python3
"""End-to-end benchmark driver of the paper (Table 1).

Sweeps (benchmark x config x arch x target) where:
  config ∈ {vanilla, fiber, trampoline}
  arch   ∈ {wse2, wse3}
  target ∈ {sim, wse}

`target=sim` compiles and runs against the SDK's fabric simulator, exactly as
the benchmark's own commands_wseN.sh does. `target=wse` compiles the same
sources locally for the full WSE-3 wafer and runs the *same, unmodified*
runner script on a CS-3 with --cmaddr: all configurations are compiled first,
then run one after another in a single job (see ../wafer.py). Hardware is
wse3-only: the paper's measurements were taken on a CS-3.

For each configuration we collect:
  - cycles_send: parsed from run.py stdout
  - num tasks (local / control / data): from csl-pp's count-tasks pass (stderr),
    reported as the max across PEs of each kind
  - prog_size:   max .text size across all PE ELFs
  - globals_size: max sum of allocated non-code sections across all PE ELFs
  - loc:         total lines in the benchmark's source dir (excludes libs)

Output: <out>/end2end/results/data.csv. Per-config build dirs live under
<out>/end2end/work/<bench>__<config>__<arch>__<target>/ (see ../common.py).
"""

import argparse
import csv
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402
import wafer  # noqa: E402

from elftools.elf.elffile import ELFFile
from elftools.elf.constants import SH_FLAGS

HERE = Path(__file__).resolve().parent
RESULTS = common.results_dir("end2end")
WORK = common.work_dir("end2end")

CORO_DIR = common.CORO_BENCHMARKS
# The vanilla baselines: the SDK's examples with the instrumentation patches,
# prepared by configure().
NON_CORO_DIR = WORK / "vanilla"
# The library copy that the ports import, as ../../benchmark-libs/... from
# src/ (../benchmark-libs/... for the collectives benchmarks).
PORT_LIBS = CORO_DIR / "benchmark-libs"
# The directory of the cslc wrapper that runs csl-pp; set by configure().
INTEGRATION_BIN = WORK / "bin"

# Overridable on the command line.
DEFAULT_SDK = common.DEFAULT_SDK


DEFAULT_EXAMPLES = NON_CORO_DIR
DEFAULT_CSL_PP = common.default_csl_pp()


def configure(csl_pp: Path, sdk: Path) -> None:
    """Write the cslc wrapper for `csl-pp` and the SDK, and prepare the
    vanilla baselines. Called before any compile."""
    global INTEGRATION_BIN
    INTEGRATION_BIN = common.make_cslc_wrapper(WORK / "bin", csl_pp, sdk)
    common.prepare_vanilla_sources(NON_CORO_DIR)

BENCHMARKS = [
    "7pt-stencil-spmv",
    "bicgstab",
    "conjugate-gradient",
    "gemm-collectives_2d",
    "gemv-collectives_2d",
    "power-method",
    "preconditioned-conjugate-gradient",
]

# Configurations that fail to compile, as Table 1 of the paper reports.
# Their failure is listed as expected, without its log.
EXPECTED_FAILURES = {
    ("gemm-collectives_2d", "trampoline"),
    ("gemv-collectives_2d", "trampoline"),
}


# Per-benchmark vanilla overrides. The SDK ships both a host-RPC variant
# (entry: layout.csl + run.py) and a device-controlled variant
# (entry: layout_<x>.csl + device_run.py) for these benchmarks. Our coro
# version is device-controlled, so we match it by patching the commands
# file in our temp work dir to point at the device variant. The SDK files
# themselves are not modified.
VANILLA_DEVICE_VARIANT = {
    "bicgstab": ("layout_bicgstab.csl", "device_run.py"),
    "conjugate-gradient": ("layout_cg.csl", "device_run.py"),
    "power-method": ("layout_power.csl", "device_run.py"),
    "preconditioned-conjugate-gradient": ("layout_pcg.csl", "device_run.py"),
}

CSV_FIELDS = [
    "benchmark",
    "config",
    "arch",
    "target",
    "rep",
    "cycles_send",
    "tasks_local",
    "tasks_control",
    "tasks_data",
    "tasks_total",
    "prog_size",
    "globals_size",
    "loc",
]

CYCLES_RE = re.compile(r"cycles_send\s*=\s*(\d+)\s*cycles?")

# csl-pp prints the task counts only when asked to.
COUNT_TASKS_FLAG = "--count-tasks"

# count-tasks pass output (one block per PE program):
#   === count-tasks: task slot accounting ===
#     local tasks   : 3   (...)
#     control tasks : 1
#     data tasks    : 2
#     total         : 6
#   ==========================================
COUNT_TASKS_BLOCK_RE = re.compile(
    r"=== count-tasks: task slot accounting ===\s*\n"
    r"\s*local tasks\s*:\s*(\d+)[^\n]*\n"
    r"\s*control tasks\s*:\s*(\d+)[^\n]*\n"
    r"\s*data tasks\s*:\s*(\d+)[^\n]*\n",
    re.MULTILINE,
)

# cslc invocation in a commands_wseN.sh — captures the full command. Ignores
# leading/trailing whitespace and supports backslash-continuation lines.
CSLC_LINE_RE = re.compile(r"^\s*cslc\b(?:[^\n\\]*\\\n)*[^\n]*", re.MULTILINE)


def discover_benchmarks(examples_dir: Path) -> list[tuple[str, Path, Path]]:
    """Return list of (name, our_dir, vanilla_dir). vanilla_dir is the source
    directory that holds the baseline commands_wseN.sh: the instrumented
    baseline prepared by configure(), else examples_dir/<name>."""
    out = []
    for name in BENCHMARKS:
        our = CORO_DIR / name
        if not our.is_dir():
            print(f"warning: missing coro-benchmarks/{name}", file=sys.stderr)
            continue
        nc = NON_CORO_DIR / name
        if nc.is_dir():
            vanilla = nc
        else:
            sdk = examples_dir / name
            if not sdk.is_dir():
                print(f"warning: no vanilla baseline found for {name}", file=sys.stderr)
                continue
            vanilla = sdk
        out.append((name, our, vanilla))
    return out


# Matches an @import_module("path", ...) or @set_tile_code(x, y, "path", ...)
# string literal — captures only the file path. Comments are stripped first.
_IMPORT_RE = re.compile(r"@import_module\s*\(\s*\"([^\"]+)\"")
_TILE_RE = re.compile(r"@set_tile_code\s*\([^,]+,[^,]+,\s*\"([^\"]+)\"")
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")


def _is_external_import(path: str) -> bool:
    """System imports (<...>), absolute paths (compiler-internal libs),
    anything under benchmark-libs/ (SDK) or libs/ (csl++), and the shared
    blas.csl utility module are treated as 'libs' and not counted toward
    benchmark LOC."""
    if path.startswith("<") and path.endswith(">"):
        return True
    if path.startswith("/"):
        return True
    if "benchmark-libs" in path or "/libs/" in path or path.startswith("libs/"):
        return True
    # blas.csl is a shared BLAS-style utility module included with most of
    # the SDK linear-algebra benchmarks; we don't count it toward LOC.
    if path.endswith("blas.csl"):
        return True
    return False


def _without_license_header(text: str) -> str:
    """`text` without a leading license header: the comment block with the
    Apache license notice that the files of github.com/Cerebras/sdk-examples
    begin with, and the blank line after it. The paper counts the programs
    without it, as the SDK ships them."""
    lines = text.splitlines(True)
    end = 0
    while end < len(lines) and lines[end].lstrip().startswith("//"):
        end += 1
    if end and "Apache License" in "".join(lines[:end]) and end < len(lines) \
            and lines[end].strip() == "":
        return "".join(lines[end + 1:])
    return text


def count_loc(entry_layout: Path) -> int:
    """Sum of line counts over the transitive closure of .csl files reachable
    from `entry_layout` via @import_module / @set_tile_code, excluding system
    imports, absolute-path imports, and the library directories."""
    seen: set[Path] = set()
    stack: list[Path] = [entry_layout.resolve()]
    total = 0
    while stack:
        f = stack.pop()
        if f in seen or not f.is_file():
            continue
        seen.add(f)
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        counted = _without_license_header(text)
        total += counted.count("\n") + (0 if counted.endswith("\n") else 1)
        stripped = _LINE_COMMENT_RE.sub("", text)
        for m in _IMPORT_RE.finditer(stripped):
            path = m.group(1)
            if _is_external_import(path):
                continue
            stack.append((f.parent / path).resolve())
        for m in _TILE_RE.finditer(stripped):
            path = m.group(1)
            if _is_external_import(path):
                continue
            stack.append((f.parent / path).resolve())
    return total


def parse_cycles(out: str) -> int | None:
    m = CYCLES_RE.search(out)
    return int(m.group(1)) if m else None


def parse_task_counts(stderr: str) -> tuple[int, int, int] | None:
    """Take the max-per-kind across all PEs that printed a count-tasks block."""
    blocks = COUNT_TASKS_BLOCK_RE.findall(stderr)
    if not blocks:
        return None
    local = max(int(b[0]) for b in blocks)
    control = max(int(b[1]) for b in blocks)
    data = max(int(b[2]) for b in blocks)
    return local, control, data


def measure_elf(elf_path: Path) -> tuple[int, int]:
    """Return (text_size, globals_size) for one PE ELF."""
    with elf_path.open("rb") as f:
        elf = ELFFile(f)
        text = 0
        globals_size = 0
        for s in elf.iter_sections():
            if not (s.header.sh_flags & SH_FLAGS.SHF_ALLOC):
                continue
            if s.name == ".text":
                text += s.header.sh_size
            else:
                globals_size += s.header.sh_size
    return text, globals_size


def aggregate_elfs(out_dir: Path) -> tuple[int, int]:
    """Max text/globals across all out_<x>_<y>.elf files."""
    elfs = sorted((out_dir / "bin").glob("out_*.elf"))
    if not elfs:
        raise RuntimeError(f"no ELFs found under {out_dir}/bin")
    max_text = 0
    max_glob = 0
    for elf in elfs:
        t, g = measure_elf(elf)
        if t > max_text:
            max_text = t
            max_glob_for_text_max = g  # noqa: unused
        if g > max_glob:
            max_glob = g
    return max_text, max_glob


def extract_cslc_command(commands_sh: Path) -> str:
    text = commands_sh.read_text()
    m = CSLC_LINE_RE.search(text)
    if not m:
        raise RuntimeError(f"no cslc invocation found in {commands_sh}")
    # Normalize line continuations to spaces.
    return re.sub(r"\\\n\s*", " ", m.group(0)).strip()


def extract_layout_path(commands_sh: Path) -> Path:
    """Return the layout .csl file the script passes to cslc (as a path
    relative to commands_sh.parent)."""
    cmd = extract_cslc_command(commands_sh)
    for tok in cmd.split():
        if tok.endswith(".csl"):
            return commands_sh.parent / tok
    raise RuntimeError(f"no .csl arg found in cslc invocation in {commands_sh}")


# cs_python invocation in a commands_wseN.sh, same continuation handling as
# CSLC_LINE_RE above. An optional `timeout N` prefix is part of the match so
# that a line already bounded for hardware is still recognised, and so a
# re-run inherits the same bound.
CSPYTHON_LINE_RE = re.compile(
    r"^\s*(?:timeout\s+(?:-k\s+\d+\s+)?\d+\s+)?cs_python\b"
    r"(?:[^\n\\]*\\\n)*[^\n]*",
    re.MULTILINE,
)


def runner_script_path(commands_sh: Path, run_command: str) -> Path | None:
    """The .py the cs_python invocation runs, if it can be located on disk."""
    for tok in run_command.split():
        if tok.endswith(".py"):
            candidate = commands_sh.parent / tok
            return candidate if candidate.is_file() else None
    return None


def extract_cspython_command(commands_sh: Path) -> str:
    """The cs_python invocation from a commands_wseN.sh, continuations joined."""
    m = CSPYTHON_LINE_RE.search(commands_sh.read_text())
    if not m:
        raise RuntimeError(f"no cs_python invocation found in {commands_sh}")
    return re.sub(r"\\\n\s*", " ", m.group(0)).strip()


def retarget_cslc_for_wafer(cslc_command: str) -> str:
    """Point a cslc invocation at the full WSE-3 fabric (see
    patch_commands_for_wse)."""
    patched = re.sub(
        r"(?<![\w-])--fabric-dims[= ]\d+,\d+",
        f"--fabric-dims={wafer.WSE3_FABRIC_DIMS}",
        cslc_command,
    )
    return re.sub(r"(?<![\w-])--arch[= ]wse\d", "--arch=wse3", patched)


# The runner command of a hardware configuration, saved next to its build.
RUNNER_FILE = "hardware_runner.txt"


def patch_commands_for_wse(commands_sh: Path) -> str:
    """Retarget our copy of commands_wse3.sh from the simulator to a real wafer
    and return the runner command, which runs later on the system (see
    run_built()). The script itself then only compiles.

    The compile is pointed at the full WSE-3 fabric. The simulator only has to
    instantiate the small rectangle the benchmark actually uses, so the scripts
    carry a minimal `--fabric-dims`; hardware always has the whole wafer and
    the routing must be laid out for it. `--fabric-offsets` and every
    `--params` stay as they are: the core rectangle is unchanged, only the
    fabric it sits in.

    The runner gets `--run-only` if it takes it and does not have it yet.
    Without it, the runner recompiles from its own arguments, which would
    silently undo the fabric-dims edit. The system adds `--cmaddr`.
    """
    text = commands_sh.read_text()

    cslc = CSLC_LINE_RE.search(text)
    if not cslc:
        raise RuntimeError(f"no cslc invocation found in {commands_sh}")
    patched = retarget_cslc_for_wafer(cslc.group(0))
    text = text[: cslc.start()] + patched + text[cslc.end() :]

    run = CSPYTHON_LINE_RE.search(text)
    if not run:
        raise RuntimeError(f"no cs_python invocation found in {commands_sh}")
    runner = re.sub(r"\\\n\s*", " ", run.group(0)).strip()
    # Only add --run-only where the runner actually takes it: the collectives
    # benchmarks never compile, and their argument parsers reject the flag.
    if not re.search(r"(?<![\w-])--run-only\b", runner):
        script = runner_script_path(commands_sh, runner)
        if script is not None and "--run-only" in script.read_text():
            runner += " --run-only"
    text = text[: run.start()] + ":" + text[run.end() :]

    commands_sh.write_text(text)
    return runner


def patch_vanilla_to_device_variant(
    commands_sh: Path, layout_name: str, runner_name: str
) -> None:
    """In-place edit our copy of commands_wse*.sh to point at the device
    variant. Replaces layout.csl → <layout_name>.csl and run.py → runner_name.
    Only touches whole-token references."""
    text = commands_sh.read_text()
    # Replace './src/layout.csl' or 'src/layout.csl' (whole token).
    text = re.sub(
        r"(?<![\w/.])((?:\./)?src/)layout\.csl(?!\w)", rf"\1{layout_name}", text
    )
    # Replace './run.py' or 'run.py' (whole token).
    text = re.sub(r"(?<![\w/.])(\./)?run\.py(?!\w)", rf"\g<1>{runner_name}", text)
    commands_sh.write_text(text)


class RunTimeout(RuntimeError):
    """A compile or run exceeded its time budget."""


def run_subprocess(
    cmd: list[str] | str,
    *,
    cwd: Path,
    env: dict,
    shell: bool = False,
    timeout: float | None = None,
) -> tuple[int, str, str]:
    """Run a command, killing its whole process group if it overruns.

    The timeout matters most on hardware: a kernel that never unblocks the
    command stream leaves the host parked in a blocking memcpy_d2h forever,
    and the sweep would hold its reserved system until someone noticed.
    Killing only the direct child is not enough -- the real work runs inside a
    container the SDK wrapper spawns, so the whole session has to go.
    """
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        out, err = proc.communicate()
        raise RunTimeout(
            f"timed out after {timeout:.0f}s: {cmd if shell else ' '.join(cmd)}\n"
            f"--stdout--\n{out[-2000:]}\n--stderr--\n{err[-2000:]}"
        )
    return proc.returncode, out, err


# Columns that come out of the compile rather than the run, keyed by
# configuration so repetitions can reuse them.
_COMPILE_COLUMNS: dict[tuple, dict] = {}
# Seconds allowed for a device run, set from --run-timeout. Applied to the
# runner alone, not to the compile: a full-wafer compile legitimately takes
# minutes, while nothing here legitimately spends minutes on device.
RUN_TIMEOUT: float = 120.0
# Backstop for a whole compile+run script, set from --compile-timeout.
SCRIPT_TIMEOUT: float = 3600.0
_COMPILE_COLUMNS_LOCK = threading.Lock()


def run_built(name, config, arch, target, cfg_dir, sdk, system, rep) -> dict:
    """Run an already-built configuration and report its cycle count: on the
    simulator here, on hardware through `system` (see ../wafer.py)."""
    with _COMPILE_COLUMNS_LOCK:
        cached = _COMPILE_COLUMNS.get((name, config, arch, target))
    if cached is None:
        raise RuntimeError(
            f"repetition {rep} of {name}/{config}/{arch}/{target} has no "
            "build from repetition 0"
        )
    work = cfg_dir / name
    if target == "wse":
        if system is None:
            raise RuntimeError("target=wse needs a system (see ../wafer.py)")
        runner = (cfg_dir / RUNNER_FILE).read_text()
        try:
            out = system.run(work, runner, RUN_TIMEOUT, stage_root=cfg_dir)
        except wafer.RunError as e:
            raise RuntimeError(
                f"run failed for {name} {config} {arch} rep={rep}: {e}"
            ) from e
    else:
        cmd_file = work / f"commands_wse{arch[-1]}.sh"
        runner = extract_cspython_command(cmd_file)
        env = os.environ.copy()
        env.pop("CSL_FLAGS", None)
        env.pop("CORO_MODE", None)
        env["PATH"] = f"{sdk}:{env['PATH']}"
        rc, out, err = run_subprocess(
            runner, cwd=work, env=env, shell=True, timeout=RUN_TIMEOUT
        )
        if rc != 0:
            raise RuntimeError(
                f"re-run failed for {name} {config} {arch} rep={rep}: rc={rc}\n"
                f"--stdout--\n{out}\n--stderr--\n{err}"
            )
    cycles = parse_cycles(out)
    if cycles is None:
        raise RuntimeError(
            f"could not parse cycles_send for {name} {config} {arch} rep={rep}"
        )
    return {**cached, "cycles_send": cycles, "rep": rep}


def prepare_work_dir(
    name: str,
    src_dir: Path,
    vanilla_dir: Path,
    config: str,
    arch: str,
    cfg_dir: Path,
    rep: int,
    examples_dir: Path | None,
) -> tuple[Path, Path]:
    """Populate a fresh `cfg_dir` with a copy of the benchmark sources and its
    libraries. Returns (work dir, its commands_wseN.sh)."""
    if cfg_dir.exists():
        shutil.rmtree(cfg_dir, ignore_errors=True)
    if cfg_dir.exists():
        # A killed run can leave NFS silly-rename files (.nfsXXXX) behind that
        # keep the tree un-removable until every handle is closed. Move the
        # remains aside rather than failing this configuration -- otherwise a
        # single interrupted sweep poisons the next one.
        stale = cfg_dir.with_name(f"{cfg_dir.name}.stale-{os.getpid()}-{rep}")
        cfg_dir.rename(stale)
        print(f"  note: moved un-removable {cfg_dir.name} to {stale.name}",
              file=sys.stderr, flush=True)
    cfg_dir.mkdir(parents=True)

    if config == "vanilla":
        source = vanilla_dir
    else:
        source = src_dir
    work = cfg_dir / name
    shutil.copytree(
        source,
        work,
        ignore=shutil.ignore_patterns(
            "out",
            "simfab_traces",
            "sim.log",
            "sim_stats.json",
            "__pycache__",
            "*.pyc",
            "wio_flows_tmpdir*",
        ),
    )

    # Set up the libraries as a sibling of the work dir. Both the vanilla
    # layouts and the csl++ ports import "../../benchmark-libs/..." from src/,
    # which resolves to ../benchmark-libs relative to the bench root: the
    # SDK's libraries for vanilla, the ports' own copy for coro configs. For
    # coro configs we symlink it (csl-pp follows symlinks natively). For
    # vanilla we COPY the SDK's libs into cfg_dir because the SDK cslc runs
    # inside a Singularity container that only sees what we bind-mount, and
    # symlinks pointing outside the cfg dir break inside the container.
    if config == "vanilla":
        # The prepared baselines have the SDK's benchmark-libs as a sibling.
        sdk_libs = vanilla_dir.parent / "benchmark-libs"
        if not sdk_libs.is_dir():
            sdk_libs = examples_dir / "benchmark-libs"
        if sdk_libs.is_dir():
            shutil.copytree(sdk_libs, cfg_dir / "benchmark-libs", symlinks=False)
    else:
        os.symlink(PORT_LIBS, cfg_dir / "benchmark-libs")

    cmd_file = work / f"commands_wse{arch[-1]}.sh"
    if not cmd_file.is_file():
        raise RuntimeError(f"missing {cmd_file}")

    # For vanilla configs where the SDK ships both a host-RPC and a
    # device-controlled variant, patch our local copy of commands_wse*.sh
    # to use the device variant (matches what the coro version runs).
    if config == "vanilla" and name in VANILLA_DEVICE_VARIANT:
        layout_name, runner_name = VANILLA_DEVICE_VARIANT[name]
        patch_vanilla_to_device_variant(cmd_file, layout_name, runner_name)
    return work, cmd_file


def compile_and_run(
    name: str,
    src_dir: Path,
    vanilla_dir: Path,
    config: str,
    arch: str,
    sdk: Path,
    csl_pp: Path,
    integration_bin: Path,
    target: str = "sim",
    system=None,
    rep: int = 0,
    examples_dir: Path | None = None,
    extra_csl_flags: str = "",
    work_root: Path | None = None,
) -> dict:
    """Compile and run one (bench, config, arch, target). Returns a row dict.

    `extra_csl_flags` is spliced into csl-pp's argv through $CSL_FLAGS, which
    is how the profiling driver turns on --coro-instrument without the cslc
    wrapper or this function having to know about it. `work_root` lets that
    driver build into its own tree so its (instrumented, therefore slower)
    builds never sit where an end2end sweep would find and reuse them.
    """
    cfg_dir = (work_root or WORK) / f"{name}__{config}__{arch}__{target}"
    # Repetitions re-run the binary that repetition 0 built. Rebuilding per
    # repetition would only repeat the compile for a run that takes seconds.
    with _COMPILE_COLUMNS_LOCK:
        built = (name, config, arch, target) in _COMPILE_COLUMNS
    if rep > 0 and built and cfg_dir.is_dir():
        return run_built(name, config, arch, target, cfg_dir, sdk, system, rep)
    work, cmd_file = prepare_work_dir(
        name, src_dir, vanilla_dir, config, arch, cfg_dir, rep, examples_dir
    )

    if target == "wse":
        # The script only compiles; the runner runs on the system later.
        (cfg_dir / RUNNER_FILE).write_text(patch_commands_for_wse(cmd_file))

    env = os.environ.copy()
    env.pop("CSL_FLAGS", None)
    env.pop("CORO_MODE", None)

    # Both paths need cs_python from the SDK; vanilla also needs real cslc.
    # Vanilla puts SDK first so the real cslc is selected; ours puts the
    # integration wrapper first so cslc dispatches to csl-pp, but cs_python
    # still resolves to the SDK below.
    base_path = f"{sdk}:{env['PATH']}"

    if config == "vanilla":
        # Run the SDK / non-coro script as-is against the real cslc.
        env["PATH"] = base_path
        # The SDK cslc runs inside a Singularity container which by default
        # only sees $PWD. The layout's `../../benchmark-libs/` import
        # escapes that, so we bind-mount cfg_dir (the parent) by setting
        # CSL_IMPORT_PATH — the SDK's sdk_debug_shell parses this and adds
        # --bind for each entry.
        env["CSL_IMPORT_PATH"] = str(cfg_dir.resolve())
        rc, out, err = run_subprocess(
            ["bash", cmd_file.name], cwd=work, env=env,
            timeout=SCRIPT_TIMEOUT,
        )
        if rc != 0:
            raise RuntimeError(
                f"vanilla compile/run failed for {name} {arch}: rc={rc}\n"
                f"--stdout--\n{out}\n--stderr--\n{err}"
            )

        # Separately reinvoke through the integration cslc wrapper with
        # --no-cslc so it stops after csl-pp's pipeline (no real cslc).
        # The wrapper supplies --system-module-sif and --cslc=...; csl-pp
        # honors --no-cslc and skips the downstream invocation.
        cslc_cmd = extract_cslc_command(cmd_file)
        rest = cslc_cmd[len("cslc") :].strip()
        wrapper_env = env.copy()
        wrapper_env["PATH"] = f"{integration_bin}:{base_path}"
        wrapper_env["CSL_FLAGS"] = f"--no-cslc {COUNT_TASKS_FLAG}"
        wrapper_cmd = f"cslc {rest}"
        rc2, out2, err2 = run_subprocess(
            wrapper_cmd, cwd=work, env=wrapper_env, shell=True,
            timeout=SCRIPT_TIMEOUT,
        )
        if rc2 != 0:
            # Don't fail the whole row: vanilla still has cycles + ELF.
            task_stderr = ""
            print(
                f"  warn: {name} {arch} vanilla csl-pp --no-cslc failed (rc={rc2})\n"
                f"  stderr: {err2.strip()[:500]}",
                file=sys.stderr,
            )
        else:
            task_stderr = err2
    else:
        # Use the integration cslc wrapper (csl-pp) and select the backend
        # through CORO_MODE (the wrapper passes --coro-mode=$CORO_MODE).
        # Integration bin comes first so `cslc` → csl-pp; SDK is still on
        # PATH so cs_python resolves.
        env["PATH"] = f"{integration_bin}:{base_path}"
        env["CORO_MODE"] = config
        env["CSL_FLAGS"] = f"{COUNT_TASKS_FLAG} {extra_csl_flags}".strip()
        rc, out, err = run_subprocess(
            ["bash", cmd_file.name], cwd=work, env=env,
            timeout=SCRIPT_TIMEOUT,
        )
        if rc != 0:
            raise RuntimeError(
                f"{config} compile/run failed for {name} {arch}: rc={rc}\n"
                f"--stdout--\n{out}\n--stderr--\n{err}"
            )
        task_stderr = err

    out_dir = work / "out"
    text, globs = aggregate_elfs(out_dir)
    cycles = None
    if target != "wse":
        cycles = parse_cycles(out)
        if cycles is None:
            # Some run.py scripts pipe to sim.log; check there too.
            sim_log = out_dir / "sim.log"
            if sim_log.is_file():
                cycles = parse_cycles(sim_log.read_text(errors="replace"))
        if cycles is None:
            raise RuntimeError(
                f"could not parse cycles_send for {name} {config} {arch}"
            )

    task_counts = parse_task_counts(task_stderr)
    if task_counts is None:
        # For vanilla without count-tasks, leave as zeros — the column is N/A.
        tasks_l = tasks_c = tasks_d = 0
    else:
        tasks_l, tasks_c, tasks_d = task_counts

    row = {
        "benchmark": name,
        "config": config,
        "arch": arch,
        "target": target,
        "rep": rep,
        "cycles_send": cycles,
        "tasks_local": tasks_l,
        "tasks_control": tasks_c,
        "tasks_data": tasks_d,
        "tasks_total": tasks_l + tasks_c + tasks_d,
        "prog_size": text,
        "globals_size": globs,
        "loc": count_loc(extract_layout_path(cmd_file)),
    }
    with _COMPILE_COLUMNS_LOCK:
        _COMPILE_COLUMNS[(name, config, arch, target)] = {
            k: v for k, v in row.items() if k != "cycles_send"
        }
    return row


#===----------------------------------------------------------------------===#
# Compile timing (--timing)
#===----------------------------------------------------------------------===#

COMPILE_CSV_FIELDS = [
    "benchmark",
    "config",
    "arch",
    "target",
    "rep",
    "total_s",
    "cslc_s",
    "cslc_runs",
    "cslc_run_s",
]

# Stands in for cslc and appends the wall-clock interval of every real cslc
# run to $CSL_TIMING_LOG. It is `cslc` on PATH for vanilla and csl-pp's
# --cslc for ours, where csl-pp may run cslc twice (round 1 measures the fiber
# stack sizes) and execs into the last run, so it cannot time it itself.
CSLC_TIMING_SHIM = """#!/bin/bash
start=$(date +%s.%N)
"$CSL_TIMING_REAL_CSLC" "$@"
rc=$?
echo "$start $(date +%s.%N)" >> "$CSL_TIMING_LOG"
exit $rc
"""


def setup_timing_bins(root: Path, integration_bin: Path) -> tuple[Path, Path]:
    """Create the `cslc` front ends of a timing sweep under `root`: the timing
    shim itself for vanilla, and for ours the integration wrapper with csl-pp's
    --cslc pointed at the shim. Returns (vanilla bin dir, ours bin dir)."""
    vanilla_bin = root / "vanilla"
    ours_bin = root / "ours"
    vanilla_bin.mkdir(parents=True, exist_ok=True)
    ours_bin.mkdir(parents=True, exist_ok=True)

    shim = vanilla_bin / "cslc"
    shim.write_text(CSLC_TIMING_SHIM)
    shim.chmod(0o755)

    wrapper_text, n = re.subn(
        r"--cslc=\S+", f"--cslc={shim}", (integration_bin / "cslc").read_text()
    )
    if n != 1:
        raise RuntimeError(
            f"expected one --cslc= in {integration_bin / 'cslc'}, found {n}"
        )
    wrapper = ours_bin / "cslc"
    wrapper.write_text(wrapper_text)
    wrapper.chmod(0o755)
    return vanilla_bin, ours_bin


def time_compile(
    name: str,
    src_dir: Path,
    vanilla_dir: Path,
    config: str,
    arch: str,
    target: str,
    sdk: Path,
    timing_bins: tuple[Path, Path],
    rep: int,
    examples_dir: Path | None,
) -> dict:
    """Run only the cslc line of the benchmark's commands_wseN.sh and time it.

    `total_s` is the whole invocation, `cslc_s` the sum over the real cslc
    runs inside it and `cslc_run_s` each run's time, in order and separated by
    ";". For ours, total_s - cslc_s is csl-pp's own (transpilation) time; for
    the stackful back-end, the first of two runs is the stack-sizing round. `target=wse` compiles for the full wafer, as the hardware sweep
    does; nothing is run, so no wafer is reserved.
    """
    cfg_dir = WORK / "timing" / f"{name}__{config}__{arch}__{target}"
    work, cmd_file = prepare_work_dir(
        name, src_dir, vanilla_dir, config, arch, cfg_dir, rep, examples_dir
    )
    cslc_cmd = extract_cslc_command(cmd_file)
    if target == "wse":
        cslc_cmd = retarget_cslc_for_wafer(cslc_cmd)

    log = cfg_dir / "cslc_times.log"
    env = os.environ.copy()
    env.pop("CSL_FLAGS", None)
    env.pop("CORO_MODE", None)
    env["CSL_TIMING_REAL_CSLC"] = str(sdk / "cslc")
    env["CSL_TIMING_LOG"] = str(log)
    vanilla_bin, ours_bin = timing_bins
    if config == "vanilla":
        env["PATH"] = f"{vanilla_bin}:{sdk}:{env['PATH']}"
        # See compile_and_run: the SDK cslc only sees bind-mounted paths.
        env["CSL_IMPORT_PATH"] = str(cfg_dir.resolve())
    else:
        env["PATH"] = f"{ours_bin}:{sdk}:{env['PATH']}"
        env["CORO_MODE"] = config
        env["CSL_FLAGS"] = COUNT_TASKS_FLAG

    start = time.monotonic()
    rc, out, err = run_subprocess(
        cslc_cmd, cwd=work, env=env, shell=True, timeout=SCRIPT_TIMEOUT
    )
    total = time.monotonic() - start
    if rc != 0:
        raise RuntimeError(
            f"{config} compile failed for {name} {arch} {target}: rc={rc}\n"
            f"--stdout--\n{out}\n--stderr--\n{err}"
        )
    runs = [
        float(end) - float(begin)
        for begin, end in (line.split() for line in log.read_text().splitlines())
    ]
    return {
        "benchmark": name,
        "config": config,
        "arch": arch,
        "target": target,
        "rep": rep,
        "total_s": f"{total:.3f}",
        "cslc_s": f"{sum(runs):.3f}",
        "cslc_runs": len(runs),
        "cslc_run_s": ";".join(f"{run:.3f}" for run in runs),
    }


def run_timing(args, benches) -> None:
    """The --timing sweep: every configuration is compiled `--reps` times,
    strictly one at a time so that concurrent jobs do not inflate the times,
    into results/compile_times.csv."""
    csv_path = RESULTS / "compile_times.csv"
    if args.force and csv_path.exists():
        csv_path.unlink()
    done = load_existing(csv_path)
    timing_bins = setup_timing_bins(WORK / "timing" / "bin", INTEGRATION_BIN)

    todo = []
    for name, our, vanilla in benches:
        for cfg in args.configs:
            for arch in args.archs:
                for target in args.targets:
                    if target == "wse" and arch != "wse3":
                        continue
                    for rep in range(args.reps):
                        if (name, cfg, arch, target, rep) not in done:
                            todo.append((name, our, vanilla, cfg, arch, target, rep))
    print(f"{len(todo)} compiles to time ({len(done)} already in {csv_path})")

    lock = threading.Lock()
    failures = []
    for i, (name, our, vanilla, cfg, arch, target, rep) in enumerate(todo):
        tag = f"[{i + 1:>3}/{len(todo)}] {name:<32} {cfg:<10} {arch} {target} r{rep}"
        try:
            row = time_compile(
                name, our, vanilla, cfg, arch, target, args.sdk, timing_bins,
                rep, args.examples,
            )
        except Exception as e:
            print(f"{tag}  FAIL", file=sys.stderr, flush=True)
            print(f"  {str(e).splitlines()[0]}", file=sys.stderr, flush=True)
            failures.append((tag, str(e)))
            continue
        append_row(csv_path, row, lock, COMPILE_CSV_FIELDS)
        print(
            f"{tag}  total={row['total_s']:>8}s  cslc={row['cslc_s']:>8}s "
            f"({row['cslc_runs']} run{'s' if row['cslc_runs'] != 1 else ''})",
            flush=True,
        )

    if failures:
        print(f"\n{len(failures)} compiles failed", file=sys.stderr)
        for tag, why in failures:
            print(f"  {tag}:", file=sys.stderr)
            for line in why.splitlines():
                print(f"    {line}", file=sys.stderr)
        sys.exit(2)


def load_existing(csv_path: Path) -> set[tuple[str, str, str, str, int]]:
    if not csv_path.exists():
        return set()
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        if "target" not in (reader.fieldnames or []):
            raise SystemExit(
                f"{csv_path} predates the target column - run with --force to "
                "regenerate (this discards existing rows)."
            )
        return {
            (r["benchmark"], r["config"], r["arch"], r["target"], int(r["rep"]))
            for r in reader
        }


def append_row(
    csv_path: Path, row: dict, lock: threading.Lock, fields: list[str] = CSV_FIELDS
):
    with lock:
        new = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdk", type=Path, default=DEFAULT_SDK)
    ap.add_argument(
        "--examples",
        type=Path,
        default=None,
        help="directory of the vanilla baselines (default: the instrumented "
        "SDK examples, prepared in the work directory)",
    )
    ap.add_argument("--csl-pp", type=Path, default=DEFAULT_CSL_PP)
    ap.add_argument(
        "--configs",
        nargs="+",
        choices=["vanilla", "fiber", "trampoline"],
        default=["vanilla", "fiber", "trampoline"],
    )
    ap.add_argument(
        "--archs", nargs="+", choices=["wse2", "wse3"], default=["wse2", "wse3"]
    )
    ap.add_argument(
        "--targets",
        nargs="+",
        choices=["sim", "wse"],
        default=["sim"],
        help="sim = SDK fabric simulator; wse = a real CS-3 (wse3 only)",
    )
    ap.add_argument(
        "--compile-timeout",
        type=float,
        default=3600.0,
        help="seconds allowed for a whole compile+run script (backstop)",
    )
    ap.add_argument(
        "--run-timeout",
        type=float,
        default=120.0,
        help="seconds allowed per compile+run. Nothing here legitimately takes "
        "minutes on device, so a run that does has hung: it never returns on "
        "its own, and on hardware it would hold the reserved system until the "
        "sweep is killed.",
    )
    ap.add_argument(
        "--reps",
        type=int,
        default=1,
        help="runs per configuration. Simulation is deterministic so 1 is "
        "enough; hardware cycle counts vary, so use several and let "
        "tabulate.py take the median.",
    )
    ap.add_argument(
        "--timing",
        action="store_true",
        help="instead of compiling and running, only time the compiles "
        "(transpilation and cslc) into results/compile_times.csv, serially "
        "and --reps times each. target=wse compiles for the full wafer but "
        "reserves nothing.",
    )
    ap.add_argument(
        "--cmaddr",
        help="address (IP:port) of a CS-3 that this host can reach. Without "
        "it, --targets wse reserves one through the appliance SDK.",
    )
    ap.add_argument("--benchmarks", nargs="*", help="restrict to these benchmark names")
    ap.add_argument(
        "--force", action="store_true", help="ignore existing rows in results/data.csv"
    )
    ap.add_argument("--jobs", "-j", type=int, default=8)
    args = ap.parse_args()
    global RUN_TIMEOUT, SCRIPT_TIMEOUT
    RUN_TIMEOUT = args.run_timeout
    SCRIPT_TIMEOUT = args.compile_timeout
    if args.examples is None:
        args.examples = DEFAULT_EXAMPLES

    if not args.csl_pp.exists():
        print(f"csl-pp not found at {args.csl_pp}", file=sys.stderr)
        sys.exit(1)
    if not (args.sdk / "cslc").is_file():
        print(f"SDK cslc not found at {args.sdk}/cslc", file=sys.stderr)
        sys.exit(1)
    if "wse" in args.targets:
        wafer.check_client(args.cmaddr)
    configure(args.csl_pp, args.sdk)

    RESULTS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    benches = discover_benchmarks(args.examples)
    if args.benchmarks:
        benches = [b for b in benches if b[0] in set(args.benchmarks)]

    if args.timing:
        run_timing(args, benches)
        return

    csv_path = RESULTS / "data.csv"
    if args.force and csv_path.exists():
        csv_path.unlink()
    done = load_existing(csv_path)

    todo = []
    for name, our, vanilla in benches:
        for cfg in args.configs:
            for arch in args.archs:
                for target in args.targets:
                    # The cluster is a CS-3; there is no wse2 hardware to run on.
                    if target == "wse" and arch != "wse3":
                        continue
                    reps = [
                        r
                        for r in range(args.reps)
                        if (name, cfg, arch, target, r) not in done
                    ]
                    if reps:
                        todo.append((name, our, vanilla, cfg, arch, target, reps))

    print(
        f"{len(benches)} benchmarks; {len(todo)} configs to run "
        f"({len(done)} already in {csv_path})"
    )

    csv_lock = threading.Lock()
    print_lock = threading.Lock()
    failures: list[tuple[tuple, str]] = []
    # Configurations that failed as the paper reports; see EXPECTED_FAILURES.
    expected: list[tuple] = []
    counter = {"done": 0}

    def worker(task, system):
        name, our, vanilla, cfg, arch, target, reps = task
        rows = []
        for rep in reps:
            try:
                if target == "wse":
                    # Built in phase 1; every repetition runs on the system.
                    rows.append(run_built(
                        name, cfg, arch, target,
                        WORK / f"{name}__{cfg}__{arch}__{target}",
                        args.sdk, system, rep,
                    ))
                    continue
                rows.append(
                    compile_and_run(
                        name,
                        our,
                        vanilla,
                        cfg,
                        arch,
                        args.sdk,
                        args.csl_pp,
                        INTEGRATION_BIN,
                        target=target,
                        rep=rep,
                        examples_dir=args.examples,
                    )
                )
            except Exception as e:
                return task, rows, str(e)
        return task, rows, None

    def build(task):
        """Compile a hardware configuration without running it."""
        name, our, vanilla, cfg, arch, target, _reps = task
        try:
            compile_and_run(name, our, vanilla, cfg, arch, args.sdk, args.csl_pp,
                            INTEGRATION_BIN, target=target, rep=0,
                            examples_dir=args.examples)
        except Exception as e:
            return task, [], str(e)
        return task, [], None

    def report(task, rows, err):
        name, _our, _van, cfg, arch, target, _reps = task
        with print_lock:
            counter["done"] += 1
            tag = (
                f"[{counter['done']:>3}/{len(todo)}] {name:<32} "
                f"{cfg:<10} {arch} {target}"
            )
            for row in rows:
                append_row(csv_path, row, csv_lock)
                rep_tag = f" r{row['rep']}" if args.reps > 1 else ""
                print(
                    f"{tag}{rep_tag}  "
                    f"cycles={row['cycles_send']:>8}  "
                    f"tasks={row['tasks_total']:>2} (l={row['tasks_local']},"
                    f"c={row['tasks_control']},d={row['tasks_data']})  "
                    f"text={row['prog_size']:>6}  "
                    f"glob={row['globals_size']:>6}  "
                    f"loc={row['loc']:>4}",
                    flush=True,
                )
            if err and (name, cfg) in EXPECTED_FAILURES:
                print(f"{tag}  expected failure (see Table 1 of the paper)",
                      flush=True)
                expected.append(task)
            elif err:
                print(f"{tag}  FAIL", file=sys.stderr, flush=True)
                print(f"  {err.splitlines()[0]}", file=sys.stderr, flush=True)
                failures.append((task, err))

    sim_todo = [t for t in todo if t[5] == "sim"]
    wse_todo = [t for t in todo if t[5] == "wse"]

    # Simulation is CPU-bound and independent per config, so it fans out.
    if sim_todo:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = {pool.submit(worker, t, None): t for t in sim_todo}
            for fut in as_completed(futs):
                report(*fut.result())

    # Hardware: every configuration is compiled first, in parallel, so that
    # the system is held for the runs alone. Scheduling a job costs minutes, a
    # run seconds, so one job serves all runs; they run one at a time, and
    # each is bounded so that a hang cannot hold the system indefinitely.
    if wse_todo:
        built_ok = []
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for task, _rows, err in pool.map(build, wse_todo):
                if err:
                    report(task, [], err)
                else:
                    built_ok.append(task)
        if built_ok:
            with wafer.system(args.sdk, args.cmaddr, WORK / "appliance") as system:
                print(f"running on {system.describe()}", flush=True)
                for task in built_ok:
                    report(*worker(task, system))

    if expected:
        print(f"\n{len(expected)} configurations failed as expected: "
              "stackless GEMM and GEMV (see Table 1 of the paper)", flush=True)
    if failures:
        print(f"\n{len(failures)} configurations failed", file=sys.stderr)
        for task, why in failures:
            print(f"  {task[0]} {task[3]} {task[4]} {task[5]}:", file=sys.stderr)
            for line in why.splitlines():
                print(f"    {line}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
