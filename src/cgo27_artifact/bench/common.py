"""Paths and tools shared by the benchmark drivers of the artifact.

The drivers write their results and build directories below the output
directory, CGO27_AE_OUT (default: ./cgo27-ae-output in the working
directory), one subdirectory per driver: <out>/<driver>/results and
<out>/<driver>/work. The package itself is never written to, so it can be
installed read-only.

csl-pp is the one installed with the artifact unless a driver is given
another one with --csl-pp. The ports import their own copy of the libraries
(benchmarks/coro-benchmarks/benchmark-libs), the one the paper measured.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent
BENCHMARKS = PACKAGE / "benchmarks"
# The seven SDK benchmarks written with the language extensions of csl++.
CORO_BENCHMARKS = BENCHMARKS / "coro-benchmarks"
# The examples of the Cerebras SDK, github.com/Cerebras/sdk-examples at tag
# v2.10.0, unmodified, and the patches that instrument the vanilla baselines.
SDK_EXAMPLES = PACKAGE / "vendor" / "sdk-examples"
VANILLA_PATCHES = BENCHMARKS / "vanilla-patches"
PAPER_DATA = PACKAGE / "paper"

# CSL_SDK_DIR names the unpacked Cerebras SDK 2.10.0, which contains cslc and
# the SDK image (*.sif).
DEFAULT_SDK = Path(os.environ.get("CSL_SDK_DIR", Path.home() / "cs_sdk-2.10.0"))


def output_root() -> Path:
    """The directory below which all drivers write."""
    return Path(os.environ.get("CGO27_AE_OUT", Path.cwd() / "cgo27-ae-output")).resolve()


def results_dir(driver: str) -> Path:
    """Where a driver writes its CSV files, tables and figures."""
    return output_root() / driver / "results"


def work_dir(driver: str) -> Path:
    """Where a driver keeps its per-configuration build directories."""
    return output_root() / driver / "work"


def default_csl_pp() -> Path:
    """csl-pp of the Python environment the artifact is installed in, else the
    first csl-pp on PATH."""
    candidate = Path(sys.executable).parent / "csl-pp"
    if candidate.is_file():
        return candidate
    found = shutil.which("csl-pp")
    return Path(found) if found else candidate


def sdk_image(sdk: Path) -> Path:
    """The SDK image (*.sif) in the SDK directory."""
    images = sorted(Path(sdk).glob("*.sif"))
    if len(images) != 1:
        raise RuntimeError(f"expected one *.sif in {sdk}, found {len(images)}")
    return images[0]


def make_cslc_wrapper(bin_dir: Path, csl_pp: Path, sdk: Path) -> Path:
    """Write <bin_dir>/cslc, which runs csl-pp in place of cslc.

    The benchmarks' own commands_wseN.sh call cslc; with bin_dir first on PATH
    they run csl-pp instead, which calls the SDK's cslc. CORO_MODE selects the
    coroutine lowering (fiber, the default, or trampoline). Returns bin_dir.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    wrapper = bin_dir / "cslc"
    wrapper.write_text(
        "#!/bin/bash\n"
        'CORO_ARGS=""\n'
        'if [ -n "${CORO_MODE:-}" ]; then\n'
        '    CORO_ARGS="--coro-mode=$CORO_MODE"\n'
        "fi\n"
        f'exec "{Path(csl_pp).resolve()}" --cslc="{Path(sdk).resolve() / "cslc"}" '
        f'--system-module-sif="{sdk_image(sdk)}" $CORO_ARGS "$@"\n'
    )
    wrapper.chmod(0o755)
    return bin_dir


def prepare_vanilla_sources(dest: Path) -> Path:
    """Copy the SDK's benchmarks to `dest` and instrument them.

    Each patch in vanilla-patches/ adds TSC timestamps around the interval
    that the paper measures, so that a vanilla baseline and its port with
    coroutines report the same interval. Returns `dest`, which then holds
    benchmark-libs/ and one directory per benchmark.
    """
    patch = shutil.which("patch")
    if patch is None:
        raise RuntimeError("the vanilla baselines need the 'patch' tool on PATH")
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(SDK_EXAMPLES / "benchmarks", dest)
    for diff in sorted(VANILLA_PATCHES.glob("*.patch")):
        with diff.open("rb") as stream:
            subprocess.run([patch, "-s", "-p1", "--no-backup-if-mismatch"],
                           stdin=stream, cwd=dest, check=True)
    return dest
