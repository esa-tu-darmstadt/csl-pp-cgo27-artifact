"""The CS-3 that the hardware runs of a sweep use.

The drivers compile every configuration on this host, for the full wafer, and
then run the benchmark's own runner script (run.py) on a CS-3. The runner is
the one that also drives the simulator; on hardware it is given the system's
address with --cmaddr. A system is obtained in one of two ways:

- Through the appliance SDK (the default). One SdkLauncher job reserves a
  CS-3 for the whole sweep. For each configuration, the build directory is
  staged into the job and the runner is started on the appliance with
  `launcher.run()`, which substitutes the system's address for %CMADDR%; the
  runner's standard output comes back as the result. This needs the appliance
  SDK's Python environment (cerebras.sdk.client) and nothing else on this host.
- Directly, with --cmaddr <address>. The runner is started on this host
  against that address, through the SDK's cs_python. This needs a host that
  can reach the system, over RoCE where the system is attached that way.

Either way the cycle counts come from the timestamp counters on the device,
so both report the same intervals. Hardware rows are wse3-only: the paper's
measurements were taken on a CS-3.
"""

import contextlib
import os
import shlex
import signal
import subprocess
import tempfile
from pathlib import Path

WSE3_FABRIC_DIMS = "762,1172"

# Proxy variables that gRPC honours. The appliance's management server is in
# the cluster's network; a web proxy configured for the login shell must not
# carry that connection, and proxies tend to drop its long-lived streams.
_PROXY_VARIABLES = ("grpc_proxy", "https_proxy", "http_proxy", "all_proxy",
                    "GRPC_PROXY", "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY")

# The RDMA NICs a directly attached system is reached over. The SDK's
# `cs_python` wrapper runs `singularity exec -C` (containall), which hides them
# from the container and leaves RoCE setup failing; APPTAINER_BIND is honoured
# even under -C. Only the paths that exist on the host are bound, since
# Apptainer rejects a missing bind source.
_RDMA_PATHS = (
    "/dev/infiniband",
    "/sys/class/infiniband",
    "/sys/class/infiniband_verbs",
)


class RunError(RuntimeError):
    """A runner failed on the system; the message carries its output."""


def check_client(cmaddr=None):
    """Fail now if a hardware sweep cannot obtain a system, rather than after
    compiling every configuration.

    Without `cmaddr`, the system is reserved through the appliance SDK, whose
    client lives in the appliance SDK's Python environment.
    """
    if cmaddr:
        return
    try:
        import cerebras.sdk.client  # noqa: F401
    except ModuleNotFoundError as e:
        raise SystemExit(
            "--targets wse reserves a CS-3 through the appliance SDK "
            f"(importing cerebras.sdk.client failed: {e}).\n"
            "Install the artifact with the extra [hardware] into a Python 3.11 "
            "environment (README.md, Installation), or pass the address of a "
            "system that this host can reach with --cmaddr."
        ) from e


def run_env(sdk_dir, env=None):
    """A `cs_python` environment that can reach a directly attached system."""
    env = dict(os.environ if env is None else env)
    env["PATH"] = f"{sdk_dir}:{env.get('PATH', '')}"
    rdma = ",".join(f"{p}:{p}" for p in _RDMA_PATHS if os.path.exists(p))
    if rdma:
        bind = env.get("APPTAINER_BIND")
        env["APPTAINER_BIND"] = f"{bind},{rdma}" if bind else rdma
    return env


def raise_memlock():
    """RoCE registers pinned memory; the default 64 KB limit is far too low."""
    import resource

    try:
        resource.setrlimit(
            resource.RLIMIT_MEMLOCK,
            (resource.RLIM_INFINITY, resource.RLIM_INFINITY),
        )
    except (ValueError, OSError):
        # Unprivileged and already capped -- let the run report the real error
        # rather than failing the whole sweep here.
        pass


class DirectSystem:
    """A system at a known address, reached from this host."""

    def __init__(self, cmaddr, sdk_dir):
        self.cmaddr = cmaddr
        self.sdk_dir = Path(sdk_dir)
        raise_memlock()

    def describe(self):
        return f"the system at {self.cmaddr}"

    def run(self, workdir, command, timeout, stage_root=None):
        """Run `command` (a cs_python invocation of the runner) in `workdir`
        against the system and return its standard output."""
        # -k escalates to SIGKILL: cs_python is a shell wrapper that execs the
        # container runtime, which does not exit on the SIGTERM plain
        # `timeout` sends.
        full = (f"timeout -k 10 {int(timeout)} {command} "
                f"--cmaddr {shlex.quote(self.cmaddr)}")
        proc = subprocess.run(full, shell=True, cwd=workdir,
                              env=run_env(self.sdk_dir), capture_output=True,
                              text=True, errors="replace")
        if proc.returncode != 0:
            raise RunError(f"runner failed with exit code {proc.returncode}\n"
                           f"--stdout--\n{proc.stdout}\n--stderr--\n{proc.stderr}")
        return proc.stdout


class LaunchedSystem:
    """A CS-3 reserved by one SdkLauncher job; runners execute on the
    appliance."""

    def __init__(self, launcher):
        self.launcher = launcher
        self._staged = set()

    def describe(self):
        return "a CS-3 reserved through the appliance SDK"

    def run(self, workdir, command, timeout, stage_root=None):
        """Stage `stage_root` (default: `workdir`) into the job once, unpack
        it there, then run `command` in the staged copy of `workdir` and return
        its standard output.

        The staged directory keeps its name on the appliance, so `stage_root`
        must have a name that is unique within the sweep. `launcher.run()`
        replaces cs_python with the appliance's Python and %CMADDR% with the
        system's address.
        """
        workdir = Path(workdir).resolve()
        root = Path(stage_root or workdir).resolve()
        unpack = ""
        if root not in self._staged:
            self.launcher.stage(str(root))
            # The client packs the directory into <root>.tar.gz next to it;
            # the appliance stores the archive as it is, so it is unpacked by
            # the first command that needs it.
            Path(f"{root}.tar.gz").unlink(missing_ok=True)
            archive = shlex.quote(f"{root.name}.tar.gz")
            unpack = f"tar -xzf {archive} && rm -f {archive} && "
            self._staged.add(root)
        remote = Path(root.name) / workdir.relative_to(root)
        full = (f"{unpack}cd {shlex.quote(str(remote))} && "
                f"timeout -k 10 {int(timeout)} {command} --cmaddr %CMADDR%")
        try:
            return self.launcher.run(full)
        except RuntimeError as e:
            raise RunError(str(e)) from e


@contextlib.contextmanager
def system(sdk_dir, cmaddr=None, logdir=None):
    """The CS-3 for the duration of the block.

    With `cmaddr`, the system at that address; otherwise one reserved through
    the appliance SDK. The appliance client writes its logs and job metadata
    (run_meta.json, <job>-cluster-details.json) to the current directory, so
    the block runs in `logdir` if one is given. The reservation is released
    when the block exits,
    including on exception: SIGTERM, SIGINT and SIGHUP are turned into
    exceptions first, because the release happens in SdkLauncher.__exit__,
    which a default-disposition signal would skip, leaving the job running
    and the system held. Proxy variables of the environment are removed for
    the duration of the job, so that the client reaches the cluster directly.
    """
    if cmaddr:
        yield DirectSystem(cmaddr, sdk_dir)
        return

    from cerebras.sdk.client import SdkLauncher

    def _raise(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")

    previous = {}
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            previous[sig] = signal.signal(sig, _raise)
        except (ValueError, OSError):
            pass  # not the main thread

    # The appliance client connects to the cluster directly, without a proxy.
    proxies = {name: os.environ.pop(name) for name in _PROXY_VARIABLES
               if name in os.environ}
    previous_cwd = os.getcwd()
    if logdir is not None:
        os.makedirs(logdir, exist_ok=True)
        os.chdir(logdir)
    with tempfile.TemporaryDirectory(prefix="wafer-job-") as tmp:
        # SdkLauncher uploads a first artifact when the job starts and derives
        # the job's working directory from its name. The configurations are
        # staged into that directory afterwards, so a small file suffices.
        first = os.path.join(tmp, "cgo27_ae_job.txt")
        with open(first, "w", encoding="utf8") as f:
            f.write("csl-pp CGO 2027 artifact: hardware sweep\n")
        try:
            # disable_version_check: the client may trail the cluster's
            # server; the commands used here are the stable ones.
            with SdkLauncher(first, simulator=False,
                             disable_version_check=True) as launcher:
                yield LaunchedSystem(launcher)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            os.chdir(previous_cwd)
            os.environ.update(proxies)
