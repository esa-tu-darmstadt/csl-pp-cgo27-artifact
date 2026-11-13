#!/usr/bin/env python3
"""Driver for the coroutine-runtime profile.

Runs the same benchmarks as ../end2end, but gives each @await_parallel site an
ordinary named wrapper, and reports where each PE's dispatched instructions
went: the fiber runtime, the task allocator, those wrappers, the system
modules, or user code. LLVM may inline the wrappers and runtime functions;
attribution follows their DWARF inline scopes in the final PE ELF.

These are deliberately SEPARATE runs from the end2end sweep, written to a
separate CSV. The named wrapper changes the input to backend optimization even
when LLVM inlines it away, so `runs.csv` keeps the profiled cycle count next to
the end2end one and makes any residual inflation visible.

Fiber only (the stackless backend has no separable runtime to attribute cycles
to, and csl-pp rejects the flag there) and simulator only (the instruction
trace comes from the simulator's CTF output; hardware has none).

The per-benchmark knowledge -- where the sources live, how each one is built
and run, which vanilla variant matches -- is reused from ../end2end/bench.py
rather than duplicated. Only the work tree and the results are separate.
"""

import argparse
import csv
import importlib.util
import os
import shutil
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
E2E = HERE.parent / "end2end"
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

RESULTS = common.results_dir("profile")
WORK = common.work_dir("profile")


def load_end2end_driver():
    """Load ../end2end/bench.py without colliding with this file's name.

    The SDK's container-hosted Python may register the executed profile driver
    as ``bench``. An ordinary ``import bench`` can then return this module
    instead of the sibling end-to-end driver even after changing ``sys.path``.
    Loading the exact file under a distinct name avoids depending on launcher
    behavior.
    """
    path = E2E / "bench.py"
    spec = importlib.util.spec_from_file_location("paper_bench_end2end", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load end-to-end benchmark driver: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e2e = load_end2end_driver()

# The trace tool, github.com/esa-tu-darmstadt/cerebras-simtracer, at the commit
# named in vendor/simtracer/COMMIT. CEREBRAS_SIMTRACER_DIR selects another
# checkout.
SIMTRACER = os.environ.get("CEREBRAS_SIMTRACER_DIR",
                           str(common.PACKAGE / "vendor" / "simtracer"))

CSV_FIELDS = ["benchmark", "arch", "tile", "function", "instructions"]
RUN_FIELDS = ["benchmark", "arch", "cycles_instrumented", "tiles", "events"]
CALL_FIELDS = ["benchmark", "arch", "caller", "callee", "entries"]
NEST_FIELDS = ["benchmark", "arch", "tile", "path", "cycles"]
TILE_FIELDS = ["benchmark", "arch", "tile", "task_cycles"]

# Separator between frames of a nested path, outermost first.
PATH_SEP = ">"

# Prefixes of functions the coroutine lowering emits. Nested attribution
# groups a runtime call under the runtime function that was entered from
# user code, so these are the only names that can open such a group.
CORO_PREFIXES = ("__fiber_runtime__", "__alloc_runtime__",
                 "__fiber_await_parallel_", "__fiber_reset_",
                 "__fiber_stack_")


def is_coro(fn: str) -> bool:
    return fn.startswith(CORO_PREFIXES)


def canonical(fn: str) -> str:
    """Resolve a linker jump thunk to the function it dispatches to.

    `trampoline.X` is a 16-byte thunk that tail-jumps to X. It is not a call
    and never returns, so treating it as a stack frame invents a parent that
    then swallows X and everything X calls. Collapsing the two names removes
    the pseudo-frame: the thunk's instructions are charged to X, and X's own
    entry does not open a second frame.
    """
    return fn[len("trampoline."):] if fn.startswith("trampoline.") else fn


def coroutine_suffix(frames) -> tuple:
    """Return the unbroken coroutine-generated suffix of logical frames.

    This makes attribution exclusive. In particular, an inlined synchronous
    callee gives ``await_parallel > user_callee`` and therefore an empty
    suffix. If that callee contains ``suspend``, only ``suspend`` survives:
    the intervening user frame still prevents charging it to the wrapper.
    """
    suffix = []
    for frame in reversed(frames):
        frame = canonical(frame)
        if not is_coro(frame):
            break
        if not suffix or suffix[-1] != frame:
            suffix.append(frame)
    return tuple(reversed(suffix))


INSTRUMENT_FLAG = "--coro-instrument=await-parallel"


def truncate_to_observed_calls(path: tuple, edges: Counter) -> tuple:
    """Drop outer frames of `path` that never actually called the next one.

    Some runtime functions never return: `set_sp_and_jmp` swaps the stack
    pointer and jumps, and `worker_entry` tail-jumps into it. Their frames
    therefore stay on the reconstructed stack for the whole lifetime of the
    fiber they launched, and a runtime function entered later -- on the
    fiber's stack, with the coroutine body's frame already popped -- lands
    directly on top of them. The stack shape alone cannot distinguish that
    from a real call.

    The entry census can. It records physical calls at function entry addresses
    and logical calls when execution enters a DWARF inline range. Keeping only
    the innermost run of frames whose adjacent pairs were observed cuts each
    path back to what really happened, without naming any symbol.
    """
    depth = len(path) - 1
    while depth > 0 and edges.get((path[depth], path[depth - 1]), 0):
        depth -= 1
    return path[depth:]


def per_tile_profile(out_dir: Path) -> tuple[dict[int, Counter],
                                             Counter,
                                             dict[int, Counter],
                                             Counter]:
    """Flat instruction counts, a call census, nested cycles, and denominators.

    Returns ``(per_tile, edges, nested, task_cycles)``.

    ``per_tile`` counts instructions per (tile, function) by instruction
    pointer against the per-tile ELF's DWARF inline scopes, falling back to its
    physical symbol table. The innermost function owns the instruction, so the
    partition remains exclusive and exhaustive.

    ``edges`` records why a function ran: physical entries come from the ELF
    entry address and inline entries from transitions into a DWARF scope.
    Thus `suspend` reached directly from a user coroutine remains distinct
    from `caller_join > suspend` even after both calls are inlined.

    ``nested`` is the flamegraph, charged in cycles. Physical call stacks and
    logical DWARF inline frames come from the coroutine-aware simtracer tool.
    Each interval between frame events is charged to the unbroken run of
    runtime frames at the top of the stack at that moment,
    keyed by the whole path. Two consequences follow from "unbroken", and both
    are the point of doing it this way:

      * a coroutine body reached from `set_sp_and_jmp` on a fresh start is a
        user frame, so it terminates the run and none of its cycles reach the
        runtime frames that started it;
      * the callee an `@await_parallel` wrapper runs synchronously is likewise
        a user frame, so the wrapper is not charged with the work it exists to
        spawn -- while `bind_task` and `caller_join`, which the wrapper calls
        directly, are charged to the path through it.

    ``task_cycles`` is the per-tile denominator: cycles during which some task
    was executing. Wall-clock span is the wrong basis because a PE that idles
    between activations would dilute every percentage.
    """
    sys.path.insert(0, SIMTRACER)
    from callstack import reconstruct  # noqa: E402
    from ctf import (  # noqa: E402
        build_elf_mapping, load_all_elf_lookups, parse_ctf_trace,
        resolve_bin_root,
    )
    import simtrace as st  # noqa: E402

    trace_dir = st.resolve_trace_dir(str(out_dir), None)
    bin_root = resolve_bin_root(str(out_dir), None)
    grid_width, _ = st.read_grid_dims(trace_dir)
    lookups, _arch = load_all_elf_lookups(bin_root, with_dwarf=True)
    if not lookups:
        raise RuntimeError(f"no ELFs with symbols under {bin_root}")
    mapping = build_elf_mapping(lookups, grid_width, quiet=True)

    per_tile: dict[int, Counter] = {}
    edges: Counter = Counter()
    inline_edges: Counter = Counter()

    def census(stream):
        """Yield events unchanged, accumulating the flat counts and the census.

        The reconstruction consumes the event stream once, so the per-symbol
        work rides along with it rather than costing a second pass over a
        trace that runs to hundreds of thousands of events per configuration.
        """
        prev: dict[int, tuple[str, int]] = {}
        prev_inline: dict[int, tuple[tuple, int]] = {}
        for evt in stream:
            lookup = lookups.get(mapping.get(evt.tile_index))
            if lookup is not None:
                addr = evt.inst_ptr * 2
                raw_name, _ = lookup.lookup(addr)
                name = canonical(raw_name)
                chain = tuple(canonical(frame)
                              for frame in lookup.inline_chain(addr))
                # An inline leaf is the source function responsible for this
                # instruction. With no applicable DWARF range preserve the
                # raw physical name (including a linker thunk) as before.
                attributed = chain[-1] if len(chain) > 1 else raw_name
                per_tile.setdefault(evt.tile_index,
                                     Counter())[attributed] += 1
                inline_before = prev_inline.get(evt.tile_index)
                common = 0
                if inline_before is not None \
                        and inline_before[1] == evt.task_color:
                    old_chain = inline_before[0]
                    limit = min(len(old_chain), len(chain))
                    while common < limit \
                            and old_chain[common] == chain[common]:
                        common += 1
                # Each newly opened DWARF frame contributes one logical
                # entry. Index zero is the physical frame and is covered by
                # the entry-address census below.
                for index in range(max(1, common), len(chain)):
                    inline_edges[(chain[index], chain[index - 1])] += 1
                prev_inline[evt.tile_index] = (chain, evt.task_color)
                before = prev.get(evt.tile_index)
                if (addr in lookup.entry_set and before is not None
                        and before[0] != name
                        and before[1] == evt.task_color):
                    edges[(name, before[0])] += 1
                prev[evt.tile_index] = (name, evt.task_color)
            yield evt

    nested: dict[int, Counter] = {}
    task_cycles: Counter = Counter()
    stacks: dict[int, list[str]] = {}
    last: dict[int, int] = {}

    def charge(tile: int, stack: list[str], cycles: int) -> None:
        """Charge an interval to the runtime frames on top of `stack`."""
        if cycles <= 0 or not stack:
            return
        # stack[0] is the task frame, so a non-empty stack means the tile was
        # executing; everything above it is the call chain.
        task_cycles[tile] += cycles
        path = coroutine_suffix(stack[1:])
        if path:
            nested.setdefault(tile, Counter())[path] += cycles

    events = parse_ctf_trace(trace_dir, want_ids=(2,), progress=None)
    for kind, tile, label, cycle in reconstruct(census(events), mapping,
                                                lookups, inline_frames=True):
        stack = stacks.setdefault(tile, [])
        before = last.get(tile)
        if before is not None:
            charge(tile, stack, cycle - before)
        last[tile] = cycle
        if kind == "O":
            stack.append(canonical(label))
        elif stack:
            stack.pop()

    # The census is only complete once the whole trace has been read, so the
    # paths are cut back here rather than while they are being accumulated.
    observed_edges = edges + inline_edges
    pruned: dict[int, Counter] = {}
    for tile, counter in nested.items():
        merged: Counter = Counter()
        for path, n in counter.items():
            merged[truncate_to_observed_calls(path, observed_edges)] += n
        pruned[tile] = merged
    return per_tile, observed_edges, pruned, task_cycles


def find_out_dir(cfg_dir: Path, name: str) -> Path:
    """The simulator output dir holding bin/*.elf and simfab_traces/."""
    candidates = sorted(cfg_dir.glob(f"{name}/**/out"))
    for cand in candidates:
        if (cand / "bin").is_dir():
            return cand
    raise RuntimeError(f"no out/ with bin/ under {cfg_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=e2e.BENCHMARKS)
    ap.add_argument("--archs", nargs="+", choices=["wse2", "wse3"],
                    default=["wse2", "wse3"])
    ap.add_argument("--csl-pp", type=Path, default=e2e.DEFAULT_CSL_PP)
    ap.add_argument("--sdk", type=Path, default=e2e.DEFAULT_SDK)
    ap.add_argument("--examples", type=Path, default=e2e.DEFAULT_EXAMPLES)
    ap.add_argument("--force", action="store_true",
                    help="re-run configs already in the CSV")
    ap.add_argument("--keep-traces", action="store_true",
                    help="keep simfab_traces/ (~150 MB per config)")
    ap.add_argument("--jobs", "-j", type=int, default=4)
    args = ap.parse_args()

    if not args.csl_pp.exists():
        raise SystemExit(f"csl-pp not found at {args.csl_pp} — build it first")

    RESULTS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / "data.csv"
    runs_path = RESULTS / "runs.csv"
    calls_path = RESULTS / "calls.csv"
    nest_path = RESULTS / "nested.csv"
    tiles_path = RESULTS / "tiles.csv"
    if args.force:
        for p in (csv_path, runs_path, calls_path, nest_path, tiles_path):
            p.unlink(missing_ok=True)

    done = set()
    if csv_path.exists():
        with csv_path.open() as f:
            done = {(r["benchmark"], r["arch"]) for r in csv.DictReader(f)}

    e2e.configure(args.csl_pp, args.sdk)
    found = {n: (s, v) for n, s, v in e2e.discover_benchmarks(args.examples)}
    missing = [b for b in args.benchmarks if b not in found]
    if missing:
        raise SystemExit(f"benchmarks not found in {args.examples}: {missing}")

    todo = [(b, a) for b in args.benchmarks for a in args.archs
            if (b, a) not in done]
    if len(todo) < len(args.benchmarks) * len(args.archs):
        print(f"skipping {len(args.benchmarks) * len(args.archs) - len(todo)} "
              f"configs already in {csv_path}")
    print(f"running {len(todo)} configs (fiber, sim, {INSTRUMENT_FLAG})",
          flush=True)

    write_lock = threading.Lock()
    print_lock = threading.Lock()
    failures = []

    def work(cfg):
        name, arch = cfg
        src, vanilla = found[name]
        started = time.time()
        row = e2e.compile_and_run(
            name, src, vanilla, "fiber", arch, args.sdk, args.csl_pp,
            e2e.INTEGRATION_BIN, target="sim", examples_dir=args.examples,
            extra_csl_flags=INSTRUMENT_FLAG, work_root=WORK,
        )
        cfg_dir = WORK / f"{name}__fiber__{arch}__sim"
        out_dir = find_out_dir(cfg_dir, name)
        per_tile, edges, nested, task_cycles = per_tile_profile(out_dir)
        if not args.keep_traces:
            shutil.rmtree(out_dir / "simfab_traces", ignore_errors=True)
        return cfg, row, per_tile, edges, nested, task_cycles, \
            time.time() - started

    def emit(cfg, row, per_tile, edges, nested, task_cycles, secs):
        name, arch = cfg
        events = sum(sum(c.values()) for c in per_tile.values())
        with write_lock:
            new = not csv_path.exists()
            with csv_path.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                if new:
                    w.writeheader()
                for tile in sorted(per_tile):
                    for fn, n in sorted(per_tile[tile].items()):
                        w.writerow({"benchmark": name, "arch": arch,
                                    "tile": tile, "function": fn,
                                    "instructions": n})
            new = not nest_path.exists()
            with nest_path.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=NEST_FIELDS)
                if new:
                    w.writeheader()
                for tile in sorted(nested):
                    for path, n in sorted(nested[tile].items()):
                        w.writerow({"benchmark": name, "arch": arch,
                                    "tile": tile,
                                    "path": PATH_SEP.join(path), "cycles": n})
            new = not tiles_path.exists()
            with tiles_path.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=TILE_FIELDS)
                if new:
                    w.writeheader()
                for tile in sorted(task_cycles):
                    w.writerow({"benchmark": name, "arch": arch, "tile": tile,
                                "task_cycles": task_cycles[tile]})
            new = not calls_path.exists()
            with calls_path.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=CALL_FIELDS)
                if new:
                    w.writeheader()
                for (callee, caller), n in sorted(edges.items()):
                    w.writerow({"benchmark": name, "arch": arch,
                                "caller": caller, "callee": callee,
                                "entries": n})
            new = not runs_path.exists()
            with runs_path.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
                if new:
                    w.writeheader()
                w.writerow({"benchmark": name, "arch": arch,
                            "cycles_instrumented": row.get("cycles_send"),
                            "tiles": len(per_tile), "events": events})
        with print_lock:
            print(f"  {name:<36} {arch}  {len(per_tile):>3} tiles  "
                  f"{events:>9,} instr  [{secs:5.0f}s]", flush=True)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(work, cfg): cfg for cfg in todo}
        for fut in as_completed(futs):
            cfg = futs[fut]
            try:
                emit(*fut.result())
            except Exception as e:  # noqa: BLE001 - reported, not swallowed
                with print_lock:
                    print(f"  {cfg[0]:<36} {cfg[1]}  FAIL: "
                          f"{str(e).strip()[:400]}", file=sys.stderr, flush=True)
                failures.append(cfg)

    if failures:
        print(f"\n{len(failures)} configs failed: {failures}", file=sys.stderr)
        sys.exit(2)
    print(f"\nwrote {csv_path}, {calls_path}, {nest_path}, {tiles_path} "
          f"and {runs_path}")


if __name__ == "__main__":
    main()
