import importlib.util
import unittest
from pathlib import Path


def _load_bench():
    path = Path(__file__).with_name("bench.py")
    spec = importlib.util.spec_from_file_location("profile_bench_tested", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_bench()


class CoroutineSuffixTest(unittest.TestCase):
    def test_inlined_sync_callee_excludes_wrapper(self):
        self.assertEqual(
            bench.coroutine_suffix(
                ["__fiber_await_parallel_0", "user_sync"]),
            (),
        )

    def test_runtime_inside_sync_callee_does_not_reinclude_wrapper(self):
        self.assertEqual(
            bench.coroutine_suffix([
                "__fiber_await_parallel_0",
                "user_sync",
                "__fiber_runtime__suspend",
            ]),
            ("__fiber_runtime__suspend",),
        )

    def test_unbroken_runtime_path_keeps_wrapper(self):
        self.assertEqual(
            bench.coroutine_suffix([
                "__fiber_await_parallel_0",
                "__fiber_runtime__caller_join",
                "__fiber_runtime__suspend",
            ]),
            (
                "__fiber_await_parallel_0",
                "__fiber_runtime__caller_join",
                "__fiber_runtime__suspend",
            ),
        )


if __name__ == "__main__":
    unittest.main()
