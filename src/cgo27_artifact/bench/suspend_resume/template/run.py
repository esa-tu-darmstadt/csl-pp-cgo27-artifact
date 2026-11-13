#!/usr/bin/env cs_python
"""Runner for the suspend/resume microbenchmark.

Launches `f_start` on the single PE, then reads the three timestamp-counter
markers back and prints them for the driver (bench.py) to parse. Identical on
the simulator and on a real wafer -- the latter only adds --cmaddr.

A marker is the low 16 bits of the timestamp counter (see pe.csl). The
counter is zeroed by enable_tsc() and every interval of interest is a few
hundred cycles, so the differences bench.py takes are exact; they are still
reduced mod 2**16 so a wrap between two markers would come out right.
"""

import argparse

import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import (  # pylint: disable=no-name-in-module
    MemcpyDataType, MemcpyOrder, SdkRuntime,
)

N_MARKERS = 3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--cmaddr", default=None)
    args = p.parse_args()

    runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
    symbol_time_buf = runner.get_id("time_buf_u16")
    runner.load()
    runner.run()
    runner.launch("f_start", nonblock=False)
    runner.launch("f_read_timestamps", nonblock=False)

    raw = np.zeros(N_MARKERS, np.uint32)
    runner.memcpy_d2h(
        raw,
        symbol_time_buf,
        0, 0, 1, 1, N_MARKERS,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_16BIT,
        order=MemcpyOrder.COL_MAJOR,
        nonblock=False,
    )
    runner.stop()

    for i, m in enumerate(raw):
        print(f"M{i} = {int(m)}")


if __name__ == "__main__":
    main()
