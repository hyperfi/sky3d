#!/usr/bin/env python3
"""Exercise real memory queries, a labeled simulation, and runtime refusal."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from preflight import GIB, assess, query


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, default=Path(__file__).parent / 'build/libsky3d_gpu.so')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results/preflight.json')
    args = parser.parse_args()
    rows = []
    for grid, states in [([24]*3, 20), ([48]*3, 208), ([64]*3, 208)]:
        data = query(args.library, grid, states)
        status, message = assess(data['required_bytes'], data['free_bytes'])
        rows.append(dict(grid=grid, states=states, **data, status=status, message=message, simulated=False))
    data = rows[1]
    for free_gib, expected in [(8, 'pressure'), (6, 'insufficient')]:
        status, message = assess(data['required_bytes'], free_gib*GIB)
        assert status == expected
        rows.append(dict(grid=data['grid'], states=data['states'], required_bytes=data['required_bytes'],
                         assessed_free_bytes=free_gib*GIB, status=status, message=message, simulated=True))
    # Pick a case that is provably rejected before the C API reads host pointers.
    # This tests the executable backend's guard without allocating nucleus arrays.
    grid, states = [64]*3, 208
    while True:
        data = query(args.library, grid, states)
        if data['required_bytes'] > data['total_bytes']:
            break
        states *= 2
    child = '''import ctypes,sys
lib=ctypes.CDLL(sys.argv[1])
lib.sky_gpu_create.argtypes=[ctypes.c_int]*4+[ctypes.c_void_p]*4
lib.sky_gpu_create.restype=ctypes.c_void_p
lib.sky_gpu_create(64,64,64,int(sys.argv[2]),None,None,None,None)
'''
    import os
    result = subprocess.run([sys.executable, '-c', child, str(args.library.resolve()), str(states)],
                            capture_output=True, text=True, env=dict(os.environ, SKY3D_GPU_MEMORY_FRACTION='0.8'))
    assert result.returncode == 3 and 'Insufficient free VRAM' in result.stderr, result
    evidence = dict(cases=rows, runtime_refusal=dict(grid=grid, states=states, returncode=result.returncode,
                    stdout=result.stdout.strip(), stderr=result.stderr.strip(),
                    note='Guard test only: array allocation and null host-pointer reads were never reached'), passed=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + '\n')
    for row in rows:
        print(row['grid'], row['states'], row['status'], 'simulation' if row['simulated'] else 'actual workspace query')
    print('Runtime refused the oversized case before allocation; all memory checks passed')


if __name__ == '__main__':
    main()
