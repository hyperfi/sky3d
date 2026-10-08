#!/usr/bin/env python3
"""Query this backend's actual cuFFT workspace and assess free VRAM headroom."""
import argparse
import ctypes
import json
import math
from pathlib import Path
import re
import struct

GIB = 1024**3


def allocation_bytes(grid, states):
    if len(grid) != 3 or any(n < 2 or n % 2 for n in grid) or states < 1:
        raise ValueError('Use positive even grid dimensions and a positive total state count')
    cells = math.prod(grid)
    if cells > (2**31 - 1)//22 or cells * 2 * states > 2**31 - 1:
        raise ValueError('Case exceeds current single-batch indexing limits')
    # Nine complex128 banks, 50 real64 fields, one weight/isospin per state.
    return cells * (2 * states * 16 * 9 + 50 * 8) + states * 12


def assess(required, free, fraction=0.8):
    if required <= 0 or free <= 0 or not 0.05 <= fraction <= 0.95:
        raise ValueError('Positive byte counts and safety fraction 0.05..0.95 required')
    if required > free * fraction:
        return 'insufficient', ('GPU case exceeds the free-VRAM safety budget. Use the parallel CPU backend. '
            'Paging or repeated chunk transfers in an alternative GPU implementation may be slower than CPU; '
            'this backend refuses oversubscription instead of paging.')
    if required > free * 0.60:
        return 'pressure', ('Limited free-VRAM headroom. Close other GPU workloads and benchmark against parallel CPU. '
            'Memory pressure may cause allocation failure; fitting in VRAM does not predict a speedup.')
    return 'fits', 'Memory fits with headroom. Only a matched complete-job benchmark can establish GPU speedup.'


def input_case(path):
    text = path.read_text()
    match = re.search(r'&grid\b(.*?)/', text, re.S | re.I)
    if not match:
        raise ValueError('Input must contain an explicit &grid section')
    grid = []
    for key in ('nx', 'ny', 'nz'):
        value = re.search(rf'\b{key}\s*=\s*(\d+)', match.group(1), re.I)
        if not value:
            raise ValueError(f'Explicit {key} required')
        grid.append(int(value.group(1)))
    # With one fragment, the actual count includes empty/unoccupied orbitals.
    fragment = re.search(r"\bfilename\s*=\s*(?:1\s*\*\s*)?['\"]([^'\"]+)['\"]", text, re.I)
    if not fragment:
        raise ValueError('Provide --states for an input without one fragment checkpoint')
    nof = re.search(r'\bnof\s*=\s*(\d+)', text, re.I)
    if not nof or int(nof.group(1)) != 1:
        raise ValueError('Automatic state count requires nof=1; pass explicit --states for other cases')
    state = (path.parent / fragment.group(1)).resolve()
    with state.open('rb') as f:
        size = struct.unpack('<i', f.read(4))[0]
        header = f.read(size)
        if size < 24 or f.read(4) != struct.pack('<i', size):
            raise ValueError('Invalid gfortran little-endian checkpoint header')
    return grid, struct.unpack_from('<i', header, 20)[0]


def query(library, grid, states):
    allocation_bytes(grid, states)  # Validate before entering CUDA/cuFFT.
    lib = ctypes.CDLL(str(library.resolve()))
    lib.sky_gpu_memory.argtypes = [ctypes.c_int] * 4 + [ctypes.POINTER(ctypes.c_uint64)]
    lib.sky_gpu_memory.restype = ctypes.c_int
    data = (ctypes.c_uint64 * 4)()
    if lib.sky_gpu_memory(*grid, states, data):
        raise RuntimeError('CUDA/cuFFT query failed; check the toolkit, GPU, and stderr diagnostic')
    required, workspace, free, total = data
    if required != allocation_bytes(grid, states) + workspace:
        raise RuntimeError('Library and preflight allocation formulas disagree; rebuild the backend')
    return dict(required_bytes=required, cufft_workspace_bytes=workspace,
                free_bytes=free, total_bytes=total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', type=int, nargs=3, metavar=('NX', 'NY', 'NZ'))
    parser.add_argument('--states', type=int, help='All propagated states, including unoccupied states; not just A')
    parser.add_argument('--input', type=Path, help='for005: reads grid and, for nof=1, fragment state count')
    parser.add_argument('--library', type=Path, default=Path(__file__).parent / 'build/libsky3d_gpu.so')
    parser.add_argument('--safety-fraction', type=float, default=0.8)
    parser.add_argument('--available-gib', type=float, help='Simulate free memory after a real workspace query; does not allocate a case')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    try:
        if args.input:
            if args.grid or args.states:
                # Explicit state count enables multi-fragment/restart/manual cases.
                if not args.states or not args.grid:
                    raise ValueError('Use both --grid and --states to override automatic input parsing')
                grid, states = args.grid, args.states
            else:
                grid, states = input_case(args.input.resolve())
        else:
            if not args.grid or args.states is None:
                raise ValueError('Use --input or both --grid and --states')
            grid, states = args.grid, args.states
        result = query(args.library, grid, states)
        free = result['free_bytes']
        if args.available_gib is not None:
            if not math.isfinite(args.available_gib) or args.available_gib <= 0:
                raise ValueError('Simulated available GiB must be finite and positive')
            free = int(args.available_gib * GIB)
        status, message = assess(result['required_bytes'], free, args.safety_fraction)
        result.update(grid=grid, states=states, status=status, message=message,
                      assessed_free_bytes=free, safety_fraction=args.safety_fraction,
                      simulated=args.available_gib is not None)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f'Preflight unavailable: {error}\nNo GPU suitability claim made.\n')
    print(('SIMULATION: ' if result['simulated'] else '') +
          f'{grid}, {states} states: required {result["required_bytes"]/GIB:.3f} GiB '
          f'(cuFFT workspace {result["cufft_workspace_bytes"]/GIB:.3f} GiB), '
          f'free {free/GIB:.3f} GiB; {status.upper()}')
    print(message)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + '\n')
    raise SystemExit(2 if status == 'insufficient' else 0)


if __name__ == '__main__':
    main()
