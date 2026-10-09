#!/usr/bin/env python3
"""Check actual plan memory, policy boundaries and refusal before GPU arrays."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from preflight import assess, query, GIB
from validate_extended import sha256, verify_build
from benchmark import checkpoint
from validate_single_gpu import static_input


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir',type=Path,required=True)
    parser.add_argument('--state',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    build,exes=verify_build(args.build_dir.resolve())
    library=args.build_dir.resolve()/'libsky3d_gpu.so'
    results=[]
    for grid,states in [([24]*3,20),([40]*3,20),([48]*3,208),([64]*3,208)]:
        data=query(library,grid,states)
        status,message=assess(data['required_bytes'],data['free_bytes'])
        results.append(dict(grid=grid,states=states,**data,status=status,message=message))
    # The 12GB example is a simulation on larger GPUs, not a hardware assertion.
    simulated_large_status,_=assess(results[-1]['required_bytes'],12*GIB)
    simulations=[]
    for free in (8*GIB,6*GIB):
        status,message=assess(7*GIB,free)
        simulations.append(dict(required_bytes=7*GIB,simulated_free_bytes=free,status=status,message=message))
        if status!='insufficient':
            raise AssertionError('Oversubscription simulation should refuse GPU')
    status,message=assess(7*GIB,10*GIB)
    if status!='pressure':
        raise AssertionError('Pressure warning boundary')
    simulations.append(dict(required_bytes=7*GIB,simulated_free_bytes=10*GIB,status=status,message=message))
    root=Path(tempfile.mkdtemp(prefix='sky3d-memory-guard-',dir=Path.home()/'.cache'))
    grid=None
    for n in (64,96,128,192,256):
        data=query(library,[n]*3,20)
        # Twice the refusal boundary against TOTAL memory makes the guard
        # independent of other applications releasing memory before launch.
        if data['required_bytes']>0.10*data['total_bytes']:
            grid=[n]*3
            break
    if grid is None:
        raise RuntimeError('No bounded guard fixture available for this GPU capacity')
    path=root/'oversized'
    path.mkdir()
    child="""import ctypes,sys
lib=ctypes.CDLL(sys.argv[1])
lib.sky_gpu_create.argtypes=[ctypes.c_int]*4+[ctypes.c_void_p]*4
lib.sky_gpu_create.restype=ctypes.c_void_p
spacing=(ctypes.c_double*3)(0.6,0.6,0.6)
psi=(ctypes.c_double*2)()
weights=(ctypes.c_double*20)(*[1.0]*20)
species=(ctypes.c_int*20)(*([1]*10+[2]*10))
# Required arrays exceed 10% of total VRAM, while the selected budget is 5%.
# State must refuse BEFORE reading or copying these minimal host fixtures.
lib.sky_gpu_create(*map(int,sys.argv[2:5]),20,spacing,psi,weights,species)
raise RuntimeError('Memory guard returned unexpectedly')
"""
    import sys
    with (path/'stdout.log').open('w') as log:
        result=subprocess.run([sys.executable,'-c',child,str(library),*map(str,grid)],
             cwd=path,stdout=log,stderr=subprocess.STDOUT,
             env=dict(os.environ,SKY3D_GPU_MEMORY_FRACTION='0.05'))
    text=(path/'stdout.log').read_text()
    if result.returncode!=3 or 'Insufficient free VRAM with safety headroom' not in text:
        raise AssertionError(f'Memory guard failed: {path}: {result.returncode}')
    # Exercise static residual/preconditioner kernels, including the first
    # occupied-space diagonalization after iteration 20. Dynamic memcheck
    # alone does not execute this path.
    static_path=root/'static-memcheck'
    static_path.mkdir()
    (static_path/'for005').write_text(static_input(10,'Sly4','VDI',steps=25))
    sanitizer=Path(os.environ.get('CUDA_HOME','/usr/local/cuda'))/'bin/compute-sanitizer'
    with (static_path/'stdout.log').open('w') as log:
        subprocess.run([str(sanitizer),'--tool','memcheck','--error-exitcode','1',str(exes['gpu'])],
            cwd=static_path,stdout=log,stderr=subprocess.STDOUT,check=True,
            env=dict(os.environ,SKY3D_BACKEND='gpu',OMP_NUM_THREADS='8',
                     OPENBLAS_NUM_THREADS='1',OMP_PROC_BIND='close',OMP_PLACES='cores'))
    state=checkpoint(static_path/'static.tdhf')
    if state['step']!=25 or 'ERROR SUMMARY: 0 errors' not in (static_path/'stdout.log').read_text():
        raise AssertionError(f'Static memcheck endpoint or error summary missing: {static_path}')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(build=build,state_sha256=sha256(args.state) if args.state else None,runner_sha256=sha256(Path(__file__)),
        cases=results,simulations=simulations,large_case_simulated_12gib_status=simulated_large_status,passed=True,
        static_memcheck=dict(directory=str(static_path),steps=25,force='Sly4',pairing='VDI',
            states=state['states'],input_sha256=sha256(static_path/'for005'),errors=0),
        runtime_refusal=dict(directory=str(path),
        returncode=result.returncode,safety_fraction=0.05,grid=grid,
        note='Reduced safety fraction exercises refusal before large device-bank allocation; this is a guard test, not a timing or physics run.')),indent=2)+'\n')
    print('Actual-plan memory and runtime refusal checks passed:',args.output,flush=True)


if __name__=='__main__':
    main()
