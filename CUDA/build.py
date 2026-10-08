#!/usr/bin/env python3
"""Build isolated CPU/reference/GPU executables in Linux/WSL, without touching Code/ objects."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-dir', type=Path, default=REPO / 'CUDA/build')
    parser.add_argument('--arch', default='native', help='CUDA architecture, native or e.g. sm_120')
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if os.name != 'posix':
        parser.error('Run the build in Linux/WSL')
    cuda = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda'))
    nvcc = cuda / 'bin/nvcc'
    if not nvcc.is_file():
        parser.error('Set CUDA_HOME to a compatible CUDA toolkit (12.8+ for RTX 5070)')
    root = args.build_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    library = root / 'libsky3d_gpu.so'
    cmd = [str(nvcc), '-O3', '-std=c++17', f'-arch={args.arch}', '-shared', '-Xcompiler=-fPIC',
           str(REPO / 'CUDA/sky_gpu.cu'), '-lcufft',
           '-Xlinker=-rpath', f'-Xlinker={cuda}/lib64', '-o', str(library)]
    with (root / 'cuda-build.log').open('w') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Built CUDA library:', library, flush=True)
    variants = {'reference': '-O3 -march=native -ffp-contract=off -fopenmp',
                'cpu': '-O3 -march=native -ffast-math -fopenmp',
                'gpu': '-O3 -march=native -ffast-math -fopenmp'}
    for name, flags in variants.items():
        path = root / name
        path.mkdir(exist_ok=True)
        # Refuse to reuse objects with potentially different compiler flags.
        # Only remove explicitly enumerated generated files inside this variant.
        for file in path.iterdir():
            if file.is_file() and file.suffix in {'.o', '.mod', '.smod'}:
                file.unlink()
        for source in (REPO / 'Code').iterdir():
            if source.suffix in {'.f90', '.f', '.data'} or source.name == 'Makefile':
                shutil.copy2(source, path / source.name)
        libs = '-lfftw3 -llapack -lopenblas'
        if name == 'gpu':
            shutil.copy2(REPO / 'CUDA/gpu_runtime.f90', path / 'gpu_runtime.f90')
            libs += f' -L"{root}" -lsky3d_gpu -Wl,-rpath,"{root}"'
        cmd = ['make', f'-j{args.jobs}', 'makeit', 'COMPILER_SKY=gfortran', 'PARALLEL_SKY=sequential.o',
               f'COMPILERFLAGS_SKY={flags}', f'LIBS_SKY={libs}', f'EXEC_SKY=sky3d.{name}']
        with (path / 'build.log').open('w') as log:
            subprocess.run(cmd, cwd=path, stdout=log, stderr=subprocess.STDOUT, check=True)
        print('Built:', path / f'sky3d.{name}', flush=True)
    sources = [p for p in (REPO / 'Code').iterdir() if p.suffix in {'.f90', '.f', '.data'} or p.name == 'Makefile']
    sources += list((REPO / 'CUDA').glob('*.f90')) + [REPO / 'CUDA/sky_gpu.cu', Path(__file__).resolve()]
    metadata = dict(arch=args.arch, cuda_home=str(cuda), variants=variants,
                    nvcc=subprocess.check_output([str(nvcc), '--version'], text=True).strip(),
                    gfortran=subprocess.check_output(['gfortran', '--version'], text=True).splitlines()[0],
                    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                    source_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=REPO, text=True).strip()),
                    executable_sha256={name:hashlib.sha256((root/name/f'sky3d.{name}').read_bytes()).hexdigest() for name in variants},
                    library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                    source_sha256={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (root / 'build.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
