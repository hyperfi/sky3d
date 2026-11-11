#!/usr/bin/env python3
"""Build isolated CPU/reference/GPU executables in Linux/WSL, without touching Code/ objects."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sysconfig

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CUDA_HOME = Path('/usr/local/cuda')


def executable(path):
    return path.is_file() and os.access(path, os.X_OK)


def cuda_library_dir(cuda):
    machine = platform.machine()
    candidates = [cuda / 'lib64', cuda / 'targets' / f'{machine}-linux' / 'lib']
    if machine == 'aarch64':
        candidates.append(cuda / 'targets/sbsa-linux/lib')
    candidates.append(cuda / 'lib' / f'{machine}-linux-gnu')
    multiarch = sysconfig.get_config_var('MULTIARCH')
    if multiarch:
        candidates.append(cuda / 'lib' / multiarch)
    candidates.append(cuda / 'lib')
    return next((path for path in candidates if (path / 'libcufft.so').is_file()), None)


def find_cuda():
    home = os.environ.get('CUDA_HOME')
    if home is not None:
        if not home:
            raise ValueError('CUDA_HOME is empty. Set it to a toolkit root or unset it for automatic discovery.')
        cuda = Path(os.path.expanduser(home)).absolute()
        nvcc = cuda / 'bin/nvcc'
        if not executable(nvcc):
            raise ValueError(f'CUDA_HOME={home!r} does not provide an executable {nvcc}. '
                             'Set CUDA_HOME to the toolkit root (the parent of bin/).')
    else:
        cuda = DEFAULT_CUDA_HOME
        nvcc = cuda / 'bin/nvcc'
        if not executable(nvcc):
            on_path = shutil.which('nvcc')
            if not on_path:
                raise ValueError(f'Could not find executable nvcc at {nvcc} or on PATH. '
                                 'Install a CUDA toolkit with cuFFT development files, then set '
                                 'CUDA_HOME to its root or add its bin directory to PATH.')
            nvcc = Path(on_path).absolute()
            cuda = nvcc.parent.parent
            # A PATH entry may be a symlink into a toolkit installed elsewhere.
            if not cuda_library_dir(cuda):
                target = nvcc.resolve()
                if target != nvcc:
                    nvcc, cuda = target, target.parent.parent
    lib_dir = cuda_library_dir(cuda)
    if not lib_dir:
        raise ValueError(f'Could not find libcufft.so under {cuda} for {nvcc}. '
                         'Install the cuFFT development files for this toolkit or select another CUDA_HOME.')
    return cuda, nvcc, lib_dir


def run_build(cmd, log_path, cwd=None):
    with log_path.open('w') as log:
        try:
            result = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        except OSError as error:
            print(f'Could not run {cmd[0]}: {error}', file=log)
            raise SystemExit(f'Could not run {cmd[0]}: {error}\nFull build log: {log_path}') from None
    if result.returncode:
        diagnostic = re.compile(r'\b(?:fatal|error)\s*[:#]|undefined reference|cannot find|not found|'
                                r'no rule to make target|permission denied', re.IGNORECASE)
        warning = re.compile(r'\bwarning\s*[:#]', re.IGNORECASE)
        with log_path.open(errors='replace') as log:
            first_error = next((line.strip() for line in log
                                if diagnostic.search(line) and not warning.search(line)), None)
        message = first_error or f'{cmd[0]} exited with status {result.returncode}'
        raise SystemExit(f'{message}\nFull build log: {log_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-dir', type=Path, default=REPO / 'CUDA/build')
    parser.add_argument('--arch', default='native', help='CUDA architecture, native or e.g. sm_120')
    parser.add_argument('--host-compiler', type=Path, metavar='PATH',
                        help='C++ host compiler for nvcc (-ccbin); defaults to nvcc selection')
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if os.name != 'posix':
        parser.error('Run the build in Linux/WSL')
    try:
        cuda, nvcc, cuda_lib = find_cuda()
    except ValueError as error:
        parser.error(str(error))
    host_compiler = args.host_compiler
    if host_compiler is not None:
        host_compiler = Path(os.path.expanduser(host_compiler)).absolute()
        if not executable(host_compiler):
            parser.error(f'--host-compiler must point to an executable C++ compiler: {host_compiler}')
    root = args.build_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(f'nvcc: {nvcc}\nCUDA architecture: {args.arch}\nBuild directory: {root}', flush=True)
    library = root / 'libsky3d_gpu.so'
    cmd = [str(nvcc), '-O3', '-std=c++17', f'-arch={args.arch}', '-shared', '-Xcompiler=-fPIC']
    if host_compiler is not None:
        cmd += ['-ccbin', str(host_compiler)]
    cmd += [str(REPO / 'CUDA/sky_gpu.cu'), '-L', str(cuda_lib), '-lcufft',
            '-Xlinker=-rpath', f'-Xlinker={cuda_lib}', '-o', str(library)]
    run_build(cmd, root / 'cuda-build.log')
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
        run_build(cmd, path / 'build.log', cwd=path)
        print('Built:', path / f'sky3d.{name}', flush=True)
    sources = [p for p in (REPO / 'Code').iterdir() if p.suffix in {'.f90', '.f', '.data'} or p.name == 'Makefile']
    sources += list((REPO / 'CUDA').glob('*.cuh')) + list((REPO / 'CUDA').glob('*.f90')) + [REPO / 'CUDA/sky_gpu.cu', Path(__file__).resolve()]
    metadata = dict(arch=args.arch, cuda_home=str(cuda), nvcc_path=str(nvcc), cuda_lib_dir=str(cuda_lib),
                    host_compiler=str(host_compiler) if host_compiler is not None else None, variants=variants,
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
