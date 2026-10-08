#!/usr/bin/env python3
"""Profile serialized TDHF stages in isolated CPU/GPU builds, outside production Code/."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np

from benchmark import REPO, checkpoint, check_observable, difference, input_text, run

FLAGS = '-O3 -march=native -ffast-math -fopenmp'
SLOTS = ['initialization', 'predictor', 'midpoint_fields_and_upload', 'corrector',
         'center_of_mass', 'diagnostics', 'final_fields_and_upload', 'restart_io',
         'cleanup', 'skyrme_inclusive', 'poisson_inclusive', 'tinfo_inclusive']
TIMER = '''MODULE checkpoint_timers
  IMPLICIT NONE
  INTEGER(8), SAVE :: ticks(12)=0,calls(12)=0
CONTAINS
  SUBROUTINE timer_begin(start)
    INTEGER(8),INTENT(OUT) :: start
    CALL SYSTEM_CLOCK(start)
  END SUBROUTINE
  SUBROUTINE timer_end(slot,start)
    INTEGER,INTENT(IN) :: slot
    INTEGER(8),INTENT(IN) :: start
    INTEGER(8) :: finish
    CALL SYSTEM_CLOCK(finish)
    ticks(slot)=ticks(slot)+finish-start
    calls(slot)=calls(slot)+1
  END SUBROUTINE
  SUBROUTINE timer_report
    INTEGER :: i,unit
    INTEGER(8) :: rate
    CALL SYSTEM_CLOCK(count_rate=rate)
    OPEN(NEWUNIT=unit,FILE='profile.csv',STATUS='REPLACE')
    WRITE(unit,'(A)') 'slot,calls,wall_seconds'
    DO i=1,12
       WRITE(unit,'(I0,A,I0,A,ES24.16)') i,',',calls(i),',',REAL(ticks(i),8)/REAL(rate,8)
    ENDDO
    CLOSE(unit)
  END SUBROUTINE
END MODULE checkpoint_timers
'''


def replace_once(text, anchor, replacement):
    if text.count(anchor) != 1:
        raise ValueError(f'Instrumentation anchor changed: {anchor!r}')
    return text.replace(anchor, replacement)


def instrument(build):
    (build/'checkpoint_timers.f90').write_text(TIMER)
    for filename in ('dynamic.f90', 'meanfield.f90', 'coulomb.f90'):
        path = build/filename
        text, count = re.subn(r'(?im)^(module\s+\w+\s*)$',
                              r'\1\n  USE checkpoint_timers, ONLY: timer_begin,timer_end', path.read_text(), count=1)
        if count != 1:
            raise ValueError(filename)
        path.write_text(text)
    path = build/'dynamic.f90'
    text = path.read_text()
    match = re.search(r'(?ims)^  SUBROUTINE dynamichf\b.*?^  END SUBROUTINE dynamichf\b', text)
    block = match.group()
    block = replace_once(block, '    ALLOCATE(ps4', '    INTEGER(8) :: stage_start\n    CALL timer_begin(stage_start)\n    ALLOCATE(ps4')
    block = replace_once(block, '    Timestepping:  DO', '    CALL timer_end(1,stage_start)\n    Timestepping:  DO')
    boundaries = [
        ('       ! correction for parallel version', '       CALL timer_begin(stage_start)\n'),
        ('       ! compute mean field and add external field', '       CALL timer_end(2,stage_start)\n       CALL timer_begin(stage_start)\n'),
        ('       ! Step 3: full time step', '       CALL timer_end(3,stage_start)\n       CALL timer_begin(stage_start)\n'),
        ('       ! Step 4: eliminate center-of-mass motion if desired', '       CALL timer_end(4,stage_start)\n       CALL timer_begin(stage_start)\n'),
        ('       ! Step 5: generating some output', '       CALL timer_end(5,stage_start)\n       CALL timer_begin(stage_start)\n'),
        ('       ! Step 6: finishing up', '       CALL timer_end(6,stage_start)\n       CALL timer_begin(stage_start)\n'),
        ('       IF(output_due(iter,mrest)) THEN', '       CALL timer_end(7,stage_start)\n       CALL timer_begin(stage_start)\n'),
        ('    END DO Timestepping', '       CALL timer_end(8,stage_start)\n'),
    ]
    for anchor, insertion in boundaries:
        block = replace_once(block, anchor, insertion+anchor)
    block = replace_once(block, '    END DO Timestepping\n', '    END DO Timestepping\n    CALL timer_begin(stage_start)\n')
    block = replace_once(block, '  END SUBROUTINE dynamichf', '    CALL timer_end(9,stage_start)\n  END SUBROUTINE dynamichf')
    path.write_text(text[:match.start()]+block+text[match.end():])
    for filename, name, anchor, slot in [('meanfield.f90', 'skyrme', '    ALLOCATE(workden(', 10),
                                        ('coulomb.f90', 'poisson', '    ALLOCATE(rho2(', 11),
                                        ('dynamic.f90', 'tinfo', '    ALLOCATE(ps1(', 12)]:
        path = build/filename
        text = path.read_text()
        match = re.search(rf'(?ims)^  SUBROUTINE {name}\b.*?^  END SUBROUTINE {name}\b', text)
        block = match.group()
        block = replace_once(block, anchor, '    INTEGER(8) :: local_start\n    CALL timer_begin(local_start)\n'+anchor)
        end = f'  END SUBROUTINE {name}'
        block = replace_once(block, end, f'    CALL timer_end({slot},local_start)\n'+end)
        path.write_text(text[:match.start()]+block+text[match.end():])
    path = build/'main3d.f90'
    text = replace_once(path.read_text(), '  USE Params', '  USE checkpoint_timers, ONLY: timer_report\n  USE Params')
    path.write_text(replace_once(text, '  CALL finish_mpi', '  CALL timer_report\n  CALL finish_mpi'))
    path = build/'Makefile'
    text = replace_once(path.read_text(), 'OBJS   = params.o', 'OBJS   = checkpoint_timers.o params.o')
    text += '\n$(filter-out checkpoint_timers.o,$(OBJS)): checkpoint_timers.o\n'
    path.write_text(text)


def build_variant(root, name, library):
    build = root/f'{name}-build'
    build.mkdir()
    for source in (REPO/'Code').iterdir():
        if source.suffix in {'.f90', '.f', '.data'} or source.name == 'Makefile':
            shutil.copy2(source, build/source.name)
    libs = '-lfftw3 -llapack -lopenblas'
    if name == 'gpu':
        shutil.copy2(REPO/'CUDA/gpu_runtime.f90', build/'gpu_runtime.f90')
        libs += f' -L"{library.parent}" -lsky3d_gpu -Wl,-rpath,"{library.parent}"'
    instrument(build)
    with (build/'build.log').open('w') as log:
        subprocess.run(['make', '-j2', 'makeit', 'COMPILER_SKY=gfortran', 'PARALLEL_SKY=sequential.o',
                        f'COMPILERFLAGS_SKY={FLAGS}', f'LIBS_SKY={libs}', 'EXEC_SKY=sky3d.profile'],
                       cwd=build, check=True, stdout=log, stderr=subprocess.STDOUT)
    return build/'sky3d.profile'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True, help='Passed benchmark.py validation JSON for this state/settings')
    parser.add_argument('--build-dir', type=Path, default=REPO/'CUDA/build')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--mxpact', type=int, default=4)
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--gpu-threads', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.name != 'posix':
        parser.error('Run in WSL/Linux')
    if min(args.steps, args.mxpact, args.cpu_threads, args.gpu_threads) < 1 or not np.isfinite(args.dt) or args.dt <= 0:
        parser.error('Positive finite run settings required')
    state = checkpoint(args.state)
    validation = json.loads(args.validation.read_text())
    state_hash = hashlib.sha256(args.state.read_bytes()).hexdigest()
    if not validation['passed'] or validation['state_sha256'] != state_hash or validation['steps'] != args.steps:
        parser.error('Validation must match the checkpoint and step count')
    if validation['dt_fm_c'] != args.dt or validation['mxpact'] != args.mxpact:
        parser.error('Validation must match timestep and Taylor order')
    library = args.build_dir.resolve()/'libsky3d_gpu.so'
    if hashlib.sha256(library.read_bytes()).hexdigest() != validation['build']['library_sha256']:
        parser.error('CUDA library differs from the validated build')
    root = Path(tempfile.mkdtemp(prefix='sky3d-stage-profile-', dir=Path.home()/'.cache'))
    (root/'initial.tdhf').symlink_to(args.state.resolve())
    print('Isolated stage profile:', root, flush=True)
    rows = []
    text = input_text(args.steps, state['grid'], spacing=state['spacing'], dt=args.dt, mxpact=args.mxpact)
    for name, threads in [('cpu', args.cpu_threads), ('gpu', args.gpu_threads)]:
        executable = build_variant(root, name, library)
        directory = root/name
        timing = run(executable, directory, text, threads, name)
        baseline = Path(next(c['run']['directory'] for c in validation['cases'] if c['case']=='standard' and c['variant']==name))
        error = difference(checkpoint(directory/'k0_restart.tdhf')['psi'], checkpoint(baseline/'k0_restart.tdhf')['psi'])
        assert error['relative_l2'] < 1e-9 and error['max_abs'] < 1e-9, error
        for filename in ('energies.res', 'quadrupoles.res', 'monopoles.res'):
            check_observable(filename, np.loadtxt(directory/filename, ndmin=2), np.loadtxt(baseline/filename, ndmin=2),
                             validation['initial_flow_atol_mev'])
        data = np.loadtxt(directory/'profile.csv', delimiter=',', skiprows=1)
        phases = [dict(stage=SLOTS[int(slot)-1], calls=int(calls), wall_seconds=float(seconds),
                       nested_in_other_stages=bool(slot >= 10)) for slot, calls, seconds in data]
        rows.append(dict(**timing, stages=phases, wavefunction_vs_uninstrumented=error,
                         executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest()))
        print(name, [(r['stage'], round(r['wall_seconds'], 3)) for r in phases[:9]], flush=True)
    result = dict(run_directory=str(root), state_sha256=state_hash, grid=list(state['grid']), spacing_fm=list(state['spacing']),
                  steps=args.steps, dt_fm_c=args.dt, mxpact=args.mxpact, flags=FLAGS, runs=rows,
                  build=json.loads((args.build_dir/'build.json').read_text()),
                  profiler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  validation_sha256=hashlib.sha256(args.validation.read_bytes()).hexdigest(),
                  source_sha256={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (REPO/'Code').glob('*.f90')},
                  limits='Instrumented wall times locate costs, not speedup evidence. Stages 1-9 do not overlap; '
                         'the last three inclusive timers overlap those stages and must not be added. '
                         'GPU stage times include synchronization and density download; fields stages include upload. '
                         'Program setup before dynamichf and loop stdout writes are outside the stage totals. '
                         'Timers are called only on the serial coordinator, including OpenMP runs.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
