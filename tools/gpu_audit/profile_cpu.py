#!/usr/bin/env python3
"""Profile copied CPU source in WSL; never modify production runs or Code/."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tempfile
import time

REPO = Path(__file__).resolve().parents[2]
FLAGS = '-O3 -msse4.2 -mfpmath=sse -ffast-math -finline-functions -funroll-loops'
TIMER = '''module audit_timers
  implicit none
  private
  public :: audit_begin, audit_end, audit_report
  integer(8), save :: ticks(10)=0, calls(10)=0
contains
  subroutine audit_begin(start)
    integer(8), intent(out) :: start
    call system_clock(start)
  end subroutine
  subroutine audit_end(slot,start)
    integer, intent(in) :: slot
    integer(8), intent(in) :: start
    integer(8) :: finish
    call system_clock(finish)
    ticks(slot)=ticks(slot)+finish-start
    calls(slot)=calls(slot)+1
  end subroutine
  subroutine audit_report
    integer :: i,u
    integer(8) :: rate
    call system_clock(count_rate=rate)
    open(newunit=u,file='profile.csv',status='replace')
    write(u,'(a)') 'slot,calls,inclusive_seconds'
    do i=1,10
      write(u,'(i0,a,i0,a,es24.16)') i,',',calls(i),',',real(ticks(i),8)/real(rate,8)
    enddo
    close(u)
  end subroutine
end module
'''
SLOTS = [
    ('meanfield.f90', 'hpsi', '    ALLOCATE(pswk(', 1),
    ('meanfield.f90', 'skyrme', '    ALLOCATE(workden(', 2),
    ('densities.f90', 'add_density', '    ALLOCATE(ps1(', 3),
    ('levels.f90', 'cdervx', '    kfac=', 4),
    ('levels.f90', 'cdervy', '    kfac=', 5),
    ('levels.f90', 'cdervz', '    kfac=', 6),
    ('coulomb.f90', 'poisson', '    ALLOCATE(rho2(', 7),
    ('moment.f90', 'moments', '    pnr=', 8),
    ('dynamic.f90', 'dynamichf', '    ALLOCATE(ps4(', 9),
    ('dynamic.f90', 'tinfo', '    ALLOCATE(ps1(', 10),
]


def build(path, instrument):
    path.mkdir()
    for p in (REPO / 'Code').iterdir():
        if p.suffix in {'.f90', '.f', '.data'} or p.name == 'Makefile':
            shutil.copy2(p, path / p.name)
    if instrument:
        (path / 'audit_timers.f90').write_text(TIMER)
        for filename in {s[0] for s in SLOTS}:
            p = path / filename
            text = p.read_text()
            text = re.sub(r'(?im)^(module\s+\w+\s*)$',
                          r'\1\n  use audit_timers, only: audit_begin, audit_end', text, count=1)
            for _, name, anchor, slot in [s for s in SLOTS if s[0] == filename]:
                pattern = rf'(?ims)^  SUBROUTINE {name}\b.*?^  END SUBROUTINE {name}\b'
                match = re.search(pattern, text)
                assert match, name
                block = match.group()
                position = block.index(anchor)
                block = (block[:position] + '    integer(8) :: audit_start\n'
                         + '    call audit_begin(audit_start)\n' + block[position:])
                # Zero-weight calls must also close the timer before returning.
                block = block.replace('    IF(weight<=0.D0) RETURN',
                                      f'    IF(weight<=0.D0) THEN\n'
                                      f'      call audit_end({slot},audit_start)\n      RETURN\n    ENDIF')
                block = block.replace(f'  END SUBROUTINE {name}',
                                      f'    call audit_end({slot},audit_start)\n  END SUBROUTINE {name}')
                text = text[:match.start()] + block + text[match.end():]
            p.write_text(text)
        p = path / 'main3d.f90'
        text = p.read_text().replace('  USE Params', '  use audit_timers, only: audit_report\n  USE Params', 1)
        p.write_text(text.replace('  CALL finish_mpi', '  call audit_report\n  CALL finish_mpi'))
        p = path / 'Makefile'
        text = p.read_text().replace('OBJS   = params.o', 'OBJS   = audit_timers.o params.o')
        text += '\n$(filter-out audit_timers.o,$(OBJS)): audit_timers.o\n'
        p.write_text(text)
    flags = FLAGS + ('' if instrument else ' -fopenmp')
    cmd = ['make', '-j2', 'makeit', 'COMPILER_SKY=gfortran', 'PARALLEL_SKY=sequential.o',
           f'COMPILERFLAGS_SKY={flags}', 'LIBS_SKY=-lfftw3 -llapack -lopenblas', 'EXEC_SKY=sky3d.audit']
    with (path / 'build.log').open('w') as log:
        subprocess.run(cmd, cwd=path, stdout=log, stderr=subprocess.STDOUT, check=True)


def run(root, exe, label, threads, steps):
    path = root / label
    path.mkdir()
    template = (REPO / 'projects/icnpa2026_20Ne_Kresolved_E2/inputs/td_k0.in').read_text()
    template = template.replace('nt=30000', f'nt={steps}')
    state = REPO / 'projects/icnpa2026_20Ne_Kresolved_E2/static/20ne_sly5_zaxis.tdhf'
    assert state.is_file(), state
    template = template.replace('../../static/20ne_sly5_zaxis.tdhf', '../initial.tdhf')
    (path / 'for005').write_text(template)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS='1',
               OMP_PROC_BIND='close', OMP_PLACES='cores')
    start = time.perf_counter()
    with (path / 'stdout.log').open('w') as log:
        subprocess.run([str(exe)], cwd=path, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    wall = time.perf_counter() - start
    final = [line for line in (path / 'energies.res').read_text().splitlines()
             if line.strip() and not line.startswith('#')][-1]
    assert abs(float(final.split()[0]) - steps * 0.2) < 1e-6
    return dict(label=label, threads=threads, steps=steps, total_wall_seconds=wall,
                final_energy_line=final, run_directory=str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--repetitions', type=int, default=3)
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix='sky3d-gpu-audit-', dir=Path.home() / '.cache'))
    (root / 'initial.tdhf').symlink_to(REPO / 'projects/icnpa2026_20Ne_Kresolved_E2/static/20ne_sly5_zaxis.tdhf')
    evidence = Path(__file__).parent / 'evidence'
    evidence.mkdir(exist_ok=True)
    print('Isolated CPU audit:', root, flush=True)
    build(root / 'profile-build', True)
    build(root / 'cpu-build', False)
    profile = run(root, root / 'profile-build/sky3d.audit', 'profile', 1, args.steps)
    data = []
    for line in (root / 'profile/profile.csv').read_text().splitlines()[1:]:
        slot, count, seconds = line.split(',')
        data.append(dict(routine=SLOTS[int(slot)-1][1], calls=int(count), inclusive_seconds=float(seconds)))
    print('Serial inclusive timings:', data, flush=True)
    # Warm up separately, then interleave complete uninstrumented runs.
    run(root, root / 'cpu-build/sky3d.audit', 'warmup', 8, 20)
    runs = []
    for repetition in range(args.repetitions):
        order = [1,4,8,20] if repetition % 2 == 0 else [20,8,4,1]
        for threads in order:
            result = run(root, root / 'cpu-build/sky3d.audit', f'r{repetition}-t{threads}', threads, args.steps)
            runs.append(result)
            print(result['label'], round(result['total_wall_seconds'], 4), flush=True)
    results = dict(date='2026-10-08', source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                   root=str(root), flags=FLAGS, grid=[24,24,24], nucleus='20Ne',
                   source_hashes={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (REPO/'Code').glob('*.f90')},
                   serial_profile=profile, inclusive_timings=data, uninstrumented_runs=runs,
                   median_wall_by_threads={t:statistics.median(r['total_wall_seconds'] for r in runs if r['threads']==t)
                                           for t in [1,4,8,20]},
                   limits='Short complete runs include FFT planning, initialization and output; serial inclusive timers overlap and cannot be added. No GPU TDHF run.')
    (evidence / 'cpu.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
