import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np

root = Path.cwd()
build = root.parent / "line40.wIP1ci"
sys.path.insert(0, str(build / "source/CUDA"))
from benchmark import check_observable, checkpoint, difference, record

seed = root.parent / "pb208-screen.8bAD7z/grid-40-cpu/dynamic.tdhf"
template = root.parent / "pb208-restart.VrrNfn/grid40-resume-1-candidate/for005"
grid = (40, 40, 40)
dt = 0.2
start_step = 20
final_step = 15_000
threads = {"gpu": 8, "cpu": 24}
report = dict(status="preparing", checkpoint=str(seed), grid=grid, spacing_fm=1.0,
              dt_fm_c=dt, start_step=start_step, start_time_fm_c=4.0,
              final_step=final_step, final_time_fm_c=3000.0,
              steps_to_evolve=final_step-start_step, threads=threads, runs={},
              source_hash="bcce3c1789f8fa92dc8653adc5f53fbf94ddd36d")

def save():
    tmp = root / "summary.tmp"
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    tmp.replace(root / "summary.json")

def checked_state(path, step):
    state = checkpoint(path)
    if (state["step"] != step or abs(state["time"]-step*dt) > 1e-7
            or state["grid"] != grid or state["nucleons"] != (126, 82)
            or state["states"] != 208):
        raise RuntimeError(f"Unexpected checkpoint header at {path}: "
                           f"step={state['step']} time={state['time']} grid={state['grid']}")
    return state

def run(backend, text):
    folder = root / backend
    folder.mkdir()
    shutil.copyfile(seed, folder / "dynamic.tdhf")
    (folder / "for005").write_text(text)
    binary = build / "candidate" / backend / f"sky3d.{backend}"
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("SKY3D_"):
            del env[key]
    env.update(SKY3D_BACKEND=backend, SKY3D_GPU_GRAPHS="1",
               OMP_NUM_THREADS=str(threads[backend]), OMP_PROC_BIND="close",
               OMP_PLACES="cores", OPENBLAS_NUM_THREADS="1")
    row = report["runs"][backend] = dict(directory=str(folder), binary=str(binary), status="running")
    report["status"] = f"running-{backend}"
    print(f"Starting {backend} in {folder}", flush=True)
    with (folder / "stdout.log").open("w") as log:
        start = time.perf_counter()
        with subprocess.Popen([str(binary)], cwd=folder, env=env, stdout=log,
                              stderr=subprocess.STDOUT) as proc:
            row.update(pid=proc.pid, started_unix=time.time())
            save()
            status = proc.wait()
    row["seconds"] = time.perf_counter() - start
    if status:
        raise RuntimeError(f"{backend} executable exited {status}")
    final = checked_state(folder / "dynamic.tdhf", final_step)
    row.update(status="complete", final_step=final["step"], final_time_fm_c=final["time"])
    save()
    print(f"Completed {backend} in {row['seconds']:.2f} s", flush=True)

def compare(initial):
    states = {backend: checked_state(root / backend / "dynamic.tdhf", final_step)
              for backend in threads}
    wave = difference(states["gpu"]["psi"], states["cpu"]["psi"])
    result = dict(wavefunction=wave, observables={}, endpoint_invariants={})
    for name in ("energies.res", "quadrupoles.res", "monopoles.res"):
        cpu, gpu = (np.loadtxt(root / backend / name, ndmin=2) for backend in ("cpu", "gpu"))
        if cpu.shape != gpu.shape or not np.array_equal(cpu[:, 0], gpu[:, 0]):
            raise RuntimeError(f"Sample grids differ in {name}")
        row = result["observables"][name] = {}
        try:
            check_observable(name, gpu, cpu)
        except AssertionError:
            row["within_existing_tolerance"] = False
        else:
            row["within_existing_tolerance"] = True
        if name == "energies.res":
            row["integrated_energy_max_abs_mev"] = float(np.max(np.abs(gpu[:, 4]-cpu[:, 4])))
    import struct
    with seed.open("rb") as stream:
        split = struct.unpack_from("<i", record(stream), 40)[0]
    species = (slice(None, split), slice(split, None))
    def grams(state):
        out = []
        for section in species:
            active = state["psi"][section][initial["attributes"][0, section] > 1e-8]
            out.append(float(np.prod(state["spacing"])) * (active.conj() @ active.T))
        return out
    initial_grams = grams(initial)
    for backend, state in states.items():
        matrices = grams(state)
        result["endpoint_invariants"][backend] = dict(
            particle_max_drift=float(np.max(np.abs(np.asarray(state["particles"])-initial["particles"]))),
            occupied_gram_max_error=float(max(np.max(np.abs(g-np.eye(len(g)))) for g in matrices)),
            occupied_gram_max_drift=float(max(np.max(np.abs(g-g0)) for g,g0 in zip(matrices,initial_grams))),
            occupation_max_drift=float(np.max(np.abs(state["attributes"][0]-initial["attributes"][0]))))
    result["cpu_gpu_agreement_passed"] = max(wave.values()) < 1e-9 and all(
        row["within_existing_tolerance"] for row in result["observables"].values())
    result["endpoint_invariants_passed"] = all(
        max(r["particle_max_drift"], r["occupied_gram_max_error"], r["occupied_gram_max_drift"]) <= 1e-6
        and r["occupation_max_drift"] <= 1e-12 for r in result["endpoint_invariants"].values())
    return result

save()
try:
    initial = checked_state(seed, start_step)
    text = template.read_text()
    for key, value in (("nt", final_step), ("mrest", 1000)):
        text, count = re.subn(rf"\b{key}=\d+", f"{key}={value}", text)
        if count != 1:
            raise RuntimeError(f"Expected one {key} in the restart input")
    if "trestart=T" not in text or "dt=0.2," not in text:
        raise RuntimeError("Unexpected restart input")
    (root / "for005").write_text(text)
    for backend in threads:
        try:
            run(backend, text)
        except Exception as error:
            report["runs"].setdefault(backend, {}).update(status="failed", error=repr(error))
            save()
            print(f"{backend} failed: {error}", flush=True)
    if all(row["status"] == "complete" for row in report["runs"].values()):
        report["gpu_speedup"] = report["runs"]["cpu"]["seconds"] / report["runs"]["gpu"]["seconds"]
        report["comparison"] = compare(initial)
        report["status"] = "complete"
    else:
        report["status"] = "failed"
    save()
except Exception as error:
    report.update(status="failed", error=repr(error))
    save()
    raise

