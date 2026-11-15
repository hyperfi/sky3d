"""Plot the Sky3D 500 fm/c CPU/GPU timing sweep."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

NUCLEI = {"O16": ("¹⁶O", "#207057"), "Sn120": ("¹²⁰Sn", "#245ea0"), "Pb208": ("²⁰⁸Pb", "#a24122")}
VARIANTS = {"cpu": ("CPU · 24 threads", "-", "o"),
            "gpu-off": ("GPU · cache off", "--", "s"),
            "gpu-on": ("GPU · cache on", ":", "^")}
FIGURES = {
    "dynamic": ("Dynamics to 500 fm/c", "Dynamics wall time (s)"),
    "static": ("Static iteration loop", "Static loop wall time (s)"),
    "total": ("Static preparation + dynamics", "Combined wall time (s)"),
}

def write_figures(summary, output, preview=False):
    if not preview and summary["status"] != "complete":
        raise ValueError("The full benchmark did not complete")
    output.mkdir(parents=True, exist_ok=True)
    lengths = summary["grid_lengths"]
    rows = {}
    for nucleus in NUCLEI:
        for length in lengths:
            case = summary["cases"].get(f"{nucleus}-{length}", {})
            static, dynamic = case.get("static", {}), case.get("dynamic", {})
            for variant in VARIANTS:
                prep = static.get("cpu" if variant == "cpu" else "gpu-off", {})
                evolve = dynamic.get(variant, {})
                prep_ok, evolve_ok = prep.get("status") == "complete", evolve.get("status") == "complete"
                row = dict(nucleus=nucleus, grid_length=length, variant=variant,
                           static_loop_seconds=prep.get("loop_seconds") if prep_ok else None,
                           static_job_seconds=prep.get("seconds") if prep_ok else None,
                           static_iterations=prep.get("iterations") if prep_ok else None,
                           static_converged=prep.get("converged") if prep_ok else None,
                           dynamic_seconds=evolve.get("seconds") if evolve_ok else None,
                           total_seconds=prep["seconds"]+evolve["seconds"] if prep_ok and evolve_ok else None,
                           dynamic_required_mib=evolve.get("required_mib") if evolve_ok else None,
                           states=evolve.get("states") if evolve_ok else prep.get("states"))
                rows[nucleus,length,variant] = row
    if not preview and any(row["total_seconds"] is None for row in rows.values()):
        raise ValueError("Missing timing data")
    for (nucleus, length, variant), row in rows.items():
        baseline = rows[nucleus,length,"gpu-off"]
        cached = variant == "gpu-on" and row["dynamic_seconds"] is not None and baseline["dynamic_seconds"] is not None
        row["cache_extra_required_mib"] = (
            row["dynamic_required_mib"] - baseline["dynamic_required_mib"] if cached else None)
        row["cache_active"] = row["cache_extra_required_mib"] > 0 if cached else None
        row["cache_dynamic_wall_time_saved_pct"] = (
            100 * (1 - row["dynamic_seconds"] / baseline["dynamic_seconds"]) if cached else None)
        row["cache_total_wall_time_saved_pct"] = (
            100 * (1 - row["total_seconds"] / baseline["total_seconds"])
            if cached and row["total_seconds"] is not None and baseline["total_seconds"] is not None else None)
    with (output/"timings.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(next(iter(rows.values()))))
        writer.writeheader()
        writer.writerows(rows.values())
    keys = {"static": "static_loop_seconds", "dynamic": "dynamic_seconds", "total": "total_seconds"}
    footers = {
        "static": "Up to 100 iterations · cache applies only to dynamics",
        "dynamic": "Shared starting checkpoint per case · 2,500 steps at 0.2 fm/c",
        "total": "Backend-specific static job + dynamics from a shared CPU checkpoint",
    }
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    for phase, (title, ylabel) in FIGURES.items():
        cache_ax = None
        if phase == "dynamic":
            fig, (ax, cache_ax) = plt.subplots(2, 1, figsize=(9.4, 7.2), sharex=True,
                                               gridspec_kw={"height_ratios":[3,1]})
            fig.subplots_adjust(left=.12, right=.97, bottom=.17, top=.85, hspace=.18)
        else:
            fig, ax = plt.subplots(figsize=(9.4, 5.6))
            fig.subplots_adjust(left=.12, right=.97, bottom=.17, top=.82)
        variants = tuple(VARIANTS) if phase != "static" else ("cpu","gpu-off")
        values = []
        for nucleus, (_, color) in NUCLEI.items():
            for variant in variants:
                _, style, marker = VARIANTS[variant]
                series = [rows[nucleus,g,variant][keys[phase]] for g in lengths]
                values.extend(v for v in series if v is not None)
                ax.plot(lengths, [float("nan") if v is None else v for v in series],
                        color=color, linestyle=style, marker=marker, lw=2, ms=6)
        if values and max(values) / min(values) >= 8:
            ax.set_yscale("log")
        ax.set(xticks=lengths, xticklabels=[f"{g}³" for g in lengths], xlabel="Simulation grid",
               ylabel=ylabel, xlim=(min(lengths)-1,max(lengths)+1))
        ax.grid(axis="y", which="major", alpha=.23)
        if cache_ax is not None:
            ax.set_xlabel("")
            cache_ax.axhline(0, color="#6b7280", lw=.8)
            for nucleus, (_, color) in NUCLEI.items():
                values = [rows[nucleus,g,"gpu-on"]["cache_dynamic_wall_time_saved_pct"] for g in lengths]
                cache_ax.plot(lengths, [float("nan") if v is None else v for v in values],
                              color=color, lw=1.7, marker="^", ms=5)
            cache_ax.set(xticks=lengths, xticklabels=[f"{g}³" for g in lengths],
                         xlabel="Simulation grid", ylabel="Time saved\nby cache (%)")
            cache_ax.grid(axis="y", alpha=.23)
        handles = [Line2D([],[],color=color,lw=3,label=label) for label,color in NUCLEI.values()]
        handles += [Line2D([],[],color="#222222",ls=VARIANTS[v][1],marker=VARIANTS[v][2],
                           label=("GPU · 8 host threads" if phase == "static" and v == "gpu-off" else VARIANTS[v][0]))
                    for v in variants]
        ax.legend(handles=handles, ncol=3, frameon=False, loc="lower left",
                  bbox_to_anchor=(0,1.015), borderaxespad=0, fontsize=9)
        fig.suptitle(title + (" · preliminary" if preview else ""), x=.12, ha="left", weight="bold")
        fig.text(.12,.041, "RTX A2000 12 GB and Xeon w7-3455 · SLy5 · 1 fm spacing\n" + footers[phase],
                 color="#4a515a", fontsize=9)
        for ext in ("png","pdf"):
            fig.savefig(output/f"{phase}-timing.{ext}",dpi=180,facecolor="white")
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--preview", action="store_true")
    args=parser.parse_args()
    write_figures(json.loads(args.summary.read_text()), args.output, args.preview)

if __name__=="__main__":
    main()

