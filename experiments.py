"""Extended experiments: algorithm comparison, GA ablation, parameter sweeps, diagnostics.

Run:  py -3.11 experiments.py        (outputs go to outputs/extended/)
All data is SYNTHETIC. Dataset seed is fixed at 7; run seeds are 0-9 (plus 100-109 for
held-out validation of the best ablation variant).
"""
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import baselines as bl
import main_timetable_ga as core
import visuals as vz
from ga_extended import run_ga_ext

EXT_DIR = os.path.join(core.OUT_DIR, "extended")
SEEDS = list(range(10))
VAL_SEEDS = list(range(100, 110))
BUDGET = core.N_EVALS_RANDOM                        # 9000 evaluations
GRID = np.arange(0, BUDGET + 1, 50)

ALGOS = ["GA", "GA (DSATUR-seeded)", "GA + local search", "Simulated annealing", "Hill climbing",
         "Random search"]
BIG_BUDGET = 4 * BUDGET
BIG = ["GA", "GA (DSATUR-seeded)", "Simulated annealing"]
ABLATION = {
    "Baseline (tournament 3, one-point, elites 2, pm 0.05)": {},
    "Tournament size 2": {"tournament": 2},
    "Tournament size 5": {"tournament": 5},
    "Uniform crossover": {"crossover": "uniform"},
    "No crossover (mutation only)": {"crossover": "none"},
    "No elitism": {"elites": 0},
    "Mutation 0.02": {"p_mut": 0.02},
    "Mutation 0.10": {"p_mut": 0.10},
    "Local search on best (3 random moves/gen)": {"ls_steps": 3, "generations": 140},
    "Local search on best (2 exam scans/gen)": {"ls_steps": 2, "ls_mode": "scan", "generations": 100},
    "DSATUR-seeded initial population": {"seeded": True},
}
MUT_SWEEP = [0.01, 0.02, 0.05, 0.08, 0.10, 0.15]
POP_SWEEP = [20, 40, 60, 100]

_DATA = None


def get_data():
    global _DATA
    if _DATA is None:
        _DATA = core.create_dataset()
    return _DATA


def _sample(evals, best):
    idx = np.searchsorted(evals, GRID, side="right") - 1
    return np.asarray(best)[np.clip(idx, 0, None)]


def job_algo(args):
    name, seed = args[:2]
    scale = args[2] if len(args) > 2 else 1            # budget multiplier (large-budget study)
    data = get_data()
    budget = BUDGET * scale
    feas = None
    if name.startswith("GA"):
        kw = {"generations": core.GENERATIONS * scale}
        if name == "GA + local search":
            kw.update(ls_steps=3, generations=140)
        if name == "GA (DSATUR-seeded)":
            kw["init"] = [bl.greedy_dsatur(data)]
        r = run_ga_ext(data, seed=seed, **kw)
        chrom, evals, feas = r["best"], r["evals"], r["first_feasible_gen"]
        trace = _sample(r["evals_hist"], r["best_hist"])
    else:
        fn = {"Simulated annealing": bl.simulated_annealing, "Hill climbing": bl.hill_climb}.get(name)
        if fn:
            chrom, t = fn(data, budget, seed)
        else:
            chrom, t = bl.random_search_trace(data, budget, seed)
        trace, evals = _sample(np.arange(1, budget + 1), t), budget
    val = core.validate_schedule(chrom, data)
    return {"name": name, "seed": seed, "fit": core.penalty_from_validation(val), "val": val,
            "trace": trace, "evals": evals, "feas": feas}


def job_ga(kwargs):
    kwargs = dict(kwargs)
    if kwargs.pop("seeded", False):
        kwargs["init"] = [bl.greedy_dsatur(get_data())]
    r = run_ga_ext(get_data(), **kwargs)
    val = core.validate_schedule(r["best"], get_data())
    return {"fit": core.penalty_from_validation(val), "val": val, "evals": r["evals"],
            "feas": r["first_feasible_gen"]}


def pmap(fn, jobs):
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as ex:
        return list(ex.map(fn, jobs, chunksize=1))


def summarize(rows):
    f = np.array([r["fit"] for r in rows], dtype=float)
    feas = [r["feas"] for r in rows if r["feas"] is not None]
    return {"mean": f.mean(), "std": f.std(), "median": float(np.median(f)), "min": f.min(),
            "max": f.max(), "zero_hard": 100 * np.mean([core.hard_total(r["val"]) == 0 for r in rows]),
            "consec": np.mean([r["val"]["consecutive"] for r in rows]),
            "load": np.mean([r["val"]["load"] for r in rows]),
            "evals": np.mean([r["evals"] for r in rows]),
            "feas": float(np.median(feas)) if feas else None}


def write_table(name, label, summaries):
    header = [label, "mean_penalty", "std", "median", "best", "worst", "zero_hard_runs_pct",
              "mean_consecutive", "mean_load", "mean_evaluations", "median_gen_first_feasible"]
    rows = [[k, f"{s['mean']:.1f}", f"{s['std']:.1f}", f"{s['median']:.1f}", f"{s['min']:.0f}",
             f"{s['max']:.0f}", f"{s['zero_hard']:.0f}", f"{s['consec']:.1f}", f"{s['load']:.1f}",
             f"{s['evals']:.0f}", "-" if s["feas"] is None else f"{s['feas']:.0f}"]
            for k, s in summaries.items()]
    core.write_csv(os.path.join(EXT_DIR, name), header, rows)
    print(f"\n{name}")
    print(" | ".join(header))
    for r in rows:
        print(" | ".join(r))
    return rows


def main():
    t0 = time.time()
    os.makedirs(EXT_DIR, exist_ok=True)
    data = get_data()
    log = []

    # 1. algorithm comparison -------------------------------------------------------------
    greedy = bl.greedy_dsatur(data)
    gv = core.validate_schedule(greedy, data)
    greedy_fit = core.penalty_from_validation(gv)
    print(f"Greedy DSATUR: penalty {greedy_fit}, validation {gv}")
    log.append(f"Greedy DSATUR (deterministic, 20 fitness-free steps): penalty {greedy_fit}, {gv}")
    rows = pmap(job_algo, [(n, s) for n in ALGOS for s in SEEDS])
    by_algo = {n: [r for r in rows if r["name"] == n] for n in ALGOS}
    algo_sum = {n: summarize(by_algo[n]) for n in ALGOS}
    write_table("ext_table_algorithms.csv", "algorithm", algo_sum)
    vz.algorithm_boxplot({n: [r["fit"] for r in by_algo[n]] for n in ALGOS}, greedy_fit, EXT_DIR)
    vz.algorithm_convergence(GRID, {n: np.array([r["trace"] for r in by_algo[n]]) for n in ALGOS},
                             greedy_fit, EXT_DIR)

    # 2. ablation ------------------------------------------------------------------------
    jobs = [dict(kw, seed=s) for kw in ABLATION.values() for s in SEEDS]
    out = pmap(job_ga, jobs)
    abl_rows = {n: out[i * len(SEEDS):(i + 1) * len(SEEDS)] for i, n in enumerate(ABLATION)}
    abl_sum = {n: summarize(r) for n, r in abl_rows.items()}
    write_table("ext_table_ablation.csv", "variant", abl_sum)
    vz.ablation_plot({n: [r["fit"] for r in rs] for n, rs in abl_rows.items()}, EXT_DIR)

    # 3. held-out validation of the best variant (seeds never used for selection) ---------
    held = {}
    for n in ["GA", "GA (DSATUR-seeded)", "Simulated annealing"]:
        held[n] = summarize(pmap(job_algo, [(n, s) for s in VAL_SEEDS]))
    write_table("ext_table_heldout_validation.csv", "algorithm (seeds 100-109)", held)
    log.append("Held-out seeds 100-109 were not used for any design choice above.")

    # 3b. same algorithms with a 4x evaluation budget --------------------------------------
    big = {n: summarize(pmap(job_algo, [(n, s, 4) for s in SEEDS])) for n in BIG}
    write_table("ext_table_large_budget.csv", f"algorithm ({BIG_BUDGET} evaluations)", big)

    # 4. parameter sweeps ----------------------------------------------------------------
    mjobs = [{"p_mut": pm, "seed": s} for pm in MUT_SWEEP for s in SEEDS]
    pjobs = [{"pop_size": ps, "seed": s} for ps in POP_SWEEP for s in SEEDS]
    mo, po = pmap(job_ga, mjobs), pmap(job_ga, pjobs)
    mut = {pm: mo[i * len(SEEDS):(i + 1) * len(SEEDS)] for i, pm in enumerate(MUT_SWEEP)}
    pop = {ps: po[i * len(SEEDS):(i + 1) * len(SEEDS)] for i, ps in enumerate(POP_SWEEP)}
    write_table("ext_table_mutation_sweep.csv", "mutation_probability",
                {str(k): summarize(v) for k, v in mut.items()})
    write_table("ext_table_population_sweep.csv", "population_size",
                {str(k): summarize(v) for k, v in pop.items()})
    vz.sweeps_plot({k: [r["fit"] for r in v] for k, v in mut.items()},
                   {k: [r["fit"] for r in v] for k, v in pop.items()}, EXT_DIR)

    # 5. diagnostics for seed 7 -----------------------------------------------------------
    main_run = run_ga_ext(data, seed=core.DATA_SEED)
    rs_c, _ = bl.random_search_trace(data, BUDGET, core.DATA_SEED)
    vz.breakdown_plot(main_run["breakdown_hist"], EXT_DIR)
    div = {"GA baseline": main_run["div_hist"]}
    for lab, kw in [("Uniform crossover", {"crossover": "uniform"}),
                    ("No crossover", {"crossover": "none"}), ("Mutation 0.15", {"p_mut": 0.15})]:
        div[lab] = run_ga_ext(data, seed=core.DATA_SEED, **kw)["div_hist"]
    vz.diversity_plot(div, EXT_DIR)
    vz.conflict_heatmap(data, main_run["best"], EXT_DIR)
    vz.student_load(data, [rs_c, main_run["best"]],
                    ["Random-search best (seed 7)", "GA best (seed 7)"], EXT_DIR)
    vz.room_utilisation(data, main_run["best"], EXT_DIR)
    log.append(f"GA seed 7: penalty {main_run['best_fit']:.0f}, first feasible generation "
               f"{main_run['first_feasible_gen']}, evaluations {main_run['evals']}")

    elapsed = time.time() - t0
    with open(os.path.join(EXT_DIR, "extended_run_summary.txt"), "w") as f:
        f.write(f"Extended experiments ({core.SYN}); dataset seed {core.DATA_SEED}; run seeds 0-9; "
                f"held-out seeds 100-109\nBudget per baseline algorithm: {BUDGET} evaluations\n")
        f.write("\n".join(log) + f"\nRuntime: {elapsed:.0f} s\n")
    print(f"\nSaved extended outputs to {EXT_DIR} (runtime {elapsed:.0f} s)")


if __name__ == "__main__":
    main()
