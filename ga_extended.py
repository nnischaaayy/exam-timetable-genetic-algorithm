"""Configurable GA used for the ablation study and parameter sweeps.

Same operators and RNG call order as core.run_ga when run with the default options,
so run_ga_ext(data, seed=7) reproduces the original run exactly.
"""
import numpy as np

import main_timetable_ga as core
from main_timetable_ga import N_EXAMS, N_SLOTS


def uniform_crossover(p1, p2, rng, p_cx=core.P_CROSSOVER):
    if rng.random() < p_cx:
        mask = rng.random(N_EXAMS) < 0.5
        return np.where(mask, p1, p2), np.where(mask, p2, p1)
    return p1.copy(), p2.copy()


def no_crossover(p1, p2, rng, p_cx=core.P_CROSSOVER):
    return p1.copy(), p2.copy()


CROSSOVERS = {"one_point": core.crossover, "uniform": uniform_crossover, "none": no_crossover}


def diversity(pop):
    """Mean pairwise normalised Hamming distance (0 = clones, ~0.93 = random)."""
    n = len(pop)
    d = (pop[:, None, :] != pop[None, :, :]).mean(axis=2)
    return float(d.sum() / (n * (n - 1)))


def local_search(c, f, data, rng, steps, mode="random"):
    """Improve one chromosome. mode 'random': single random-gene moves, keep improvements.
    mode 'scan': pick a random exam and try every other slot, keep the best (14 evals/step).
    Returns (chromosome, fitness, evals)."""
    evals = 0
    if mode == "scan":
        for _ in range(steps):
            e = int(rng.integers(N_EXAMS))
            for s in range(N_SLOTS):
                if s == c[e]:
                    continue
                trial = c.copy()
                trial[e] = s
                ft = core.fitness(trial, data)[0]
                evals += 1
                if ft < f:
                    c, f = trial, ft
        return c, f, evals
    for _ in range(steps):
        e = int(rng.integers(N_EXAMS))
        s = int(rng.integers(N_SLOTS - 1))
        trial = c.copy()
        trial[e] = s if s < c[e] else s + 1
        ft = core.fitness(trial, data)[0]
        evals += 1
        if ft < f:
            c, f = trial, ft
    return c, f, evals


def run_ga_ext(data, pop_size=core.POP_SIZE, generations=core.GENERATIONS, p_cx=core.P_CROSSOVER,
               p_mut=core.P_MUTATION, tournament=core.TOURNAMENT_SIZE, elites=core.ELITES,
               crossover="one_point", ls_steps=0, ls_mode="random", seed=core.DATA_SEED,
               init=None):
    cx = CROSSOVERS[crossover]
    rng = np.random.default_rng(seed)
    pop = core.initialize_population(rng, pop_size)
    if init is not None:                         # optional seeded individuals
        pop[:len(init)] = np.asarray(init)
    fit = np.array([core.fitness(c, data)[0] for c in pop], dtype=float)
    evals = pop_size
    best_f, best_c = fit.min(), pop[np.argmin(fit)].copy()
    hist = {"best": [], "mean": [], "evals": [], "div": [], "breakdown": []}

    def record():
        _, b = core.fitness(best_c, data)           # diagnostic only, not counted as an eval
        hist["best"].append(best_f)
        hist["mean"].append(fit.mean())
        hist["evals"].append(evals)
        hist["div"].append(diversity(pop))
        hist["breakdown"].append((b["clashes"], b["room_failures"], b["consecutive"], b["load"]))

    record()
    for _ in range(generations):
        order = np.argsort(fit, kind="stable")
        new_pop = [pop[i].copy() for i in order[:elites]]
        new_fit = [fit[i] for i in order[:elites]]
        while len(new_pop) < pop_size:
            p1 = core.tournament_select(pop, fit, rng, tournament)
            p2 = core.tournament_select(pop, fit, rng, tournament)
            for child in cx(p1, p2, rng, p_cx):
                child = core.mutate(child, rng, p_mut)
                new_pop.append(child)
                new_fit.append(core.fitness(child, data)[0])
                evals += 1
        pop = np.array(new_pop[:pop_size])
        fit = np.array(new_fit[:pop_size], dtype=float)
        if ls_steps:
            i = int(np.argmin(fit))
            pop[i], fit[i], ev = local_search(pop[i], fit[i], data, rng, ls_steps, ls_mode)
            evals += ev
        if fit.min() < best_f:
            best_f, best_c = fit.min(), pop[np.argmin(fit)].copy()
        record()
    brk = hist["breakdown"]
    feasible = next((g for g, b in enumerate(brk) if b[0] + b[1] == 0), None)
    return {"best": best_c, "best_fit": best_f, "best_hist": np.array(hist["best"]),
            "mean_hist": np.array(hist["mean"]), "evals_hist": np.array(hist["evals"]),
            "div_hist": np.array(hist["div"]), "breakdown_hist": np.array(brk),
            "evals": evals, "first_feasible_gen": feasible}
