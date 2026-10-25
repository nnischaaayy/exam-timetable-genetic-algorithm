"""Competing heuristics evaluated on the same data, fitness and validator as the GA."""
import numpy as np

import main_timetable_ga as core
from main_timetable_ga import N_EXAMS, N_SLOTS, N_PERIODS, ROOM_CAPS


def _neighbour(c, rng):
    n = c.copy()
    e = int(rng.integers(N_EXAMS))
    s = int(rng.integers(N_SLOTS - 1))
    n[e] = s if s < c[e] else s + 1          # always a different slot
    return n


def random_search_trace(data, n_evals, seed):
    rng = np.random.default_rng(seed)
    best_f, best_c, trace = np.inf, None, np.empty(n_evals)
    for i in range(n_evals):
        c = rng.integers(0, N_SLOTS, size=N_EXAMS)
        f = core.fitness(c, data)[0]
        if f < best_f:
            best_f, best_c = f, c
        trace[i] = best_f
    return best_c, trace


def hill_climb(data, n_evals, seed):
    """First-improvement hill climbing with sideways moves, single random start."""
    rng = np.random.default_rng(seed)
    cur = rng.integers(0, N_SLOTS, size=N_EXAMS)
    f = core.fitness(cur, data)[0]
    trace = np.empty(n_evals)
    trace[0] = f
    for i in range(1, n_evals):
        cand = _neighbour(cur, rng)
        fc = core.fitness(cand, data)[0]
        if fc <= f:
            cur, f = cand, fc
        trace[i] = f
    return cur, trace


def simulated_annealing(data, n_evals, seed, t0=500.0, t1=1.0):
    """Geometric cooling, single-gene neighbourhood, Metropolis acceptance."""
    rng = np.random.default_rng(seed)
    cur = rng.integers(0, N_SLOTS, size=N_EXAMS)
    f = core.fitness(cur, data)[0]
    best_c, best_f = cur.copy(), f
    trace = np.empty(n_evals)
    trace[0] = f
    for i in range(1, n_evals):
        t = t0 * (t1 / t0) ** (i / (n_evals - 1))
        cand = _neighbour(cur, rng)
        fc = core.fitness(cand, data)[0]
        if fc <= f or rng.random() < np.exp(-(fc - f) / t):
            cur, f = cand, fc
            if f < best_f:
                best_c, best_f = cur.copy(), f
        trace[i] = best_f
    return best_c, trace


def _slot_feasible(sizes, caps):
    free = list(range(len(caps)))
    for s in sorted(sizes, reverse=True):
        for r in free:
            if caps[r] >= s:
                free.remove(r)
                break
        else:
            return False
    return True


def greedy_dsatur(data):
    """Deterministic constructive heuristic: DSATUR-style exam ordering, each exam goes to the
    feasible slot with the lowest incremental clash/consecutive/load cost."""
    C, sizes, caps = data["conflict"].copy(), data["sizes"], ROOM_CAPS
    np.fill_diagonal(C, 0)
    chrom = np.full(N_EXAMS, -1, dtype=np.int64)
    wdeg = C.sum(axis=1)
    for _ in range(N_EXAMS):
        un = np.flatnonzero(chrom < 0)
        sat = [len({int(chrom[j]) for j in np.flatnonzero((C[e] > 0) & (chrom >= 0))}) for e in un]
        key = [(sat[k], wdeg[e]) for k, e in enumerate(un)]
        e = int(un[max(range(len(un)), key=lambda k: key[k])])
        best_s, best_cost = None, np.inf
        for s in range(N_SLOTS):
            inside = [int(sizes[j]) for j in np.flatnonzero(chrom == s)] + [int(sizes[e])]
            if not _slot_feasible(inside, caps):
                continue
            cost = 0
            for j in np.flatnonzero((chrom >= 0) & (C[e] > 0)):
                sj = int(chrom[j])
                if sj == s:
                    cost += core.W_CLASH * C[e, j]
                if sj // N_PERIODS == s // N_PERIODS:
                    cost += core.W_LOAD * C[e, j]
                    if abs(sj % N_PERIODS - s % N_PERIODS) == 1:
                        cost += core.W_CONSEC * C[e, j]
            if cost < best_cost:
                best_s, best_cost = s, cost
        chrom[e] = best_s
    return chrom
