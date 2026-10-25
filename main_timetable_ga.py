"""University exam timetable optimisation with a Genetic Algorithm.

All data is SYNTHETIC (generated once with numpy default_rng(7)).
Run:  python main_timetable_ga.py      (needs only numpy and matplotlib)
"""
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------- constants
DATA_SEED = 7
N_EXAMS = 20
N_STUDENTS = 150
N_DAYS = 5
N_PERIODS = 3
N_SLOTS = N_DAYS * N_PERIODS            # 15
ROOM_CAPS = [30, 60, 90, 120]           # ascending
N_PROGRAMS = 4                          # synthetic cohorts (5 exams each)
P_OWN_PROGRAM = 0.85                    # chance an exam pick is from own cohort
MIN_EXAMS_PER_STUDENT, MAX_EXAMS_PER_STUDENT = 3, 5

W_CLASH, W_ROOM, W_CONSEC, W_LOAD = 1000, 1000, 10, 2

POP_SIZE, GENERATIONS = 60, 150
P_CROSSOVER, P_MUTATION = 0.8, 0.05
TOURNAMENT_SIZE, ELITES = 3, 2
N_EVALS_RANDOM = POP_SIZE * GENERATIONS  # 9000
STUDY_SEEDS = list(range(10))

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SYN = "SYNTHETIC data"


# ------------------------------------------------------------------- dataset
def create_dataset():
    rng = np.random.default_rng(DATA_SEED)
    prog_of_exam = np.arange(N_EXAMS) * N_PROGRAMS // N_EXAMS
    while True:
        enroll = np.zeros((N_STUDENTS, N_EXAMS), dtype=np.int64)
        for s in range(N_STUDENTS):
            prog = rng.integers(N_PROGRAMS)
            w = np.where(prog_of_exam == prog,
                         P_OWN_PROGRAM / np.sum(prog_of_exam == prog),
                         (1 - P_OWN_PROGRAM) / np.sum(prog_of_exam != prog))
            k = rng.integers(MIN_EXAMS_PER_STUDENT, MAX_EXAMS_PER_STUDENT + 1)
            enroll[s, rng.choice(N_EXAMS, size=k, replace=False, p=w)] = 1
        sizes = enroll.sum(axis=0)
        if sizes.min() >= 1 and sizes.max() <= max(ROOM_CAPS):
            break
    C = enroll.T @ enroll                       # C[i,j] = students taking both
    return {
        "enroll": enroll,
        "enroll_f": enroll.astype(np.float64),
        "sizes": sizes,
        "room_caps": np.array(ROOM_CAPS),
        "C_upper": np.triu(C, 1),
        "conflict": C,
        "student_exams": [np.flatnonzero(enroll[s]).tolist() for s in range(N_STUDENTS)],
    }


# ------------------------------------------------------------- room allocator
def assign_rooms(chromosome, data):
    """Greedy best-fit per slot, largest exam first. Returns (room per exam or -1, failures)."""
    sizes, caps = data["sizes"], data["room_caps"]
    rooms = np.full(N_EXAMS, -1, dtype=np.int64)
    failures = 0
    for slot in np.unique(chromosome):
        exams = np.flatnonzero(chromosome == slot)
        exams = exams[np.argsort(-sizes[exams], kind="stable")]
        free = list(range(len(caps)))           # indices ascending by capacity
        for e in exams:
            for r in free:
                if caps[r] >= sizes[e]:
                    rooms[e] = r
                    free.remove(r)
                    break
            else:
                failures += 1
    return rooms, failures


# ----------------------------------------------------------------- validator
def validate_schedule(chromosome, data):
    """Independent plain-loop check of a decoded timetable (no matrix trick)."""
    rooms, _ = assign_rooms(chromosome, data)
    caps, sizes = data["room_caps"], data["sizes"]
    slot_of = [int(x) for x in chromosome]
    clashes = consec = load = 0
    for exams in data["student_exams"]:
        per_day = {}
        for a in range(len(exams)):
            ea = exams[a]
            per_day[slot_of[ea] // N_PERIODS] = per_day.get(slot_of[ea] // N_PERIODS, 0) + 1
            for b in range(a + 1, len(exams)):
                eb = exams[b]
                if slot_of[ea] == slot_of[eb]:
                    clashes += 1
                elif (slot_of[ea] // N_PERIODS == slot_of[eb] // N_PERIODS
                      and abs(slot_of[ea] % N_PERIODS - slot_of[eb] % N_PERIODS) == 1):
                    consec += 1
        for cnt in per_day.values():
            load += max(0, cnt - 1)
    used, room_conflicts, cap_viol, unplaced = {}, 0, 0, 0
    for e in range(N_EXAMS):
        if rooms[e] < 0:
            unplaced += 1
            continue
        key = (slot_of[e], int(rooms[e]))
        if key in used:
            room_conflicts += 1
        used[key] = e
        if caps[rooms[e]] < sizes[e]:
            cap_viol += 1
    return {"student_clashes": clashes, "room_conflicts": room_conflicts,
            "capacity_violations": cap_viol, "unplaced_exams": unplaced,
            "consecutive": consec, "load": load}


def hard_total(v):
    return v["student_clashes"] + v["room_conflicts"] + v["capacity_violations"] + v["unplaced_exams"]


def penalty_from_validation(v):
    return (W_CLASH * v["student_clashes"] + W_ROOM * (v["unplaced_exams"] + v["room_conflicts"]
            + v["capacity_violations"]) + W_CONSEC * v["consecutive"] + W_LOAD * v["load"])


# ------------------------------------------------------------------- fitness
def fitness(chromosome, data):
    """Vectorised penalty (lower is better). Returns (total, breakdown)."""
    day = chromosome // N_PERIODS
    per = chromosome % N_PERIODS
    cu = data["C_upper"]
    same_slot = chromosome[:, None] == chromosome[None, :]
    adjacent = (day[:, None] == day[None, :]) & (np.abs(per[:, None] - per[None, :]) == 1)
    clashes = int((cu * same_slot).sum())
    consec = int((cu * adjacent).sum())
    counts = data["enroll_f"] @ np.eye(N_DAYS)[day]          # students x days
    load = int(np.maximum(counts - 1, 0).sum())
    _, room_fail = assign_rooms(chromosome, data)
    total = W_CLASH * clashes + W_ROOM * room_fail + W_CONSEC * consec + W_LOAD * load
    return total, {"clashes": clashes, "room_failures": room_fail,
                   "consecutive": consec, "load": load}


# ----------------------------------------------------------------- GA parts
def initialize_population(rng, pop_size):
    return rng.integers(0, N_SLOTS, size=(pop_size, N_EXAMS))


def tournament_select(pop, fit, rng, k=TOURNAMENT_SIZE):
    idx = rng.integers(0, len(pop), size=k)
    return pop[idx[np.argmin(fit[idx])]]


def crossover(p1, p2, rng, p_cx=P_CROSSOVER):
    if rng.random() < p_cx:
        cut = int(rng.integers(1, N_EXAMS))
        c1 = np.concatenate([p1[:cut], p2[cut:]])
        c2 = np.concatenate([p2[:cut], p1[cut:]])
    else:
        c1, c2 = p1.copy(), p2.copy()
    return c1, c2


def mutate(c, rng, p_mut=P_MUTATION):
    c = c.copy()
    mask = rng.random(N_EXAMS) < p_mut
    c[mask] = rng.integers(0, N_SLOTS, size=int(mask.sum()))
    return np.clip(c, 0, N_SLOTS - 1)            # repair


def run_ga(data, pop_size=POP_SIZE, generations=GENERATIONS, p_cx=P_CROSSOVER,
           p_mut=P_MUTATION, seed=DATA_SEED):
    rng = np.random.default_rng(seed)
    pop = initialize_population(rng, pop_size)
    fit = np.array([fitness(c, data)[0] for c in pop], dtype=float)
    best_hist, mean_hist = [fit.min()], [fit.mean()]
    best_so_far = (fit.min(), pop[np.argmin(fit)].copy())
    for _ in range(generations):
        order = np.argsort(fit, kind="stable")
        new_pop = [pop[i].copy() for i in order[:ELITES]]
        new_fit = [fit[i] for i in order[:ELITES]]
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fit, rng)
            p2 = tournament_select(pop, fit, rng)
            for child in crossover(p1, p2, rng, p_cx):
                child = mutate(child, rng, p_mut)
                new_pop.append(child)
                new_fit.append(fitness(child, data)[0])
        pop = np.array(new_pop[:pop_size])
        fit = np.array(new_fit[:pop_size], dtype=float)
        if fit.min() < best_so_far[0]:
            best_so_far = (fit.min(), pop[np.argmin(fit)].copy())
        best_hist.append(best_so_far[0])
        mean_hist.append(fit.mean())
    return best_so_far[1], np.array(best_hist), np.array(mean_hist)


def random_search(data, n_evals=N_EVALS_RANDOM, seed=DATA_SEED):
    rng = np.random.default_rng(seed)
    best_f, best_c = np.inf, None
    for _ in range(n_evals):
        c = rng.integers(0, N_SLOTS, size=N_EXAMS)
        f = fitness(c, data)[0]
        if f < best_f:
            best_f, best_c = f, c
    return best_c, best_f


# ------------------------------------------------------------------- output
def exam_name(e):
    return f"E{e + 1:02d}"


def room_name(r):
    return f"R{ROOM_CAPS[r]}"


def timetable_rows(chromosome, data):
    rooms, _ = assign_rooms(chromosome, data)
    rows = []
    for e in np.argsort(chromosome, kind="stable"):
        s = int(chromosome[e])
        r = int(rooms[e])
        rows.append([s // N_PERIODS + 1, s % N_PERIODS + 1, exam_name(e), int(data["sizes"][e]),
                     room_name(r) if r >= 0 else "UNPLACED", ROOM_CAPS[r] if r >= 0 else 0])
    return rows


def print_timetable(chromosome, data):
    print(f"\nFinal timetable ({SYN})")
    print(f"{'Day':>3} {'Period':>6} {'Exam':>5} {'Students':>8} {'Room':>9} {'Capacity':>8}")
    for r in timetable_rows(chromosome, data):
        print(f"{r[0]:>3} {r[1]:>6} {r[2]:>5} {r[3]:>8} {r[4]:>9} {r[5]:>8}")


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=220)
    plt.close(fig)


def plot_convergence(best_hist, mean_hist):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    g = np.arange(len(best_hist))
    ax.plot(g, best_hist, lw=2, label="Best-so-far penalty")
    ax.plot(g, mean_hist, lw=1.3, ls="--", label="Population mean penalty")
    ax.set_yscale("log")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Penalty (log scale, lower is better)")
    ax.set_title(f"GA convergence, seed {DATA_SEED} ({SYN})")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    save_fig(fig, "fig1_convergence.png")


def plot_boxplot(ga_f, rs_f):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.boxplot([ga_f, rs_f])
    ax.set_xticklabels(["Genetic Algorithm", "Random search"])
    ax.set_yscale("log")
    ax.set_ylabel("Final penalty (log scale, lower is better)")
    ax.set_title(f"GA vs random search, 10 seeds, 9000-eval budget\n({SYN})", fontsize=10)
    ax.grid(alpha=0.3, axis="y", which="both")
    save_fig(fig, "fig2_baseline_boxplot.png")


def plot_grid(chromosome, data):
    rooms, _ = assign_rooms(chromosome, data)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, N_PERIODS)
    ax.set_ylim(N_DAYS, 0)
    for d in range(N_DAYS):
        for p in range(N_PERIODS):
            ax.add_patch(plt.Rectangle((p, d), 1, 1, fill=(d + p) % 2 == 0,
                                       fc="#eef3fb", ec="#555555", lw=0.8))
            exams = np.flatnonzero(chromosome == d * N_PERIODS + p)
            txt = "\n".join(f"{exam_name(e)} ({data['sizes'][e]}) - "
                            f"{room_name(rooms[e]) if rooms[e] >= 0 else 'NONE'}" for e in exams)
            ax.text(p + 0.5, d + 0.5, txt or "-", ha="center", va="center", fontsize=8.5)
    ax.set_xticks(np.arange(N_PERIODS) + 0.5)
    ax.set_xticklabels([f"Period {p + 1}" for p in range(N_PERIODS)])
    ax.set_yticks(np.arange(N_DAYS) + 0.5)
    ax.set_yticklabels([f"Day {d + 1}" for d in range(N_DAYS)])
    ax.xaxis.tick_top()
    ax.set_title(f"GA exam timetable: exam (students) - room ({SYN})", pad=28)
    save_fig(fig, "fig3_timetable_grid.png")


def plot_sweep(sweep):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = list(sweep)
    ax.boxplot([sweep[k] for k in labels])
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylabel("Final penalty (log scale, lower is better)")
    ax.set_xlabel("Configuration")
    ax.set_title(f"Parameter sweep: population size / mutation rate, 10 seeds ({SYN})", fontsize=10)
    ax.grid(alpha=0.3, axis="y", which="both")
    save_fig(fig, "fig4_param_sweep.png")


# --------------------------------------------------------------- self tests
def self_tests(data):
    print("\n=== Pass tests ===")
    print(f"[1] Dataset: enrollment {data['enroll'].shape}, exam sizes {data['sizes'].tolist()}")
    ok1 = data["sizes"].max() <= max(ROOM_CAPS) and data["sizes"].min() >= 1
    print(f"    largest exam {data['sizes'].max()} <= largest room {max(ROOM_CAPS)}: {ok1}")
    assert ok1
    bad = validate_schedule(np.zeros(N_EXAMS, dtype=np.int64), data)
    print(f"[2] Validator, all exams in slot 0: {bad}")
    assert bad["student_clashes"] > 50
    rng = np.random.default_rng(123)
    for _ in range(100):
        c = rng.integers(0, N_SLOTS, size=N_EXAMS)
        _, b = fitness(c, data)
        v = validate_schedule(c, data)
        assert b["clashes"] == v["student_clashes"] and b["room_failures"] == v["unplaced_exams"]
        assert b["consecutive"] == v["consecutive"] and b["load"] == v["load"]
        assert v["room_conflicts"] == 0 and v["capacity_violations"] == 0
        assert fitness(c, data)[0] == penalty_from_validation(v)
    print("[3] Fitness matches validator on 100 random chromosomes: True")
    for _ in range(100):
        c = rng.integers(0, N_SLOTS, size=N_EXAMS)
        rooms, _ = assign_rooms(c, data)
        pairs = [(int(c[e]), int(rooms[e])) for e in range(N_EXAMS) if rooms[e] >= 0]
        assert len(pairs) == len(set(pairs))
        assert all(data["room_caps"][rooms[e]] >= data["sizes"][e]
                   for e in range(N_EXAMS) if rooms[e] >= 0)
    print("[4] Room allocator: no repeated (slot, room), capacity respected: True")
    ch, bh, mh = run_ga(data, pop_size=20, generations=1, seed=1)
    assert len(ch) == N_EXAMS and ch.min() >= 0 and ch.max() < N_SLOTS
    _, bh, _ = run_ga(data, pop_size=20, generations=30, seed=1)
    assert np.all(np.diff(bh) <= 0)
    print("[5] GA operators: length/range OK, best-so-far never worsens: True")


# ---------------------------------------------------------------------- main
def fmt_ms(x):
    return f"{np.mean(x):.1f} +/- {np.std(x):.1f}"


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    data = create_dataset()
    self_tests(data)

    # --- main run
    best, best_hist, mean_hist = run_ga(data, seed=DATA_SEED)
    val = validate_schedule(best, data)
    print("\n[6] Full run (seed 7): penalty "
          f"{best_hist[0]:.0f} (gen 0 best) -> {best_hist[-1]:.0f}, hard violations = {hard_total(val)}")
    zero_gen = int(np.argmax(best_hist < W_CLASH)) if (best_hist < W_CLASH).any() else None
    print(f"    first generation with zero hard violations: {zero_gen}")
    print_timetable(best, data)
    print("\nValidation summary:")
    for k, v in val.items():
        print(f"  {k:<20}{v}")
    assert hard_total(val) == 0, "GA result has hard violations; tune parameters"

    # --- baseline (same data, same validator)
    initial = initialize_population(np.random.default_rng(DATA_SEED), POP_SIZE)[0]
    rs_best, rs_f = random_search(data, N_EVALS_RANDOM, DATA_SEED)
    v_init, v_rs = validate_schedule(initial, data), validate_schedule(rs_best, data)

    # --- study over seeds 0-9
    ga_runs, rs_runs = [], []
    for s in STUDY_SEEDS:
        c, bh, _ = run_ga(data, seed=s)
        ga_runs.append((bh[-1], validate_schedule(c, data)))
        c, f = random_search(data, N_EVALS_RANDOM, s)
        rs_runs.append((f, validate_schedule(c, data)))
    ga_f = [r[0] for r in ga_runs]
    rs_f_all = [r[0] for r in rs_runs]
    print("\nSeed study (seeds 0-9):")
    print(f"  GA     penalty {fmt_ms(ga_f)}, zero-hard runs "
          f"{100 * np.mean([hard_total(r[1]) == 0 for r in ga_runs]):.0f}%")
    print(f"  Random penalty {fmt_ms(rs_f_all)}, zero-hard runs "
          f"{100 * np.mean([hard_total(r[1]) == 0 for r in rs_runs]):.0f}%")

    # --- parameter sweep (pop x mutation), 10 seeds each
    sweep = {}
    for pop, pm in [(30, 0.05), (30, 0.10), (60, 0.05), (60, 0.10)]:
        if (pop, pm) == (POP_SIZE, P_MUTATION):
            sweep[f"pop{pop}, pm{pm}"] = ga_f
        else:
            sweep[f"pop{pop}, pm{pm}"] = [run_ga(data, pop_size=pop, p_mut=pm, seed=s)[1][-1]
                                          for s in STUDY_SEEDS]
    print("\nParameter sweep (final penalty, mean +/- std over 10 seeds):")
    for k, v in sweep.items():
        print(f"  {k:<16}{fmt_ms(v)}")

    # --- figures
    plot_convergence(best_hist, mean_hist)
    plot_boxplot(ga_f, rs_f_all)
    plot_grid(best, data)
    plot_sweep(sweep)

    # --- tables
    rows = timetable_rows(best, data)
    write_csv(os.path.join(OUT_DIR, "table1_timetable.csv"),
              ["day", "period", "exam", "candidates", "room", "room_capacity"], rows)
    vkeys = ["student_clashes", "room_conflicts", "capacity_violations", "unplaced_exams",
             "consecutive", "load"]
    write_csv(os.path.join(OUT_DIR, "table2_validation.csv"),
              ["student_clashes", "room_conflicts", "capacity_violations", "unplaced_exams",
               "consecutive_penalty", "load_penalty"], [[val[k] for k in vkeys]])
    hdr = ["method", "scope", "total_penalty", "student_clashes", "room_conflicts",
           "capacity_violations", "unplaced_exams", "consecutive", "load", "zero_hard_runs_pct"]

    def single(name, chrom, v):
        return [name, "seed 7 single run", penalty_from_validation(v)] + [v[k] for k in vkeys] + ["-"]

    def multi(name, runs):
        cols = [fmt_ms([penalty_from_validation(r[1]) for r in runs])]
        cols += [fmt_ms([r[1][k] for r in runs]) for k in vkeys]
        pct = 100 * np.mean([hard_total(r[1]) == 0 for r in runs])
        return [name, "10 seeds (mean +/- std)"] + cols + [f"{pct:.0f}"]

    write_csv(os.path.join(OUT_DIR, "table3_baseline.csv"), hdr, [
        single("Initial random schedule", initial, v_init),
        single("Random search best (9000 evals)", rs_best, v_rs),
        single("GA best", best, val),
        multi("Random search", rs_runs),
        multi("GA", ga_runs),
    ])
    print("\nTable 3 (baseline comparison):")
    with open(os.path.join(OUT_DIR, "table3_baseline.csv")) as f:
        print(f.read())

    elapsed = time.time() - t0
    with open(os.path.join(OUT_DIR, "run_summary.txt"), "w") as f:
        f.write(f"Exam timetable GA - run summary ({SYN})\n")
        f.write(f"Dataset seed: {DATA_SEED}; main GA seed: {DATA_SEED}; study seeds 0-9\n")
        f.write(f"Dataset: {N_EXAMS} exams, {N_STUDENTS} students ({MIN_EXAMS_PER_STUDENT}-"
                f"{MAX_EXAMS_PER_STUDENT} exams each, {N_PROGRAMS} cohorts, {P_OWN_PROGRAM} own-cohort "
                f"probability), {N_SLOTS} slots ({N_DAYS} days x {N_PERIODS} periods), "
                f"rooms {ROOM_CAPS}\n")
        f.write(f"Exam sizes: {data['sizes'].tolist()}\n")
        f.write(f"GA: population {POP_SIZE}, generations {GENERATIONS}, tournament {TOURNAMENT_SIZE}, "
                f"one-point crossover p={P_CROSSOVER}, per-gene mutation p={P_MUTATION}, "
                f"elites {ELITES}\n")
        f.write(f"Random search budget: {N_EVALS_RANDOM} evaluations "
                f"(GA uses {POP_SIZE + GENERATIONS * (POP_SIZE - ELITES)} fitness evaluations)\n")
        f.write(f"Penalty = {W_CLASH}*student_clashes + {W_ROOM}*room_failures + "
                f"{W_CONSEC}*consecutive + {W_LOAD}*load\n")
        f.write("Consecutive = student exam pairs in adjacent periods of the same day. "
                "Load = sum over student-days of max(0, exams that day - 1).\n")
        f.write(f"Main GA: best penalty {best_hist[-1]:.0f}, hard violations {hard_total(val)}, "
                f"validation {val}\n")
        f.write(f"GA 10 seeds: {fmt_ms(ga_f)}; Random 10 seeds: {fmt_ms(rs_f_all)}\n")
        f.write(f"Runtime: {elapsed:.1f} s\n")
    print(f"\nSaved outputs to {OUT_DIR}  (runtime {elapsed:.1f} s)")


if __name__ == "__main__":
    main()
