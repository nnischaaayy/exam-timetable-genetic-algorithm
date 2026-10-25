# University Exam Timetable Optimisation with a Genetic Algorithm

All data is **synthetic** (dataset seed 7): 20 exams, 150 students, 15 slots (5 days x 3 periods), 4 rooms (30/60/90/120 seats).
Author: Nischay Rehal, B.Tech CSE (Data Science), Amity University Uttar Pradesh. Course: Soft Computing.

## Files

| File | Purpose |
|---|---|
| `main_timetable_ga.py` | Core project: dataset, room allocator, independent validator, fitness, GA, random-search baseline, core figures/tables |
| `ga_extended.py` | Configurable GA (tournament size, crossover type, elitism, mutation, local search, seeding) that records diversity and per-term penalties |
| `baselines.py` | Competing methods: hill climbing, simulated annealing, greedy DSATUR constructive heuristic, random search with traces |
| `experiments.py` | Algorithm comparison, GA ablation, parameter sweeps, held-out validation, 4x-budget study, diagnostic figures (parallel) |
| `visuals.py` | Extended figures |
| `tests/test_timetable.py` | 10 unit tests (dataset, validator vs fitness, allocator, operators, determinism, GA reproduces core run) |

## Run (use the Windows Python 3.11 that has numpy and matplotlib)

```
py -3.11 main_timetable_ga.py      # core run, ~1 min  -> outputs/
py -3.11 experiments.py            # extended study, ~3 min -> outputs/extended/
py -3.11 -m unittest discover -s tests -v
```

## Interactive web app (for presenting)

The web pages live in `docs/` so they can be published with GitHub Pages (Settings, Pages, deploy from branch `main`, folder `/docs`).
Start at `docs/index.html`, a scrolling overview that opens with the problem and a live GA solving it, then walks through why the
problem is hard, the genetic algorithm, scoring, learning, the result, an honest comparison, the correctness checks and how to run it.

`docs/lab.html` runs the GA live in the browser next to a 6-step guided tour: conflict graph, timetable grid, chromosome strip,
convergence chart, per-student view and a comparison with the Python results. Double-click `docs/index.html` to open the site
(or `py -3.11 -m http.server 8765 --directory docs` and visit http://localhost:8765). `docs/data.js` holds the exported synthetic
dataset (regenerate with `py -3.11 export_app_data.py`). The browser GA uses a different random generator from Python, so its single
runs differ from the Python seed-7 run; the algorithm and fitness definitions are the same.

`docs/results.html` is the exam-schedule result page: the GA timetable from the Python seed-7 run shown as a weekly calendar,
a filterable datesheet (subject, programme, date, session, room, seats used) and a per-student schedule check, with the
validation recomputed live in the page. Subject names, dates and times are illustrative labels on the synthetic data.

## Outputs

The `outputs/` folder is not stored in the repository; running the scripts below creates it.
Install the two dependencies with `pip install -r requirements.txt` first.

`outputs/`: fig1-fig4, table1-table3, `run_summary.txt` (core project).
`outputs/extended/`: fig5 algorithm comparison, fig6 convergence per evaluation, fig7 ablation, fig8 penalty breakdown,
fig9 diversity, fig10 conflict heatmap, fig11 student load, fig12 room utilisation, fig13 parameter sensitivity,
`ext_table_*.csv`, `extended_run_summary.txt`.

## Honest summary of findings (10 seeds, ~9000 evaluations unless stated)

- The GA reaches 0 hard violations in 100% of runs; random search in 0%.
- Mean penalty: GA 629, SA 671, hill climbing 730, random search 8841. The deterministic greedy DSATUR heuristic scores 566, which is
  better than the plain GA on average. Seeding the GA's initial population with the DSATUR schedule gives 539 (held-out seeds: 549 vs SA 617 vs plain GA 696).
- With a 4x budget SA (475) overtakes both GA variants (519 and 505). The GA is competitive, not dominant, on this problem.
- Ablation: elitism, crossover, tournament size 3 (size 2 is clearly worse) and mutation near 0.05 each matter; mutation 0.10+ or no elitism clearly hurt.
- Two local-search add-ons did not beat the baseline GA; they are kept in the ablation as negative results.
- Limitations: synthetic data, simplified constraints, one dataset instance, 10 seeds per setting.
