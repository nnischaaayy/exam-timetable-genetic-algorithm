import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import baselines as bl
import main_timetable_ga as core
from ga_extended import run_ga_ext


class TimetableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = core.create_dataset()

    def test_dataset(self):
        d = self.data
        self.assertEqual(d["enroll"].shape, (core.N_STUDENTS, core.N_EXAMS))
        self.assertGreaterEqual(d["sizes"].min(), 1)
        self.assertLessEqual(d["sizes"].max(), max(core.ROOM_CAPS))
        self.assertTrue(((d["enroll"].sum(axis=1) >= 3) & (d["enroll"].sum(axis=1) <= 5)).all())

    def test_dataset_is_deterministic(self):
        self.assertTrue((core.create_dataset()["enroll"] == self.data["enroll"]).all())

    def test_validator_flags_bad_schedule(self):
        v = core.validate_schedule(np.zeros(core.N_EXAMS, dtype=np.int64), self.data)
        self.assertGreater(v["student_clashes"], 100)
        self.assertGreater(v["unplaced_exams"], 0)

    def test_fitness_matches_validator(self):
        rng = np.random.default_rng(5)
        for _ in range(100):
            c = rng.integers(0, core.N_SLOTS, size=core.N_EXAMS)
            total, b = core.fitness(c, self.data)
            v = core.validate_schedule(c, self.data)
            self.assertEqual(b["clashes"], v["student_clashes"])
            self.assertEqual(b["room_failures"], v["unplaced_exams"])
            self.assertEqual(b["consecutive"], v["consecutive"])
            self.assertEqual(b["load"], v["load"])
            self.assertEqual(total, core.penalty_from_validation(v))

    def test_allocator(self):
        rng = np.random.default_rng(6)
        for _ in range(100):
            c = rng.integers(0, core.N_SLOTS, size=core.N_EXAMS)
            rooms, _ = core.assign_rooms(c, self.data)
            pairs = [(int(c[e]), int(rooms[e])) for e in range(core.N_EXAMS) if rooms[e] >= 0]
            self.assertEqual(len(pairs), len(set(pairs)))
            for e in range(core.N_EXAMS):
                if rooms[e] >= 0:
                    self.assertGreaterEqual(core.ROOM_CAPS[rooms[e]], self.data["sizes"][e])

    def test_operators(self):
        rng = np.random.default_rng(1)
        a, b = rng.integers(0, 15, 20), rng.integers(0, 15, 20)
        for c in core.crossover(a, b, rng, 1.0):
            self.assertEqual(len(c), core.N_EXAMS)
        m = core.mutate(a, rng, 1.0)
        self.assertTrue(((m >= 0) & (m < core.N_SLOTS)).all())

    def test_ga_monotone_and_deterministic(self):
        r1 = run_ga_ext(self.data, pop_size=20, generations=20, seed=3)
        r2 = run_ga_ext(self.data, pop_size=20, generations=20, seed=3)
        self.assertTrue((np.diff(r1["best_hist"]) <= 0).all())
        self.assertTrue((r1["best"] == r2["best"]).all())

    def test_extended_ga_reproduces_core_ga(self):
        _, best_hist, _ = core.run_ga(self.data, pop_size=20, generations=15, seed=7)
        r = run_ga_ext(self.data, pop_size=20, generations=15, seed=7)
        self.assertTrue(np.allclose(best_hist, r["best_hist"]))

    def test_all_ga_variants_run(self):
        for kw in [{"crossover": "uniform"}, {"crossover": "none"}, {"elites": 0}, {"ls_steps": 2}]:
            r = run_ga_ext(self.data, pop_size=10, generations=5, seed=2, **kw)
            self.assertEqual(len(r["best"]), core.N_EXAMS)

    def test_baselines(self):
        for fn in (bl.hill_climb, bl.simulated_annealing, bl.random_search_trace):
            c, trace = fn(self.data, 200, 1)
            self.assertEqual(len(c), core.N_EXAMS)
            self.assertTrue((np.diff(trace) <= 0).all())
        g = bl.greedy_dsatur(self.data)
        self.assertTrue(((g >= 0) & (g < core.N_SLOTS)).all())


if __name__ == "__main__":
    unittest.main()
