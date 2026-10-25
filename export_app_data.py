"""Exports the synthetic dataset and measured reference results to docs/data.js for the web app."""
import json
import os

import main_timetable_ga as core


def main():
    d = core.create_dataset()
    data = {
        "nExams": core.N_EXAMS, "nSlots": core.N_SLOTS, "nPeriods": core.N_PERIODS,
        "nDays": core.N_DAYS, "roomCaps": core.ROOM_CAPS,
        "sizes": [int(x) for x in d["sizes"]],
        "students": d["student_exams"],
        "weights": {"clash": core.W_CLASH, "room": core.W_ROOM, "consec": core.W_CONSEC,
                    "load": core.W_LOAD},
        # measured in Python (outputs/extended/ext_table_algorithms.csv, 10 seeds, ~9000 evals)
        "ref": {"Random search": 8841, "Hill climbing": 730, "Simulated annealing": 671,
                "GA (Python)": 629, "Greedy DSATUR": 566, "GA seeded with DSATUR": 539},
        "pythonSeed7": 600,
    }
    # final timetable of the main run (seed 7), with rooms and independent validation
    best, hist, _ = core.run_ga(d, seed=core.DATA_SEED)
    rooms, _ = core.assign_rooms(best, d)
    val = core.validate_schedule(best, d)
    data["result"] = {"slots": [int(x) for x in best], "rooms": [int(x) for x in rooms],
                      "penalty": int(core.penalty_from_validation(val)),
                      "validation": {k: int(v) for k, v in val.items()}}
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "data.js"), "w") as f:
        f.write("// SYNTHETIC dataset exported from main_timetable_ga.create_dataset() (seed 7)\n")
        f.write("const DATA = " + json.dumps(data) + ";\n")
    print("wrote docs/data.js")


if __name__ == "__main__":
    main()

