import csv
import os

import Run


QUICK_MODE = True
BETA_COOP_VALUES = [0.02, 0.05, 0.10]
ETA_COOP_VALUES = [0.05, 0.10, 0.15]
SEEDS = [42] if QUICK_MODE else [42, 43, 44]
MAX_ITER = 1000 if QUICK_MODE else 25000
EVAL_STEPS = 300 if QUICK_MODE else 3000
OUTPUT_CSV = os.path.join("results", "coop_sensitivity.csv")


def configure_run_module():
    Run.FAST_TEST = QUICK_MODE
    Run.SEEDS = SEEDS
    Run.MAX_ITER = MAX_ITER
    Run.EVAL_STEPS = EVAL_STEPS
    Run.REWARD_MODE = "cooperative"


def main():
    configure_run_module()
    rows = []

    for beta in BETA_COOP_VALUES:
        for eta in ETA_COOP_VALUES:
            Run.BETA_COOP = beta
            Run.ETA_COOP = eta

            for seed in SEEDS:
                print(f"\n=== Cooperative sensitivity: BETA_COOP={beta}, ETA_COOP={eta}, seed={seed} ===\n")
                metrics = Run.run_simulation(seed)
                row = {
                    "BETA_COOP": beta,
                    "ETA_COOP": eta,
                }
                row.update(metrics)
                rows.append(row)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["BETA_COOP", "ETA_COOP"]
    with open(OUTPUT_CSV, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved cooperative sensitivity results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
