import csv
import os
import shutil

import Run


QUICK_MODE = True
D2LS_WEIGHT_VALUES = [0.00, 0.02, 0.05, 0.10]
SEEDS = [42] if QUICK_MODE else [42, 43, 44]
MAX_ITER = 1000 if QUICK_MODE else 25000
EVAL_STEPS = 300 if QUICK_MODE else 3000
BETA_COOP = 0.10
ETA_COOP = 0.10
USE_CLOCK = True
OUTPUT_CSV = os.path.join("results", "d2ls_sensitivity.csv")
ARTIFACT_NAMES = [
    "metrics_seed_{seed}.csv",
    "training_logs_seed_{seed}.npz",
    "paper_results.pdf",
]


def configure_run_module():
    Run.FAST_TEST = QUICK_MODE
    Run.SEEDS = SEEDS
    Run.MAX_ITER = MAX_ITER
    Run.EVAL_STEPS = EVAL_STEPS
    Run.REWARD_MODE = "cooperative"
    Run.BETA_COOP = BETA_COOP
    Run.ETA_COOP = ETA_COOP
    Run.USE_CLOCK = USE_CLOCK


def weight_dir_name(weight):
    return f"d2ls_weight_{weight:.2f}"


def archive_run_artifacts(weight, seed):
    output_dir = os.path.join("results", weight_dir_name(weight), f"seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)

    for artifact_pattern in ARTIFACT_NAMES:
        artifact_name = artifact_pattern.format(seed=seed)
        source_path = os.path.abspath(artifact_name)
        destination_path = os.path.abspath(os.path.join(output_dir, artifact_name))

        if not os.path.exists(source_path):
            print(f"[WARN] Missing artifact for D2LS weight={weight:.2f}, seed={seed}: {artifact_name}")
            continue

        if os.path.exists(destination_path):
            os.remove(destination_path)

        shutil.move(source_path, destination_path)
        print(f"Saved artifact: {destination_path}")


def main():
    configure_run_module()
    rows = []

    for weight in D2LS_WEIGHT_VALUES:
        Run.D2LS_PENALTY_WEIGHT = weight

        for seed in SEEDS:
            print(f"\n=== D2LS sensitivity: weight={weight:.2f}, seed={seed} ===\n")
            metrics = Run.run_simulation(seed)
            row = {
                "D2LS_PENALTY_WEIGHT": weight,
                "USE_CLOCK": USE_CLOCK,
                "BETA_COOP": BETA_COOP,
                "ETA_COOP": ETA_COOP,
            }
            row.update(metrics)
            rows.append(row)
            archive_run_artifacts(weight, seed)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "D2LS_PENALTY_WEIGHT",
        "USE_CLOCK",
        "BETA_COOP",
        "ETA_COOP",
    ]
    with open(OUTPUT_CSV, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved D2LS sensitivity results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
