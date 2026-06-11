import csv
import os
import shutil

import Run


QUICK_MODE = True
ENERGY_WEIGHT_VALUES = [0.10, 0.30, 0.50]
SEEDS = [42] if QUICK_MODE else [42, 43, 44]
MAX_ITER = 1000 if QUICK_MODE else 25000
EVAL_STEPS = 300 if QUICK_MODE else 3000
BETA_COOP = 0.10
ETA_COOP = 0.10
D2LS_PENALTY_WEIGHT = 0.05
LAMBDA_COLLISION = 0.20
USE_CLOCK = True
OUTPUT_CSV = os.path.join("results", "energy_sensitivity.csv")
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
    Run.D2LS_PENALTY_WEIGHT = D2LS_PENALTY_WEIGHT
    Run.LAMBDA_COLLISION = LAMBDA_COLLISION
    Run.ENV_COLLISION_PENALTY_VAL = None


def weight_dir_name(weight):
    return f"energy_weight_{weight:.2f}"


def archive_run_artifacts(weight, seed):
    output_dir = os.path.join("results", weight_dir_name(weight), f"seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)

    for artifact_pattern in ARTIFACT_NAMES:
        artifact_name = artifact_pattern.format(seed=seed)
        source_path = os.path.abspath(artifact_name)
        destination_path = os.path.abspath(os.path.join(output_dir, artifact_name))

        if not os.path.exists(source_path):
            print(f"[WARN] Missing artifact for energy weight={weight:.2f}, seed={seed}: {artifact_name}")
            continue

        if os.path.exists(destination_path):
            os.remove(destination_path)

        shutil.move(source_path, destination_path)
        print(f"Saved artifact: {destination_path}")


def main():
    configure_run_module()
    rows = []

    for energy_weight in ENERGY_WEIGHT_VALUES:
        Run.ENERGY_PENALTY_WEIGHT = energy_weight

        for seed in SEEDS:
            print(f"\n=== Energy sensitivity: ENERGY_PENALTY_WEIGHT={energy_weight:.2f}, seed={seed} ===\n")
            metrics = Run.run_simulation(seed)
            row = {
                "ENERGY_PENALTY_WEIGHT": energy_weight,
                "REWARD_MODE": Run.REWARD_MODE,
                "USE_CLOCK": USE_CLOCK,
                "BETA_COOP": BETA_COOP,
                "ETA_COOP": ETA_COOP,
                "D2LS_PENALTY_WEIGHT": D2LS_PENALTY_WEIGHT,
                "LAMBDA_COLLISION": LAMBDA_COLLISION,
            }
            row.update(metrics)
            rows.append(row)
            archive_run_artifacts(energy_weight, seed)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "ENERGY_PENALTY_WEIGHT",
        "REWARD_MODE",
        "USE_CLOCK",
        "BETA_COOP",
        "ETA_COOP",
        "D2LS_PENALTY_WEIGHT",
        "LAMBDA_COLLISION",
    ]
    with open(OUTPUT_CSV, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved energy sensitivity results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
