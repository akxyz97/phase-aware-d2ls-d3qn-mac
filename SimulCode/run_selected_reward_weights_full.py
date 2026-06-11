import csv
import os
import shutil

import Run


SELECTED_CASES = [
    {
        "case_name": "default",
        "LAMBDA_COLLISION": 0.20,
        "ENERGY_PENALTY_WEIGHT": 0.30,
    },
    {
        "case_name": "collision_0.10_energy_0.30",
        "LAMBDA_COLLISION": 0.10,
        "ENERGY_PENALTY_WEIGHT": 0.30,
    },
    {
        "case_name": "collision_0.20_energy_0.10",
        "LAMBDA_COLLISION": 0.20,
        "ENERGY_PENALTY_WEIGHT": 0.10,
    },
]
SEEDS = [42, 43, 44]
MAX_ITER = 25000
EVAL_STEPS = 3000
REWARD_MODE = "cooperative"
BETA_COOP = 0.10
ETA_COOP = 0.10
D2LS_PENALTY_WEIGHT = 0.05
USE_CLOCK = True
OUTPUT_CSV = os.path.join("results", "selected_reward_weights_full.csv")
ARTIFACT_NAMES = [
    "metrics_seed_{seed}.csv",
    "training_logs_seed_{seed}.npz",
    "paper_results.pdf",
]


def configure_run_module():
    Run.FAST_TEST = False
    Run.SEEDS = SEEDS
    Run.MAX_ITER = MAX_ITER
    Run.EVAL_STEPS = EVAL_STEPS
    Run.REWARD_MODE = REWARD_MODE
    Run.BETA_COOP = BETA_COOP
    Run.ETA_COOP = ETA_COOP
    Run.D2LS_PENALTY_WEIGHT = D2LS_PENALTY_WEIGHT
    Run.USE_CLOCK = USE_CLOCK
    Run.ENV_COLLISION_PENALTY_VAL = None


def case_dir_name(case_name):
    return f"reward_weights_{case_name}"


def archive_run_artifacts(case_name, seed):
    output_dir = os.path.join("results", case_dir_name(case_name), f"seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)

    for artifact_pattern in ARTIFACT_NAMES:
        artifact_name = artifact_pattern.format(seed=seed)
        source_path = os.path.abspath(artifact_name)
        destination_path = os.path.abspath(os.path.join(output_dir, artifact_name))

        if not os.path.exists(source_path):
            print(f"[WARN] Missing artifact for case={case_name}, seed={seed}: {artifact_name}")
            continue

        if os.path.exists(destination_path):
            os.remove(destination_path)

        shutil.move(source_path, destination_path)
        print(f"Saved artifact: {destination_path}")


def main():
    configure_run_module()
    rows = []

    for case in SELECTED_CASES:
        case_name = case["case_name"]
        lambda_collision = case["LAMBDA_COLLISION"]
        energy_penalty_weight = case["ENERGY_PENALTY_WEIGHT"]

        Run.LAMBDA_COLLISION = lambda_collision
        Run.ENERGY_PENALTY_WEIGHT = energy_penalty_weight
        Run.ENV_COLLISION_PENALTY_VAL = None

        for seed in SEEDS:
            print(
                f"\n=== Selected reward weights: case={case_name}, "
                f"LAMBDA_COLLISION={lambda_collision:.2f}, "
                f"ENERGY_PENALTY_WEIGHT={energy_penalty_weight:.2f}, seed={seed} ===\n"
            )
            metrics = Run.run_simulation(seed)
            row = {
                "case_name": case_name,
                "LAMBDA_COLLISION": lambda_collision,
                "ENERGY_PENALTY_WEIGHT": energy_penalty_weight,
                "REWARD_MODE": REWARD_MODE,
                "USE_CLOCK": USE_CLOCK,
                "BETA_COOP": BETA_COOP,
                "ETA_COOP": ETA_COOP,
                "D2LS_PENALTY_WEIGHT": D2LS_PENALTY_WEIGHT,
            }
            row.update(metrics)
            rows.append(row)
            archive_run_artifacts(case_name, seed)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "case_name",
        "LAMBDA_COLLISION",
        "ENERGY_PENALTY_WEIGHT",
        "REWARD_MODE",
        "USE_CLOCK",
        "BETA_COOP",
        "ETA_COOP",
        "D2LS_PENALTY_WEIGHT",
    ]
    with open(OUTPUT_CSV, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved selected reward-weight full results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
