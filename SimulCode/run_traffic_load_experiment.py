import csv
import os
import shutil

import Run


QUICK_MODE = True
TRAFFIC_LOADS = [
    ("low", 0.10),
    ("medium", 0.30),
    ("high", 0.50),
]
SEEDS = [42] if QUICK_MODE else [42, 43, 44]
MAX_ITER = 1000 if QUICK_MODE else 25000
EVAL_STEPS = 300 if QUICK_MODE else 3000

USE_CLOCK = True
REWARD_MODE = "cooperative"
BETA_COOP = 0.10
ETA_COOP = 0.10
D2LS_PENALTY_WEIGHT = 0.05
LAMBDA_COLLISION = 0.20
ENERGY_PENALTY_WEIGHT = 0.10

OUTPUT_CSV = os.path.join("results", "traffic_load_experiment.csv")
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
    Run.USE_CLOCK = USE_CLOCK
    Run.REWARD_MODE = REWARD_MODE
    Run.BETA_COOP = BETA_COOP
    Run.ETA_COOP = ETA_COOP
    Run.D2LS_PENALTY_WEIGHT = D2LS_PENALTY_WEIGHT
    Run.LAMBDA_COLLISION = LAMBDA_COLLISION
    Run.ENERGY_PENALTY_WEIGHT = ENERGY_PENALTY_WEIGHT
    Run.ENV_COLLISION_PENALTY_VAL = None


def traffic_dir_name(traffic_load, aloha_prob):
    return f"traffic_{traffic_load}_{aloha_prob:.2f}"


def archive_run_artifacts(traffic_load, aloha_prob, seed):
    output_dir = os.path.join(
        "results",
        traffic_dir_name(traffic_load, aloha_prob),
        f"seed_{seed}",
    )
    os.makedirs(output_dir, exist_ok=True)

    for artifact_pattern in ARTIFACT_NAMES:
        artifact_name = artifact_pattern.format(seed=seed)
        source_path = os.path.abspath(artifact_name)
        destination_path = os.path.abspath(os.path.join(output_dir, artifact_name))

        if not os.path.exists(source_path):
            print(
                f"[WARN] Missing artifact for traffic={traffic_load}, "
                f"ALOHA_PROB={aloha_prob:.2f}, seed={seed}: {artifact_name}"
            )
            continue

        if os.path.exists(destination_path):
            os.remove(destination_path)

        shutil.move(source_path, destination_path)
        print(f"Saved artifact: {destination_path}")


def main():
    configure_run_module()
    rows = []

    for traffic_load, aloha_prob in TRAFFIC_LOADS:
        Run.ALOHA_PROB_OVERRIDE = aloha_prob

        for seed in SEEDS:
            print(
                f"\n=== Traffic load experiment: "
                f"{traffic_load}, ALOHA_PROB={aloha_prob:.2f}, seed={seed} ===\n"
            )
            metrics = Run.run_simulation(seed)
            row = {
                "traffic_load": traffic_load,
                "ALOHA_PROB": aloha_prob,
                "USE_CLOCK": USE_CLOCK,
                "REWARD_MODE": REWARD_MODE,
                "BETA_COOP": BETA_COOP,
                "ETA_COOP": ETA_COOP,
                "D2LS_PENALTY_WEIGHT": D2LS_PENALTY_WEIGHT,
                "LAMBDA_COLLISION": LAMBDA_COLLISION,
                "ENERGY_PENALTY_WEIGHT": ENERGY_PENALTY_WEIGHT,
            }
            row.update(metrics)
            rows.append(row)
            archive_run_artifacts(traffic_load, aloha_prob, seed)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "traffic_load",
        "ALOHA_PROB",
        "USE_CLOCK",
        "REWARD_MODE",
        "BETA_COOP",
        "ETA_COOP",
        "D2LS_PENALTY_WEIGHT",
        "LAMBDA_COLLISION",
        "ENERGY_PENALTY_WEIGHT",
    ]
    with open(OUTPUT_CSV, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved traffic-load experiment results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
