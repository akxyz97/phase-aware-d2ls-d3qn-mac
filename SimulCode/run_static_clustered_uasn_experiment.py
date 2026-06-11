import csv
import os
import shutil

import Run


QUICK_MODE = True
CLUSTER_SCENARIOS = [
    ("clustered_small", 1),
    ("clustered_medium", 2),
    ("clustered_large", 3),
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
CLUSTER_MEMBER_TX_PROB = 0.30

OUTPUT_CSV = os.path.join("results", "static_clustered_uasn_experiment.csv")
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
    Run.ALOHA_PROB_OVERRIDE = None


def archive_run_artifacts(scenario, seed):
    output_dir = os.path.join("results", scenario, f"seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)

    for artifact_pattern in ARTIFACT_NAMES:
        artifact_name = artifact_pattern.format(seed=seed)
        source_path = os.path.abspath(artifact_name)
        destination_path = os.path.abspath(os.path.join(output_dir, artifact_name))

        if not os.path.exists(source_path):
            print(f"[WARN] Missing artifact for scenario={scenario}, seed={seed}: {artifact_name}")
            continue

        if os.path.exists(destination_path):
            os.remove(destination_path)

        shutil.move(source_path, destination_path)
        print(f"Saved artifact: {destination_path}")


def main():
    configure_run_module()
    rows = []

    for scenario, members_per_cluster in CLUSTER_SCENARIOS:
        Run.CLUSTER_MEMBERS_PER_CLUSTER = members_per_cluster
        Run.CLUSTER_MEMBER_TX_PROB = CLUSTER_MEMBER_TX_PROB

        for seed in SEEDS:
            print(
                f"\n=== Static clustered UASN: {scenario}, "
                f"members_per_cluster={members_per_cluster}, seed={seed} ===\n"
            )
            metrics = Run.run_simulation(seed)
            row = {
                "scenario": scenario,
                "members_per_cluster": members_per_cluster,
                "total_member_nodes": members_per_cluster * 2,
                "cluster_member_tx_prob": CLUSTER_MEMBER_TX_PROB,
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
            archive_run_artifacts(scenario, seed)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "scenario",
        "members_per_cluster",
        "total_member_nodes",
        "cluster_member_tx_prob",
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

    print(f"Saved static clustered UASN results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
