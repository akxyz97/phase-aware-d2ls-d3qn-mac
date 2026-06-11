import csv
import os
import shutil

import Run


QUICK_MODE = True
REWARD_MODES = ["independent", "cooperative"]
SEEDS = [42] if QUICK_MODE else [42, 43, 44]
MAX_ITER = 1000 if QUICK_MODE else 25000
EVAL_STEPS = 300 if QUICK_MODE else 3000
OUTPUT_CSV = os.path.join("results", "reward_mode_comparison.csv")
ARTIFACT_NAMES = [
    "metrics_seed_{seed}.csv",
    "training_logs_seed_{seed}.npz",
    "paper_results.pdf",
]

FIELDNAMES = [
    "reward_mode",
    "seed",
    "agent1_throughput",
    "agent2_throughput",
    "sum_drl_throughput",
    "tdma_throughput",
    "aloha_throughput",
    "total_coexist_throughput",
    "jain_fairness",
    "d2ls_agent1",
    "d2ls_agent2",
    "total_collisions",
    "collision_rate",
    "collisions_per_1k",
    "energy_per_bit_agent1",
    "energy_per_bit_agent2",
    "eval_agents_throughput",
    "eval_coexist_throughput",
    "eval_collisions_per_1k",
]


def configure_run_module():
    Run.FAST_TEST = QUICK_MODE
    Run.SEEDS = SEEDS
    Run.MAX_ITER = MAX_ITER
    Run.EVAL_STEPS = EVAL_STEPS


def archive_run_artifacts(reward_mode, seed):
    output_dir = os.path.join("results", reward_mode, f"seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)

    for artifact_pattern in ARTIFACT_NAMES:
        artifact_name = artifact_pattern.format(seed=seed)
        source_path = os.path.abspath(artifact_name)
        destination_path = os.path.abspath(os.path.join(output_dir, artifact_name))

        if not os.path.exists(source_path):
            print(f"[WARN] Missing artifact for {reward_mode} seed {seed}: {artifact_name}")
            continue

        if os.path.exists(destination_path):
            os.remove(destination_path)

        shutil.move(source_path, destination_path)
        print(f"Saved artifact: {destination_path}")


def main():
    configure_run_module()
    rows = []

    for reward_mode in REWARD_MODES:
        Run.REWARD_MODE = reward_mode

        for seed in SEEDS:
            print(f"\n=== Running reward mode: {reward_mode}, seed: {seed} ===\n")
            metrics = Run.run_simulation(seed)
            rows.append({field: metrics[field] for field in FIELDNAMES})
            archive_run_artifacts(reward_mode, seed)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved reward mode comparison to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
