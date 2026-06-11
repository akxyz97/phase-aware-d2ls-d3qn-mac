import csv
import os
import shutil

import Run


SELECTED_COEFFICIENTS = [
    (0.10, 0.10),
    (0.02, 0.10),
]
SEEDS = [42, 43, 44]
MAX_ITER = 25000
EVAL_STEPS = 3000
OUTPUT_CSV = os.path.join("results", "selected_coop_full.csv")
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
    Run.REWARD_MODE = "cooperative"


def coefficient_dir_name(beta, eta):
    return f"coop_beta_{beta:.2f}_eta_{eta:.2f}"


def archive_run_artifacts(beta, eta, seed):
    output_dir = os.path.join("results", coefficient_dir_name(beta, eta), f"seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)

    for artifact_pattern in ARTIFACT_NAMES:
        artifact_name = artifact_pattern.format(seed=seed)
        source_path = os.path.abspath(artifact_name)
        destination_path = os.path.abspath(os.path.join(output_dir, artifact_name))

        if not os.path.exists(source_path):
            print(f"[WARN] Missing artifact for beta={beta:.2f}, eta={eta:.2f}, seed={seed}: {artifact_name}")
            continue

        if os.path.exists(destination_path):
            os.remove(destination_path)

        shutil.move(source_path, destination_path)
        print(f"Saved artifact: {destination_path}")


def main():
    configure_run_module()
    rows = []

    for beta, eta in SELECTED_COEFFICIENTS:
        Run.BETA_COOP = beta
        Run.ETA_COOP = eta

        for seed in SEEDS:
            print(f"\n=== Full cooperative run: BETA_COOP={beta:.2f}, ETA_COOP={eta:.2f}, seed={seed} ===\n")
            metrics = Run.run_simulation(seed)
            row = {
                "BETA_COOP": beta,
                "ETA_COOP": eta,
            }
            row.update(metrics)
            rows.append(row)
            archive_run_artifacts(beta, eta, seed)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["BETA_COOP", "ETA_COOP"]
    with open(OUTPUT_CSV, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved selected cooperative full results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
