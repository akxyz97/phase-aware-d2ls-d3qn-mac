import csv
import os
import shutil

import Run


QUICK_MODE = True
PHASE_SETTINGS = [False, True]
SEEDS = [42] if QUICK_MODE else [42, 43, 44]
MAX_ITER = 1000 if QUICK_MODE else 25000
EVAL_STEPS = 300 if QUICK_MODE else 3000
BETA_COOP = 0.10
ETA_COOP = 0.10
OUTPUT_CSV = os.path.join("results", "phase_ablation.csv")
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


def phase_dir_name(use_clock):
    return "phase_clock_true" if use_clock else "phase_clock_false"


def archive_run_artifacts(use_clock, seed):
    output_dir = os.path.join("results", phase_dir_name(use_clock), f"seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)

    for artifact_pattern in ARTIFACT_NAMES:
        artifact_name = artifact_pattern.format(seed=seed)
        source_path = os.path.abspath(artifact_name)
        destination_path = os.path.abspath(os.path.join(output_dir, artifact_name))

        if not os.path.exists(source_path):
            print(f"[WARN] Missing artifact for USE_CLOCK={use_clock}, seed={seed}: {artifact_name}")
            continue

        if os.path.exists(destination_path):
            os.remove(destination_path)

        shutil.move(source_path, destination_path)
        print(f"Saved artifact: {destination_path}")


def main():
    configure_run_module()
    rows = []

    for use_clock in PHASE_SETTINGS:
        Run.USE_CLOCK = use_clock

        for seed in SEEDS:
            print(f"\n=== Phase ablation: USE_CLOCK={use_clock}, seed={seed} ===\n")
            metrics = Run.run_simulation(seed)
            row = {
                "USE_CLOCK": use_clock,
                "BETA_COOP": BETA_COOP,
                "ETA_COOP": ETA_COOP,
            }
            row.update(metrics)
            rows.append(row)
            archive_run_artifacts(use_clock, seed)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["USE_CLOCK", "BETA_COOP", "ETA_COOP"]
    with open(OUTPUT_CSV, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved phase ablation results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
