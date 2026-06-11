import csv
import os
import random
import time
from collections import defaultdict

import numpy as np


QUICK_MODE = True
SEEDS = [42] if QUICK_MODE else [42, 43, 44]
MAX_ITER = 1000 if QUICK_MODE else 25000

N_SLOTS = 6
N_OFFSETS = 3
ALOHA_PROB = 0.3
FAMA_REQUEST_PROB = 0.3
ENERGY_PER_TRANSMIT = 0.5
ENERGY_PER_RECEIVE = 0.2
ENERGY_PER_IDLE_LISTEN = 0.05
TDMA_PATTERN = [0, 1, 2, 3, 4, 5]

OUTPUT_CSV = os.path.join("results", "slotted_fama_baseline.csv")
FIELDNAMES = [
    "seed",
    "max_iter",
    "fama_throughput",
    "tdma_throughput",
    "aloha_throughput",
    "total_throughput",
    "estimated_collisions",
    "collisions_per_1k",
    "estimated_energy_per_bit",
    "runtime_seconds",
]


def run_baseline(seed, max_iter=MAX_ITER):
    random.seed(seed)
    np.random.seed(seed)

    tdma_counter = 0
    fama_success_log = []
    tdma_success_log = []
    aloha_success_log = []
    estimated_collisions = 0
    fama_energy = 0.0
    start_time = time.time()

    for _ in range(max_iter):
        transmissions = defaultdict(list)

        tdma_slot = TDMA_PATTERN[tdma_counter % len(TDMA_PATTERN)]
        tdma_offset = 0
        tdma_counter += 1
        transmissions[(tdma_slot, tdma_offset)].append("tdma")

        aloha_transmits = random.random() < ALOHA_PROB
        if aloha_transmits:
            aloha_slot = random.randint(0, N_SLOTS - 1)
            aloha_offset = random.randint(0, N_OFFSETS - 1)
            transmissions[(aloha_slot, aloha_offset)].append("aloha")

        fama_attempts = random.random() < FAMA_REQUEST_PROB
        if fama_attempts:
            fama_slot = random.randint(0, N_SLOTS - 1)
            fama_offset = random.randint(0, N_OFFSETS - 1)
            transmissions[(fama_slot, fama_offset)].append("fama")
            fama_energy += ENERGY_PER_TRANSMIT
        else:
            fama_energy += ENERGY_PER_IDLE_LISTEN

        fama_success = 0
        tdma_success = 0
        aloha_success = 0

        for occupants in transmissions.values():
            if len(occupants) > 1:
                estimated_collisions += 1
                continue

            node = occupants[0]
            if node == "fama":
                fama_success = 1
            elif node == "tdma":
                tdma_success = 1
            elif node == "aloha":
                aloha_success = 1

        if fama_success:
            fama_energy += ENERGY_PER_RECEIVE

        fama_success_log.append(fama_success)
        tdma_success_log.append(tdma_success)
        aloha_success_log.append(aloha_success)

    runtime_seconds = time.time() - start_time
    fama_successes = float(np.sum(fama_success_log))
    fama_throughput = float(np.mean(fama_success_log))
    tdma_throughput = float(np.mean(tdma_success_log))
    aloha_throughput = float(np.mean(aloha_success_log))
    total_throughput = fama_throughput + tdma_throughput + aloha_throughput
    estimated_energy_per_bit = fama_energy / fama_successes if fama_successes > 0 else "N/A"

    return {
        "seed": seed,
        "max_iter": max_iter,
        "fama_throughput": fama_throughput,
        "tdma_throughput": tdma_throughput,
        "aloha_throughput": aloha_throughput,
        "total_throughput": total_throughput,
        "estimated_collisions": estimated_collisions,
        "collisions_per_1k": (estimated_collisions / max_iter) * 1000.0,
        "estimated_energy_per_bit": estimated_energy_per_bit,
        "runtime_seconds": runtime_seconds,
    }


def print_summary(rows):
    print("\nSimplified Slotted FAMA baseline summary (mean +/- std)")
    for metric in [
        "fama_throughput",
        "tdma_throughput",
        "aloha_throughput",
        "total_throughput",
        "collisions_per_1k",
        "runtime_seconds",
    ]:
        values = np.array([row[metric] for row in rows], dtype=float)
        print(f"{metric}: {np.mean(values):.4f} +/- {np.std(values):.4f}")

    energy_values = [
        row["estimated_energy_per_bit"]
        for row in rows
        if row["estimated_energy_per_bit"] != "N/A"
    ]
    if energy_values:
        values = np.array(energy_values, dtype=float)
        print(f"estimated_energy_per_bit: {np.mean(values):.4f} +/- {np.std(values):.4f}")
    else:
        print("estimated_energy_per_bit: N/A")


def main():
    rows = [run_baseline(seed) for seed in SEEDS]

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved simplified Slotted FAMA baseline results to {OUTPUT_CSV}")
    print_summary(rows)


if __name__ == "__main__":
    main()
