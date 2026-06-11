import csv
import os
import random
import time

import numpy as np

import env as dr_env_module
from Brain import DQN, set_seed
from env import ENVIRONMENT


QUICK_MODE = False
SEEDS = [42] if QUICK_MODE else [42, 43, 44]
MAX_ITER = 1000 if QUICK_MODE else 25000
OUTPUT_CSV = "baseline_dr_dlma_metrics.csv"
ENERGY_PER_TRANSMIT = 0.5
FIELDNAMES = [
    "seed",
    "max_iter",
    "agent_throughput",
    "tdma_throughput",
    "aloha_throughput",
    "total_throughput",
    "estimated_collisions",
    "estimated_collisions_per_1k",
    "estimated_energy_per_bit",
    "transmit_count",
    "runtime_seconds",
]


def run_baseline(seed, max_iter=MAX_ITER):
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)

    # DR-DLMA keeps TDMA position in a module-level global.
    dr_env_module.TDMA_counter = 0

    env = ENVIRONMENT(state_size=60)
    dqn_agent = DQN(
        env.state_size,
        env.n_actions,
        env.n_nodes,
        state_action_memory_size=2 * env.DRL_delay,
        memory_size=500,
        replace_target_iter=200,
        batch_size=32,
        learning_rate=0.01,
        gamma=0.95,
        epsilon=0.1,
        epsilon_min=0.001,
        epsilon_decay=0.996,
    )

    agent_reward_list = []
    tdma_reward_list = []
    aloha_reward_list = []
    total_reward_list = []
    delayed_observation_list = []
    agent_state_list = []
    agent_action_list = []

    transmit_count = 0
    counter2 = 0
    state = env.reset()
    start_time = time.time()

    for i in range(max_iter):
        agent_action = dqn_agent.choose_action(state)
        if agent_action == 1:
            transmit_count += 1

        observation, reward, agent_reward, tdma_reward, aloha_reward = env.step(agent_action)

        if i < env.DRL_delay * 2 - 1:
            next_state = env.reset()
        else:
            next_state = np.concatenate([
                agent_state_list[counter2][2:],
                [agent_action_list[counter2], observation],
            ])
            counter2 += 1

        if i >= env.DRL_delay * 2 - 1:
            dqn_agent.store_transition(state, agent_action, reward, next_state)

        agent_state_list.append(state)
        agent_action_list.append(agent_action)
        delayed_observation_list.append(observation)
        agent_reward_list.append(agent_reward)
        tdma_reward_list.append(tdma_reward)
        aloha_reward_list.append(aloha_reward)
        total_reward_list.append(reward)

        if i > 200:
            dqn_agent.learn()

        state = next_state

    runtime = time.time() - start_time
    agent_successes = float(np.sum(agent_reward_list))
    energy_used = transmit_count * ENERGY_PER_TRANSMIT
    energy_per_bit = energy_used / agent_successes if agent_successes > 0 else "N/A"

    # The baseline does not expose collision events directly. A delayed observation of
    # -1 is the available signal for a failed DRL transmission due to TDMA/ALOHA overlap.
    estimated_collisions = int(np.sum(np.array(delayed_observation_list) == -1))
    estimated_collisions_per_1k = (estimated_collisions / max_iter) * 1000.0

    return {
        "seed": seed,
        "max_iter": max_iter,
        "agent_throughput": float(np.mean(agent_reward_list)) if agent_reward_list else "N/A",
        "tdma_throughput": float(np.mean(tdma_reward_list)) if tdma_reward_list else "N/A",
        "aloha_throughput": float(np.mean(aloha_reward_list)) if aloha_reward_list else "N/A",
        "total_throughput": float(np.mean(total_reward_list)) if total_reward_list else "N/A",
        "estimated_collisions": estimated_collisions,
        "estimated_collisions_per_1k": estimated_collisions_per_1k,
        "estimated_energy_per_bit": energy_per_bit,
        "transmit_count": transmit_count,
        "runtime_seconds": runtime,
    }


def write_summary(rows, output_csv=OUTPUT_CSV):
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_csv)
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def print_summary(rows):
    print("\nDR-DLMA baseline summary (mean +/- std)")
    for metric in [
        "agent_throughput",
        "tdma_throughput",
        "aloha_throughput",
        "total_throughput",
        "estimated_collisions_per_1k",
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


if __name__ == "__main__":
    rows = []
    for seed in SEEDS:
        print(f"\n=== Running DR-DLMA baseline: seed={seed}, max_iter={MAX_ITER} ===\n")
        rows.append(run_baseline(seed=seed, max_iter=MAX_ITER))

    path = write_summary(rows)
    print(f"Saved DR-DLMA comparison summary to {path}")
    print_summary(rows)
