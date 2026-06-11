import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import numpy as np
from Brain import DQN, set_seed
from env import ENVIRONMENT
import throughput as throughput
import warnings
import csv
import sys
import subprocess

# --- Q2 SCIENTIFIC CONFIG ---
USE_CLOCK = True             # Final model uses phase-aware state
FAST_TEST = False            # False = full final experiment, True = quick test only
SEEDS = [42, 43, 44] if not FAST_TEST else [42]
MAX_ITER = 25000 if not FAST_TEST else 1000
EVAL_STEPS = 3000 if not FAST_TEST else 300
TDMA_CHANGE_STEP = 12000     # Dynamic adaptability test

# Final selected reward / sensitivity settings
D2LS_PENALTY_WEIGHT = 0.05
ENV_COLLISION_PENALTY_VAL = None   # Keep env.py default collision penalty = 1.0
ENERGY_PENALTY_WEIGHT = 0.10       # Final selected value from full validation
ALOHA_PROB_OVERRIDE = None
CLUSTER_MEMBERS_PER_CLUSTER = None
CLUSTER_MEMBER_TX_PROB = None

# Balanced Reward Shaping
REWARD_MODE = "cooperative"        # Final selected mode
LAMBDA_COLLISION = 0.20            # Final selected collision penalty
BETA_DEFER = 0.03                  # Keep existing defer bonus
BETA_COOP = 0.10                   # Final selected cooperative success bonus
ETA_COOP = 0.10                    # Final selected cooperative collision penalty

DEBUG = True
warnings.filterwarnings("ignore", category=UserWarning)


def run_simulation(s):
    # =================================================================================
    #                               INITIALIZATION
    # =================================================================================
    if REWARD_MODE not in {"independent", "cooperative"}:
        raise ValueError(f"Unknown REWARD_MODE: {REWARD_MODE}")

    # 1. Initialize Env with Interference ENABLED
    # This ensures agents learn to handle legacy nodes (TDMA/ALOHA) from the start.
    env = ENVIRONMENT(state_size=63, n_slots=6, random_seed=s)
    if D2LS_PENALTY_WEIGHT is not None:
        env.d2ls_penalty_weight = D2LS_PENALTY_WEIGHT
    if ENV_COLLISION_PENALTY_VAL is not None:
        env.collision_penalty_val = ENV_COLLISION_PENALTY_VAL
    if ENERGY_PENALTY_WEIGHT is not None:
        env.energy_penalty_weight = ENERGY_PENALTY_WEIGHT
    if ALOHA_PROB_OVERRIDE is not None:
        env.aloha_prob = ALOHA_PROB_OVERRIDE
    if CLUSTER_MEMBERS_PER_CLUSTER is not None:
        env.cluster_members_per_cluster = CLUSTER_MEMBERS_PER_CLUSTER
    if CLUSTER_MEMBER_TX_PROB is not None:
        env.cluster_member_tx_prob = CLUSTER_MEMBER_TX_PROB
    env.baselines_coexist = True
    print(f"Training STARTED. Baselines Coexist: {env.baselines_coexist}")
    print(f"Reward mode: {REWARD_MODE}")
    print(f"D2LS penalty weight: {env.d2ls_penalty_weight}")
    print(f"Run collision penalty weight: {LAMBDA_COLLISION}")
    print(f"Env collision penalty value: {env.collision_penalty_val}")
    print(f"Energy penalty weight: {env.energy_penalty_weight}")
    print(f"ALOHA probability: {env.aloha_prob}")
    print(f"Cluster members per cluster: {env.cluster_members_per_cluster}")
    print(f"Cluster member TX probability: {env.cluster_member_tx_prob}")

    env_delay_for_state_construction = env.DRL_delay

    agent_config = {
        "state_size": env.state_size,
        "n_actions": env.n_slots * env.n_offsets + 1,  # +1 for Idle
        "state_action_memory_size": env_delay_for_state_construction,
        "memory_size": 5000,
        "replace_target_iter": 200,
        "batch_size": 32,
        "learning_rate": 0.001,
        "gamma": 0.95,
        "epsilon": 1.0,
        "epsilon_min": 0.02,
        "epsilon_decay": 0.9995,
    }

    # Deterministic init for reproducibility
    set_seed(s);
    agent1 = DQN(**agent_config)
    set_seed(s+1);
    agent2 = DQN(**agent_config)

    # Initial states
    state1_current, state2_current = env.reset()

    # === CSV Logger ===
    csv_file = open(f"metrics_seed_{s}.csv", "w", newline="")
    csvw = csv.writer(csv_file)
    csvw.writerow([
        "step", "avg_raw_r1", "avg_raw_r2", "avg_tdma", "avg_aloha",
        "collisions_last_win", "energy_efficiency"
    ])

    # --- Logging lists ---
    agent1_raw_reward_log, agent2_raw_reward_log = [], []
    tdma_reward_log, aloha_reward_log = [], []
    total_network_log = []

    # =========================
    # LOGGING ARRAYS (ADD THIS)
    # =========================
    steps_log = []
    throughput_log = []
    collision_log = []
    energy_log = []
    fairness_log = []
    delay_log = []

    # Store window-averaged logs for plotting
    avg_r1_log, avg_r2_log = [], []
    avg_tdma_log, avg_aloha_log = [], []
    avg_collisions_log = []

    state_hist_1, state_hist_2 = [state1_current], [state2_current]
    action_hist_1, action_hist_2 = [], []
    shaped_reward_hist_1, shaped_reward_hist_2 = [], []

    last_collision_total = 0
    if not hasattr(env, "coexistence_log"): env.coexistence_log = []

    start_time = time.time()

    # =================================================================================
    #                               TRAINING LOOP
    # =================================================================================
    for i in range(MAX_ITER):

        # DYNAMIC EVENT: Change TDMA Pattern
        if i == TDMA_CHANGE_STEP:
            print(f"\n--- STEP {i}: DYNAMIC CHANGE ---")
            env.change_tdma_pattern([0, 1])

            # Reset exploration to help AI find new slots
            agent1.epsilon = 0.5
            agent2.epsilon = 0.5
            print("Exploration reset to 0.5 for re-learning!")

        # --- Agents choose actions ---
        action1 = agent1.choose_action(state1_current)
        action2 = agent2.choose_action(state2_current)
        action_hist_1.append(action1)
        action_hist_2.append(action2)

        # --- Environment step ---
        (obs1_delayed, obs2_delayed, _,
         shaped_r1_delayed, shaped_r2_delayed,
         tdma_r_delayed, aloha_r_delayed,
         raw_r1_delayed, raw_r2_delayed) = env.step(action1, action2)

        # --- Reward Shaping (using delayed info) ---
        info = getattr(env, "last_info_delayed", {})
        r1_shaped = shaped_r1_delayed
        r2_shaped = shaped_r2_delayed

        # Small penalties/bonuses to encourage fairness and reduce collisions
        if info.get("both_tx_collision", False):
            r1_shaped -= LAMBDA_COLLISION
            r2_shaped -= LAMBDA_COLLISION

        sti = info.get("single_tx_idx", None)
        if sti is not None:
            if sti == 0:
                r2_shaped += BETA_DEFER
            elif sti == 1:
                r1_shaped += BETA_DEFER

        if REWARD_MODE == "cooperative":
            delayed_total_drl_successes = raw_r1_delayed + raw_r2_delayed
            delayed_total_drl_collision_participation = info.get("agent_collision_participation", 0)

            cooperative_bonus = BETA_COOP * delayed_total_drl_successes
            cooperative_penalty = ETA_COOP * delayed_total_drl_collision_participation
            cooperative_adjustment = cooperative_bonus - cooperative_penalty

            r1_shaped += cooperative_adjustment
            r2_shaped += cooperative_adjustment

        shaped_reward_hist_1.append(r1_shaped)
        shaped_reward_hist_2.append(r2_shaped)

        if i >= env.DRL_delay - 1:
            agent1_raw_reward_log.append(raw_r1_delayed)
            agent2_raw_reward_log.append(raw_r2_delayed)
            tdma_reward_log.append(tdma_r_delayed)
            aloha_reward_log.append(aloha_r_delayed)
            total_network_log.append(raw_r1_delayed + raw_r2_delayed + tdma_r_delayed + aloha_r_delayed)

        # Calculate time phase (The Clock)
        current_phase = (env.current_step % env.n_slots) / float(env.n_slots)
        denom = env.n_slots * env.n_offsets
        def norm_a(a, d): return 0.0 if a == d else (a / d) * 2.0 - 1.0

        # Construct 3-feature state [Action, Observation, Clock]
        # Note: We roll by -3 now to make room for the Clock!
        state1_next = np.roll(state1_current, -3)
        state1_next[-3] = norm_a(action1, denom)
        state1_next[-2] = obs1_delayed
        state1_next[-1] = current_phase if USE_CLOCK else 0.0

        state2_next = np.roll(state2_current, -3)
        state2_next[-3] = norm_a(action2, denom)
        state2_next[-2] = obs2_delayed
        state2_next[-1] = current_phase if USE_CLOCK else 0.0

        # --- Store transitions ---
        transition_idx = i - env.DRL_delay + 1
        if transition_idx >= 0:
            s1_k = state_hist_1[transition_idx]
            a1_k = action_hist_1[transition_idx]
            r1_k = shaped_reward_hist_1[i]
            s1_kp1 = state_hist_1[transition_idx + 1]

            s2_k = state_hist_2[transition_idx]
            a2_k = action_hist_2[transition_idx]
            r2_k = shaped_reward_hist_2[i]
            s2_kp1 = state_hist_2[transition_idx + 1]

            agent1.store_transition(s1_k, a1_k, r1_k, s1_kp1)
            agent2.store_transition(s2_k, a2_k, r2_k, s2_kp1)

        state_hist_1.append(state1_next)
        state_hist_2.append(state2_next)
        state1_current, state2_current = state1_next, state2_next

        # --- Learning ---
        if i > 200:
            if agent1.memory_counter > agent1.batch_size + agent1.state_action_memory_size:
                agent1.learn()
            if agent2.memory_counter > agent2.batch_size + agent2.state_action_memory_size:
                agent2.learn()

        # === 500-step logging ===
        if (i + 1) % 500 == 0:
            n_win = min(500, len(agent1_raw_reward_log))
            if n_win > 0:
                avg_raw_r1 = np.mean(agent1_raw_reward_log[-n_win:])
                avg_raw_r2 = np.mean(agent2_raw_reward_log[-n_win:])
                avg_tdma = np.mean(tdma_reward_log[-n_win:])
                avg_aloha = np.mean(aloha_reward_log[-n_win:])
            else:
                avg_raw_r1 = avg_raw_r2 = avg_tdma = avg_aloha = 0.0

            col_500 = env.total_collisions - last_collision_total
            last_collision_total = env.total_collisions

            # Simple efficiency metric for CSV
            total_success = np.sum(agent1_raw_reward_log[-n_win:]) + np.sum(agent2_raw_reward_log[-n_win:])
            efficiency_metric = total_success / (col_500 + 1.0)

            print(
                f"Step {i + 1}: R1={avg_raw_r1:.3f}, R2={avg_raw_r2:.3f}, TDMA={avg_tdma:.3f}, ALOHA={avg_aloha:.3f}, Cols={col_500}")

            avg_r1_log.append(avg_raw_r1)
            avg_r2_log.append(avg_raw_r2)
            avg_tdma_log.append(avg_tdma)
            avg_aloha_log.append(avg_aloha)
            avg_collisions_log.append(col_500)

            csvw.writerow([i + 1, avg_raw_r1, avg_raw_r2, avg_tdma, avg_aloha, col_500, efficiency_metric])
            csv_file.flush()

            # =========================
            # STORE METRICS (ADD THIS)
            # =========================
            steps_log.append(i + 1)

            # Throughput (sum of agents)
            current_throughput = avg_raw_r1 + avg_raw_r2
            throughput_log.append(current_throughput)

            # Collisions
            collision_log.append(col_500)

            # Energy (average of agents)
            if hasattr(env, "agent1_total_energy_log") and hasattr(env, "agent2_total_energy_log"):
                energy_used_1 = np.sum(env.agent1_total_energy_log[-n_win:])
                energy_used_2 = np.sum(env.agent2_total_energy_log[-n_win:])
            else:
                energy_used_1 = n_win * env.energy_per_transmit * np.mean(env.agent1_tx_attempts_log[-n_win:])
                energy_used_2 = n_win * env.energy_per_transmit * np.mean(env.agent2_tx_attempts_log[-n_win:])

            total_success = np.sum(agent1_raw_reward_log[-n_win:]) + np.sum(agent2_raw_reward_log[-n_win:])

            if total_success > 0:
                current_energy = (energy_used_1 + energy_used_2) / total_success
            else:
                current_energy = 0
            energy_log.append(current_energy)

            # Fairness (you already compute Jain index)
            jain_index = throughput.jains_index([avg_raw_r1, avg_raw_r2]) if (avg_raw_r1 + avg_raw_r2) > 0 else 0.0
            fairness_log.append(jain_index)

            # Delay (D2LS average)
            d1 = np.mean(env.agent1_delay_log[-n_win:]) if len(env.agent1_delay_log) > 0 else 0
            d2 = np.mean(env.agent2_delay_log[-n_win:]) if len(env.agent2_delay_log) > 0 else 0
            current_delay = (d1 + d2) / 2
            delay_log.append(current_delay)

    # =================================================================================
    #                               FINAL REPORTING
    # =================================================================================

    print("\n" + "=" * 60)
    print("               SIMULATION COMPLETE - GENERATING REPORT")
    print("=" * 60)

    np.savez(f"training_logs_seed_{s}.npz",
             steps=steps_log,
             throughput=throughput_log,
             collisions=collision_log,
             energy=energy_log,
             fairness=fairness_log,
             delay=delay_log)

    # --- 1. Scenario Labeling ---
    scenario_label = "Static Coexistence"
    if 'TDMA_CHANGE_STEP' in globals() and TDMA_CHANGE_STEP < MAX_ITER:
        scenario_label = "Dynamic TDMA Change (Adaptability Test)"
    if not env.baselines_coexist:
        scenario_label = "Agents Only (No Interference)"

    # --- 2. Training Phase Metrics (Last 5000 Steps) ---
    lookback = 5000

    # Slice logs safely
    r1_log = agent1_raw_reward_log[-lookback:] if len(agent1_raw_reward_log) > 0 else [0]
    r2_log = agent2_raw_reward_log[-lookback:] if len(agent2_raw_reward_log) > 0 else [0]
    tdma_log = tdma_reward_log[-lookback:] if len(tdma_reward_log) > 0 else [0]
    aloha_log = aloha_reward_log[-lookback:] if len(aloha_reward_log) > 0 else [0]
    net_log = total_network_log[-lookback:] if len(total_network_log) > 0 else [0]

    # Averages
    avg_r1 = np.mean(r1_log)
    avg_r2 = np.mean(r2_log)
    avg_tdma = np.mean(tdma_log)
    avg_aloha = np.mean(aloha_log)

    sum_drl_throughput = avg_r1 + avg_r2
    coexist_throughput = np.mean(net_log)

    # Efficiency vs Theoretical Cap (assuming cap=4.0 for 4-slot equivalent)
    theoretical_max = 4.0
    efficiency_vs_cap = (coexist_throughput / theoretical_max) * 100.0

    # Jain's Fairness Index (for Agents)
    if sum_drl_throughput > 0:
        jfi = throughput.jains_index([avg_r1, avg_r2])
    else:
        jfi = 0.0

    # D2LS (Delay Since Last Success) & Jitter
    d2ls1_data = np.array(env.agent1_delay_log) if len(env.agent1_delay_log) > 1 else np.array([0.0])
    d2ls2_data = np.array(env.agent2_delay_log) if len(env.agent2_delay_log) > 1 else np.array([0.0])

    mean_d2ls1 = np.mean(d2ls1_data)
    mean_d2ls2 = np.mean(d2ls2_data)

    # Jitter = mean of absolute differences between consecutive delays
    jitter1 = np.mean(np.abs(np.diff(d2ls1_data))) if len(d2ls1_data) > 1 else 0.0
    jitter2 = np.mean(np.abs(np.diff(d2ls2_data))) if len(d2ls2_data) > 1 else 0.0

    # Collision Metrics (Training)
    total_cols = env.total_collisions
    col_rate_step = total_cols / MAX_ITER
    col_rate_1k = (env.collision_events / MAX_ITER) * 1000.0

    # Energy Efficiency (Joules per successful bit/packet)
    total_success_1 = np.sum(agent1_raw_reward_log)
    total_success_2 = np.sum(agent2_raw_reward_log)
    energy_consumed_1 = 1000.0 - env.agent_energy_1
    energy_consumed_2 = 1000.0 - env.agent_energy_2

    epb_1 = energy_consumed_1 / (total_success_1 + 1e-9)
    epb_2 = energy_consumed_2 / (total_success_2 + 1e-9)

    agent1_energy_breakdown = {
        "tx": getattr(env, "agent1_tx_energy", energy_consumed_1),
        "rx": getattr(env, "agent1_rx_energy", 0.0),
        "idle": getattr(env, "agent1_idle_listen_energy", 0.0),
        "sleep": getattr(env, "agent1_sleep_energy", 0.0),
    }
    agent2_energy_breakdown = {
        "tx": getattr(env, "agent2_tx_energy", energy_consumed_2),
        "rx": getattr(env, "agent2_rx_energy", 0.0),
        "idle": getattr(env, "agent2_idle_listen_energy", 0.0),
        "sleep": getattr(env, "agent2_sleep_energy", 0.0),
    }

    # --- 3. Evaluation Phase (Greedy) ---
    print(f"[-] Running Greedy Evaluation ({EVAL_STEPS} steps)...")

    eval_steps = EVAL_STEPS
    env.baselines_coexist = True  # Ensure interference isON for rigorous testing

    agent1.epsilon = 0.0
    agent2.epsilon = 0.0

    eval_r1_sum = 0
    eval_r2_sum = 0
    eval_net_sum = 0

    cols_before_eval = env.collision_events

    for _ in range(eval_steps):
        # Greedy actions
        a1 = agent1.choose_action_greedy(state1_current)
        a2 = agent2.choose_action_greedy(state2_current)

        (o1d, o2d, _, sr1d, sr2d, td, al, r1d, r2d) = env.step(a1, a2)

        denom = env.n_slots * env.n_offsets

        # Calculate time phase (The Clock)
        current_phase = (env.current_step % env.n_slots) / float(env.n_slots)
        def norm_a(a, d): return 0.0 if a == d else (a / d) * 2.0 - 1.0

        # We roll by -3 here as well
        state1_current = np.roll(state1_current, -3)
        state1_current[-3] = norm_a(a1, denom)
        state1_current[-2] = o1d
        state1_current[-1] = current_phase if USE_CLOCK else 0.0

        state2_current = np.roll(state2_current, -3)
        state2_current[-3] = norm_a(a2, denom)
        state2_current[-2] = o2d
        state2_current[-1] = current_phase if USE_CLOCK else 0.0

        eval_r1_sum += r1d
        eval_r2_sum += r2d
        eval_net_sum += (r1d + r2d + td + al)

    # Calculate Eval Averages
    eval_throughput_agents = (eval_r1_sum + eval_r2_sum) / eval_steps
    eval_throughput_coexist = eval_net_sum / eval_steps

    eval_col_events = env.collision_events - cols_before_eval
    eval_col_per_1k = (eval_col_events / eval_steps) * 1000.0

    # --- 4. PRINT SUMMARY ---
    print("\n" + "-" * 60)
    print(f"SCENARIO: {scenario_label}")
    print("-" * 60)

    print(f"TRAINING METRICS (Last {lookback} steps):")
    print(f" > Avg Raw Reward (A1):        {avg_r1:.4f}")
    print(f" > Avg Raw Reward (A2):        {avg_r2:.4f}")
    print(f" > Sum DRL Throughput:         {sum_drl_throughput:.4f}")
    print(f" > Jain's Fairness Index:      {jfi:.4f}")
    print("-" * 30)
    print(f" > Avg TDMA Throughput:        {avg_tdma:.4f}")
    print(f" > Avg ALOHA Throughput:       {avg_aloha:.4f}")
    print(f" > Total Coexist Throughput:   {coexist_throughput:.4f}")
    print(f" > Efficiency vs Cap ({theoretical_max}): {efficiency_vs_cap:.2f}%")

    print("\nDELAY & STABILITY:")
    print(f" > Agent 1 D2LS Mean:          {mean_d2ls1:.2f} (Jitter: {jitter1:.2f})")
    print(f" > Agent 2 D2LS Mean:          {mean_d2ls2:.2f} (Jitter: {jitter2:.2f})")

    print("\nCOLLISIONS & ENERGY (Entire Run):")
    print(f" > Total Collisions:           {total_cols}")
    print(f" > Collision Rate (per step):  {col_rate_step:.4f}")
    print(f" > Collisions (per 1k steps):  {col_rate_1k:.2f}")
    print(f" > Energy/Bit (Agent 1):       {epb_1:.4f}")
    print(f" > Energy/Bit (Agent 2):       {epb_2:.4f}")
    print(
        f" > Agent 1 Energy Breakdown:   "
        f"Tx={agent1_energy_breakdown['tx']:.4f}, "
        f"Rx={agent1_energy_breakdown['rx']:.4f}, "
        f"Idle={agent1_energy_breakdown['idle']:.4f}, "
        f"Sleep={agent1_energy_breakdown['sleep']:.4f}"
    )
    print(
        f" > Agent 2 Energy Breakdown:   "
        f"Tx={agent2_energy_breakdown['tx']:.4f}, "
        f"Rx={agent2_energy_breakdown['rx']:.4f}, "
        f"Idle={agent2_energy_breakdown['idle']:.4f}, "
        f"Sleep={agent2_energy_breakdown['sleep']:.4f}"
    )

    print(f"\nEVALUATION PHASE (Greedy, {eval_steps} steps):")
    print(f" > Agents-Only Throughput:     {eval_throughput_agents:.4f}")
    print(f" > Coexistence Throughput:     {eval_throughput_coexist:.4f}")
    print(f" > Eval Collisions/1k:         {eval_col_per_1k:.2f}")

    print("=" * 60)
    print(f"Total Runtime: {time.time() - start_time:.2f} seconds")
    print("=" * 60 + "\n")

    # Final Plot
    # Try using the new plotting function, fall back to old if needed
    try:
        throughput.plot_results(
            avg_r1_log, avg_r2_log,
            avg_tdma_log, avg_aloha_log,
            avg_collisions_log, step_interval=500
        )
        print("Results plotted to paper_results.pdf")
    except AttributeError:
        # Fallback for old throughput.py
        print("Using legacy plot function...")
        throughput.plot_with_collisions_and_d2ls(
            avg_r1_log, avg_r2_log, avg_tdma_log, avg_aloha_log,
            avg_collisions_log, avg_d2ls1_log=[], avg_d2ls2_log=[],
            step_interval=500, save_path="plot.pdf"
        )

    csv_file.close()

    # =========================
    # AUTO PLOT AFTER TRAINING
    # =========================
    try:
        print("Generating additional plots...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plot_script = os.path.join(script_dir, "plot_results.py")
        plot_input = os.path.join(script_dir, "training_logs.npz")
        if os.path.exists(plot_input):
            # Use the same interpreter that ran Run.py (argument list avoids Windows quoting issues)
            subprocess.run([sys.executable, plot_script], check=False)
        else:
            print(f"[WARN] Skipping additional plots: missing {plot_input}")
    except Exception as e:
        print(f"[WARN] Failed to auto-generate additional plots: {e}")

    return {
        "reward_mode": REWARD_MODE,
        "seed": s,
        "agent1_throughput": float(avg_r1),
        "agent2_throughput": float(avg_r2),
        "sum_drl_throughput": float(sum_drl_throughput),
        "tdma_throughput": float(avg_tdma),
        "aloha_throughput": float(avg_aloha),
        "total_coexist_throughput": float(coexist_throughput),
        "jain_fairness": float(jfi),
        "d2ls_agent1": float(mean_d2ls1),
        "d2ls_agent2": float(mean_d2ls2),
        "total_collisions": int(total_cols),
        "collision_rate": float(col_rate_step),
        "collisions_per_1k": float(col_rate_1k),
        "energy_per_bit_agent1": float(epb_1),
        "energy_per_bit_agent2": float(epb_2),
        "eval_agents_throughput": float(eval_throughput_agents),
        "eval_coexist_throughput": float(eval_throughput_coexist),
        "eval_collisions_per_1k": float(eval_col_per_1k),
    }


def main():
    all_run_results = []

    for s in SEEDS:
        print(f"\n{'='*60}")
        print(f"               RUNNING SEED: {s}")
        print(f"{'='*60}\n")

        metrics = run_simulation(s)
        evaluation_throughput = metrics["eval_agents_throughput"]

        # Store result for this seed
        all_run_results.append(evaluation_throughput)
        print(f">>> Seed {s} completed. Eval Throughput: {evaluation_throughput:.4f}\n")

    # AFTER ALL SEEDS: Print the Scientific Mean
    print(f"\nFinal Result: {np.mean(all_run_results):.4f} +/- {np.std(all_run_results):.4f}")


if __name__ == "__main__":
    main()
