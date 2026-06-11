import matplotlib.pyplot as plt
import numpy as np


# --- 1. The Missing Function (Added Here) ---
def jains_index(values):
    """
    Calculates Jain's Fairness Index.
    Range: [1/n, 1]. 1 = Perfect Fairness.
    """
    values = np.array(values, dtype=float)
    if len(values) == 0:
        return 0.0

    sum_of_values = float(np.sum(values))
    sum_of_squares = float(np.sum(values ** 2))

    if sum_of_squares == 0.0:
        return 0.0

    numerator = sum_of_values ** 2
    denominator = float(len(values)) * sum_of_squares

    return float(numerator) / float(denominator)


# --- 2. The Plotting Function ---

def plot_results(r1, r2, tdma, aloha, cols, step_interval=500):
    steps = (np.arange(len(r1)) + 1) * step_interval

    plt.figure(figsize=(12, 5))

    # --- Subplot 1: Throughput ---
    plt.subplot(1, 2, 1)

    # CHANGE: Added 'linestyle' and explicit colors for high contrast
    plt.plot(steps, r1, label="Agent 1 (D-DQN)", linestyle='-', color='blue', linewidth=2)
    plt.plot(steps, r2, label="Agent 2 (D-DQN)", linestyle='--', color='orange', linewidth=2)
    plt.plot(steps, tdma, label="TDMA (Legacy)", linestyle='-.', color='green', alpha=0.8)
    plt.plot(steps, aloha, label="ALOHA (Legacy)", linestyle=':', color='red', alpha=0.8)

    # Dynamic Event Line
    plt.axvline(x=12000, color='black', linestyle='-', linewidth=1.5, label="Topology Change")

    plt.xlabel("Training Steps")
    plt.ylabel("Success Rate (Throughput)")
    plt.title("Adaptability to Dynamic Interference")
    plt.legend(loc='lower right', fontsize='small')
    plt.grid(True, alpha=0.3)

    # --- Subplot 2: Collisions ---
    plt.subplot(1, 2, 2)
    plt.plot(steps, cols, color='black', linestyle='-', label="Collisions")
    plt.axvline(x=12000, color='red', linestyle='--', linewidth=1.5)
    plt.xlabel("Training Steps")
    plt.ylabel("Collision Count (per window)")
    plt.title("Collision Rate Stability")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("paper_results.pdf")
    print("Plot saved successfully to paper_results.pdf (Grayscale Friendly)")

# Backward compatibility wrapper (in case old code calls the old name)
def plot_with_collisions_and_d2ls(agent1_rewards, agent2_rewards, tdma_rewards, aloha_rewards,
                                  collision_counts, d2ls_delays_1, d2ls_delays_2,
                                  step_interval=500, save_path=None):
    # Redirect to new function
    plot_results(agent1_rewards, agent2_rewards, tdma_rewards, aloha_rewards, collision_counts, step_interval)
