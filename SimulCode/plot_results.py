import numpy as np
import matplotlib.pyplot as plt
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_NPZ_PATH = os.path.join(_HERE, "training_logs.npz")

data = np.load(_NPZ_PATH)

steps = data["steps"]
throughput = data["throughput"]
collisions = data["collisions"]
energy = data["energy"]
fairness = data["fairness"]
delay = data["delay"]

# =========================
# (B) COLLISION RATE GRAPH
# =========================
plt.figure()
plt.plot(steps, collisions)
plt.xlabel("Steps")
plt.ylabel("Collisions")
plt.title("Collision Rate vs Steps")
plt.grid()
plt.savefig("collision_plot.png")

# =========================
# (C) ENERGY GRAPH
# =========================
plt.figure()
plt.plot(steps, energy)
plt.xlabel("Steps")
plt.ylabel("Energy per Bit")
plt.title("Energy Efficiency vs Steps")
plt.grid()
plt.savefig("energy_plot.png")

# =========================
# (D) FAIRNESS GRAPH
# =========================
plt.figure()
plt.plot(steps, fairness)
plt.xlabel("Steps")
plt.ylabel("Jain Fairness Index")
plt.title("Fairness vs Steps")
plt.grid()
plt.savefig("fairness_plot.png")

# =========================
# (E) DELAY GRAPH
# =========================
plt.figure()
plt.plot(steps, delay)
plt.xlabel("Steps")
plt.ylabel("D2LS Delay")
plt.title("Delay vs Steps")
plt.grid()
plt.savefig("delay_plot.png")

# =========================
# (F) ADAPTATION GRAPH
# =========================
plt.figure()
plt.plot(steps, throughput)

# Mark dynamic change point
plt.axvline(x=12000, linestyle='--')

plt.xlabel("Steps")
plt.ylabel("Throughput")
plt.title("Adaptation to Dynamic Change")
plt.grid()
plt.savefig("adaptation_plot.png")

print("All plots saved successfully!")
