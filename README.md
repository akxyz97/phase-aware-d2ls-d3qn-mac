# Phase-Aware D2LS-D3QN MAC for Underwater Acoustic Networks

This repository contains the simulation code and experimental results for a **Phase-Aware Cooperative D2LS-D3QN MAC protocol** for Underwater Acoustic Sensor Networks (UASNs).

The project focuses on adaptive MAC-layer channel access under long underwater acoustic propagation delay, delayed feedback, packet collisions, partial observability, and energy constraints. The proposed method combines Dueling Double Deep Q-Network (D3QN) learning, phase-aware state representation, Delay-to-Last-Success (D2LS) fairness control, cooperative reward shaping, and energy-aware MAC evaluation.

---

## Overview

Underwater Acoustic Sensor Networks are useful for marine monitoring, underwater exploration, disaster warning, and tactical surveillance. However, efficient MAC protocol design is difficult because underwater acoustic communication suffers from:

- long propagation delay
- delayed ACK and reward feedback
- limited bandwidth
- high collision probability
- partial observability
- strict energy constraints

Traditional MAC protocols such as ALOHA and TDMA are simple, but they are less adaptive in dynamic underwater acoustic environments. This repository provides a DRL-based MAC simulation framework designed to improve throughput, fairness, collision avoidance, and energy efficiency.

---

## Main Contributions

1. **Phase-aware state representation**  
   A normalized slot-phase indicator is added to the agent state to help the DRL model learn periodic channel access behavior.

2. **D2LS-based fairness mechanism**  
   Delay-to-Last-Success is used as a fairness-aware reward component to reduce long access gaps among learning nodes.

3. **Cooperative reward shaping**  
   A team-level reward adjustment is introduced to improve coordination among independently learning DRL agents.

4. **Energy-aware MAC evaluation**  
   Transmission, reception, idle listening, and sleep energy are considered during MAC-layer performance evaluation.

5. **Baseline and ablation experiments**  
   The proposed method is compared with TDMA, ALOHA, Slotted FAMA, and DR-DLMA. Additional experiments evaluate phase-awareness, D2LS weight, cooperation parameters, energy penalty, collision penalty, and traffic load.

---

## Repository Structure

```text
phase-aware-d2ls-d3qn-mac/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── SimulCode/
│   ├── Brain.py
│   ├── env.py
│   ├── Run.py
│   ├── throughput.py
│   ├── plot_results.py
│   ├── run_phase_ablation.py
│   ├── run_d2ls_sensitivity.py
│   ├── run_coop_sensitivity.py
│   ├── run_energy_sensitivity.py
│   ├── run_collision_sensitivity.py
│   ├── run_reward_mode_comparison.py
│   ├── run_traffic_load_experiment.py
│   ├── run_static_clustered_uasn_experiment.py
│   ├── run_slotted_fama_baseline.py
│   │
│   ├── nodes/
│   │   ├── ALOHA.py
│   │   └── TDMA.py
│   │
│   └── baselines/
│       └── DR_DLMA/
│           ├── Brain.py
│           ├── env.py
│           ├── Run.py
│           ├── throughput.py
│           ├── run_dr_dlma_comparison.py
│           ├── nodes/
│           └── rewards/
│
└── results/
    ├── final_model/
    ├── cooperative/
    ├── independent/
    ├── phase_clock_true/
    ├── phase_clock_false/
    ├── traffic_low_0.10/
    ├── traffic_medium_0.30/
    ├── traffic_high_0.50/
    └── summary CSV files
```

---

## Core Files

| File | Purpose |
|---|---|
| `SimulCode/Run.py` | Main training and evaluation script |
| `SimulCode/env.py` | Underwater acoustic MAC simulation environment |
| `SimulCode/Brain.py` | D3QN agent implementation |
| `SimulCode/throughput.py` | Throughput and performance metric calculation |
| `SimulCode/plot_results.py` | Result plotting and visualization |
| `SimulCode/nodes/ALOHA.py` | ALOHA baseline implementation |
| `SimulCode/nodes/TDMA.py` | TDMA baseline implementation |
| `SimulCode/baselines/DR_DLMA/` | DR-DLMA baseline implementation |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/akxyz97/phase-aware-d2ls-d3qn-mac.git
cd phase-aware-d2ls-d3qn-mac
```

Create and activate a Conda environment:

```bash
conda create -n d2ls_d3qn_mac python=3.10
conda activate d2ls_d3qn_mac
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Main Simulation

From the repository root:

```bash
cd SimulCode
python Run.py
```

---

## Run Additional Experiments

From the `SimulCode/` folder:

```bash
python run_phase_ablation.py
python run_d2ls_sensitivity.py
python run_coop_sensitivity.py
python run_energy_sensitivity.py
python run_collision_sensitivity.py
python run_reward_mode_comparison.py
python run_traffic_load_experiment.py
python run_static_clustered_uasn_experiment.py
python run_slotted_fama_baseline.py
```

To run the DR-DLMA baseline:

```bash
cd SimulCode/baselines/DR_DLMA
python run_dr_dlma_comparison.py
```

---

## Evaluation Metrics

The simulation records the following metrics:

- throughput
- packet collisions
- Jain fairness index
- delay-to-last-success
- energy per bit
- transmission energy
- reception energy
- idle listening energy
- sleep energy
- baseline comparison performance

---

## Result Files

The `results/` directory contains summary CSV files and seed-wise experiment results.

| Result File / Folder | Description |
|---|---|
| `results/final_model/` | Final proposed model results using multiple seeds |
| `results/reward_mode_comparison.csv` | Independent and cooperative reward comparison |
| `results/phase_ablation.csv` | Phase-aware state ablation results |
| `results/d2ls_sensitivity.csv` | D2LS reward weight sensitivity |
| `results/coop_sensitivity.csv` | Cooperative reward sensitivity |
| `results/energy_sensitivity.csv` | Energy penalty sensitivity |
| `results/collision_sensitivity.csv` | Collision penalty sensitivity |
| `results/traffic_load_experiment.csv` | Traffic load experiment |
| `results/static_clustered_uasn_experiment.csv` | Static clustered UASN experiment |
| `results/slotted_fama_baseline.csv` | Slotted FAMA baseline |
| `results/dr_dlma/baseline_dr_dlma_metrics.csv` | DR-DLMA baseline results |

---

## Method Summary

The proposed method formulates underwater MAC access as a decentralized partially observable decision-making problem. Each learning node is controlled by a D3QN agent and makes access decisions using local historical observations.

The state includes recent action-observation history and a normalized slot-phase indicator. The phase-aware feature helps the agent understand periodic access behavior caused by slotted access and scheduled baseline traffic.

The reward function combines successful transmission reward, collision penalty, D2LS fairness reward, energy-aware shaping, and cooperative adjustment. This allows the agents to learn channel access decisions that balance throughput, fairness, and energy efficiency.

---

## Reproducibility

The repository includes selected seed-wise result files for repeated experiments. Generated cache files, model checkpoints, temporary logs, and PDF figures are excluded to keep the repository lightweight.

---

## Dependencies

```text
numpy
pandas
matplotlib
torch
```

---

## Citation

If this repository is useful for your research, please cite the related thesis or future publication:

```text
Abdul Ali Khan, Research on Phase-Aware Cooperative Deep Reinforcement Learning-Based MAC Protocol Design for Underwater Acoustic Networks, Master's Dissertation, Hohai University, 2026.
```

---

## Author

**Abdul Ali Khan**  
College of Information Science and Engineering  
Hohai University  
Nanjing, China

---

## License

This repository is shared for academic and research purposes. A formal license may be added later.
