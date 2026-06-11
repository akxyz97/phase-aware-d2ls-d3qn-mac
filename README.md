Phase-aware Cooperative D2LS-D3QN MAC for Underwater Acoustic Networks

This repository contains the simulation code and experiment results for a Phase-aware Cooperative D2LS-D3QN-based Medium Access Control (MAC) protocol for Underwater Acoustic Sensor Networks (UASNs).

The project focuses on adaptive channel access under long underwater acoustic propagation delay, delayed feedback, collision risk, and energy constraints. The proposed model uses Dueling Double Deep Q-Network (D3QN) agents with phase-aware state representation, Delay-to-Last-Success (D2LS) reward shaping, cooperative reward adjustment, and MAC-layer energy accounting.



Research Background

Underwater Acoustic Sensor Networks are important for marine environmental monitoring, underwater exploration, disaster warning, and tactical surveillance. However, MAC protocol design in underwater acoustic networks is difficult because acoustic communication suffers from long propagation delay, limited bandwidth, high collision probability, partial observability, and strict energy constraints.

Traditional MAC protocols such as ALOHA and TDMA are simple, but they often suffer from random collisions, static scheduling, and weak adaptability in delayed underwater feedback environments. This project addresses these problems using a deep reinforcement learning-based adaptive MAC framework.



Main Features

•	Phase-aware state representation for learning periodic channel access patterns

•	Dueling Double Deep Q-Network (D3QN)-based MAC decision-making

•	Delay-aware learning with delayed ACK/reward feedback

•	Delay-to-Last-Success (D2LS) reward term for temporal fairness

•	Cooperative reward adjustment for multi-agent coordination

•	MAC-layer energy accounting including transmission, reception, idle listening, and sleep states

•	Baseline comparison with TDMA, ALOHA, Slotted FAMA, and DR-DLMA

•	Sensitivity analysis for reward weights, D2LS weight, cooperation parameters, and traffic load

•	Static clustered UASN scenario evaluation

•	Multi-seed experiment results for reliability



Repository Structure

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

│   ├── plot\_results.py

│   ├── run\_collision\_sensitivity.py

│   ├── run\_coop\_sensitivity.py

│   ├── run\_d2ls\_sensitivity.py

│   ├── run\_energy\_sensitivity.py

│   ├── run\_phase\_ablation.py

│   ├── run\_reward\_mode\_comparison.py

│   ├── run\_selected\_coop\_full.py

│   ├── run\_selected\_d2ls\_full.py

│   ├── run\_selected\_reward\_weights\_full.py

│   ├── run\_slotted\_fama\_baseline.py

│   ├── run\_static\_clustered\_uasn\_experiment.py

│   ├── run\_traffic\_load\_experiment.py

│   │

│   ├── nodes/

│   │   ├── ALOHA.py

│   │   └── TDMA.py

│   │

│   └── baselines/

│       └── DR\_DLMA/

│           ├── Brain.py

│           ├── env.py

│           ├── Run.py

│           ├── throughput.py

│           ├── run\_dr\_dlma\_comparison.py

│           ├── nodes/

│           └── rewards/

│

└── results/

&#x20;   ├── final\_model/

&#x20;   ├── cooperative/

&#x20;   ├── independent/

&#x20;   ├── phase\_clock\_true/

&#x20;   ├── phase\_clock\_false/

&#x20;   ├── traffic\_low\_0.10/

&#x20;   ├── traffic\_medium\_0.30/

&#x20;   ├── traffic\_high\_0.50/

&#x20;   └── summary CSV files



Core Files

File	Description

SimulCode/Run.py	Main training and evaluation runner

SimulCode/env.py	Underwater acoustic MAC simulation environment

SimulCode/Brain.py	D3QN agent implementation

SimulCode/throughput.py	Throughput and performance-related calculations

SimulCode/plot\_results.py	Result plotting script

SimulCode/nodes/ALOHA.py	ALOHA baseline node

SimulCode/nodes/TDMA.py	TDMA baseline node

SimulCode/baselines/DR\_DLMA/	DR-DLMA baseline implementation



Installation

Clone the repository:

git clone https://github.com/akxyz97/phase-aware-d2ls-d3qn-mac.git

cd phase-aware-d2ls-d3qn-mac

Create a Python environment:

conda create -n d2ls\_d3qn\_mac python=3.10

conda activate d2ls\_d3qn\_mac

Install dependencies:

pip install -r requirements.txt

Running the Main Model

Go to the simulation folder:

cd SimulCode

Run the main D2LS-D3QN-MAC simulation:

python Run.py



Running Experiment Scripts

From inside the SimulCode/ folder, the following scripts can be used for additional experiments:

python run\_reward\_mode\_comparison.py

python run\_phase\_ablation.py

python run\_d2ls\_sensitivity.py

python run\_coop\_sensitivity.py

python run\_energy\_sensitivity.py

python run\_collision\_sensitivity.py

python run\_traffic\_load\_experiment.py

python run\_static\_clustered\_uasn\_experiment.py

python run\_slotted\_fama\_baseline.py



Running DR-DLMA Baseline

cd SimulCode/baselines/DR\_DLMA

python run\_dr\_dlma\_comparison.py



Results

The results/ folder contains summary CSV files and seed-wise experimental metrics.

Main result files include:

File / Folder	Description

results/final\_model/	Final proposed model results for seeds 42, 43, and 44

results/reward\_mode\_comparison.csv	Independent vs cooperative reward comparison

results/phase\_ablation.csv	Phase-aware state ablation results

results/d2ls\_sensitivity.csv	D2LS weight sensitivity results

results/coop\_sensitivity.csv	Cooperative reward parameter sensitivity results

results/energy\_sensitivity.csv	Energy reward weight sensitivity results

results/collision\_sensitivity.csv	Collision penalty sensitivity results

results/traffic\_load\_experiment.csv	Traffic-load robustness experiment

results/static\_clustered\_uasn\_experiment.csv	Static clustered UASN scenario results

results/slotted\_fama\_baseline.csv	Slotted FAMA baseline results

results/dr\_dlma/baseline\_dr\_dlma\_metrics.csv	DR-DLMA baseline results



Main Metrics

The simulation evaluates the following metrics:

•	DRL throughput

•	Total coexistence throughput

•	Jain fairness index

•	Packet collisions

•	Delay-to-Last-Success (D2LS)

•	Energy per bit

•	Transmission, reception, idle listening, and sleep energy

•	Baseline comparison performance



Method Summary

The proposed MAC protocol models underwater channel access as a decentralized partially observable decision-making problem. Each learning node is controlled by a D3QN agent and selects its MAC action using local history information.

The state representation includes historical action-observation information and a normalized slot-phase indicator. The slot-phase feature helps the agent recognize periodic access patterns, especially when coexisting with scheduled traffic such as TDMA.

The reward function includes individual transmission success, collision penalty, energy-aware reward shaping, D2LS-based temporal fairness, and cooperative reward adjustment. The cooperative component provides a team-level coordination signal without requiring explicit inter-agent message exchange.



Reproducibility Notes

The final experiments are evaluated using multiple random seeds. The seed-wise metric files are included in the results/ folder. Heavy training logs, generated PDF figures, cache files, and model checkpoints are intentionally excluded from the repository to keep it lightweight.

Requirements



The main dependencies are:

numpy

pandas

matplotlib

torch



Citation

If this repository helps your research, please cite the related thesis or paper when available.

Abdul Ali Khan, Research on Phase-Aware Cooperative Deep Reinforcement Learning-Based MAC Protocol Design for Underwater Acoustic Networks, Master's Dissertation, Hohai University, 2026.



Author

Abdul Ali Khan

College of Information Science and Engineering

Hohai University

Nanjing, China



License

This repository is shared for academic and research purposes. A formal license may be added later.



