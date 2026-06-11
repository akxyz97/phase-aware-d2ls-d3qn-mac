

Traditional MAC protocols such as ALOHA and TDMA are simple, but they are not sufficiently adaptive in dynamic underwater acoustic environments. This project investigates a deep reinforcement learning-based MAC design using D3QN agents, phase-aware state representation, D2LS fairness control, cooperative reward shaping, and energy-aware learning.



\---



\## Main Contributions



This repository includes the implementation of:



1\. \*\*Phase-aware state representation\*\*

&#x20;  A normalized slot-phase indicator is added to the agent state to help the DRL model learn periodic channel access behavior.



2\. \*\*D2LS-based fairness mechanism\*\*

&#x20;  Delay-to-Last-Success is used to encourage temporal fairness among learning nodes.



3\. \*\*Cooperative reward shaping\*\*

&#x20;  A team-level reward adjustment is introduced to improve coordination among independent DRL agents.



4\. \*\*Energy-aware MAC evaluation\*\*

&#x20;  Transmission, reception, idle listening, and sleep energy are considered in the simulation.



