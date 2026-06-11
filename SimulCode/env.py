import numpy as np
import random
from collections import defaultdict


class ENVIRONMENT:
    def __init__(self, state_size=60, n_slots=4, random_seed=None):
        if state_size <= 0: raise ValueError("state_size must be a positive integer.")
        if n_slots <= 0: raise ValueError("n_slots must be a positive integer.")

        self.state_size = state_size
        self.n_slots = n_slots
        self.n_offsets = 3
        self.snr_db_mean = 12.0
        self.snr_db_std = 3.0
        self.rx_sensitivity_db = -90

        self.successful_payload_slots = 0
        self.total_slots = 0

        self.theoretical_cap = getattr(self, "theoretical_cap", 0.9)
        self.cap_bonus_beta = 0.05
        self.cap_bonus_clip = 1.0

        # --- DYNAMIC TDMA SETUP ---
        # Initially, TDMA uses all slots (heavy traffic)
        self.tdma_pattern = list(range(self.n_slots))
        self.tdma_counter = 0

        # Probability that ALOHA transmits
        self.aloha_prob = 0.3
        self.cluster_members_per_cluster = 0
        self.cluster_member_tx_prob = self.aloha_prob

        self.agent_distance = 1000
        self.slot_duration = 0.2
        self.signal_speed = 1500
        self.frequency_khz = 25

        self.DRL_delay = int(np.ceil((self.agent_distance / self.signal_speed) / self.slot_duration))
        if self.DRL_delay == 0: self.DRL_delay = 1

        self.energy_per_transmit = 0.5
        self.energy_per_receive = 0.2
        self.energy_per_idle_listen = 0.05
        self.energy_per_sleep = 0.005
        self.energy_penalty_weight = 0.3
        self.agent_energy_1 = 1000.0
        self.agent_energy_2 = 1000.0

        self.agent1_priority = 0.7
        self.agent2_priority = 0.3

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.current_step = 0
        self.last_success_step_1 = 0
        self.last_success_step_2 = 0
        self.collision_penalty_val = 1.0
        self.d2ls_penalty_weight = 0.05

        # *** IMPORTANT: Default to TRUE for training with interference ***
        self.baselines_coexist = True

        self.last_info_delayed = {"both_tx_collision": False, "single_tx_idx": None}
        self.reset_logs()

    def change_tdma_pattern(self, new_pattern):
        """Called by Run.py to simulate environment change."""
        print(f"\n[ENV] !!! CHANGING TDMA PATTERN FROM {self.tdma_pattern} TO {new_pattern} !!!")
        self.tdma_pattern = new_pattern
        self.tdma_counter = 0  # Reset counter to align with new pattern

    def reset_logs(self):
        self.raw_agent1_reward_log = []
        self.raw_agent2_reward_log = []
        self.tdma_reward_log = []
        self.aloha_reward_log = []
        self.shaped_agent1_reward_log = []
        self.shaped_agent2_reward_log = []
        self.observation1_log = []
        self.observation2_log = []
        self.energy_log_1 = []
        self.energy_log_2 = []
        self.agent1_total_energy_log = []
        self.agent2_total_energy_log = []
        self.collisions_this_step = 0
        self.total_collisions = 0
        self.agent_collision_count = 0
        self.collision_events = 0
        self.agent1_tx_attempts_log = []
        self.agent2_tx_attempts_log = []
        self.agent1_collided_log = []
        self.agent2_collided_log = []
        self.agent1_delay_log = []
        self.agent2_delay_log = []
        self.current_step = 0
        self.last_success_step_1 = 0
        self.last_success_step_2 = 0
        self.observation_counter = 0
        self.agent1_tx_energy = 0.0
        self.agent2_tx_energy = 0.0
        self.agent1_rx_energy = 0.0
        self.agent2_rx_energy = 0.0
        self.agent1_idle_listen_energy = 0.0
        self.agent2_idle_listen_energy = 0.0
        self.agent1_sleep_energy = 0.0
        self.agent2_sleep_energy = 0.0
        self.agent1_total_energy = 0.0
        self.agent2_total_energy = 0.0

    def reset(self):
        self.tdma_counter = 0
        self.agent_energy_1 = 1000.0
        self.agent_energy_2 = 1000.0
        self.reset_logs()
        return np.zeros(self.state_size, dtype=float), np.zeros(self.state_size, dtype=float)

    def get_channel_gain(self, distance_m, frequency_khz):
        f_ghz = frequency_khz / 1000.0
        absorption_factor = 0.01
        spreading_loss = 20 * np.log10(distance_m) if distance_m > 0 else 0
        absorption_loss = absorption_factor * distance_m / 1000 * f_ghz ** 2
        pathloss_db = spreading_loss + absorption_loss
        fading_db = np.random.normal(0.0, self.snr_db_std)
        return -(pathloss_db) + self.snr_db_mean + fading_db

    def is_successful(self, gain_db, sensitivity_db=None):
        if sensitivity_db is None:
            sensitivity_db = self.rx_sensitivity_db
        return gain_db >= sensitivity_db

    @property
    def idle_action_index(self):
        return self.n_slots * self.n_offsets

    def decode_action(self, a):
        if a == self.idle_action_index:
            return True, None, None
        slot = a // self.n_offsets
        offset = a % self.n_offsets
        return False, slot, offset

    def step(self, action1, action2):
        self.current_step += 1
        self.observation_counter += 1
        self.collisions_this_step = 0
        self.total_slots += 1

        a1_idle, selected_slot1, offset_index1 = self.decode_action(action1)
        a2_idle, selected_slot2, offset_index2 = self.decode_action(action2)

        tx1 = (not a1_idle)
        tx2 = (not a2_idle)
        self.agent1_tx_attempts_log.append(1 if tx1 else 0)
        self.agent2_tx_attempts_log.append(1 if tx2 else 0)

        transmissions = defaultdict(list)
        if not a1_idle:
            transmissions[(selected_slot1, offset_index1)].append('agent1')
        if not a2_idle:
            transmissions[(selected_slot2, offset_index2)].append('agent2')

        # --- Baselines (TDMA / ALOHA) ---
        # Robust check to ensure TDMA pattern is valid
        if not self.tdma_pattern:
            tdma_slot = 0  # Fallback
        else:
            tdma_slot = self.tdma_pattern[self.tdma_counter % len(self.tdma_pattern)]

        tdma_offset = (self.tdma_counter // max(1, len(self.tdma_pattern))) % self.n_offsets
        self.tdma_counter = (self.tdma_counter + 1) % (max(1, len(self.tdma_pattern)) * self.n_offsets)

        aloha_transmits = random.random() < self.aloha_prob
        aloha_slot = random.randint(0, self.n_slots - 1) if aloha_transmits else None
        aloha_offset = random.randint(0, self.n_offsets - 1) if aloha_transmits else None

        gain1 = self.get_channel_gain(self.agent_distance, self.frequency_khz)
        gain2 = self.get_channel_gain(self.agent_distance, self.frequency_khz)
        gain_tdma = self.get_channel_gain(self.agent_distance, self.frequency_khz)
        gain_aloha = self.get_channel_gain(self.agent_distance, self.frequency_khz)

        r1_raw = r2_raw = tdma_success_raw = aloha_success_raw = 0
        agent1_collided = agent2_collided = False
        tdma_collided = False
        aloha_collided = False
        self.collisions_this_step = 0

        # *** CRITICAL FOR Q2 PAPER: Always coexist during training ***
        if self.baselines_coexist:
            transmissions[(tdma_slot, tdma_offset)].append('tdma')
            if aloha_transmits:
                transmissions[(aloha_slot, aloha_offset)].append('aloha')

        for cluster_idx in range(2):
            for member_idx in range(self.cluster_members_per_cluster):
                if random.random() < self.cluster_member_tx_prob:
                    slot = random.randint(0, self.n_slots - 1)
                    offset = random.randint(0, self.n_offsets - 1)
                    transmissions[(slot, offset)].append(
                        f"cluster{cluster_idx + 1}_member{member_idx + 1}"
                    )

        # Collision Resolution
        for (slot, offset), who in transmissions.items():
            if len(who) > 1:
                self.collision_events += 1
                if 'agent1' in who: agent1_collided = True; self.collisions_this_step += 1; self.agent_collision_count += 1
                if 'agent2' in who: agent2_collided = True; self.collisions_this_step += 1; self.agent_collision_count += 1
                if 'tdma' in who: tdma_collided = True
                if 'aloha' in who: aloha_collided = True
            else:
                if who[0] == 'agent1' and self.is_successful(gain1): r1_raw = 1
                if who[0] == 'agent2' and self.is_successful(gain2): r2_raw = 1
                if who[0] == 'tdma' and self.is_successful(gain_tdma): tdma_success_raw = 1
                if who[0] == 'aloha' and self.is_successful(gain_aloha): aloha_success_raw = 1

        if not self.baselines_coexist:
            if self.is_successful(gain_tdma): tdma_success_raw = 1
            if aloha_transmits and self.is_successful(gain_aloha): aloha_success_raw = 1

        self.total_collisions += self.collisions_this_step

        if r1_raw == 1:
            self.last_success_step_1 = self.current_step
        else:
            self.agent1_delay_log.append(self.current_step - self.last_success_step_1)

        if r2_raw == 1:
            self.last_success_step_2 = self.current_step
        else:
            self.agent2_delay_log.append(self.current_step - self.last_success_step_2)

        if r1_raw == 1: self.successful_payload_slots += 1
        if r2_raw == 1: self.successful_payload_slots += 1

        base1 = float(r1_raw) * self.agent1_priority
        base2 = float(r2_raw) * self.agent2_priority
        shaped1 = base1
        shaped2 = base2

        if not a1_idle:
            energy_cost1 = self.energy_per_transmit
            shaped1 -= self.energy_penalty_weight * energy_cost1

        if not a2_idle:
            energy_cost2 = self.energy_per_transmit
            shaped2 -= self.energy_penalty_weight * energy_cost2

        if r1_raw == 0 and agent1_collided: shaped1 -= self.collision_penalty_val
        if r2_raw == 0 and agent2_collided: shaped2 -= self.collision_penalty_val

        if r1_raw == 0: shaped1 -= self.d2ls_penalty_weight * (self.current_step - self.last_success_step_1)
        if r2_raw == 0: shaped2 -= self.d2ls_penalty_weight * (self.current_step - self.last_success_step_2)

        current_payloads_per_slot = float(r1_raw + r2_raw)
        cap_track = 0.0
        if self.theoretical_cap > 0:
            cap_track = min(current_payloads_per_slot / self.theoretical_cap, self.cap_bonus_clip)
        cap_bonus = self.cap_bonus_beta * cap_track
        shaped1 += cap_bonus
        shaped2 += cap_bonus

        channel_has_success = (r1_raw + r2_raw + tdma_success_raw + aloha_success_raw) > 0

        if tx1:
            step_energy_1 = self.energy_per_transmit
            self.agent1_tx_energy += step_energy_1
        elif channel_has_success:
            step_energy_1 = self.energy_per_receive
            self.agent1_rx_energy += step_energy_1
        else:
            step_energy_1 = self.energy_per_idle_listen
            self.agent1_idle_listen_energy += step_energy_1

        if tx2:
            step_energy_2 = self.energy_per_transmit
            self.agent2_tx_energy += step_energy_2
        elif channel_has_success:
            step_energy_2 = self.energy_per_receive
            self.agent2_rx_energy += step_energy_2
        else:
            step_energy_2 = self.energy_per_idle_listen
            self.agent2_idle_listen_energy += step_energy_2

        self.agent1_total_energy += step_energy_1
        self.agent2_total_energy += step_energy_2
        self.agent_energy_1 = max(0.0, self.agent_energy_1 - step_energy_1)
        self.agent_energy_2 = max(0.0, self.agent_energy_2 - step_energy_2)

        self.agent1_collided_log.append(1 if agent1_collided else 0)
        self.agent2_collided_log.append(1 if agent2_collided else 0)

        self.raw_agent1_reward_log.append(r1_raw)
        self.raw_agent2_reward_log.append(r2_raw)
        self.shaped_agent1_reward_log.append(shaped1)
        self.shaped_agent2_reward_log.append(shaped2)
        self.tdma_reward_log.append(tdma_success_raw)
        self.aloha_reward_log.append(aloha_success_raw)
        self.observation1_log.append(float(r1_raw))
        self.observation2_log.append(float(r2_raw))
        self.energy_log_1.append(self.agent_energy_1)
        self.energy_log_2.append(self.agent_energy_2)
        self.agent1_total_energy_log.append(step_energy_1)
        self.agent2_total_energy_log.append(step_energy_2)

        min_log_len = self.DRL_delay
        if self.observation_counter >= min_log_len:
            idx = -self.DRL_delay
            if len(self.observation1_log) < self.DRL_delay:
                self.last_info_delayed = {"both_tx_collision": False, "single_tx_idx": None}
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            obs1_delayed = self.observation1_log[idx]
            obs2_delayed = self.observation2_log[idx]
            shaped_r1_delayed = self.shaped_agent1_reward_log[idx]
            shaped_r2_delayed = self.shaped_agent2_reward_log[idx]
            raw_r1_delayed = self.raw_agent1_reward_log[idx]
            raw_r2_delayed = self.raw_agent2_reward_log[idx]
            tdma_r_delayed = self.tdma_reward_log[idx]
            aloha_r_delayed = self.aloha_reward_log[idx]
            overall_reward_delayed = raw_r1_delayed + raw_r2_delayed

            try:
                tx1_d = bool(self.agent1_tx_attempts_log[idx])
                tx2_d = bool(self.agent2_tx_attempts_log[idx])
                col1_d = bool(self.agent1_collided_log[idx])
                col2_d = bool(self.agent2_collided_log[idx])
            except IndexError:
                tx1_d = tx2_d = col1_d = col2_d = False

            both_tx_collision_d = bool(tx1_d and tx2_d and col1_d and col2_d)
            single_tx_idx_d = None
            if (tx1_d ^ tx2_d):
                if raw_r1_delayed == 1:
                    single_tx_idx_d = 0
                elif raw_r2_delayed == 1:
                    single_tx_idx_d = 1

            self.last_info_delayed = {
                "both_tx_collision": both_tx_collision_d,
                "single_tx_idx": single_tx_idx_d,
                "agent_collision_participation": int(col1_d) + int(col2_d),
            }

            return (obs1_delayed, obs2_delayed,
                    overall_reward_delayed,
                    shaped_r1_delayed,
                    shaped_r2_delayed,
                    tdma_r_delayed,
                    aloha_r_delayed,
                    raw_r1_delayed,
                    raw_r2_delayed)
        else:
            self.last_info_delayed = {"both_tx_collision": False, "single_tx_idx": None}
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
