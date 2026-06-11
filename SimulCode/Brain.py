import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


# from collections import deque # REMOVED: No longer needed for nimble training placeholders

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MLPModel(nn.Module):  # Renamed from DenseNetModel
    def __init__(self, state_size, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        return self.model(x)

# ==== NEW: Dueling MLP head ====
class DuelingMLPModel(nn.Module):
    def __init__(self, state_size, n_actions, hidden=64, depth=5):
        super().__init__()
        layers = []
        in_dim = state_size
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        self.backbone = nn.Sequential(*layers)
        self.adv = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))
        self.val = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self._init_weights()

    def _init_weights(self):
        for m in list(self.backbone) + list(self.adv) + list(self.val):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        h = self.backbone(x)
        A = self.adv(h)
        V = self.val(h)
        return V + A - A.mean(dim=1, keepdim=True)


class DQN(nn.Module):
    def __init__(self, state_size, n_actions,
                 state_action_memory_size, memory_size=500, replace_target_iter=200,
                 batch_size=32, learning_rate=0.01, gamma=0.9, epsilon=1, epsilon_min=0.01, epsilon_decay=0.995):
        super().__init__()
        self.state_size = state_size
        self.n_actions = n_actions

        # self.n_offsets = 2 # This is an env property, agent doesn't need to know directly here.

        self.memory = np.zeros((memory_size, state_size * 2 + 2))  # s, a, r, s'
        self.memory_counter = 0
        self.state_action_memory_size = state_action_memory_size  # Used for sampling delay in learn()
        self.replace_target_iter = replace_target_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # ==== CHANGED: use DuelingMLPModel instead of MLPModel ====
        self.model = DuelingMLPModel(state_size, n_actions).to(device)
        self.target_model = DuelingMLPModel(state_size, n_actions).to(device)
        self.replace_target_parameters()

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.learn_step_counter = 0
        self.last_loss = 0.0  # NEW: latest training loss for logging

    # ==== NEW: soft target update (tau) ====
    def soft_update(self, tau=0.01):
        with torch.no_grad():
            for p, pt in zip(self.model.parameters(), self.target_model.parameters()):
                pt.data.mul_(1 - tau).add_(tau * p.data)

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if np.random.random() < self.epsilon:  # Epsilon-greedy
            action = np.random.randint(0, self.n_actions)
        else:
            self.model.eval()
            with torch.no_grad():
                actions_value = self.model(state)
            action = torch.argmax(actions_value).item()
            self.model.train()

        # Decay epsilon after action choice
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        return action

    # Use this for evaluation to avoid epsilon updates
    def choose_action_greedy(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.model.eval()
        with torch.no_grad():
            q = self.model(state)
        action = torch.argmax(q, dim=1).item()
        self.model.train()
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.concatenate((s, [a, r], s_))
        index = self.memory_counter % self.memory.shape[0]
        self.memory[index, :] = transition
        self.memory_counter += 1

    def replace_target_parameters(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def learn(self):
        if self.memory_counter < self.batch_size or self.memory_counter < self.state_action_memory_size:
            return

        if self.learn_step_counter % self.replace_target_iter == 0:
            self.replace_target_parameters()
        self.learn_step_counter += 1

        max_mem = min(self.memory_counter, self.memory.shape[0])
        valid_start_indices = np.arange(max_mem - self.state_action_memory_size)
        if len(valid_start_indices) == 0:
            return

        # 70% recent (last 30% of buffer), 30% older
        max_mem = min(self.memory_counter, self.memory.shape[0])
        recent_portion = int(0.3 * max_mem)
        recent_mask = valid_start_indices >= (max_mem - recent_portion)
        recent_idx = valid_start_indices[recent_mask]
        old_idx = valid_start_indices[~recent_mask]
        k_recent = int(self.batch_size * 0.7)
        k_old = self.batch_size - k_recent
        pick_recent = np.random.choice(recent_idx, size=min(k_recent, len(recent_idx)),
                                       replace=len(recent_idx) < k_recent) if len(recent_idx) > 0 else np.array([],
                                                                                                                dtype=int)
        pick_old = np.random.choice(old_idx, size=min(k_old, len(old_idx)),
                                    replace=len(old_idx) < k_old) if len(old_idx) > 0 else np.array([], dtype=int)
        if len(pick_recent) + len(pick_old) < self.batch_size:
            # fallback to uniform if buffer is tiny
            sample_indices = np.random.choice(valid_start_indices, self.batch_size,
                                              replace=len(valid_start_indices) < self.batch_size)
        else:
            sample_indices = np.concatenate([pick_recent, pick_old])


        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []

        for idx in sample_indices:
            s = self.memory[idx, :self.state_size]
            a = int(self.memory[idx, self.state_size])
            r_total = 0.0
            discount = 1.0
            for step in range(self.state_action_memory_size):
                r = self.memory[idx + step, self.state_size + 1]
                r_total += discount * r
                discount *= self.gamma
            s_ = self.memory[idx + self.state_action_memory_size - 1, -self.state_size:]

            state_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r_total)
            next_state_batch.append(s_)

        state_batch = torch.FloatTensor(np.array(state_batch)).to(device)
        action_batch = torch.LongTensor(np.array(action_batch)).to(device)
        reward_batch = torch.FloatTensor(np.array(reward_batch)).to(device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(device)

        q_eval = self.model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        # ...existing code...
        B = state_batch.size(0)
        q_next_target = self.target_model(next_state_batch).detach()
        q_next_online_actions = torch.argmax(self.model(next_state_batch), dim=1)
        batch_idx = torch.arange(B, device=next_state_batch.device, dtype=torch.long)
        q_next_selected = q_next_target[batch_idx, q_next_online_actions]
        # ...existing code...
        q_target = reward_batch + (self.gamma ** self.state_action_memory_size) * q_next_selected

        loss = self.loss_fn(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)  # NEW
        self.optimizer.step()
        self.last_loss = float(loss.item())  # NEW
        # ==== NEW: also do a gentle soft update each learn ====
        self.soft_update(tau=0.01)
