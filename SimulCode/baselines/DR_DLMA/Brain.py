import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# GPU Run
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

class DenseNetModel(nn.Module):
    def __init__(self, state_size, n_actions):
        super().__init__()
        self.state_size = state_size
        self.n_actions = n_actions
        self.model = self.build_model()
        self.initialize_weights()


    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_actions)
        )

    def initialize_weights(self):
        base_seed = 247
        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.Linear):
                torch.manual_seed(base_seed + i)
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        return self.model(x)


class DQN(nn.Module):
    def __init__(self, state_size, n_actions, n_nodes,
                 state_action_memory_size, memory_size=500, replace_target_iter=200, batch_size=32, learning_rate=0.01,
                 gamma=0.9, epsilon=1, epsilon_min=0.01, epsilon_decay=0.995):
        super().__init__()
        set_seed(42)  # Set global seed

        # Hyperparameters
        self.state_size = state_size
        self.n_actions = n_actions
        self.n_nodes = n_nodes
        self.state_action_memory_size = state_action_memory_size
        self.memory_size = memory_size
        self.replace_target_iter = replace_target_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = np.zeros((self.memory_size, self.state_size * 2 + 2))

        # Temporary parameters
        self.learn_state_action_counter = 0
        self.learn_step_counter = 0
        self.memory_counter = 0

        # Build models
        self.model = DenseNetModel(state_size, n_actions).to(device)
        self.target_model = DenseNetModel(state_size, n_actions).to(device)
        self.replace_target_parameters()

        # Optimizer and loss function
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        with torch.no_grad():
            action_values = self.model(state)
            return torch.argmax(action_values).item()

    def store_transition(self, s, a, r, s_):  # s_: next_state
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.concatenate((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def replace_target_parameters(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.replace_target_parameters()
        self.learn_step_counter += 1

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        sample_index1 = []
        sample_index2 = []
        for i in sample_index:
            if i >= self.memory_size - self.state_action_memory_size:
                if self.memory_counter > self.memory_size:
                    a11 = np.random.randint(0, self.memory_size - self.state_action_memory_size)
                    sample_index1.append(a11)
                    sample_index2.append(a11 + self.state_action_memory_size - 1)
                else:
                    a12 = np.random.randint(0, self.memory_counter - self.state_action_memory_size)
                    sample_index1.append(a12)
                    sample_index2.append(a12 + self.state_action_memory_size - 1)
            else:
                sample_index1.append(i)
                sample_index2.append(i + self.state_action_memory_size - 1)

        batch_memory1 = self.memory[sample_index1, :]
        batch_memory2 = self.memory[sample_index2, :]

        state = torch.FloatTensor(batch_memory1[:, :self.state_size]).to(device)
        action = torch.LongTensor(batch_memory1[:, self.state_size].astype(int)).to(device)
        reward = torch.FloatTensor(batch_memory2[:, self.state_size + 1]).to(device)
        next_state = torch.FloatTensor(batch_memory2[:, -self.state_size:]).to(device)

        q_eval = self.model(state)
        q_next = self.target_model(next_state).detach()
        q_target = q_eval.clone().detach()  # Added .detach() to prevent in-place autograd errors
        batch_index = torch.arange(self.batch_size, dtype=torch.long).to(device)
        q_target[batch_index, action] = reward + self.gamma * torch.max(q_next, dim=1)[0]

        self.optimizer.zero_grad()
        loss = self.loss_fn(q_eval, q_target)
        loss.backward()
        self.optimizer.step()
