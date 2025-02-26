import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameter
BATCH_SIZE = 8192
BUFFER_SIZE = 100000
GAMMA = 0.99
LR = 0.0001
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE = 10


class OptimizedReplayBuffer:
    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        self.index = 0
        self.size = 0

        # pre-allocation of memory
        self.states = np.zeros((capacity, *state_shape), dtype=np.int64)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.int64)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    def push(self, state, action, reward, next_state, done):
        idx = self.index % self.capacity
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state if next_state is not None else -1
        self.dones[idx] = done
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self):
        return self.size


class ResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class DuelingDQN(nn.Module):
    def __init__(self, input_shape, action_size):
        super(DuelingDQN, self).__init__()
        self.input_shape = input_shape
        self.action_size = action_size

        # Feature extraction layers
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.res1 = ResidualBlock3D(32)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.res2 = ResidualBlock3D(64)

        # Calculate linear layer input size with corrected dimension ordering (pipes, rows, cols)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_shape[2], input_shape[0], input_shape[1])
            dummy_output = self.feature_extraction(dummy_input)
            linear_input_size = dummy_output.view(1, -1).size(1)

        # Value stream (dropout removed for RL stability)
        self.value_stream = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            # nn.Dropout(0.2),  # Dropout removed for more stable Q-value estimates
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Advantage stream (dropout removed for RL stability)
        self.advantage_stream = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            # nn.Dropout(0.2),  # Dropout removed for more stable Q-value estimates
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

    def feature_extraction(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.res2(x)
        return x

    def forward(self, x):
        features = self.feature_extraction(x)
        features = features.view(features.size(0), -1)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals


class DQNAgent:
    def __init__(self, input_shape, action_size):
        self.input_shape = input_shape
        self.action_size = action_size
        self.epsilon = EPS_START

        # Initialize networks
        self.policy_net = DuelingDQN(input_shape, action_size).to(device)
        self.target_net = DuelingDQN(input_shape, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR, weight_decay=1e-5)

        # Initialize optimized replay buffer in memory
        self.memory = OptimizedReplayBuffer(BUFFER_SIZE, input_shape)

        # Pre-allocation of preprocess buffer: shape (batch, channel, pipes, rows, cols)
        self.preprocess_buffer = torch.empty(
            (BATCH_SIZE, 1, input_shape[2], input_shape[0], input_shape[1]),
            device=device,
            dtype=torch.float32
        )

    def preprocess_batch(self, states):
        with np.errstate(divide='ignore', invalid='ignore'):
            log_states = np.log2(np.where(states > 0, states, 1)) / 11.0
        # Convert from (batch, rows, cols, pipes) to (batch, pipes, rows, cols)
        log_states = np.transpose(log_states, (0, 3, 1, 2))
        log_states = np.expand_dims(log_states, 1)  # (batch, channel, pipes, rows, cols)

        # Use the preallocated buffer to avoid repeated memory allocation
        self.preprocess_buffer[:len(states)] = torch.from_numpy(log_states).float()
        return self.preprocess_buffer[:len(states)]

    def preprocess_single_state(self, state):
        with np.errstate(divide='ignore', invalid='ignore'):
            processed = np.log2(np.where(state > 0, state, 1)) / 11.0
        processed = np.transpose(processed, (2, 0, 1))  # (pipes, rows, cols)
        processed = np.expand_dims(processed, 0)         # (1, rows, cols, pipes)
        processed = np.expand_dims(processed, 1)         # (1, channel, rows, cols, pipes)
        return torch.from_numpy(processed).float().to(device)

    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.policy_net(state)
                return q_values.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_size)]],
                                device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        # Sample batches
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        state_batch = self.preprocess_batch(states)
        next_state_batch = self.preprocess_batch(next_states)

        action_batch = torch.tensor(actions, device=device).unsqueeze(1)
        reward_batch = torch.tensor(rewards, device=device)
        done_batch = torch.tensor(dones, device=device)

        # Q-value calculation
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)

        next_q_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_q_values[~done_batch] = self.target_net(next_state_batch[~done_batch]).max(1)[0]

        target_q_values = (next_q_values * GAMMA) + reward_batch

        # Loss Computation and Parameter Optimization
        loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        # Removed torch.cuda.empty_cache() for improved training speed

    def update_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())