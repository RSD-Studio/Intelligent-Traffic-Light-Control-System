"""
Advanced DQN Models with Enhanced Features
Implements: Double DQN, Dueling DQN, Prioritized Replay, Noisy Networks, LSTM
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer using Sum Tree"""

    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0=uniform, 1=full prioritization)
            beta: Importance sampling exponent
            beta_increment: Beta annealing rate
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6  # Small constant to avoid zero priority

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        """Add experience with maximum priority"""
        max_priority = self.priorities.max() if self.size > 0 else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample batch with prioritized sampling"""
        if self.size < batch_size:
            batch_size = self.size

        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)

        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize

        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*experiences)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, priorities):
        """Update priorities based on TD errors"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon

    def __len__(self):
        return self.size


class NoisyLinear(nn.Module):
    """Noisy Linear Layer for exploration"""

    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))

        # Factorized noise
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initialize parameters"""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        """Reset noise for both weight and bias"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        """Factorized Gaussian noise"""
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x):
        """Forward pass with noisy weights"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


class DuelingQNetwork(nn.Module):
    """Dueling DQN Architecture with separate value and advantage streams"""

    def __init__(self, state_size, action_size, hidden_size=512, use_noisy=False, use_batch_norm=True):
        super(DuelingQNetwork, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.use_noisy = use_noisy
        self.use_batch_norm = use_batch_norm

        # Shared feature extractor
        if use_noisy:
            self.feature_layer = NoisyLinear(state_size, hidden_size)
        else:
            self.feature_layer = nn.Linear(state_size, hidden_size)

        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_size)

        self.dropout1 = nn.Dropout(0.2)

        # Value stream
        if use_noisy:
            self.value_stream = nn.Sequential(
                NoisyLinear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                NoisyLinear(hidden_size // 2, 1)
            )
        else:
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1)
            )

        # Advantage stream
        if use_noisy:
            self.advantage_stream = nn.Sequential(
                NoisyLinear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                NoisyLinear(hidden_size // 2, action_size)
            )
        else:
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, action_size)
            )

    def forward(self, state):
        """Forward pass computing Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))"""
        # Shared features
        x = self.feature_layer(state)

        if self.use_batch_norm and state.size(0) > 1:  # BatchNorm needs batch size > 1
            x = self.bn1(x)

        x = F.relu(x)
        x = self.dropout1(x)

        # Value and advantage streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

    def reset_noise(self):
        """Reset noise for noisy layers"""
        if self.use_noisy:
            if isinstance(self.feature_layer, NoisyLinear):
                self.feature_layer.reset_noise()

            for module in self.value_stream:
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

            for module in self.advantage_stream:
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


class LSTMQNetwork(nn.Module):
    """LSTM-based Q-Network for temporal modeling"""

    def __init__(self, state_size, action_size, hidden_size=256, lstm_layers=2):
        super(LSTMQNetwork, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers

        # LSTM for temporal processing
        self.lstm = nn.LSTM(state_size, hidden_size, lstm_layers, batch_first=True)

        # Dueling architecture on top of LSTM
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )

    def forward(self, state_sequence, hidden=None):
        """
        Forward pass with sequence input
        Args:
            state_sequence: (batch, seq_len, state_size) or (batch, state_size)
        """
        # Handle single state vs sequence
        if len(state_sequence.shape) == 2:
            state_sequence = state_sequence.unsqueeze(1)  # Add sequence dimension

        # LSTM processing
        lstm_out, hidden = self.lstm(state_sequence, hidden)

        # Use last output
        x = lstm_out[:, -1, :]

        # Dueling architecture
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values, hidden

    def init_hidden(self, batch_size, device):
        """Initialize hidden state"""
        h0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)


class ImprovedDQNAgent:
    """Enhanced DQN Agent with all improvements"""

    def __init__(self,
                 state_size=15,  # Enhanced state size
                 action_size=2,
                 hidden_size=512,
                 learning_rate=0.0001,
                 gamma=0.99,
                 tau=0.005,  # Soft update parameter
                 memory_size=50000,
                 batch_size=64,
                 device=None,
                 use_double_dqn=True,
                 use_dueling=True,
                 use_noisy=True,
                 use_per=True,
                 use_lstm=False,
                 alpha=0.6,  # PER alpha
                 beta=0.4):  # PER beta
        """
        Initialize improved DQN agent

        Args:
            state_size: Enhanced state dimension
            action_size: Action dimension
            hidden_size: Network hidden layer size
            learning_rate: Optimizer learning rate
            gamma: Discount factor
            tau: Soft update coefficient
            memory_size: Replay buffer size
            batch_size: Training batch size
            device: Computing device
            use_double_dqn: Use Double DQN
            use_dueling: Use Dueling architecture
            use_noisy: Use Noisy Networks
            use_per: Use Prioritized Experience Replay
            use_lstm: Use LSTM architecture
            alpha: PER prioritization exponent
            beta: PER importance sampling exponent
        """
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.use_double_dqn = use_double_dqn
        self.use_dueling = use_dueling
        self.use_noisy = use_noisy
        self.use_per = use_per
        self.use_lstm = use_lstm

        # Device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Create networks
        if use_lstm:
            self.q_network = LSTMQNetwork(state_size, action_size, hidden_size).to(self.device)
            self.target_network = LSTMQNetwork(state_size, action_size, hidden_size).to(self.device)
        else:
            self.q_network = DuelingQNetwork(
                state_size, action_size, hidden_size,
                use_noisy=use_noisy, use_batch_norm=True
            ).to(self.device)
            self.target_network = DuelingQNetwork(
                state_size, action_size, hidden_size,
                use_noisy=use_noisy, use_batch_norm=True
            ).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        if use_per:
            self.memory = PrioritizedReplayBuffer(memory_size, alpha=alpha, beta=beta)
        else:
            from models.dqn import ReplayBuffer
            self.memory = ReplayBuffer(memory_size)

        # For LSTM
        self.hidden_state = None
        self.sequence_buffer = deque(maxlen=10)  # Store last 10 states for LSTM

        # Training step counter
        self.training_step = 0

    def get_action(self, state, training=True):
        """Select action using learned policy"""
        # For LSTM, use sequence
        if self.use_lstm:
            self.sequence_buffer.append(state)
            state_sequence = list(self.sequence_buffer)

            # Pad if necessary
            while len(state_sequence) < 10:
                state_sequence.insert(0, state_sequence[0])

            state_tensor = torch.FloatTensor(state_sequence).unsqueeze(0).to(self.device)

            self.q_network.eval()
            with torch.no_grad():
                if self.hidden_state is None:
                    self.hidden_state = self.q_network.init_hidden(1, self.device)
                q_values, self.hidden_state = self.q_network(state_tensor, self.hidden_state)
            self.q_network.train()

            return q_values.argmax().item()

        # For regular network
        if isinstance(state, tuple) or isinstance(state, list):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Reset noise for Noisy Networks
        if self.use_noisy and training:
            self.q_network.reset_noise()

        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        self.q_network.train()

        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        """Train on batch from replay buffer"""
        if len(self.memory) < self.batch_size:
            return 0

        # Sample batch
        if self.use_per:
            states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
            weights_tensor = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            weights_tensor = torch.ones(self.batch_size, 1).to(self.device)

        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Get current Q-values
        if self.use_lstm:
            current_q_values, _ = self.q_network(states_tensor)
        else:
            current_q_values = self.q_network(states_tensor)
        current_q_values = current_q_values.gather(1, actions_tensor)

        # Get target Q-values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use Q-network to select action, target network to evaluate
                if self.use_lstm:
                    next_q_values_select, _ = self.q_network(next_states_tensor)
                    next_actions = next_q_values_select.argmax(1, keepdim=True)
                    next_q_values_eval, _ = self.target_network(next_states_tensor)
                    next_q_values = next_q_values_eval.gather(1, next_actions)
                else:
                    next_q_values_select = self.q_network(next_states_tensor)
                    next_actions = next_q_values_select.argmax(1, keepdim=True)
                    next_q_values_eval = self.target_network(next_states_tensor)
                    next_q_values = next_q_values_eval.gather(1, next_actions)
            else:
                # Standard DQN
                if self.use_lstm:
                    next_q_values, _ = self.target_network(next_states_tensor)
                else:
                    next_q_values = self.target_network(next_states_tensor)
                next_q_values = next_q_values.max(1)[0].unsqueeze(1)

            target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values

        # Compute loss with importance sampling weights
        td_errors = target_q_values - current_q_values
        loss = (weights_tensor * td_errors.pow(2)).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)  # Gradient clipping
        self.optimizer.step()

        # Update priorities for PER
        if self.use_per:
            priorities = td_errors.abs().detach().cpu().numpy().flatten()
            self.memory.update_priorities(indices, priorities)

        # Soft update target network
        self.soft_update_target_network()

        self.training_step += 1

        return loss.item()

    def soft_update_target_network(self):
        """Soft update: θ_target = τ*θ_local + (1-τ)*θ_target"""
        for target_param, local_param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def update_target_network(self):
        """Hard update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'config': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'hidden_size': self.hidden_size,
                'use_double_dqn': self.use_double_dqn,
                'use_dueling': self.use_dueling,
                'use_noisy': self.use_noisy,
                'use_per': self.use_per,
                'use_lstm': self.use_lstm
            }
        }, filepath)

    def load_model(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)

        # Load configuration if available and recreate networks with correct architecture
        if 'config' in checkpoint:
            config = checkpoint['config']
            # Use hidden_size from checkpoint, or fall back to self.hidden_size
            checkpoint_hidden_size = config.get('hidden_size', self.hidden_size)

            # Recreate networks with the saved configuration
            if config.get('use_lstm', False):
                self.q_network = LSTMQNetwork(
                    config['state_size'],
                    config['action_size'],
                    checkpoint_hidden_size
                ).to(self.device)
                self.target_network = LSTMQNetwork(
                    config['state_size'],
                    config['action_size'],
                    checkpoint_hidden_size
                ).to(self.device)
            else:
                self.q_network = DuelingQNetwork(
                    config['state_size'],
                    config['action_size'],
                    checkpoint_hidden_size,
                    use_noisy=config.get('use_noisy', True),
                    use_batch_norm=True
                ).to(self.device)
                self.target_network = DuelingQNetwork(
                    config['state_size'],
                    config['action_size'],
                    checkpoint_hidden_size,
                    use_noisy=config.get('use_noisy', True),
                    use_batch_norm=True
                ).to(self.device)

            # Update agent config
            self.use_noisy = config.get('use_noisy', True)
            self.use_double_dqn = config.get('use_double_dqn', True)
            self.use_dueling = config.get('use_dueling', True)
            self.use_lstm = config.get('use_lstm', False)

            # Recreate optimizer with new network parameters
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)
