import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random


class ReplayBuffer:
    """Experience replay buffer for storing and sampling experiences."""
    
    def __init__(self, capacity=10000):
        """Initialize a replay buffer with fixed capacity.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences randomly.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)


class QNetwork(nn.Module):
    """Neural network for approximating Q-function."""
    
    def __init__(self, state_size, action_size, hidden_size=400):
        """Initialize the Q-network.
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            hidden_size: Size of hidden layer
        """
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, state):
        """Forward pass through the network.
        
        Args:
            state: State tensor
            
        Returns:
            Q-values for each action
        """
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """Deep Q-Network agent for traffic light control."""
    
    def __init__(self, 
                 state_size=3,  # (X1, X2, L)
                 action_size=2,  # 0: Continue, 1: Switch
                 hidden_size=400,
                 learning_rate=0.001,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 memory_size=10000,
                 batch_size=64,
                 device=None):
        """Initialize the DQN agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            hidden_size: Size of hidden layer
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Rate of exploration decay
            memory_size: Size of replay buffer
            batch_size: Batch size for training
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Set device (CPU or GPU)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create Q-network and target network
        self.q_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.update_target_network()
        
        # Create optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Create replay buffer
        self.memory = ReplayBuffer(memory_size)
    
    def update_target_network(self):
        """Copy weights from Q-network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_action(self, state, training=True):
        """Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether the agent is in training mode
            
        Returns:
            Selected action
        """
        # Exploration: select random action
        if training and np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        
        # Exploitation: select best action
        # Convert state to tensor
        if isinstance(state, tuple):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        else:
            state_tensor = torch.FloatTensor(state).to(self.device)
            
        # Set network to evaluation mode
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        # Set network back to training mode
        self.q_network.train()
        
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """Train the model using experiences from replay buffer.
        
        Returns:
            Loss value
        """
        if len(self.memory) < self.batch_size:
            return 0  # Not enough samples for training
        
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Get current Q-values
        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor)
        
        # Get target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor).max(1)[0].unsqueeze(1)
            target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        # self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save_model(self, filepath):
        """Save the Q-network model.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load_model(self, filepath):
        """Load a saved Q-network model.
        
        Args:
            filepath: Path to the saved model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']


class FixedCycleAgent:
    """Fixed-cycle traffic light control policy."""
    
    def __init__(self, cycle_length=20):
        """Initialize the fixed-cycle agent.
        
        Args:
            cycle_length: Number of steps in a complete cycle
        """
        self.cycle_length = cycle_length
        self.current_step = 0
    
    def get_action(self, state, training=False):
        """Return action based on fixed cycle.
        
        Args:
            state: Current state (ignored)
            training: Whether the agent is in training mode
            
        Returns:
            Action to take (0: continue, 1: switch)
        """
        # Switch the light every cycle_length/2 steps
        self.current_step = (self.current_step + 1) % self.cycle_length
        
        if self.current_step == 0 or self.current_step == self.cycle_length // 2:
            return 1  # Switch
        else:
            return 0  # Continue
    
    def remember(self, *args):
        """Placeholder to match DQNAgent interface."""
        pass
    
    def replay(self):
        """Placeholder to match DQNAgent interface."""
        return 0
    
    def update_target_network(self):
        """Placeholder to match DQNAgent interface."""
        pass


class OptimalThresholdAgent:
    """Optimal threshold policy for single intersection control.
    
    The optimal policy is a thresholding policy:
    - If (X1 - X2) > threshold, give green to direction 1 (west-east)
    - If (X2 - X1) > threshold, give green to direction 2 (north-south)
    - Otherwise, maintain current phase
    """
    
    def __init__(self, threshold=5):
        """Initialize the optimal threshold agent.
        
        Args:
            threshold: Queue difference threshold for switching
        """
        self.threshold = threshold
    
    def get_action(self, state, training=False):
        """Return action based on optimal threshold policy.
        
        Args:
            state: Current state (X1, X2, light_config)
            training: Whether the agent is in training mode
            
        Returns:
            Action to take (0: continue, 1: switch)
        """
        x1, x2, light_config = state
        
        # Light configuration: 0 (green for X1), 1 (yellow for X1), 
        #                      2 (green for X2), 3 (yellow for X2)
        
        if light_config == 0:  # Green for X1
            # Consider switching to yellow for X1 if X2 is much longer than X1
            if x2 - x1 > self.threshold:
                return 1  # Switch to yellow for X1
            else:
                return 0  # Continue green for X1
                
        elif light_config == 1:  # Yellow for X1
            # Must switch to green for X2
            return 1
            
        elif light_config == 2:  # Green for X2
            # Consider switching to yellow for X2 if X1 is much longer than X2
            if x1 - x2 > self.threshold:
                return 1  # Switch to yellow for X2
            else:
                return 0  # Continue green for X2
                
        elif light_config == 3:  # Yellow for X2
            # Must switch to green for X1
            return 1
    
    def remember(self, *args):
        """Placeholder to match DQNAgent interface."""
        pass
    
    def replay(self):
        """Placeholder to match DQNAgent interface."""
        return 0
    
    def update_target_network(self):
        """Placeholder to match DQNAgent interface."""
        pass
