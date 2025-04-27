"""Implementation of Deep Q-learning routing algorithm."""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import List, Dict, Tuple
import networkx as nx

class DeepQLearningRouter:
    """Implements Deep Q-learning algorithm for routing in dynamic networks."""
    
    def __init__(self, graph: nx.Graph, action_size: int, learning_rate: float = 0.001,
                 discount_factor: float = 0.99, exploration_rate: float = 1.0,
                 exploration_decay: float = 0.999, batch_size: int = 32, memory_size: int = 1000):
        """Initialize Deep Q-learning router with hyperparameters."""
        self.state_size = max(len(list(graph.neighbors(node))) for node in graph.nodes) + 2
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_min = 0.01
        self.exploration_decay = exploration_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self) -> nn.Module:
        """Build neural network model for Q-value approximation."""
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model
        
    def update_target_model(self):
        """Update target model with weights from main model."""
        self.target_model.load_state_dict(self.model.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy."""
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()
        
    def replay(self):
        """Train model on batch of experiences from memory."""
        if len(self.memory) < self.batch_size:
            return
            
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in minibatch]))
        actions = torch.LongTensor(np.array([t[1] for t in minibatch]))
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch]))
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch]))
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch]))
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).detach().max(1)[0]
        target = rewards + (1 - dones) * self.discount_factor * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        self.exploration_rate = max(self.exploration_min, 
                                  self.exploration_rate * self.exploration_decay)
        
    def get_next_node(self, graph: nx.Graph, current: int, target: int) -> int:
        """Get next node using deep Q-learning policy."""
        state = self._get_state_representation(graph, current, target)
        action = self.act(state)
        neighbors = list(graph.neighbors(current))
        if action < len(neighbors):
            return neighbors[action]
        else:
            # Fallback: return a random neighbor if action is out of bounds
            return random.choice(neighbors) if neighbors else None
        
    def _get_state_representation(self, graph: nx.Graph, current: int, target: int) -> np.ndarray:
        """Convert graph state to neural network input vector."""
        neighbors = list(graph.neighbors(current))
        state = np.zeros(self.state_size)
        state[0] = current
        state[1] = target
        for i, neighbor in enumerate(neighbors[:self.state_size - 2]):
            state[2 + i] = graph.edges[current, neighbor]['weight']
        return state
