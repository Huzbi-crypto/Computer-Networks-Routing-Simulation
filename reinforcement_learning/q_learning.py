"""Implementation of Q-learning routing algorithm."""

import random
import numpy as np
from typing import Dict, List, Tuple
import networkx as nx

class QLearningRouter:
    """Implements Q-learning algorithm for routing in dynamic networks."""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9, 
                 exploration_rate: float = 0.1):
        """Initialize Q-learning router with hyperparameters."""
        self.learning_rate = learning_rate
        self.discount_factor = 0.9
        self.exploration_rate = exploration_rate
        self.q_table = {}  # State-action value function
        self.visited_states = set()
        
    def get_next_node(self, graph: nx.Graph, current: int, target: int) -> int:
        """Get next node using Q-learning policy (epsilon-greedy)."""
        state = (current, target)
        possible_actions = list(graph.neighbors(current))
        
        if not possible_actions:
            return None
            
        # Exploration
        if random.random() < self.exploration_rate:
            return random.choice(possible_actions)
            
        # Exploitation
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in possible_actions}
            
        return max(self.q_table[state].items(), key=lambda x: x[1])[0]
        
    def update_q_value(self, state: Tuple[int, int], action: int, reward: float, 
                      next_state: Tuple[int, int]):
        """Update Q-value using Bellman equation."""
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0
            
        max_next_q = max(self.q_table.get(next_state, {}).values(), default=0)
        current_q = self.q_table[state][action]
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
    def get_reward(self, current: int, next_node: int, target: int, 
                  graph: nx.Graph) -> float:
        """Calculate reward for taking action in current state."""
        if next_node == target:
            return 1.0  # Positive reward for reaching target
        if not graph.has_edge(current, next_node):
            return -1.0  # Negative reward for invalid action
            
        # Reward based on inverse of distance
        distance = graph.edges[current, next_node]['weight']
        return 1.0 / (distance + 1e-6)
