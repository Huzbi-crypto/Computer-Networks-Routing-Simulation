"""Implementation of shortest path routing algorithms."""

import networkx as nx
from typing import Optional, List, Dict, Tuple
import numpy as np

class ShortestPathRouter:
    """Implements Dijkstra's and Floyd-Warshall algorithms for routing."""
    
    def __init__(self, method: str = "dijkstra"):
        """Initialize router with specified algorithm method."""
        self.method = method
        self.distance_matrix = None
        self.next_hop = None
        
    def get_next_node(self, graph: nx.Graph, current: int, target: int) -> Optional[int]:
        """Get next node in path from current to target using selected method."""
        if self.method == "dijkstra":
            return self._dijkstra_next_node(graph, current, target)
        elif self.method == "floyd":
            if self.distance_matrix is None:
                self._precompute_floyd_warshall(graph)
            return self._floyd_next_node(current, target)
        else:
            raise ValueError(f"Unknown routing method: {self.method}")
            
    def _dijkstra_next_node(self, graph: nx.Graph, current: int, target: int) -> Optional[int]:
        """Get next node using Dijkstra's algorithm."""
        try:
            path = nx.dijkstra_path(graph, current, target, weight='weight')
            return path[1] if len(path) > 1 else None
        except nx.NetworkXNoPath:
            return None
            
    def _precompute_floyd_warshall(self, graph: nx.Graph):
        """Precompute distance and next hop matrices for Floyd-Warshall."""
        nodes = sorted(graph.nodes())
        n = len(nodes)
        self.distance_matrix = np.full((n, n), np.inf)
        self.next_hop = np.zeros((n, n), dtype=int)
        
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if u == v:
                    self.distance_matrix[i][j] = 0
                    self.next_hop[i][j] = j
                elif graph.has_edge(u, v):
                    self.distance_matrix[i][
                        j] = graph.edges[u, v]['weight']
                    self.next_hop[i][j] = j
                    
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if self.distance_matrix[i][j] > self.distance_matrix[i][k] + self.distance_matrix[k][j]:
                        self.distance_matrix[i][j] = self.distance_matrix[i][k] + self.distance_matrix[k][j]
                        self.next_hop[i][j] = self.next_hop[i][k]
                        
    def _floyd_next_node(self, current: int, target: int) -> Optional[int]:
        """Get next node using precomputed Floyd-Warshall matrices."""
        if self.distance_matrix is None:
            raise RuntimeError("Floyd-Warshall matrices not computed")
        return self.next_hop[current][target] if self.next_hop[current][target] != target else None
