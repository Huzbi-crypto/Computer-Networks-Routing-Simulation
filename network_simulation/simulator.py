"""Network simulation core functionality."""

import random
import networkx as nx
from typing import List, Dict, Tuple
import time

class NetworkSimulator:
    """Simulates a dynamic network with changing topology and packet routing."""
    
    def __init__(self, num_nodes: int = 10):
        """Initialize network with given number of nodes."""
        self.num_nodes = num_nodes
        self.graph = self._initialize_network()
        self.packets = []
        self.metrics = {
            'delivery_times': [],
            'congestion': [],
            'dropped_packets': 0
        }
        
    def _initialize_network(self) -> nx.Graph:
        """Create initial random network topology."""
        graph = nx.erdos_renyi_graph(self.num_nodes, p=0.3)
        for u, v in graph.edges():
            graph.edges[u, v]['weight'] = random.uniform(0.1, 1.0)
            graph.edges[u, v]['active'] = True
        return graph
        
    def update_topology(self):
        """Randomly update network topology by toggling edges."""
        for u, v in self.graph.edges():
            if random.random() < 0.3:  # 30% chance to toggle edge (increased from 10%)
                self.graph.edges[u, v]['active'] = not self.graph.edges[u, v]['active']
            elif not self.graph.edges[u, v]['active'] and random.random() < 0.5:
                # 50% chance to reactivate inactive edges
                self.graph.edges[u, v]['active'] = True
                
    def generate_packets(self, count: int = 1):
        """Generate new packets with random source and destination."""
        for _ in range(count):
            src = random.randint(0, self.num_nodes - 1)
            dest = random.randint(0, self.num_nodes - 1)
            while dest == src:
                dest = random.randint(0, self.num_nodes - 1)
            self.packets.append({
                'source': src,
                'destination': dest,
                'path': [],
                'start_time': time.time(),
                'delivered': False
            })
            
    def run_simulation(self, num_packets: int, router, visualizer=None, max_steps=1000):
        """Run complete simulation with given parameters.
        
        Args:
            num_packets: Number of packets to simulate
            router: Routing algorithm to use
            visualizer: Optional visualization object
            max_steps: Maximum number of steps before stopping (default: 1000)
        """
        print(f"Starting simulation with {num_packets} packets (max {max_steps} steps)")
        self.generate_packets(num_packets)
        
        step = 0
        while any(not p['delivered'] for p in self.packets) and step < max_steps:
            step += 1
            self.update_topology()
            active_packets = [p for p in self.packets if not p['delivered']]
            
            if step % 10 == 0:
                delivered = len([p for p in self.packets if p['delivered']])
                print(f"Step {step}: {delivered}/{num_packets} packets delivered")
            
            for packet in active_packets:
                current_node = packet['source'] if not packet['path'] else packet['path'][-1]
                next_node = router.get_next_node(self.graph, current_node, packet['destination'])
                
                if next_node is not None:
                    packet['path'].append(next_node)
                    if next_node == packet['destination']:
                        packet['delivered'] = True
                        delivery_time = time.time() - packet['start_time']
                        self.metrics['delivery_times'].append(delivery_time)
            
            if visualizer:
                visualizer.update(self.graph, self.packets, self.metrics, step)
                
        if step >= max_steps:
            delivered = len([p for p in self.packets if p['delivered']])
            print(f"Simulation stopped after {max_steps} steps ({delivered}/{num_packets} packets delivered)")
        else:
            print("Simulation completed successfully")
        return self.metrics
