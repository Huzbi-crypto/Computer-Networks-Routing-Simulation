"""Visualization module for dynamic routing simulation."""

import matplotlib
matplotlib.use('Qt5Agg')  # Use PyQt6 backend
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List
import numpy as np
from PyQt6 import QtWidgets

class Visualizer:
    """Handles visualization of network state and performance metrics."""
    
    def __init__(self):
        """Initialize visualizer with empty figures."""
        print("Initializing visualizer...")
        self.app = QtWidgets.QApplication.instance()
        if not self.app:
            print("Creating new QApplication instance")
            self.app = QtWidgets.QApplication([])
            
        print("Setting up matplotlib figures")
        plt.ion()  # Interactive mode on
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.canvas.manager.set_window_title('Network Simulation Visualizer')
        self.fig.canvas.manager.window.raise_()
        self.metric_history = {
            'delivery_times': [],
            'congestion': [],
            'dropped_packets': []
        }
        print("Visualizer initialized successfully")
        
    def update(self, graph: nx.Graph, packets: List[Dict], metrics: Dict, step: int):
        """Update visualization with current network state and metrics."""
        print(f"\n=== Visualizing step {step} ===")
        try:
            print("Updating network view...")
            self._update_network_view(graph, packets)
            print("Updating metrics view...")
            self._update_metrics_view(metrics)
            
            print("Rendering visualization...")
            plt.draw()
            print("Processing Qt events...")
            self.app.processEvents()
            plt.pause(0.1)
            print("Visualization update complete")
                
        except Exception as e:
            print(f"Visualization error: {str(e)}")
            try:
                print("Attempting to save fallback image...")
                self.fig.savefig(f'network_step_{step}.png')
                print(f"Saved fallback visualization to network_step_{step}.png")
            except Exception as save_error:
                print(f"Failed to save visualization: {str(save_error)}")
            self.close()
        
    def _update_network_view(self, graph: nx.Graph, packets: List[Dict]):
        """Draw network topology and packet routes."""
        self.ax1.clear()
        
        # Draw nodes
        pos = nx.spring_layout(graph)
        nx.draw_networkx_nodes(graph, pos, ax=self.ax1, node_size=300)
        
        # Draw active edges
        active_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['active']]
        nx.draw_networkx_edges(graph, pos, edgelist=active_edges, ax=self.ax1, 
                              edge_color='green', width=2)
        
        # Draw inactive edges
        inactive_edges = [(u, v) for u, v, d in graph.edges(data=True) if not d['active']]
        nx.draw_networkx_edges(graph, pos, edgelist=inactive_edges, ax=self.ax1, 
                              edge_color='red', width=1, style='dashed')
        
        # Draw packet paths
        for packet in packets:
            if len(packet['path']) > 1:
                path_edges = list(zip(packet['path'][:-1], packet['path'][1:]))
                nx.draw_networkx_edges(graph, pos, edgelist=path_edges, ax=self.ax1, 
                                      edge_color='blue', width=1.5, alpha=0.5)
        
        self.ax1.set_title('Network Topology')
        
    def _update_metrics_view(self, metrics: Dict):
        """Update performance metrics plots."""
        self.ax2.clear()
        
        # Update metric history
        for key in self.metric_history:
            if key in metrics:
                self.metric_history[key].append(np.mean(metrics[key]) if metrics[key] else 0)
        
        # Plot delivery times
        if self.metric_history['delivery_times']:
            self.ax2.plot(self.metric_history['delivery_times'], label='Avg Delivery Time')
        
        # Plot congestion
        if self.metric_history['congestion']:
            self.ax2.plot(self.metric_history['congestion'], label='Congestion')
            
        # Plot dropped packets
        if self.metric_history['dropped_packets']:
            self.ax2.plot(self.metric_history['dropped_packets'], label='Dropped Packets')
            
        self.ax2.set_title('Performance Metrics')
        self.ax2.set_xlabel('Time Step')
        self.ax2.legend()
        self.ax2.grid(True)
        
    def close(self):
        """Close visualization windows."""
        try:
            print("Saving final visualization screenshot...")
            self.fig.savefig('network_final.png')
            print("Saved final visualization to network_final.png")
        except Exception as e:
            print(f"Error saving final screenshot: {str(e)}")
        plt.ioff()
        plt.close()
