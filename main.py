"""Main script for dynamic routing simulation."""

import argparse
import time
import random
import numpy as np
import networkx as nx
from network_simulation import NetworkSimulator
from routing_algorithms.shortest_path import ShortestPathRouter
from reinforcement_learning.q_learning import QLearningRouter
from reinforcement_learning.deep_q_learning import DeepQLearningRouter
from visualization.visualizer import Visualizer

def main(args=None):
    """Run the dynamic routing simulation.
    Args:
        args: Optional list of command line arguments (for programmatic calls)
    """
    parser = argparse.ArgumentParser(description='Dynamic Routing Simulation')
    parser.add_argument('--nodes', type=int, default=10, help='Number of nodes in network')
    parser.add_argument('--packets', type=int, default=100, help='Number of packets to simulate')
    parser.add_argument('--algorithm', type=str, default='dijkstra', 
                       choices=['dijkstra', 'floyd', 'qlearning', 'deepq'], 
                       help='Routing algorithm to use')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    parser.add_argument('--max_steps', type=int, default=1000, 
                       help='Maximum simulation steps (default: 1000)')
    args = parser.parse_args(args)

    # Initialize components
    network = NetworkSimulator(args.nodes)
    visualizer = Visualizer() if args.visualize else None

    # Select routing algorithm
    if args.algorithm == 'dijkstra':
        router = ShortestPathRouter('dijkstra')
    elif args.algorithm == 'floyd':
        router = ShortestPathRouter('floyd')
    elif args.algorithm == 'qlearning':
        router = QLearningRouter()
    elif args.algorithm == 'deepq':
        router = DeepQLearningRouter()
    else:
        raise ValueError(f'Unknown algorithm: {args.algorithm}')

    # Run simulation
    start_time = time.time()
    metrics = network.run_simulation(args.packets, router, visualizer, args.max_steps)
    duration = time.time() - start_time

    # Print results
    print(f'Simulation completed in {duration:.2f} seconds')
    print(f'Average delivery time: {np.mean(metrics["delivery_times"]):.2f} seconds')
    print(f'Total packets delivered: {len(metrics["delivery_times"])}')

    if args.visualize:
        visualizer.close()

if __name__ == '__main__':
    main()
