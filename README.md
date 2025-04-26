# Dynamic Routing Simulation

A Python project that simulates different routing algorithms on dynamic networks with visualization capabilities.

## Features

- Dynamic network simulation with changing topology
- Multiple routing algorithms:
  - Dijkstra's algorithm
  - Floyd-Warshall algorithm  
  - Q-learning reinforcement learning
  - Deep Q-learning neural networks
- Real-time visualization (PyQt6 backend)
- Performance metrics tracking
- Easy-to-use interface for all skill levels

## Getting Started

### For Non-Technical Users

1. **Install Python**:
   - Download from [python.org](https://www.python.org/downloads/)
   - Check "Add Python to PATH" during installation

2. **Run the Simulation**:
   - Windows: Double-click `run_simulation.bat`
   - Mac/Linux: Run `./run_simulation.sh` in terminal
   - Or simply: `python main.py`

### For Developers

```bash
# Setup virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run test simulation
python test_simulation.py

# Run custom simulation
python main.py --nodes 20 --packets 100 --algorithm qlearning --visualize
```

## Command Line Options

| Parameter      | Description                          | Default |
|----------------|--------------------------------------|---------|
| `--nodes`      | Number of nodes in network           | 10      |
| `--packets`    | Number of packets to simulate        | 100     |
| `--algorithm`  | Routing algorithm to use             | dijkstra |
| `--visualize`  | Enable real-time visualization       | Off     |
| `--max_steps`  | Maximum simulation steps             | 1000    |

Available algorithms: `dijkstra`, `floyd`, `qlearning`, `deepq`

## Output Files

- `network_final.png`: Final visualization screenshot
- `network_step_*.png`: Step-by-step screenshots (if visualization fails)

## Troubleshooting

1. **Visualization Issues**:
   - Try simpler test: `python test_simulation.py`
   - Check console for error messages
   - Screenshots are saved as fallback

2. **Missing Packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Need Help?**  
   Contact: [Your Email/Contact Info]

## Project Structure

```
├── network_simulation/    # Core simulation logic
├── routing_algorithms/    # Pathfinding implementations  
├── reinforcement_learning/ # RL algorithms
├── visualization/         # PyQt6 visualization
├── main.py                # Main entry point
├── test_simulation.py     # Test script
├── run_simulation.*       # Platform-specific run scripts
└── requirements.txt       # Dependencies
```

## License

MIT License
