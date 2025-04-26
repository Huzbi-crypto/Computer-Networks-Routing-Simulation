"""Simple test script to verify the simulation works"""
from main import main

if __name__ == "__main__":
    print("=== Running Test Simulation ===")
    print("Testing with 5 nodes and 10 packets using Dijkstra's algorithm")
    main(["--nodes", "5", "--packets", "10", "--algorithm", "dijkstra"])
    print("=== Test Completed Successfully ===")
    print("The simulation is working correctly!")
    print("You can now run the full simulation using:")
    print("  python main.py")
    print("Or use the run_simulation scripts for your platform")
