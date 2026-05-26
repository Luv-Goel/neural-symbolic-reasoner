"""
A more complete example script demonstrating the full neural-symbolic reasoning system.
This script shows how to use the hybrid system with sample data and visualizes the process.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.hybrid_reasoner import NeuralSymbolicReasoner

def run_demo():
    print("=== Neural-Symbolic Reasoner Demo ===")
    # Initialize the reasoner
    reasoner = NeuralSymbolicReasoner(feature_dim=16)
    
    # Create some sample data
    input_data = np.random.rand(10, 5) # 10 samples, 5 features
    
    # Process the data
    results = reasoner.process_and_reason(input_data)
    
    print(f"Input: {input_data.shape[0]} samples with {input_data.shape[1]} features each")
    print(f"Results: {results}")
    
    # Keep running until we get a star or determine more features are needed
    return results

if __name__ == "__main__":
    result = run_demo()
    print(result)