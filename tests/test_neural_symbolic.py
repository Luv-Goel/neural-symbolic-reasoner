"""
Unit tests for the Neural-Symbolic Reasoner project.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.neural_symbolic.feature_extractor import FeatureExtractor, build_feature_extractor_model
from src.reasoning.symbolic_reasoner import KnowledgeBase, SymbolicReasoner
from src.utils.hybrid_reasoner import NeuralSymbolicReasoner

def test_feature_extractor():
    """Tests the FeatureExtractor layer."""
    print("Testing FeatureExtractor...")
    layer = FeatureExtractor(embedding_dim=16)
    input_data = np.random.rand(5, 10) # batch_size=5, input_dim=10
    
    # Build the layer
    layer.build(input_data.shape)
    
    # Check that weights are created
    assert hasattr(layer, 'dense1')
    assert hasattr(layer, 'dense2')
    
    # Forward pass
    output = layer(input_data)
    assert output.shape == (5, 16) # batch_size, embedding_dim
    print("  FeatureExtractor test passed.")

def test_build_feature_extractor_model():
    """Tests the model building function."""
    print("Testing build_feature_extractor_model...")
    input_shape = (20,)
    model = build_feature_extractor_model(input_shape, embedding_dim=32)
    
    assert model.input_shape == (None, 20)
    assert model.output_shape == (None, 32)
    print("  build_feature_extractor_model test passed.")

def test_knowledge_base():
    """Tests the KnowledgeBase class."""
    print("Testing KnowledgeBase...")
    kb = KnowledgeBase()
    
    # Add facts
    kb.add_fact("fact_a")
    kb.add_fact("fact_b")
    assert kb.query("fact_a") == True
    assert kb.query("fact_c") == False
    
    # Add a rule: if fact_a and fact_b are true, then fact_c is true
    kb.add_rule(["fact_a", "fact_b"], "fact_c")
    
    # Initially, fact_c should not be inferred
    assert kb.query("fact_c") == False
    
    # Perform inference
    inferred = kb.infer()
    assert "fact_c" in inferred
    assert kb.query("fact_c") == True
    print("  KnowledgeBase test passed.")

def test_symbolic_reasoner():
    """Tests the SymbolicReasoner class."""
    print("Testing SymbolicReasoner...")
    reasoner = SymbolicReasoner()
    
    # Add knowledge
    reasoner.add_knowledge(
        facts=["is_bird(Tweety)", "has_feathers(Tweety)"],
        rules=[
            (["is_bird(X)"], "can_fly(X)"), # Simple rule
            (["has_feathers(X)", "is_bird(X)"], "is_vertebrate(X)")
        ]
    )
    
    # Add a fact to trigger the rules
    reasoner.add_knowledge(facts=["is_bird(Tweety)"])
    
    # Deduce
    new_facts = reasoner.deduce()
    assert "can_fly(Tweety)" in new_facts
    assert "is_vertebrate(Tweety)" in new_facts
    
    # Query
    assert reasoner.ask("can_fly(Tweety)") == True
    assert reasoner.ask("is_vertebrate(Tweety)") == True
    print("  SymbolicReasoner test passed.")

def test_neural_symbolic_reasoner():
    """Tests the hybrid NeuralSymbolicReasoner."""
    print("Testing NeuralSymbolicReasoner...")
    ns_reasoner = NeuralSymbolicReasoner(feature_dim=8)
    
    # Create dummy input data
    input_data = np.random.rand(4, 6) # 4 samples, 6 features
    
    # Process and reason
    results = ns_reasoner.process_and_reason(input_data)
    
    # Check output structure
    assert "extracted_features" in results
    assert "inferred_facts" in results
    assert "all_facts" in results
    assert results["extracted_features"].shape == (4, 8)
    
    # Check that some reasoning occurred
    # The hybrid reasoner adds facts based on the mean of the first feature
    # So either "high_feature_activity" or "low_feature_activity" should be present
    all_facts = results["all_facts"]
    assert "high_feature_activity" in all_facts or "low_feature_activity" in all_facts
    print("  NeuralSymbolicReasoner test passed.")

if __name__ == '__main__':
    test_feature_extractor()
    test_build_feature_extractor_model()
    test_knowledge_base()
    test_symbolic_reasoner()
    test_neural_symbolic_reasoner()
    print("\nAll tests passed!")