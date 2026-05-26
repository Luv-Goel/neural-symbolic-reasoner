"""
Symbolic Reasoning Module

This module provides a framework for symbolic logic and reasoning, enabling:
- Knowledge representation
- Logical inference engines
- Rule-based systems
"""

from typing import List, Any, Dict, Tuple

class KnowledgeBase:
    """
    A simple in-memory knowledge base to store facts and rules.
    """
    def __init__(self):
        self.facts = set()
        self.rules = [] # List of tuples: (antecedent_conditions, consequent)

    def add_fact(self, fact: Any):
        """Adds a simple fact to the knowledge base."""
        self.facts.add(fact)

    def add_rule(self, antecedent: List[Any], consequent: Any):
        """Adds a rule to the knowledge base.
        Antecedent is a list of facts/conditions that must be true.
        Consequent is the fact that can be inferred.
        """
        self.rules.append((antecedent, consequent))

    def infer(self) -> List[Any]:
        """Performs forward chaining inference to derive new facts."""
        newly_inferred = []
        changed = True
        while changed:
            changed = False
            for antecedent, consequent in self.rules:
                # Check if all antecedent conditions are met by current facts
                all_conditions_met = True
                for condition in antecedent:
                    if condition not in self.facts:
                        all_conditions_met = False
                        break
                
                if all_conditions_met and consequent not in self.facts:
                    self.facts.add(consequent)
                    newly_inferred.append(consequent)
                    changed = True
            # Break if no new facts were inferred in this pass
            if not changed:
                break
        return newly_inferred

    def query(self, fact: Any) -> bool:
        """Checks if a fact exists in the knowledge base."""
        return fact in self.facts

class SymbolicReasoner:
    """
    A class to manage knowledge and perform inference.
    """
    def __init__(self):
        self.kb = KnowledgeBase()

    def add_knowledge(self, facts: List[Any] = None, rules: List[Tuple[List[Any], Any]] = None):
        """Adds facts and rules to the knowledge base."""
        if facts:
            for fact in facts:
                self.kb.add_fact(fact)
        if rules:
            for antecedent, consequent in rules:
                self.kb.add_rule(antecedent, consequent)

    def ask(self, fact: Any) -> bool:
        """Queries the knowledge base."""
        return self.kb.query(fact)

    def deduce(self) -> List[Any]:
        """Triggers inference and returns newly derived facts."""
        return self.kb.infer()

if __name__ == '__main__':
    # Example Usage:
    reasoner = SymbolicReasoner()
    reasoner.add_knowledge(
        facts=["is_mammal(Socrates)", "is_greek(Socrates)"],
        rules=[
            (["is_mammal(X)", "is_human(X)"], "is_mortal(X)"),
            (["is_greek(X)", "is_human(X)"], "likes_olives(X)")
        ]
    )
    
    # Add more facts to enable inference
    reasoner.add_knowledge(facts=["is_human(Socrates)"])
    
    # Deduce new facts
    new_facts = reasoner.deduce()
    print("Newly inferred facts:", new_facts)
    
    # Query knowledge base
    print("Does Socrates like olives?", reasoner.ask("likes_olives(Socrates)"))
    print("Is Socrates mortal?", reasoner.ask("is_mortal(Socrates)"))
    print("Does Socrates like feta?", reasoner.ask("likes_feta(Socrates)"))
