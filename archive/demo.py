"""
Quick demonstration of the Temporal Memory Graph vs Baseline
"""
from temporal_memory_graph import TemporalMemoryGraph
from baseline_memory_graph import BaselineMemoryGraph
from datetime import datetime, timedelta
import time


def demo():
    print("="*60)
    print("TEMPORAL MEMORY GRAPH - LIVE DEMONSTRATION")
    print("="*60)
    
    # Initialize both systems
    print("\\n Initializing systems...")
    baseline = BaselineMemoryGraph()
    temporal = TemporalMemoryGraph(decay_rate=0.1, strengthening_delta=0.3)
    current_time = datetime.now()
    
    # Phase 1: Add initial facts
    print("\\n Adding initial facts about Alice...")
    baseline.add_fact("Alice", "lives_in", "New York")
    baseline.add_fact("Alice", "works_at", "Google")
    baseline.add_fact("Alice", "knows", "Python")
    
    temporal.add_fact("Alice", "lives_in", "New York", time=current_time)
    temporal.add_fact("Alice", "works_at", "Google", time=current_time)
    temporal.add_fact("Alice", "knows", "Python", time=current_time)
    
    print(" Initial facts added to both systems")
    
    # Phase 2: Show initial query
    print("\\n Initial query: 'Alice'")
    baseline_result = baseline.query("Alice", k=5)
    temporal_result = temporal.query("Alice", k=5, time=current_time)
    
    print(f"   Baseline: {[(r['relation'], r['object']) for r in baseline_result]}")
    print(f"   Temporal: {[(r['relation'], r['object'], '{:.2f}'.format(r['weight'])) for r in temporal_result]}")
    
    # Phase 3: Alice moves
    print("\\n Alice moves to Boston (Day 10)")
    current_time += timedelta(days=10)
    baseline.add_fact("Alice", "lives_in", "Boston")
    temporal.add_fact("Alice", "lives_in", "Boston", time=current_time)
    
    # Query immediately
    baseline_result = baseline.query("Alice", k=5)
    temporal_result = temporal.query("Alice", k=5, time=current_time)
    
    print(f"   Baseline: {[(r['relation'], r['object']) for r in baseline_result]}")
    print(f"   Temporal: {[(r['relation'], r['object'], '{:.2f}'.format(r['weight'])) for r in temporal_result]}")
    
    # Phase 4: Demonstrate weight strengthening
    print("\\n Querying 'Alice' skill multiple times...")
    for i in range(5):
        current_time += timedelta(hours=1)
        temporal_result = temporal.query("Alice", k=5, time=current_time)
        python_weight = next(
            (r['weight'] for r in temporal_result if r['object'] == 'Python'), 
            0
        )
        print(f"   Query {i+1}: Python weight = {python_weight:.3f}")
    
    # Phase 5: Demonstrate decay
    print("\\n No queries for 30 days...")
    current_time += timedelta(days=30)
    temporal_result = temporal.query("Alice", k=5, time=current_time)
    
    print(f"   After decay: {[(r['relation'], r['object'], '{:.3f}'.format(r['weight'])) for r in temporal_result]}")
    
    # Phase 6: Show statistics
    print("\\nFinal Statistics:")
    print(f"   Temporal Graph: {temporal.get_statistics()}")
    
    print("\\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)


if __name__ == "__main__":
    demo()
