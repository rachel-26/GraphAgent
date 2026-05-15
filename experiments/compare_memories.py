from datetime import datetime, timedelta
import numpy as np

from memory import BaseMemory, MemoryItem, TemporalGraphMemory, VectorMemory

def run_experiment():
    print("="*60)
    print("RESEARCH FRAMEWORK: Memory System Comparison")
    print("="*60)

    # Initialize memory systems
    systems = {
        "Vector Memory": VectorMemory(),
        "Temporal Graph Memory": TemporalGraphMemory(decay_rate=0.01, strengthening_delta=0.2)
    }

    # Setup timeline
    start_time = datetime.now()
    
    # Dataset
    memories_to_add = [
        MemoryItem(
            text="Alice lives in New York",
            timestamp=start_time,
            metadata={"subject": "Alice", "relation": "lives_in", "object": "New York"}
        ),
        MemoryItem(
            text="Bob knows Python",
            timestamp=start_time,
            metadata={"subject": "Bob", "relation": "knows", "object": "Python"}
        ),
        MemoryItem(
            text="Bob works at TechCorp",
            timestamp=start_time,
            metadata={"subject": "Bob", "relation": "works_at", "object": "TechCorp"}
        )
    ]

    print("\n[1] Adding initial memories to all systems...")
    for name, system in systems.items():
        for memory in memories_to_add:
            system.add_memory(memory)
    print("Memories successfully added.")

    # Time passes
    current_time = start_time + timedelta(days=5)
    
    # Alice moves
    new_memory = MemoryItem(
        text="Alice lives in Boston",
        timestamp=current_time,
        metadata={"subject": "Alice", "relation": "lives_in", "object": "Boston"}
    )
    
    print(f"\n[2] Time advances to Day 5. New event: {new_memory.text}")
    for name, system in systems.items():
        system.update(current_time)
        system.add_memory(new_memory)

    # Time passes again
    current_time = start_time + timedelta(days=10)
    
    print("\n[3] Time advances to Day 10. Querying 'Alice lives_in'...")
    query = "Where does Alice live? (Alice lives_in)"
    
    print("\n--- Retrieval Results ---")
    for name, system in systems.items():
        system.update(current_time)
        results = system.retrieve(query, k=2)
        
        print(f"\nSystem: {name}")
        for i, res in enumerate(results):
            mem = res["memory"]
            score = res["score"]
            print(f"  {i+1}. [{score:.3f}] {mem.text} (Time: {mem.timestamp.strftime('%Y-%m-%d %H:%M')})")

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_experiment()
