from datetime import datetime, timedelta
from collections import defaultdict
import copy

from memory.base_memory import MemoryItem
from memory.temporal_graph_memory import TemporalGraphMemory
from memory.vector_memory import VectorMemory
from memory.hybrid_graph_memory import HybridGraphMemory
from benchmark.datasets import load_tasks
from benchmark.evaluator import Evaluator

def run_experiment():
    print("="*60)
    print("RESEARCH EXPERIMENT RUNNER")
    print("Hypothesis: Temporal graph memory outperforms standard vector retrieval")
    print("in environments with evolving and conflicting user facts.")
    print("="*60)

    # Initialize memory systems
    # We will instantiate them fresh for each task to avoid state bleed
    system_classes = {
        "VectorMemory": lambda: VectorMemory(),
        "OldTemporalGraph": lambda: TemporalGraphMemory(decay_rate=0.01, strengthening_delta=0.2),
        "HybridGraphMemory": lambda: HybridGraphMemory(decay_rate=0.01, strengthening_delta=0.2, alpha_semantic=0.7, beta_temporal=0.3)
    }

    tasks = load_tasks()
    start_time = datetime(2025, 1, 1, 0, 0, 0)

    # Metrics storage: { system_name: { category: { metric: [scores] } } }
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    print(f"Loaded {len(tasks)} benchmark tasks.\n")

    for system_name, system_factory in system_classes.items():
        print(f"Evaluating {system_name}...")
        
        for task in tasks:
            # Instantiate a fresh system for the task
            system = system_factory()
            
            # Feed events
            for event in task.events:
                event_time = start_time + timedelta(hours=event.timestamp * 24) # interpreting timestamp as days
                
                # Update system time to process decay
                system.update(event_time)
                
                # Create and add MemoryItem
                mem_item = MemoryItem(
                    text=event.text,
                    timestamp=event_time,
                    metadata=event.metadata
                )
                system.add_memory(mem_item)
            
            # Retrieve
            # Let's say retrieval happens 1 day after the last event
            retrieval_time = start_time + timedelta(hours=(task.events[-1].timestamp + 1) * 24)
            system.update(retrieval_time)
            retrieved = system.retrieve(task.query, k=5)
            
            # Evaluate
            r_at_k = Evaluator.recall_at_k(retrieved, task.ground_truth, k=5)
            mrr = Evaluator.mrr(retrieved, task.ground_truth)
            contradiction = Evaluator.contradiction_accuracy(retrieved, task.ground_truth)
            
            # Save metrics
            results[system_name][task.category]["Recall@5"].append(r_at_k)
            results[system_name][task.category]["MRR"].append(mrr)
            results[system_name][task.category]["Contradiction_Acc"].append(contradiction)
            
            # Memory Efficiency
            # A simple metric: number of edges (graph) vs size of memory (vector)
            if "Graph" in system_name:
                final_size = system.graph.number_of_edges()
            else:
                final_size = len(system.memories)
            results[system_name][task.category]["Memory_Size"].append(final_size)

    # Print Report
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # We will compute the average of each metric per category per system
    metrics_to_report = ["Recall@5", "MRR", "Contradiction_Acc", "Memory_Size"]
    
    categories = sorted(list(set(task.category for task in tasks)))
    
    for category in categories:
        print(f"\n[ Category: {category.upper()} ]")
        # Print header
        header = f"{'System':<25}" + "".join([f"{m:>18}" for m in metrics_to_report])
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        
        for system_name in system_classes.keys():
            row_str = f"{system_name:<25}"
            for metric in metrics_to_report:
                scores = results[system_name][category][metric]
                avg = sum(scores) / len(scores) if scores else 0
                if metric == "Memory_Size":
                    row_str += f"{avg:>18.1f}"
                else:
                    row_str += f"{avg:>18.2f}"
            print(row_str)

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_experiment()
