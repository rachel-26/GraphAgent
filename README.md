# Memory Systems Research Framework

This project is an experimental research framework designed to rigorously benchmark and compare different memory architectures for AI agents. The current focus is evaluating **Temporal Graph Memory** against standard **Vector Memory** (RAG) paradigms.

The core hypothesis this framework tests is:
> *Temporal graph memory outperforms standard vector retrieval in environments with evolving and conflicting user facts.*

## Project Structure

The codebase is organized into a clean, reproducible research structure:

```text
GraphMemory/
├── memory/                   # Memory System Implementations
│   ├── base_memory.py        # Abstract interface (BaseMemory, MemoryItem)
│   ├── temporal_graph_memory.py # Graph-based memory with time-decay
│   └── vector_memory.py      # Standard embeddings-based memory baseline
│
├── benchmark/                # Evaluation Infrastructure
│   ├── datasets.py           # Synthetic ground-truth tasks
│   └── evaluator.py          # Objective evaluation metrics (Recall@K, MRR, etc.)
│
├── experiments/              # Experiment Runners
│   └── compare_memories.py   # Core benchmarking loop
│
└── main.py                   # Entry point
```

## Key Components

### 1. The Memory Interface
All memory systems inherit from `BaseMemory` and accept standardized `MemoryItem`s. This ensures tests are perfectly controlled: same tasks, same queries, same load, only the backend changes.

### 2. Benchmark Datasets
The `benchmark/datasets.py` module defines ground-truth tasks across critical memory categories:
- **Static Facts**: Basic, unchanging retrieval.
- **Conflicting Facts**: Testing contradiction resolution when newer truths overwrite older ones.
- **Long-Term Retention**: Testing memory survival amidst noise over long delays.
- **Reinforcement**: Testing if frequently accessed memories survive forgetting mechanisms.

### 3. Evaluation Metrics
We rely on rigorous, objective evaluation metrics (`benchmark/evaluator.py`):
- **Recall@K**: Is the correct answer in the top-K retrieved memories?
- **Mean Reciprocal Rank (MRR)**: How early does the correct memory appear?
- **Contradiction Accuracy**: Does the system retrieve the *newest* truth as the #1 rank when faced with conflicts?
- **Memory Efficiency**: What is the final storage footprint (edges vs. items) after the task completes?

## Getting Started

### Installation

Ensure you have Python 3.8+ installed, then run:

```bash
pip install -r requirements.txt
```

*Note: The `VectorMemory` utilizes `sentence-transformers`. If it fails to install on your environment, the system gracefully falls back to a mocked TF-IDF vectorizer so experiments can still run.*

### Running the Benchmark

To execute the core experiment loop and view the evaluation matrix:

```bash
python main.py
```

This will run all configured memory systems against the synthetic benchmark tasks, applying simulated time jumps, and output a detailed comparative matrix of the results.

## Future Directions
- Implementing LLM-based metadata extraction (parsing `MemoryItem.text` directly into graph nodes/edges).
- Scaling the benchmark dataset with more complex, real-world conversational datasets.
- Exploring hybrid retrieval methods combining embeddings with graph traversal.
