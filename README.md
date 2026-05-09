# Temporal Edge Weighting Memory Graph

This project implements a Memory Graph with **Temporal Edge Weighting** designed to improve context retrieval over time compared to a standard, static baseline memory graph. It simulates how memory strengthens with frequent access and decays over time when unused.

## Key Features

- **Temporal Decay**: Facts that are not accessed for a long period gradually lose their weight.
- **Access Strengthening**: Frequently queried or retrieved facts have their edge weights strengthened.
- **Automatic Forgetting**: Edges that drop below a defined forgetting threshold are automatically pruned from the graph.
- **Dynamic Adaptability**: The system natively adapts to changing facts (e.g., when a person's location changes) by prioritizing the most recent and frequently reinforced information.

## Core Files

- `temporal_memory_graph.py`: Contains the `TemporalMemoryGraph` implementation, utilizing `networkx` to build a time-aware memory graph.
- `baseline_memory_graph.py`: Contains the `BaselineMemoryGraph`, a static graph used for comparing against the temporal implementation.
- `demo.py`: A quick, step-by-step interactive demonstration of the temporal dynamics (adding facts, querying, decaying, and strengthening).
- `experiment.py`: A comprehensive simulation that tests both graphs across three scenarios:
  1. **Changing Facts**: How well the system remembers updated data (e.g., Alice moves to different cities).
  2. **Frequency Pattern**: How frequently accessed facts get higher retrieval priority.
  3. **Forgetting Mechanism**: Ensuring unused facts are properly pruned from memory.
- `requirements.txt`: Python dependencies.

## Installation

Ensure you have Python 3.8+ installed, then run:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Run the Live Demo
To see a quick demonstration of facts being added, queried, and decayed:

```bash
python demo.py
```

### 2. Run the Full Experiment
To run the automated test scenarios and generate visualization charts:

```bash
python experiment.py
```

This will run all three scenarios and produce:
- `experiment_results.png`: Visual charts comparing the Baseline vs Temporal graph accuracy, weight evolution, and forgetting edge counts.
- `experiment_data.json`: The raw experiment data.

## Architecture & Concepts

The `TemporalMemoryGraph` is built on top of `networkx.MultiDiGraph`.
- **`decay_rate`**: Determines how fast memory weights decay per hour.
- **`strengthening_delta`**: Determines how much a fact's weight increases upon successful retrieval.
- **`forgetting_threshold`**: The minimum weight boundary. If an edge's weight drops below this value, the fact is "forgotten" and the edge is removed from the graph.

This setup is highly beneficial for Agentic AI and LLM memory pipelines, ensuring the context window is populated with the most relevant, recent, and frequently used information rather than being flooded with outdated facts.
