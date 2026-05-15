import networkx as nx
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any
import math

from .base_memory import BaseMemory, MemoryItem

class TemporalEdge:
    """Represents a weighted, temporal edge in the memory graph"""
    def __init__(self, source: str, target: str, relation: str, weight: float = 1.0, created_at: Optional[datetime] = None):
        self.source = source
        self.target = target
        self.relation = relation
        self.weight = weight
        self.created_at = created_at or datetime.now()
        self.last_accessed = self.created_at
        self.last_decayed = self.created_at
        self.access_count = 0
        self.memory_item = None # Can store a reference to the original MemoryItem if desired

    def to_dict(self):
        return {
            'source': self.source,
            'target': self.target,
            'relation': self.relation,
            'weight': self.weight,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count
        }

class TemporalGraphMemory(BaseMemory):
    """
    A memory graph with temporal edge weighting that conforms to the BaseMemory interface.
    Edges strengthen with use and decay over time.
    """
    
    def __init__(
        self, 
        decay_rate: float = 0.01,  # per hour
        strengthening_delta: float = 0.1,
        forgetting_threshold: float = 0.1,
        max_weight: float = 10.0,
        min_weight: float = 0.01
    ):
        self.graph = nx.MultiDiGraph()
        self.decay_rate = decay_rate
        self.strengthening_delta = strengthening_delta
        self.forgetting_threshold = forgetting_threshold
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.current_time = datetime.now()
        
    def add_memory(self, memory: MemoryItem):
        """Add a standardized memory item to the graph."""
        self.current_time = memory.timestamp
        
        # We expect metadata to hold the triplets for this basic graph implementation
        # e.g. {"subject": "Rachel", "relation": "likes", "object": "Python"}
        subject = memory.metadata.get("subject")
        relation = memory.metadata.get("relation")
        obj = memory.metadata.get("object")
        
        if not (subject and relation and obj):
            # If triplets aren't provided, we can't properly add to the graph without NLP extraction.
            # For now, we skip or could handle differently.
            return
            
        weight = memory.importance
        
        # Add nodes if they don't exist
        if subject not in self.graph:
            self.graph.add_node(subject, type='entity')
        if obj not in self.graph:
            self.graph.add_node(obj, type='entity')
            
        existing_edge = self._find_edge(subject, obj, relation)
        if existing_edge is not None:
            # Update existing edge
            edge_data = self.graph[subject][obj][existing_edge]
            edge_data['temporal'].weight = max(
                edge_data['temporal'].weight, weight
            )
            edge_data['temporal'].last_accessed = self.current_time
            edge_data['temporal'].memory_item = memory
        else:
            # Add new edge
            temporal_edge = TemporalEdge(
                source=subject,
                target=obj,
                relation=relation,
                weight=weight,
                created_at=self.current_time
            )
            temporal_edge.memory_item = memory
            
            self.graph.add_edge(
                subject, obj, 
                key=relation,
                temporal=temporal_edge
            )
            
    def _find_edge(self, source: str, target: str, relation: str) -> Optional[str]:
        if source in self.graph and target in self.graph[source]:
            for key in self.graph[source][target]:
                if key == relation:
                    return key
        return None
        
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve facts by extracting entities from the query using a naive substring match.
        """
        # Apply decay to all edges first using the current global time state
        self._apply_decay()
        
        # 1. Naive Entity Extraction: find nodes in graph that appear in the query
        query_lower = query.lower()
        matched_entities = []
        for node in self.graph.nodes():
            if str(node).lower() in query_lower:
                matched_entities.append(node)
                
        # 2. Collect facts around the matched entities
        facts = []
        visited_edges = set()
        
        for entity in matched_entities:
            # Outgoing
            if entity in self.graph:
                for target, edges in self.graph[entity].items():
                    for relation, edge_data in edges.items():
                        temporal = edge_data['temporal']
                        edge_id = (entity, target, relation)
                        if edge_id not in visited_edges:
                            facts.append(temporal)
                            visited_edges.add(edge_id)
            
            # Incoming
            for source, target, edges in self.graph.in_edges(entity, data=True):
                if source != entity:  # Avoid self loops
                    temporal = edges['temporal']
                    relation = edges.get('relation', temporal.relation) # key fallback
                    edge_id = (source, entity, relation)
                    if edge_id not in visited_edges:
                        facts.append(temporal)
                        visited_edges.add(edge_id)
                        
        # 3. Sort by weight and take top-k
        facts.sort(key=lambda x: x.weight, reverse=True)
        top_facts = facts[:k]
        
        # 4. Strengthen and Format
        results = []
        for temporal in top_facts:
            # Strengthen accessed edge
            self._strengthen_edge(temporal.source, temporal.target, temporal.relation)
            
            results.append({
                "memory": temporal.memory_item, # The standardized item
                "score": temporal.weight,
                "metadata": {
                    "source": temporal.source,
                    "relation": temporal.relation,
                    "target": temporal.target,
                    "access_count": temporal.access_count,
                    "last_accessed": temporal.last_accessed.isoformat()
                }
            })
            
        return results

    def update(self, current_time: datetime):
        """Standard update method to step time forward and trigger decay."""
        self.current_time = current_time
        self._apply_decay()
        
    def _apply_decay(self):
        """Apply exponential decay to all edges based on time since last decay"""
        current = self.current_time
        edges_to_remove = []
        
        for source, target, key, data in self.graph.edges(data=True, keys=True):
            temporal = data['temporal']
            
            # Use last_decayed instead of last_accessed to prevent exponential double-decay
            hours_since_decay = (current - temporal.last_decayed).total_seconds() / 3600
            if hours_since_decay <= 0:
                continue
                
            decay_factor = math.exp(-self.decay_rate * hours_since_decay)
            temporal.weight *= decay_factor
            temporal.weight = max(temporal.weight, self.min_weight)
            temporal.last_decayed = current
            
            if temporal.weight < self.forgetting_threshold:
                edges_to_remove.append((source, target, key))
        
        for source, target, key in edges_to_remove:
            self.graph.remove_edge(source, target, key)
            
    def _strengthen_edge(self, source: str, target: str, relation: str):
        """Strengthen an edge that was successfully retrieved"""
        edge_key = self._find_edge(source, target, relation)
        if edge_key is not None:
            temporal = self.graph[source][target][edge_key]['temporal']
            temporal.weight = min(
                temporal.weight + self.strengthening_delta * (1 - temporal.weight / self.max_weight),
                self.max_weight
            )
            temporal.access_count += 1
            temporal.last_accessed = self.current_time
            temporal.last_decayed = self.current_time
