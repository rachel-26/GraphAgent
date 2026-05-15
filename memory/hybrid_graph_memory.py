import networkx as nx
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any
import math
import logging

from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logging.warning("sentence_transformers not found. Using a mock TfIdf-based SentenceTransformer for demonstration.")
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    class SentenceTransformer:
        def __init__(self, model_name: str):
            self.vectorizer = TfidfVectorizer()
            self.is_fitted = False
            
        def encode(self, texts):
            is_string = isinstance(texts, str)
            if is_string:
                texts = [texts]
                
            if not self.is_fitted:
                self.vectorizer.fit(texts + ["Alice", "Bob", "New York", "Boston", "Chicago", "Python", "Rust", "Go", "TechCorp"])
                self.is_fitted = True
                
            arr = self.vectorizer.transform(texts).toarray()
            return arr[0] if is_string else arr

from .base_memory import BaseMemory, MemoryItem

class HybridEdge:
    """Represents a weighted, temporal, and semantic edge in the memory graph"""
    def __init__(self, source: str, target: str, relation: str, weight: float = 1.0, created_at: Optional[datetime] = None):
        self.source = source
        self.target = target
        self.relation = relation
        self.weight = weight
        self.created_at = created_at or datetime.now()
        self.last_accessed = self.created_at
        self.last_decayed = self.created_at
        self.access_count = 0
        self.memory_item = None

class HybridGraphMemory(BaseMemory):
    """
    A memory graph combining temporal decay/reinforcement with semantic embeddings.
    Implements contradiction resolution by actively penalizing stale conflicting facts.
    """
    
    def __init__(
        self, 
        model_name: str = 'all-MiniLM-L6-v2',
        decay_rate: float = 0.01,
        strengthening_delta: float = 0.2,
        forgetting_threshold: float = 0.1,
        max_weight: float = 10.0,
        min_weight: float = 0.01,
        alpha_semantic: float = 0.7,
        beta_temporal: float = 0.3
    ):
        self.graph = nx.MultiDiGraph()
        self.model = SentenceTransformer(model_name)
        self.decay_rate = decay_rate
        self.strengthening_delta = strengthening_delta
        self.forgetting_threshold = forgetting_threshold
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.alpha_semantic = alpha_semantic
        self.beta_temporal = beta_temporal
        self.current_time = datetime.now()
        
    def add_memory(self, memory: MemoryItem):
        self.current_time = memory.timestamp
        
        # Add embedding to MemoryItem if missing
        if memory.embedding is None:
            memory.embedding = self.model.encode(memory.text)
            
        subject = memory.metadata.get("subject")
        relation = memory.metadata.get("relation")
        obj = memory.metadata.get("object")
        
        if not (subject and relation and obj):
            return
            
        weight = memory.importance
        
        if subject not in self.graph:
            self.graph.add_node(subject, type='entity')
        if obj not in self.graph:
            self.graph.add_node(obj, type='entity')
            
        # Contradiction Resolution:
        # If the same subject and relation exist, penalize the old facts heavily
        if subject in self.graph:
            for target, edges in self.graph[subject].items():
                for key, edge_data in edges.items():
                    if key == relation and target != obj:
                        # Found a contradiction! Suppress the old edge heavily.
                        old_temporal = edge_data['temporal']
                        old_temporal.weight *= 0.1  # Severe decay penalty
                        old_temporal.last_decayed = self.current_time
                        
        existing_edge = self._find_edge(subject, obj, relation)
        if existing_edge is not None:
            edge_data = self.graph[subject][obj][existing_edge]
            edge_data['temporal'].weight = max(
                edge_data['temporal'].weight, weight
            )
            edge_data['temporal'].last_accessed = self.current_time
            edge_data['temporal'].memory_item = memory
        else:
            temporal_edge = HybridEdge(
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
        self._apply_decay()
        
        query_embedding = self.model.encode(query)
        
        # 1. Semantic + Temporal Retrieval
        # We will iterate through all edges since we don't have an inverted index 
        # (Fine for research prototype scale)
        facts = []
        
        for source, target, key, data in self.graph.edges(data=True, keys=True):
            temporal = data['temporal']
            mem_item = temporal.memory_item
            
            # Semantic Score
            semantic_score = float(cosine_similarity([query_embedding], [mem_item.embedding])[0][0])
            
            # Normalize temporal score slightly for the hybrid equation
            # (assuming max_weight is around 1-10, we cap it or log it, but for simplicity we just use it)
            # Let's bound the temporal score to roughly 0-1 for a fair mix
            normalized_temporal = min(temporal.weight / 1.0, 1.0) 
            
            # Hybrid Score
            hybrid_score = (self.alpha_semantic * semantic_score) + (self.beta_temporal * normalized_temporal)
            
            facts.append({
                "edge": temporal,
                "semantic_score": semantic_score,
                "temporal_score": normalized_temporal,
                "hybrid_score": hybrid_score
            })
            
        # 2. Sort by hybrid score
        facts.sort(key=lambda x: x["hybrid_score"], reverse=True)
        top_facts = facts[:k]
        
        # 3. Format and Strengthen
        results = []
        for fact in top_facts:
            temporal = fact["edge"]
            
            # Reinforcement only if semantic match is strong enough to avoid reinforcing random edges
            if fact["semantic_score"] > 0.5:
                self._strengthen_edge(temporal.source, temporal.target, temporal.relation)
            
            # Generate interpretability reason
            reason = "strong semantic match" if fact["semantic_score"] > 0.8 else "recently reinforced" if fact["temporal_score"] > 0.8 else "weak match"
            
            results.append({
                "memory": temporal.memory_item,
                "score": fact["hybrid_score"],
                "metadata": {
                    "source": temporal.source,
                    "relation": temporal.relation,
                    "target": temporal.target,
                    "semantic_score": round(fact["semantic_score"], 3),
                    "temporal_score": round(fact["temporal_score"], 3),
                    "reason": reason
                }
            })
            
        return results

    def update(self, current_time: datetime):
        self.current_time = current_time
        self._apply_decay()
        
    def _apply_decay(self):
        current = self.current_time
        edges_to_remove = []
        
        for source, target, key, data in self.graph.edges(data=True, keys=True):
            temporal = data['temporal']
            
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
