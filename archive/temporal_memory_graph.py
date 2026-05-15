import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
import json
from dataclasses import dataclass, field
from collections import defaultdict
import math


@dataclass
class TemporalEdge:
    """Represents a weighted, temporal edge in the memory graph"""
    source: str
    target: str
    relation: str
    weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    
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


class TemporalMemoryGraph:
    """
    A memory graph with temporal edge weighting.
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
        """
        Args:
            decay_rate: Exponential decay rate per hour
            strengthening_delta: How much weight increases on access
            forgetting_threshold: Edges below this weight are pruned
            max_weight: Maximum possible edge weight
            min_weight: Minimum weight after decay (before pruning)
        """
        self.graph = nx.MultiDiGraph()
        self.decay_rate = decay_rate
        self.strengthening_delta = strengthening_delta
        self.forgetting_threshold = forgetting_threshold
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.current_time = datetime.now()
        
    def add_fact(self, subject: str, relation: str, obj: str, weight: float = 1.0, time: Optional[datetime] = None):
        """Add a new fact as a subject-relation-object triple"""
        if time:
            self.current_time = time
        
        # Add nodes if they don't exist
        if subject not in self.graph:
            self.graph.add_node(subject, type='entity')
        if obj not in self.graph:
            self.graph.add_node(obj, type='entity')
            
        # Check if this exact edge already exists
        existing_edge = self._find_edge(subject, obj, relation)
        if existing_edge is not None:
            # Update existing edge
            edge_data = self.graph[subject][obj][existing_edge]
            edge_data['temporal'].weight = max(
                edge_data['temporal'].weight, weight
            )
            edge_data['temporal'].last_accessed = self.current_time
        else:
            # Add new edge
            temporal_edge = TemporalEdge(
                source=subject,
                target=obj,
                relation=relation,
                weight=weight,
                created_at=self.current_time,
                last_accessed=self.current_time
            )
            self.graph.add_edge(
                subject, obj, 
                key=relation,
                temporal=temporal_edge
            )
    
    def _find_edge(self, source: str, target: str, relation: str) -> Optional[str]:
        """Find an existing edge key between two nodes"""
        if source in self.graph and target in self.graph[source]:
            for key in self.graph[source][target]:
                if key == relation:
                    return key
        return None
    
    def query(self, entity: str, k: int = 5, time: Optional[datetime] = None) -> List[Dict]:
        """
        Retrieve the most relevant facts about an entity.
        Returns top-k facts weighted by temporal importance.
        
        Args:
            entity: The entity to query about
            k: Number of facts to return
            time: Current time (for decay calculation)
        """
        if time:
            self.current_time = time
        
        # Apply decay to all edges first
        self._apply_decay()
        
        # Collect all connected facts
        facts = []
        
        # Outgoing edges
        if entity in self.graph:
            for target, edges in self.graph[entity].items():
                for relation, edge_data in edges.items():
                    temporal = edge_data['temporal']
                    facts.append({
                        'subject': entity,
                        'relation': relation,
                        'object': target,
                        'weight': temporal.weight,
                        'access_count': temporal.access_count,
                        'last_accessed': temporal.last_accessed,
                        'direction': 'outgoing'
                    })
        
        # Incoming edges
        for source, target, edges in self.graph.in_edges(entity, data=True):
            if source != entity:  # Avoid double counting self-loops
                temporal = edges['temporal']
                facts.append({
                    'subject': source,
                    'relation': edges.get('relation', 'related_to'),
                    'object': entity,
                    'weight': temporal.weight,
                    'access_count': temporal.access_count,
                    'last_accessed': temporal.last_accessed,
                    'direction': 'incoming'
                })
        
        # Sort by weight and return top-k
        facts.sort(key=lambda x: x['weight'], reverse=True)
        top_facts = facts[:k]
        
        # Strengthen accessed edges
        for fact in top_facts:
            self._strengthen_edge(
                fact['subject'], 
                fact['object'], 
                fact['relation']
            )
        
        return top_facts
    
    def _apply_decay(self):
        """Apply exponential decay to all edges based on time since last access"""
        current = self.current_time
        
        edges_to_remove = []
        
        for source, target, key, data in self.graph.edges(data=True, keys=True):
            temporal = data['temporal']
            hours_since_access = (current - temporal.last_accessed).total_seconds() / 3600
            
            # Exponential decay
            decay_factor = math.exp(-self.decay_rate * hours_since_access)
            temporal.weight *= decay_factor
            temporal.weight = max(temporal.weight, self.min_weight)
            
            # Mark for removal if below threshold
            if temporal.weight < self.forgetting_threshold:
                edges_to_remove.append((source, target, key))
        
        # Remove forgotten edges
        for source, target, key in edges_to_remove:
            self.graph.remove_edge(source, target, key)
    
    def _strengthen_edge(self, source: str, target: str, relation: str):
        """Strengthen an edge that was successfully retrieved"""
        edge_key = self._find_edge(source, target, relation)
        if edge_key is not None:
            temporal = self.graph[source][target][edge_key]['temporal']
            # Strengthen with diminishing returns
            temporal.weight = min(
                temporal.weight + self.strengthening_delta * (1 - temporal.weight / self.max_weight),
                self.max_weight
            )
            temporal.access_count += 1
            temporal.last_accessed = self.current_time
    
    def get_statistics(self) -> Dict:
        """Get current statistics of the memory graph"""
        edges = []
        for source, target, key, data in self.graph.edges(data=True, keys=True):
            temporal = data['temporal']
            edges.append(temporal.weight)
        
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'mean_weight': np.mean(edges) if edges else 0,
            'std_weight': np.std(edges) if edges else 0,
            'min_weight': np.min(edges) if edges else 0,
            'max_weight': np.max(edges) if edges else 0,
            'forgotten_edges': [e for e in edges if e < self.forgetting_threshold]
        }
    
    def save(self, filepath: str):
        """Save the graph to a JSON file"""
        data = {
            'nodes': list(self.graph.nodes(data=True)),
            'edges': [],
            'config': {
                'decay_rate': self.decay_rate,
                'strengthening_delta': self.strengthening_delta,
                'forgetting_threshold': self.forgetting_threshold,
                'max_weight': self.max_weight,
                'min_weight': self.min_weight
            }
        }
        
        for source, target, key, edge_data in self.graph.edges(data=True, keys=True):
            temporal = edge_data['temporal']
            data['edges'].append({
                'source': source,
                'target': target,
                'key': key,
                'temporal': temporal.to_dict()
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'TemporalMemoryGraph':
        """Load a graph from a JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        config = data['config']
        graph = cls(**config)
        
        for node, node_data in data['nodes']:
            graph.graph.add_node(node, **node_data)
        
        for edge in data['edges']:
            temporal_data = edge['temporal']
            temporal_edge = TemporalEdge(
                source=temporal_data['source'],
                target=temporal_data['target'],
                relation=temporal_data['relation'],
                weight=temporal_data['weight'],
                created_at=datetime.fromisoformat(temporal_data['created_at']),
                last_accessed=datetime.fromisoformat(temporal_data['last_accessed']),
                access_count=temporal_data['access_count']
            )
            graph.graph.add_edge(
                edge['source'],
                edge['target'],
                key=edge['key'],
                temporal=temporal_edge
            )
        
        return graph
