import networkx as nx
from typing import List, Dict, Optional
from datetime import datetime


class BaselineMemoryGraph:
    """
    A standard memory graph without temporal dynamics.
    All edges are treated equally during retrieval.
    """
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        
    def add_fact(self, subject: str, relation: str, obj: str):
        """Add a new fact as a subject-relation-object triple"""
        if subject not in self.graph:
            self.graph.add_node(subject)
        if obj not in self.graph:
            self.graph.add_node(obj)
            
        self.graph.add_edge(subject, obj, key=relation)
    
    def query(self, entity: str, k: int = 5) -> List[Dict]:
        """
        Retrieve facts about an entity.
        Returns all connected facts without weighting (first k alphabetical).
        """
        facts = []
        
        # Outgoing edges
        if entity in self.graph:
            for target in self.graph[entity]:
                for relation in self.graph[entity][target]:
                    facts.append({
                        'subject': entity,
                        'relation': relation,
                        'object': target,
                        'direction': 'outgoing'
                    })
        
        # Incoming edges
        for source in self.graph:
            if source != entity and entity in self.graph[source]:
                for relation in self.graph[source][entity]:
                    facts.append({
                        'subject': source,
                        'relation': relation,
                        'object': entity,
                        'direction': 'incoming'
                    })
        
        # Simple alphabetical sorting by relation (no weighting)
        facts.sort(key=lambda x: x['relation'])
        return facts[:k]
    
    def get_statistics(self) -> Dict:
        """Get current statistics of the memory graph"""
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges()
        }
