from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MemoryItem:
    text: str
    timestamp: datetime
    importance: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseMemory(ABC):
    """
    Abstract base class for all memory systems in the research framework.
    This interface ensures uniform evaluation across different memory strategies.
    """

    @abstractmethod
    def add_memory(self, memory: MemoryItem):
        """
        Store a new memory item in the backend.
        
        Args:
            memory (MemoryItem): The standardized memory item object containing text, 
                                 timestamp, importance, and metadata.
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant memories for a given query.
        
        Args:
            query (str): The search query (semantic string or keyword).
            k (int): Number of memory items to return.
            
        Returns:
            List[Dict]: A list of dictionaries with standardized keys:
                        [
                            {
                                "memory": MemoryItem,
                                "score": float,
                                "metadata": dict
                            },
                            ...
                        ]
        """
        pass

    @abstractmethod
    def update(self, current_time: datetime):
        """
        Trigger periodic updates to the memory system (e.g., temporal decay, consolidation).
        
        Args:
            current_time (datetime): The external current time to simulate deterministic progression.
        """
        pass
