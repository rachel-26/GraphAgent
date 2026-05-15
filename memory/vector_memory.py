from sklearn.metrics.pairwise import cosine_similarity
import logging
import numpy as np
from typing import List, Dict, Any
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logging.warning("sentence_transformers not found. Using a mock TfIdf-based SentenceTransformer for demonstration.")
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    class SentenceTransformer:
        def __init__(self, model_name: str):
            self.vectorizer = TfidfVectorizer()
            self.is_fitted = False
            self.corpus = []
            
        def encode(self, texts):
            is_string = isinstance(texts, str)
            if is_string:
                texts = [texts]
                
            if not self.is_fitted:
                self.vectorizer.fit(texts + ["Alice", "Bob", "New York", "Boston", "Chicago", "Python", "TechCorp"])
                self.is_fitted = True
                
            arr = self.vectorizer.transform(texts).toarray()
            return arr[0] if is_string else arr

from .base_memory import BaseMemory, MemoryItem

class VectorMemory(BaseMemory):
    """
    A minimal vector baseline memory using SentenceTransformers for embeddings 
    and cosine similarity for retrieval.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.memories: List[MemoryItem] = []
        self.embeddings: List[np.ndarray] = []
        self.current_time = datetime.now()
        
    def add_memory(self, memory: MemoryItem):
        """
        Embed the memory text and store it alongside the original MemoryItem.
        """
        embedding = self.model.encode(memory.text)
        
        self.memories.append(memory)
        self.embeddings.append(embedding)
        self.current_time = memory.timestamp
        
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve memory items using cosine similarity between the query 
        and stored embeddings.
        """
        if not self.memories:
            return []
            
        query_embedding = self.model.encode([query])
        memory_embeddings = np.array(self.embeddings)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, memory_embeddings)[0]
        
        # Get top k indices
        top_k_indices = similarities.argsort()[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            score = float(similarities[idx])
            memory_item = self.memories[idx]
            results.append({
                "memory": memory_item,
                "score": score,
                "metadata": {
                    "method": "cosine_similarity"
                }
            })
            
        return results
        
    def update(self, current_time: datetime):
        """
        Standard update method. Vector memories typically do not decay natively 
        like temporal graphs, but we update the internal clock to remain consistent.
        """
        self.current_time = current_time
