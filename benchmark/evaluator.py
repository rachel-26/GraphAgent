from typing import List, Dict, Any

class Evaluator:
    @staticmethod
    def recall_at_k(retrieved: List[Dict[str, Any]], ground_truth: str, k: int = 5) -> int:
        """
        Was the correct answer in the top-k retrieved memories?
        Returns 1 if yes, 0 if no.
        """
        top_k = retrieved[:k]
        for item in top_k:
            mem_text = item["memory"].text
            if ground_truth.lower() in mem_text.lower():
                return 1
        return 0

    @staticmethod
    def mrr(retrieved: List[Dict[str, Any]], ground_truth: str) -> float:
        """
        Mean Reciprocal Rank (MRR).
        Returns 1/rank of the first correct memory.
        """
        for i, item in enumerate(retrieved):
            mem_text = item["memory"].text
            if ground_truth.lower() in mem_text.lower():
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def contradiction_accuracy(retrieved: List[Dict[str, Any]], ground_truth: str) -> int:
        """
        Did the system retrieve the newest truth as the #1 rank?
        Returns 1 if the first retrieved item contains the ground truth, 0 otherwise.
        """
        if not retrieved:
            return 0
        mem_text = retrieved[0]["memory"].text
        if ground_truth.lower() in mem_text.lower():
            return 1
        return 0
