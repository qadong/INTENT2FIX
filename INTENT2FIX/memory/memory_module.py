import json
from typing import List, Dict
from rank_bm25 import BM25Okapi
from .memory import Memory
import logging

logger = logging.getLogger(__name__)
class MemoryModule:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.memories: List[Memory] = []
        self.bm25_problem = None
        self.bm25_repro = None
        self.bm25_test = None
        self.bm25_expected_repro = None
        self.load()

    def load(self):
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
                self.memories = [Memory(**item) for item in data]
            
            if self.memories:
                self._initialize_bm25()
        except FileNotFoundError:
            self.memories = []

    def _initialize_bm25(self):
        # Create a corpus for each field
        corpus_problem = [m.problem_statement for m in self.memories]
        corpus_repro = [m.reproduction_result for m in self.memories]
        # For test_files, concatenate the content of all files into a single string for indexing
        corpus_test = ["\n".join(file['content'] for file in m.test_files) for m in self.memories]
        corpus_expected_repro = [m.expected_reproduction_result for m in self.memories]

        # Tokenize each corpus
        tokenized_corpus_problem = [doc.split(" ") for doc in corpus_problem]
        tokenized_corpus_repro = [doc.split(" ") for doc in corpus_repro]
        tokenized_corpus_test = [doc.split(" ") for doc in corpus_test]
        tokenized_corpus_expected_repro = [doc.split(" ") for doc in corpus_expected_repro]

        # Initialize a BM25 instance for each field
        self.bm25_problem = BM25Okapi(tokenized_corpus_problem)
        self.bm25_repro = BM25Okapi(tokenized_corpus_repro)
        self.bm25_test = BM25Okapi(tokenized_corpus_test)
        self.bm25_expected_repro = BM25Okapi(tokenized_corpus_expected_repro)

    def save(self):
        with open(self.file_path, 'w') as f:
            data = [memory.model_dump() for memory in self.memories]
            json.dump(data, f, indent=4)

    def add(self, memory: Memory):
        """
        Adds a new memory or updates an existing one based on instance_id.
        This "upsert" logic prevents duplicates and allows enriching memories.
        """
        existing_memory = next((m for m in self.memories if m.instance_id == memory.instance_id), None)

        if existing_memory:
            logger.info(f"Updating existing memory for instance_id: {memory.instance_id}")
            # Update fields of the existing memory with non-empty values from the new one
            for field, value in memory.model_dump().items():
                if value and getattr(existing_memory, field) != value:
                    setattr(existing_memory, field, value)
        else:
            logger.info(f"Adding new memory for instance_id: {memory.instance_id}")
            self.memories.append(memory)

        # Re-initialize BM25 with the updated data and save
        self._initialize_bm25()
        self.save()

    def search(self, query_text: str, weights: Dict[str, float], top_n: int = 2) -> List[Memory]:
        if not self.memories or not self.bm25_problem:
            return []

        tokenized_query = query_text.split(" ")

        # Get scores from each BM25 instance
        scores_problem = self.bm25_problem.get_scores(tokenized_query)
        scores_repro = self.bm25_repro.get_scores(tokenized_query)
        scores_test = self.bm25_test.get_scores(tokenized_query)
        scores_expected_repro = self.bm25_expected_repro.get_scores(tokenized_query)

        # Calculate weighted sum of scores for each memory
        combined_scores = []
        for i in range(len(self.memories)):
            score = (
                scores_problem[i] * weights.get('problem_statement', 0.5) +
                scores_repro[i] * weights.get('reproduction_result', 0.2) +
                scores_test[i] * weights.get('test_files', 0.1) +
                scores_expected_repro[i] * weights.get('expected_reproduction_result', 0.2)
            )
            combined_scores.append(score)

        # Get the indices of the top-n scores
        top_n_indices = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i], reverse=True)[:top_n]

        # Return the corresponding memories
        return [self.memories[i] for i in top_n_indices]

    def get_all(self) -> List[Memory]:
        return self.memories
