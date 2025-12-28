import json
import os
import numpy as np
import time
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Ensure env vars are loaded for API key
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path=env_path)

from src.config import CONFIG

@dataclass
class Case:
    """
    Represents a stored experience in the Case Bank.
    """
    state_description: str  # The task or query description
    plan: List[str]         # The sequence of high-level actions/plan steps
    code: Optional[str] = None # The code generated/executed (if any)
    result: str = ""        # The final answer or outcome
    reward: float = 0.0     # Success score (0.0 to 1.0) for the case
    metadata: Dict[str, Any] = field(default_factory=dict) # Extra info like timestamp, tools used

class CaseBank:
    """
    Manages storage and retrieval of past cases using Google GenAI Embeddings.
    """
    def __init__(self, storage_path: Optional[str] = None):
        # Use config defaults if not provided
        self.storage_path = storage_path or CONFIG.get("case_bank", {}).get("path", "memory.json")
        self.model_name = CONFIG.get("embedding", {}).get("model", "text-embedding-004")
        
        # API Client
        api_key_env = CONFIG.get("llm", {}).get("api_key_env", "GOOGLE_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            print(f"Warning: {api_key_env} not found. Embeddings will fail.")
            self.client = None
        else:
            self.client = genai.Client(api_key=api_key)

        self.cases: List[Case] = []
        self.embeddings = None
        
        # Load existing cases
        self._load_memory()
        
        # We process embeddings lazily or on load if needed
        # Unlike local models, we don't "load" the model, but we might want to ensure we have embeddings for loaded cases.
        if self.cases and self.embeddings is None:
             self._update_embeddings()

    def _load_memory(self):
        """Loads cases from JSON file."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.cases = [Case(**item) for item in data]
                print(f"Loaded {len(self.cases)} cases from memory.")
            except Exception as e:
                print(f"Error loading memory: {e}. Starting with empty CaseBank.")
                self.cases = []
        else:
            self.cases = []

    def save_memory(self):
        """Saves current cases to JSON file."""
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump([asdict(c) for c in self.cases], f, indent=2)
            print("Memory saved successfully.")
        except Exception as e:
            print(f"Error saving memory: {e}")

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Helper to encode a list of texts using Google GenAI."""
        if not self.client or not texts:
            return np.array([])
            
        try:
            # The new SDK syntax based on user request/docs
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=texts
            )
            
            # Inspect structure if needed, but let's try to extract values safely
            embeddings_list = []
            if hasattr(result, 'embeddings'):
                for max_idx, emb in enumerate(result.embeddings):
                     # If emb is an object with 'values', use that. 
                     # If it's a list, use it directly.
                     if hasattr(emb, "values"):
                         embeddings_list.append(emb.values)
                     else:
                         embeddings_list.append(emb)
            
            return np.array(embeddings_list, dtype=np.float32)

        except Exception as e:
            print(f"Error calling embedding API: {e}")
            return np.array([])

    def _update_embeddings(self):
        """Updates internal vector store from current cases."""
        if not self.cases:
            self.embeddings = None
            return
        
        texts = [c.state_description for c in self.cases]
        print(f"Encoding {len(texts)} cases with {self.model_name}...")
        self.embeddings = self._encode(texts)

    def add_case(self, case: Case):
        """Adds a new case to the bank and updates embeddings."""
        self.cases.append(case)
        self.save_memory()
        
        # Incremental update: encode just the new case and stack
        if self.client:
            new_emb = self._encode([case.state_description])
            if new_emb.size > 0:
                if self.embeddings is None or self.embeddings.size == 0:
                    self.embeddings = new_emb
                else:
                    self.embeddings = np.vstack([self.embeddings, new_emb])

    def retrieve_similar(self, query: str, k: int = 3) -> List[Case]:
        """
        Retrieves top-k most similar cases to the query.
        """
        if not self.client or self.embeddings is None or len(self.cases) == 0:
            return []
        
        query_emb = self._encode([query])
        if query_emb.size == 0:
             return []
        
        # Cosine similarity
        # embeddings: (N, D), query_emb: (1, D)
        
        # Normalize embeddings (if not already normalized by API? usually they are, but safe to do it)
        emb_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        query_norm = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        
        scores = np.dot(emb_norm, query_norm.T).flatten() # (N,)
        
        # Get top k indices
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            if scores[idx] > 0.0: # Only return somewhat positive matches
                 # Add the score to the result object or logging technically
                 # For now just return the case
                results.append(self.cases[idx])
                
        return results

if __name__ == "__main__":
    # Simple test
    bank = CaseBank(storage_path="test_memory_google.json")
    
    # Add dummy case if empty
    if not bank.cases:
        c1 = Case(
            state_description="Calculate fibonacci of 10",
            plan=["Write a python function for fibonacci", "Run it with input 10"],
            code="def fib(n): ...",
            result="55",
            reward=1.0
        )
        bank.add_case(c1)
        print("Added test case.")

    # Retrieve
    query = "How to calculate fibonacci number?"
    results = bank.retrieve_similar(query, k=1)
    print(f"Query: {query}")
    for res in results:
        print(f"Retrieved: {res.state_description} | Result: {res.result}")
        
    # Clean up test file
    if os.path.exists("test_memory_google.json"):
        os.remove("test_memory_google.json")
