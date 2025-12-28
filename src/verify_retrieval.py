import json
import numpy as np
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory import CaseBank, Case
from typing import List

def load_seed_memory(path: str) -> List[Case]:
    cases = []
    print(f"Loading seed memory from {path}...")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
                
                # key mapping:
                # question -> state_description
                # plan (json str) -> plan (list of strings)
                # reward -> result
                
                question = data.get("question", "")
                raw_plan = data.get("plan", "[]")
                reward = data.get("reward", 0)
                
                # Parse plan
                try:
                    plan_json = json.loads(raw_plan)
                    # Handle both list and object wrapper
                    if isinstance(plan_json, dict) and "plan" in plan_json:
                        steps = [step["description"] for step in plan_json["plan"]]
                    elif isinstance(plan_json, list):
                         steps = [step["description"] for step in plan_json]
                    else:
                        steps = [str(raw_plan)]
                except:
                    steps = [str(raw_plan)]
                
                result_str = "Success" if reward == 1 else "Failure"
                
                case = Case(
                    state_description=question,
                    plan=steps,
                    result=result_str
                )
                cases.append(case)
            except Exception as e:
                print(f"Error parsing line: {e}")
                
    print(f"Loaded {len(cases)} cases.")
    return cases

def main():
    # create a temporary in-memory bank or use a temp file
    # For simplicity, we just use CaseBank but strictly controlled
    
    # 1. Load Data
    cases = load_seed_memory("src/seed_memory.jsonl")
    
    # 2. Init Bank (Transient)
    # We pass a None storage_path to avoid overwriting real memory.json 
    # BUT CaseBank currently enforces a path. Let's start with a temp file.
    temp_memory = "seed_memory_test.json"
    if os.path.exists(temp_memory):
        os.remove(temp_memory)
        
    bank = CaseBank(storage_path=temp_memory)
    # Clear any existing (should be empty for new file)
    bank.cases = [] 
    
    # 3. Add cases (this will trigger embedding one by one, which is slow but fine for 8 cases)
    # A better way would be batch add, but add_case is what we have.
    print("Populating CaseBank (this calls the Embedding API)...")
    for case in cases:
        bank.add_case(case)
        print(f".", end="", flush=True)
    print("\nDone.")

    # 4. Test Queries
    queries = [
        "What is the capital of a specific county?", # Similar to Case 1
        "Who is the writer of Agatha Christie?",    # Similar to Case 7
        "Calculate fibonacci number"                # Out of distribution
    ]
    
    print("\n--- Running Retrieval Tests ---")
    for q in queries:
        print(f"\nQuery: {q}")
        results = bank.retrieve_similar(q, k=3)
        for i, case in enumerate(results):
            # Calculate score manually since retrieve_similar currently returns just cases
            # In a real verification script we'd want access to the score.
            # For now, just printing the rank is enough proof of concept.
            print(f"[{i+1}] {case.state_description} | Result: {case.result}")

    # Cleanup
    if os.path.exists(temp_memory):
        os.remove(temp_memory)

if __name__ == "__main__":
    main()
