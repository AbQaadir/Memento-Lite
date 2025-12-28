from src.agent import MementoAgent
import os

print("Verifying Memento Implementation...")

# Check if API Key exists
if not os.getenv("GOOGLE_API_KEY"):
    print("WARNING: GOOGLE_API_KEY not found. Please set it to run full tests.")
    # We will skip full run if no key, just check import
else:
    print("GOOGLE_API_KEY found.")

try:
    agent = MementoAgent(memory_path="test_memory.json")
    print("MementoAgent initialized successfully.")
    
    if os.getenv("GOOGLE_API_KEY"):
        task = "What is the result of 15 * 12 + 4?"
        print(f"Running test task: {task}")
        result = agent.run(task)
        print("Test Run Completed.")
        print(f"Result: {result}")
        if "184" in result:
             print("SUCCESS: Calculation seems correct.")
        else:
             print("WARNING: Result might be incorrect, check logs.")
    
except Exception as e:
    print(f"Verification Failed: {e}")
    raise e
