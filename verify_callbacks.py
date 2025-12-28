from src.agent import MementoAgent, AgentCallback
import time

class MockCallback(AgentCallback):
    def on_log(self, message: str):
        print(f"[LOG] {message}")
    
    def on_plan(self, plan: list):
        print(f"[PLAN] {plan}")

    def on_subtask_start(self, subtask: str):
        print(f"[START] {subtask}")

    def on_subtask_end(self, result: str):
        print(f"[END] {result}")

    def on_tool_start(self, tool: str, args: str):
        print(f"[TOOL START] {tool}({args})")

    def on_tool_end(self, result: str):
        print(f"[TOOL END] {result}")

if __name__ == "__main__":
    print("Initializing Agent...")
    agent = MementoAgent()
    
    print("Running with callbacks...")
    # We expect this to fail with 429 if API is hit, but we should see logs before that
    try:
        agent.run("Calculate 2 + 2", callback=MockCallback())
    except Exception as e:
        print(f"Agent stopped (expectedly): {e}")

    print("Callback verification finished.")
