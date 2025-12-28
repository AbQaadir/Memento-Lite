import os
import json
from google import genai
from typing import List, Dict, Any, Tuple, Optional
from src.memory import CaseBank, Case
from src.tools import TOOL_REGISTRY, Tool
from src.config import CONFIG

# Configure API Key (Load from env)
from dotenv import load_dotenv
# Explicitly look for just .env in project root to avoid confusion
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path=env_path)

API_KEY_ENV = CONFIG.get("llm", {}).get("api_key_env", "GOOGLE_API_KEY")
API_KEY = os.getenv(API_KEY_ENV)

class SubtaskMemory:
    """
    Subtask Memory: Stores the decomposed plan and tracks the progress of each subtask.
    Acts as the 'To-Do List' for the agent.
    """
    def __init__(self):
        # List of {"task": str, "status": "pending"|"completed", "result": str}
        self.subtasks: List[Dict[str, Any]] = []

    def set_plan(self, plan: List[str]):
        """Initializes the memory with a new plan."""
        self.subtasks = [{"task": t, "status": "pending", "result": None} for t in plan]

    def get_next_pending_subtask(self) -> Optional[Dict[str, Any]]:
        """Returns the next subtask that needs to be executed."""
        for t in self.subtasks:
            if t["status"] == "pending":
                return t
        return None

    def update_subtask_result(self, task_description: str, result: str):
        """Marks a subtask as completed and stores its result."""
        for t in self.subtasks:
            if t["task"] == task_description:
                t["status"] = "completed"
                t["result"] = result
                break

    def get_summary(self) -> str:
        """Returns a string summary of the plan's progress."""
        summary = ""
        for i, t in enumerate(self.subtasks):
            icon = "[x]" if t["status"] == "completed" else "[ ]"
            summary += f"{icon} Step {i+1}: {t['task']}\n"
            if t["result"]:
                summary += f"    Result: {t['result']}\n"
        return summary
    
    def is_all_completed(self) -> bool:
        return all(t["status"] == "completed" for t in self.subtasks)

class ToolMemory:
    """
    Tool Memory: Stores the history of tool executions.
    Prevents redundant calls and provides detailed context for the Executor.
    """
    def __init__(self):
        # List of {"tool": str, "args": str, "result": str}
        self.logs: List[Dict[str, str]] = []

    def add_log(self, tool_name: str, args: str, result: str):
        self.logs.append({
            "tool": tool_name,
            "args": args,
            "result": result
        })

    def get_context(self) -> str:
        """Returns a formatted string of all tool interactions."""
        if not self.logs:
            return "(No tools used yet)"
        
        context = ""
        for log in self.logs:
            context += f"Action: {log['tool']}({log['args']})\nResult: {log['result']}\n---\n"
        return context

class AgentCallback:
    """Base class for handling agent events (Observability)."""
    def on_plan(self, plan: List[str]): pass
    def on_subtask_start(self, subtask: str): pass
    def on_subtask_end(self, result: str): pass
    def on_tool_start(self, tool: str, args: str): pass
    def on_tool_end(self, result: str): pass
    def on_log(self, message: str): pass

import time
from google.api_core import exceptions

def retry_with_backoff(max_retries=5, initial_delay=2):
    """
    Decorator to retry a function if it raises a ResourceExhausted (429) error.
    Uses exponential backoff.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions.ResourceExhausted as e:
                    # 429 Error
                    print(f"  [Rate Limit] Hit quota. Sleeping for {delay}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                    delay *= 2 # Exponential backoff
                except Exception as e:
                    # Reraise other errors immediately or handle them?
                    # For now, let's just reraise non-quota errors to be safe.
                    # Unless it's a transient 500? But 429 is our main concern.
                    raise e
            raise exceptions.ResourceExhausted("Max retries exceeded")
        return wrapper
    return decorator

class Planner:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or CONFIG.get("llm", {}).get("model", "gemini-flash-latest")
        self.client = None
        if API_KEY:
             self.client = genai.Client(api_key=API_KEY)
        else:
             print(f"WARNING: {API_KEY_ENV} not found. Planner will not work.")

    @retry_with_backoff()
    def _call_model(self, prompt):
        return self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

    def generate_plan(self, task: str, retrieved_cases: List[Case]) -> List[str]:
        """Generates a high-level plan based on the task and similar past cases."""
        if not self.client:
             return [f"Solve the task: {task}"]
             
        # Construct Prompt
        prompt = f"Task: {task}\n\n"
        
        if retrieved_cases:
            prompt += "Here are some similar past cases and their plans that might help:\n"
            for i, case in enumerate(retrieved_cases):
                prompt += f"--- Case {i+1} ---\n"
                prompt += f"Task: {case.state_description}\n"
                prompt += f"Plan: {json.dumps(case.plan)}\n"
                prompt += f"Result: {case.result}\n"
            prompt += "\n"
        
        prompt += (
            "Based on the above (if any), create a concise, step-by-step plan to solve the current task.\n"
            "Return ONLY a JSON array of strings, where each string is a subtask.\n"
            "Example: [\"Search for X\", \"Analyze Y\", \"Summary Z\"]\n"
            "Do not include markdown formatting like ```json ... ```."
        )

        try:
            response = self._call_model(prompt)
            text = response.text.strip()
            # Cleanup markdown code blocks if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text.rsplit("\n", 1)[0]
            text = text.replace("```json", "").replace("```", "").strip()
            
            plan = json.loads(text)
            if isinstance(plan, list):
                return plan
            else:
                return [text] # Fallback
        except Exception as e:
            print(f"Error generating plan: {e}")
            return [f"Solve the task: {task}"] # Fallback plan

class Executor:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or CONFIG.get("llm", {}).get("model", "gemini-flash-latest")
        self.client = None
        if API_KEY:
             self.client = genai.Client(api_key=API_KEY)
        
        # Load tools from config if possible, else default
        config_tools = CONFIG.get("tools", [])
        if config_tools:
            self.tools = {name: tool for name, tool in TOOL_REGISTRY.items() if name in config_tools}
        else:
            self.tools = TOOL_REGISTRY

    @retry_with_backoff()
    def _call_model(self, prompt):
        return self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

    def execute_subtask(self, subtask: str, subtask_mem: SubtaskMemory, tool_mem: ToolMemory, callback: Optional[AgentCallback] = None) -> str:
        """Executes a single subtask using available tools."""
        if not self.client:
             return "Error: No API Key."
             
        tool_desc = "\n".join([f"- {name}: {t.description()}" for name, t in self.tools.items()])
        
        plan_context = subtask_mem.get_summary()
        tool_context = tool_mem.get_context()
        
        prompt = (
            f"Current Subtask TO EXECUTE: {subtask}\n\n"
            f"--- Overall Plan Status ---\n{plan_context}\n\n"
            f"--- Tool Usage History ---\n{tool_context}\n\n"
            f"--- Available Tools ---\n{tool_desc}\n\n"
            "You need to complete the 'Current Subtask'. You can USE a tool by responding in the following JSON format ONLY:\n"
            "{ \"tool\": \"tool_name\", \"arguments\": \"argument_string\" }\n"
            "If you have the answer directly or after using tools, respond in JSON:\n"
            "{ \"final_answer\": \"your explanation or answer\" }\n"
            "Do not use markdown blocks."
        )

        max_steps = 5
        current_step = 0
        execution_log = "" 

        while current_step < max_steps:
            try:
                if callback: callback.on_log(f"Thinking (Attempt {current_step+1})...")
                
                response = self._call_model(prompt + f"\n\n--- Current Steps ---\n{execution_log}")
                text = response.text.strip()
                
                # Cleanup
                if text.startswith("```"): text = text.split("\n", 1)[1]
                if text.endswith("```"): text = text.rsplit("\n", 1)[0]
                text = text.replace("```json", "").replace("```", "").strip()
                
                try:
                    action = json.loads(text)
                except json.JSONDecodeError:
                    return text

                if "final_answer" in action:
                    return action["final_answer"]
                
                if "tool" in action:
                    tool_name = action["tool"]
                    tool_args = action["arguments"]
                    
                    tool = self.tools.get(tool_name)
                    if tool:
                        if callback: callback.on_tool_start(tool_name, tool_args)
                        print(f"  [Executor] Calling {tool_name} with {tool_args}...")
                        
                        result = tool.execute(tool_args)
                        
                        if callback: callback.on_tool_end(result)
                        
                        tool_mem.add_log(tool_name, tool_args, result)
                        execution_log += f"Action: Using {tool_name} with {tool_args}\nResult: {result}\n\n"
                    else:
                        execution_log += f"Action: Using {tool_name}\nResult: Tool not found.\n\n"
                else:
                    return f"Error: Invalid LLM response: {text}"

            except Exception as e:
                execution_log += f"Error: {e}\n"
            
            current_step += 1

        return "Subtask execution max steps reached. Last log:\n" + execution_log

class MementoAgent:
    def __init__(self, memory_path: Optional[str] = None, memory_instance: Optional[CaseBank] = None):
        if memory_instance:
            self.memory = memory_instance
        else:
            self.memory = CaseBank(storage_path=memory_path)
        self.planner = Planner()
        self.executor = Executor()

    def run(self, task: str, callback: Optional[AgentCallback] = None) -> str:
        """Orchestrates the agent's workflow."""
        if callback: callback.on_log(f"Starting Task: {task}")
        print(f"--- Starting Memento Agent for Task: {task} ---")
        
        subtask_memory = SubtaskMemory()
        tool_memory = ToolMemory()

        # 1. Retrieve
        if callback: callback.on_log("Retrieving cases...")
        retrieved_cases = self.memory.retrieve_similar(task, k=2)
        if retrieved_cases:
            print(f"Retrieved {len(retrieved_cases)} similar cases.")
            if callback: callback.on_log(f"Retrieved {len(retrieved_cases)} cases.")
        
        # 2. Plan
        if callback: callback.on_log("Generating plan...")
        plan_list = self.planner.generate_plan(task, retrieved_cases)
        subtask_memory.set_plan(plan_list) 
        
        if callback: callback.on_plan(plan_list)
        print(f"Generated Plan: {plan_list}")
        
        # 3. Execute
        full_result = []
        
        while not subtask_memory.is_all_completed():
            current_subtask_item = subtask_memory.get_next_pending_subtask()
            if not current_subtask_item:
                break
            
            subtask_desc = current_subtask_item["task"]
            print(f"Executing Step: {subtask_desc}")
            
            if callback: callback.on_subtask_start(subtask_desc)
            
            result = self.executor.execute_subtask(subtask_desc, subtask_memory, tool_memory, callback)
            
            if callback: callback.on_subtask_end(result)
            
            print(f"  Result: {result}")
            
            subtask_memory.update_subtask_result(subtask_desc, result)
            full_result.append(f"Step: {subtask_desc}\nResult: {result}")

        final_result = "\n".join(full_result)
        
        # 4. Store
        if callback: callback.on_log("Storing experience...")
        new_case = Case(
            state_description=task,
            plan=plan_list,
            result=final_result,
            reward=1.0 
        )
        self.memory.add_case(new_case)
        print("Case stored in Long-Term Case Memory.")
        
        return final_result
