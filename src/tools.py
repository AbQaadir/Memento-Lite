import subprocess
import sys
import abc
from typing import Any, Dict, Optional
from duckduckgo_search import DDGS

class Tool(abc.ABC):
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def description(self) -> str:
        pass

    @abc.abstractmethod
    def execute(self, **kwargs) -> str:
        pass

class SearchTool(Tool):
    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def name(self) -> str:
        return "web_search"

    def description(self) -> str:
        return "Search the web for information using DuckDuckGo. Arguments: query (str)."

    def execute(self, query: str) -> str:
        try:
            # Fix for Streamlit/Asyncio: "Event loop is closed"
            # We force a new loop for this synchronous wrapper call if needed
            import asyncio
            
            # Create a new loop for this thread if one doesn't exist or is closed
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    raise RuntimeError("Loop closed")
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))
            
            if not results:
                return "No results found."
            
            formatted_results = []
            for r in results:
                formatted_results.append(f"Title: {r.get('title')}\nLink: {r.get('href')}\nSnippet: {r.get('body')}")
            
            return "\n\n".join(formatted_results)
        except Exception as e:
            return f"Error gathering search results: {e}"

class CalculatorTool(Tool):
    def name(self) -> str:
        return "calculator"

    def description(self) -> str:
        return "Evaluate a mathematical expression. Arguments: expression (str). Example: calculator('2 + 2 * 10'). Note: Use '**' for power, not '^'."

    def execute(self, expression: str) -> str:
        try:
             # Basic eval as requested
            return str(eval(expression))
        except Exception as e:
            return f"Error evaluating expression: {e}"

class PythonREPLTool(Tool):
    def name(self) -> str:
        return "python_repl"

    def description(self) -> str:
        return "Execute Python code. Use for data analysis or complex calculations. Arguments: code (str)."

    def execute(self, code: str) -> str:
        # VERY UNSAFE: Executes arbitrary code. 
        # In a real "Memento" agent, this would be sandboxed.
        try:
            # Capture stdout
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                # We use a shared local scope if we want state persistence, 
                # but here we'll keep it stateless for simplicity per call or minimal persistence.
                # Let's use a fresh scope for safety/simplicity unless state is required.
                exec_globals = {}
                exec(code, exec_globals)
            
            output = f.getvalue()
            return output if output.strip() else "(Code executed with no output)"
        except Exception as e:
            return f"Error executing code: {e}"

# Tool Registry
TOOL_REGISTRY: Dict[str, Tool] = {
    "web_search": SearchTool(),
    "calculator": CalculatorTool(),
    "python_repl": PythonREPLTool(),
}

def get_tool(name: str) -> Optional[Tool]:
    return TOOL_REGISTRY.get(name)
