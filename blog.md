# 🧠 Memento: Giving LLM Agents "Muscle Memory"

*December 27, 2025* | *By Qaadir*

![Memento System Diagram](./images/memento.png)

---

### 🛑 The Problem: Groundhog Day for AI Agents

Imagine a genius mathematician who wakes up every morning with total amnesia. You ask them to solve a complex theorem. They meticulously work it out, deriving every step from scratch. It takes them all day.

Great. Now, the next day, you ask them the exact same question.

**They start from zero again.** They don't remember the shortcuts they found or the dead ends they hit yesterday. They are brilliant, but they are trapped in a loop.

This is the state of most LLM Agents today. They have high intelligence (reasoning) but zero **procedural memory**. They don't learn from experience; they just re-compute the world every single time.

**What if your agent could just... remember?**

Enter **Memento**.

---

## 💡 The Solution: Indexing Experience (Successes AND Failures)

We usually use **RAG** to give agents access to *data*. **Memento uses RAG to give agents access to *experience*.**

It introduces a **Case Bank**—a long-term memory of past trajectories.
When a new task comes in, the agent asks: *"Have I solved something like this before?"*

If the answer is yes, it retrieves the **Plan** from that past success. Crucially, strictly following the research, Memento also keeps a **"Burn Book" of failures**. It learns not just what to do, but what *not* to do, effectively simulating **Fine-Tuning** without ever touching a single model weight.

![Architecture](./images/architecture.png)

---

## � The "Real" Science (Research vs. MVP)

While I built a streamlined version, the **Memento Research Paper** goes much deeper.

### 1. The "Smart Coach" (Parametric Memory & Q-Learning) 🧠
In my implementation, I use **Cosine Similarity** (finding the most similar past task). This is called "Non-Parametric Memory."

The paper's major contribution is **Parametric Memory**. They don't just find similar cases; they learn a **Retrieval Policy**.
*   **Mechanism**: They train a **Q-function** (approximated by a **two-layer MLP**) using **online Soft Q-Learning**.
*   **The Goal**: Instead of picking the *closest* case, the model predicts `Q(s, c)`—the probability that past case `c` will help solve current task `s`.
*   **The Impact**: This acts like a smart coach. Even if an old case looks textually different, the Q-function might realize its *logic* is exactly what you need, surfacing relevant experience that a simple vector search would miss.

### 2. The Burn Book (Failures) 📉
The paper treats the agent as a **Memory-Augmented MDP**. It doesn't just store wins; it stores losses. By retrieving failed trajectories, the agent performs "retrospective analysis" to prune bad branches before it even tries them.

### 3. The Results (GAIA Benchmarks) 🏆
Does this simpler approach actually work? Yes.
Memento achieved **Top-1 on the GAIA validation set (87.88%)**, beating heavily fine-tuned models.

> **Pro Tip from the Paper**: Surprisingly, a "fast" planner (like GPT-4o or Gemini Flash) often outperforms "slow reasoning" models (like o1) in this architecture because the *memory* provides the reasoning structure, removing the need for deep introspection.

---

## 🛠️ My Implementation (The MVP)

I built a "Lite" version of Memento using **Python**, **Google Gemini**, and **Streamlit**.

### 1. The Hippocampus (Memory Bank) 💾

I stuck to the **Non-Parametric** approach for simplicity and speed. It uses **Google's `gemini-embedding-001`** model to convert tasks into vectors.

We use **Lazy Loading** so the app starts instantly and only warms up the AI model when you actually chat.

```python
# src/memory.py
from google import genai
import numpy as np

class CaseBank:
    def retrieve_similar(self, query: str, k: int = 3) -> List[Case]:
        # 1. Encode the current task (Lazy Loaded!)
        query_emb = self._encode([query])
        
        # 2. Find the closest past vectors (Cosine Similarity)
        # scores = (A . B) / (|A| |B|)
        scores = np.dot(self.embeddings, query_emb.T).flatten()
        
        # 3. Return the top K winners
        top_k_indices = np.argsort(scores)[-k:][::-1]
        return [self.cases[i] for i in top_k_indices]
```

### 2. The Planner (The Brain) 🧠

The `Planner` constructs a prompt labeled with **Traj** (Trajectory), injecting the retrieved cases as examples.

```python
# src/agent.py
def generate_plan(self, task: str, retrieved_cases: List[Case]) -> List[str]:
    prompt = f"Task: {task}\n\n"
    
    if retrieved_cases:
        prompt += "Here are similar past cases that SUCCEEDED:\n"
        for i, case in enumerate(retrieved_cases):
            prompt += f"--- Path {i+1} ---\n"
            prompt += f"Old Task: {case.state_description}\n"
            prompt += f"Plan Used: {json.dumps(case.plan)}\n"
    
    prompt += "Based on this, create a checklist for the current task."
    
    # Call Gemini Flash to generate the plan
    return self.client.models.generate_content(..., contents=prompt)
```

### 3. The Executor & MCP 🛠️

The executor takes the checklist and starts working. In a full production system, we would use the **Model Context Protocol (MCP)** to standardize tool access. For this MVP, I built a simple local tool registry.

```python
# src/tools.py
class CalculatorTool(Tool):
    def execute(self, expression: str) -> str:
        # Simple, direct, and effective
        return str(eval(expression))
```

### 4. Verification & Testing 🧪

To make this rigorous, I adopted the testing framework from the official Memento paper. I created a seed memory set (`seed_memory.jsonl`) mimicking the original "dummy memo" data and built a verification script `verify_retrieval.py`.

This allows us to deterministically test:
1.  **Recall**: Does the agent find the exact case for a known query?
2.  **Generalization**: Does it find relevant cases for *similar* but new queries?

```python
# src/verify_retrieval.py
def main():
    # Load the official seed data structure
    cases = load_seed_memory("src/seed_memory.jsonl")
    bank = CaseBank(storage_path="temp_test.json")
    
    # Run Benchmark
    query = "What is the capital of a specific county?"
    results = bank.retrieve_similar(query)
    
    # Verify we found the 'Columbia County' case
    assert results[0].state_description == "What is the capital of Columbia County?"
```

---

## 🖥️ The "Glass Box" UI

I didn't want a black-box agent. I used **Streamlit** to build a split-screen interface where you can see the agent's brain in real-time.

*   **Left**: Chat Interface.
*   **Right**: The "Glass Box" internals.
    *   **🧠 Hippocampus**: Shows exactly which past memories were triggered.
    *   **📝 Status**: A live checklist of what the agent is doing right now.
    *   **🛠️ Tools**: Raw logs of tool inputs and outputs.

---

## 🚀 How to Run It

Want to try it yourself? It's easy to get started.

### 1. Prerequisites
You'll need `uv` (or just pip) and a **Google API Key**.

### 2. Configuration
Create a `.env` file in the root:
```bash
GOOGLE_API_KEY=your_key_here
```

And check `config.yaml` to tweak settings:
```yaml
llm:
  model: gemini-flash-latest

embedding:
  provider: google
  model: gemini-embedding-001
```

### 3. Run the App
Launch the Streamlit interface:

```bash
uv run streamlit run src/app.py
```

The app will open in your browser, and you can start "training" your agent just by talking to it!

---

## 🧩 The Missing Piece

For too long, we've focused on making LLMs smarter (better reasoning) or more knowledgeable (larger RAG context). Memento demonstrates that **Experience** is the third, missing pillar.

By equipping our agents with simple procedural memory, we stop them from solving the same puzzles from scratch every day. We don't need massive compute or complex fine-tuning to fix this; we just need to let them write things down.

It turns out the most powerful upgrade for an AI agent isn't a bigger brain—it's a notebook.

*Give Memento a try, and stop explaining the same thing to your agent twice.*
