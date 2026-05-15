# LangGraph Human-in-the-Loop (HITL) with React + FastAPI

# 1. Introduction

This document explains how Human-in-the-Loop (HITL) works in LangGraph using:

- LangGraph
- FastAPI
- React

The example demonstrates:

1. Starting a graph
2. Pausing execution for human approval
3. Resuming execution with feedback
4. Returning final results

This is the same architecture used in:
- AI agents
- ATS systems
- Medical AI
- Legal AI
- Finance workflows

---

# 2. Core HITL Concepts

LangGraph Human-in-the-Loop is based on 3 important concepts.

---

# 2.1 interrupt()

`interrupt()` pauses graph execution and waits for human input.

Example:

```python
feedback = interrupt(
    {
        "answer": state["answer"],
        "message": "Approve or edit"
    }
)
````

When this executes:

* graph stops
* state is saved
* frontend receives interrupt response

---

# 2.2 Command(resume=...)

Used to continue the graph after human feedback.

Example:

```python
Command(
    resume="Add Kubernetes explanation"
)
```

This resumes graph execution from the paused node.

---

# 2.3 Checkpointing

LangGraph must save graph state while paused.

Example:

```python
memory = MemorySaver()

agent = graph.compile(
    checkpointer=memory
)
```

Without checkpointing:

* graph cannot resume properly

---

# 3. Architecture

System architecture:

```text
React Frontend
       ↓
FastAPI Backend
       ↓
LangGraph Agent
       ↓
interrupt()
       ↓
Graph pauses
       ↓
Human feedback
       ↓
Command(resume=...)
       ↓
Graph resumes
```

---

# 4. Backend Code (FastAPI + LangGraph)

# 4.1 Install Dependencies

```bash
pip install langgraph fastapi uvicorn
```

---

# 4.2 backend.py

```python
from fastapi import FastAPI

from langgraph.graph import (
    StateGraph,
    END
)

from langgraph.types import (
    interrupt,
    Command
)

from langgraph.checkpoint.memory import (
    MemorySaver
)

from typing import TypedDict


app = FastAPI()


# ============================================
# STATE
# ============================================

class AgentState(TypedDict):

    task: str

    answer: str

    feedback: str


# ============================================
# NODE 1
# GENERATE ANSWER
# ============================================

def generate_node(state):

    state["answer"] = (
        f"AI answer for: {state['task']}"
    )

    return state


# ============================================
# NODE 2
# HUMAN REVIEW
# ============================================

def human_node(state):

    feedback = interrupt(

        {
            "answer":
            state["answer"],

            "message":
            "Approve or edit"
        }

    )

    state["feedback"] = feedback

    return state


# ============================================
# NODE 3
# FINALIZE
# ============================================

def final_node(state):

    state["answer"] += (
        f" | Feedback: {state['feedback']}"
    )

    return state


# ============================================
# BUILD GRAPH
# ============================================

graph = StateGraph(AgentState)

graph.add_node(
    "generate",
    generate_node
)

graph.add_node(
    "human",
    human_node
)

graph.add_node(
    "final",
    final_node
)

graph.set_entry_point(
    "generate"
)

graph.add_edge(
    "generate",
    "human"
)

graph.add_edge(
    "human",
    "final"
)

graph.add_edge(
    "final",
    END
)


# ============================================
# MEMORY CHECKPOINT
# ============================================

memory = MemorySaver()

agent = graph.compile(
    checkpointer=memory
)


# ============================================
# API 1
# START GRAPH
# ============================================

@app.post("/start")
def start(data: dict):

    config = {
        "configurable": {
            "thread_id": "1"
        }
    }

    result = agent.invoke(

        {
            "task": data["task"]
        },

        config=config
    )

    return result


# ============================================
# API 2
# RESUME GRAPH
# ============================================

@app.post("/resume")
def resume(data: dict):

    config = {
        "configurable": {
            "thread_id": "1"
        }
    }

    result = agent.invoke(

        Command(
            resume=data["feedback"]
        ),

        config=config
    )

    return result
```

---

# 5. Backend Explanation

# 5.1 State Management

```python
class AgentState(TypedDict):
```

Stores:

* task
* AI answer
* human feedback

LangGraph passes this state between nodes.

---

# 5.2 generate_node()

```python
def generate_node(state):
```

Creates AI-generated answer.

Example:

```text
AI answer for: Explain Docker
```

---

# 5.3 human_node()

```python
feedback = interrupt(...)
```

This is the core HITL functionality.

When executed:

* graph pauses
* state saved in memory
* frontend receives interrupt response

---

# 5.4 final_node()

Uses human feedback to finalize result.

---

# 5.5 thread_id

```python
"thread_id": "1"
```

Very important.

LangGraph uses this to:

* restore state
* continue paused workflows

If thread_id changes:

* graph cannot resume

---

# 6. Frontend Code (React)

# 6.1 Create React App

```bash
npm create vite@latest
```

Install dependencies:

```bash
npm install
```

---

# 6.2 App.jsx

```jsx
import { useState } from "react"

function App() {

  const [task, setTask] = useState("")

  const [draft, setDraft] = useState("")

  const [feedback, setFeedback] = useState("")

  const [finalAnswer, setFinalAnswer] =
    useState("")


  // ============================================
  // START GRAPH
  // ============================================

  async function startGraph() {

    const res = await fetch(

      "http://localhost:8000/start",

      {
        method: "POST",

        headers: {
          "Content-Type":
          "application/json"
        },

        body: JSON.stringify({
          task
        })
      }
    )

    const data = await res.json()

    console.log(data)


    // ============================================
    // INTERRUPT RESPONSE
    // ============================================

    if (data.__interrupt__) {

      setDraft(
        data.__interrupt__[0].value.answer
      )
    }
  }


  // ============================================
  // RESUME GRAPH
  // ============================================

  async function resumeGraph() {

    const res = await fetch(

      "http://localhost:8000/resume",

      {
        method: "POST",

        headers: {
          "Content-Type":
          "application/json"
        },

        body: JSON.stringify({
          feedback
        })
      }
    )

    const data = await res.json()

    console.log(data)

    setFinalAnswer(data.answer)
  }


  return (

    <div style={{ padding: 40 }}>

      <h1>LangGraph HITL Demo</h1>

      <input
        value={task}
        onChange={(e)=>
          setTask(e.target.value)
        }
        placeholder="Enter task"
      />

      <button onClick={startGraph}>
        Start
      </button>


      <hr />


      <h2>AI Draft</h2>

      <p>{draft}</p>


      <input
        value={feedback}
        onChange={(e)=>
          setFeedback(e.target.value)
        }
        placeholder="Human feedback"
      />

      <button onClick={resumeGraph}>
        Resume Graph
      </button>


      <hr />


      <h2>Final Answer</h2>

      <p>{finalAnswer}</p>

    </div>
  )
}

export default App
```

---

# 7. Frontend Explanation

# 7.1 startGraph()

Calls:

```text
POST /start
```

Backend starts LangGraph execution.

Graph pauses at:

```python
interrupt()
```

Frontend receives:

```json
{
  "__interrupt__": ...
}
```

The AI draft is displayed.

---

# 7.2 resumeGraph()

Calls:

```text
POST /resume
```

Backend resumes graph using:

```python
Command(resume=...)
```

Graph continues execution.

---

# 7.3 UI Flow

User:

1. enters task
2. clicks Start
3. sees AI draft
4. adds feedback
5. resumes graph
6. sees final answer

---

# 8. Running the Project

# Start Backend

```bash
uvicorn backend:app --reload
```

Runs on:

```text
http://localhost:8000
```

---

# Start React Frontend

```bash
npm run dev
```

Runs on:

```text
http://localhost:5173
```

---

# 9. Example Workflow

# Input

```text
Explain Docker
```

---

# Graph Execution

```text
generate_node()
↓
human_node()
↓
interrupt()
```

Graph pauses.

---

# Frontend Displays

```text
AI answer for: Explain Docker
```

---

# Human Feedback

```text
Add Kubernetes explanation
```

---

# Resume Graph

```python
Command(
    resume="Add Kubernetes explanation"
)
```

---

# Final Output

```text
AI answer for: Explain Docker
| Feedback: Add Kubernetes explanation
```

---

# 10. Why This Matters

This architecture is used in production AI systems because:

* autonomous AI cannot always be trusted
* humans must approve critical decisions
* workflows require oversight

---

# 11. Real-World Applications

## Recruitment

Recruiters approve:

* rankings
* candidate shortlists
* hiring recommendations

---

## Medical AI

Doctors approve:

* diagnoses
* treatment plans

---

## Legal AI

Lawyers approve:

* generated contracts
* legal summaries

---

## Finance

Analysts approve:

* fraud decisions
* transaction approvals

---

# 12. Key Takeaways

LangGraph HITL requires:

1. `interrupt()`
2. `Command(resume=...)`
3. checkpointing
4. thread_id
5. frontend approval UI

---

# 13. Conclusion

This project demonstrates a real Human-in-the-Loop architecture using:

* LangGraph
* FastAPI
* React

The system supports:

* graph pausing
* human approval
* graph resuming
* state persistence
* interactive AI workflows

This is the foundation for building production-grade AI agents.

```
```
