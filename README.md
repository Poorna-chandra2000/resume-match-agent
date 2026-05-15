# AI Resume Matching Agent using LangGraph and Hybrid RAG

# Introduction

This project is an AI-powered Resume Matching and Applicant Tracking System (ATS) Agent built using:

- LangGraph
- LangChain
- Hybrid RAG (Retrieval-Augmented Generation)
- ChromaDB
- BM25
- HuggingFace Embeddings
- Groq LLM
- Streamlit

The system intelligently retrieves, analyzes, ranks, and explains candidate resumes based on a given job description.

Unlike traditional ATS systems that rely only on keyword matching, this system combines:

1. Semantic Search
2. Keyword Search
3. LLM-based reasoning
4. Agentic workflows

to perform advanced candidate screening and ranking.

---

# Problem Statement

Traditional ATS systems suffer from multiple limitations:

- Exact keyword dependency
- Poor semantic understanding
- Weak explainability
- No conversational interaction
- No reasoning capabilities
- No human feedback integration

This project solves those limitations using:

- Hybrid Retrieval
- LangGraph Agents
- LLM-based reasoning
- Human-in-the-loop workflows

---

# System Architecture

Workflow:

START
→ Parse Job Description
→ Extract Requirements
→ Hybrid Resume Retrieval
→ Candidate Ranking
→ Report Generation
→ Human Feedback Loop
→ END

---

# Core Features

# 1. Hybrid RAG Retrieval

The retrieval system combines:

## Semantic Search

Uses:
- HuggingFace Embeddings
- ChromaDB Vector Database

This helps retrieve resumes that are semantically similar even if exact keywords are missing.

Example:

Job Description:
"Backend developer with API experience"

Resume:
"Built REST microservices"

Semantic search understands both are related.

---

## Keyword Search

Uses:
- BM25 ranking algorithm

This improves:
- exact skill matching
- ATS-style keyword precision

Example:
- Docker
- Kubernetes
- AWS

---

## Combined Hybrid Retrieval

The system merges:
- semantic retrieval
- keyword retrieval

Then removes duplicates.

This produces more accurate candidate retrieval.

---

# 2. LangGraph Agent Workflow

The project uses LangGraph to create a stateful AI agent workflow.

Each stage is represented as a node.

---

# Workflow Nodes

## Parse JD Node

Purpose:
- receives job description
- initializes workflow state

---

## Extract Requirements Node

Uses LLM to identify:
- must-have skills
- nice-to-have skills
- experience requirements

Example Output:

```json
{
  "must_have": ["Python", "FastAPI"],
  "nice_to_have": ["AWS"],
  "experience": "3+ years"
}
````

---

## Search Resumes Node

Uses:

* ChromaDB semantic retrieval
* BM25 keyword retrieval

Retrieves top matching resumes.

---

## Rank Candidates Node

Uses LLM reasoning to:

* evaluate resume relevance
* compare skills
* score candidates
* identify strengths and gaps

---

## Generate Report Node

Creates:

* final ATS report
* ranking explanation
* candidate insights

---

## Human Feedback Node

Allows recruiter feedback such as:

* Prioritize AWS more heavily
* Focus on leadership experience
* Reduce frontend importance

The agent then re-ranks candidates dynamically.

---

# 3. Explainability

The system explains:

* Why candidate matched
* Why candidate ranked higher
* Missing skills
* Hiring recommendation

Example:

Strengths:

* Strong Python backend experience
* Docker and Kubernetes expertise
* Microservices architecture

Gaps:

* Limited AWS exposure
* No CI/CD experience

Recommendation:
Strong Hire

---

# 4. Conversational Recruitment

The system supports natural language recruiter queries.

Examples:

## Query 1

Find Python developers with FastAPI and Docker.

---

## Query 2

Compare top 3 candidates.

---

## Query 3

Why did Candidate A rank higher than Candidate B?

---

## Query 4

Generate interview questions for candidate 2.

---

# 5. Streaming Workflow

The Streamlit UI streams LangGraph execution in real time.

This helps visualize:

* node execution
* workflow progress
* reasoning flow

---

# Technologies Used

| Technology             | Purpose                      |
| ---------------------- | ---------------------------- |
| LangGraph              | Agent workflow orchestration |
| LangChain              | LLM framework                |
| ChromaDB               | Vector database              |
| BM25                   | Keyword retrieval            |
| HuggingFace Embeddings | Semantic embeddings          |
| Groq LLM               | Fast inference               |
| Streamlit              | UI                           |
| Python                 | Backend                      |

---

# Folder Structure

resume_matching_agent/

├── app.py
├── agent.py
├── rag.py
├── tools.py
├── config.py
├── app_state.py
├── requirements.txt
├── chroma_db/
└── resumes/

---

# File Explanations

# app.py

Main Streamlit UI.

Responsibilities:

* takes recruiter input
* streams workflow execution
* displays reports
* handles feedback

---

# agent.py

Contains:

* LangGraph workflow
* state management
* nodes
* conditional edges
* memory checkpointing

---

# rag.py

Contains:

* file loaders
* text chunking
* embedding generation
* ChromaDB creation
* BM25 indexing
* hybrid retrieval logic

---

# tools.py

Contains LLM-based tools:

* extract requirements
* rank candidates
* compare candidates
* generate interview questions

---

# config.py

Contains:

* API key setup
* LLM initialization
* embedding model initialization

---

# app_state.py

Initializes:

* vector database
* BM25
* document loading

---

# Hybrid Retrieval Explanation

# Semantic Retrieval

```python
retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": top_k}
)
```

Uses:

* Maximum Marginal Relevance (MMR)

Benefits:

* diversity in results
* reduced duplicate retrieval
* better coverage

---

# Keyword Retrieval

```python
keyword_scores = bm25.get_scores(
    job_description.lower().split()
)
```

Ranks documents using:

* exact keyword relevance

---

# Merging Results

```python
merged = semantic_docs + keyword_docs
```

Combines:

* semantic understanding
* keyword precision

---

# LangGraph State

The system maintains workflow state using TypedDict.

Example:

```python
class AgentState(TypedDict):

    messages: List[str]

    job_description: str

    requirements: str

    retrieved_candidates: List

    shortlisted_candidates: List

    ranked_candidates: str

    report: str

    feedback: str
```

This allows:

* memory persistence
* state transitions
* iterative refinement

---

# Conditional Edges

The workflow supports dynamic reranking.

Example:

```python
def should_rerank(state):

    feedback = state.get("feedback", "")

    if feedback.strip():
        return "Rank Candidates"

    return END
```

If recruiter provides feedback:

* workflow loops back
* candidates are reranked

This demonstrates:

* real agentic behavior
* adaptive workflows

---

# Memory Checkpointing

Uses:

```python
MemorySaver()
```

Benefits:

* preserves conversation state
* enables iterative interactions
* supports conversational workflows

---

# Installation

# 1. Create Virtual Environment

Windows:

```bash
python -m venv .venv
```

Activate:

```bash
.venv\Scripts\activate
```

Linux/Mac:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

# 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 3. Install PyTorch

CPU Version:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

GPU Version:
[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

---

# Add API Key

Open:

```python
config.py
```

Replace:

```python
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"
```

with your actual Groq API key.

---

# Add Resumes

Create folder:

```bash
resumes/
```

Add:

* PDF resumes
* DOCX resumes
* TXT resumes

---

# Run Application

```bash
streamlit run app.py
```

---

# Example Workflow

Recruiter enters:

```text
Looking for Python Backend Developer
with FastAPI, Docker, Kubernetes,
and AWS experience.
```

System:

1. extracts skills
2. retrieves resumes
3. ranks candidates
4. generates explanations
5. supports feedback reranking

---

# Example Feedback Loop

Recruiter:

```text
Prioritize Kubernetes more heavily.
```

Agent:

* re-ranks candidates
* updates scores
* explains ranking changes

---

# Example Interview Question Generation

The system can generate:

* technical questions
* behavioral questions
* project-based questions

based on candidate resumes.

---

# Future Improvements

Possible enhancements:

* Multi-agent architecture
* Resume upload UI
* PostgreSQL integration
* Recruiter authentication
* Email scheduling
* Candidate tracking dashboard
* Fine-tuned ranking models
* Long-term recruiter memory

---

# Conclusion

This project demonstrates:

* Agentic AI workflows
* Hybrid RAG architecture
* Conversational AI
* Explainable AI
* Human-in-the-loop systems
* Real-world ATS automation

The system combines retrieval, reasoning, ranking, explainability, and iterative refinement into a production-style recruitment AI agent.

---

# Author

Poorna Chandra S

```
```
