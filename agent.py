from typing import TypedDict, List

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from rag import hybrid_search

from tools import (
    extract_requirements,
    rank_candidates
)


class AgentState(TypedDict):

    messages: List[str]
    job_description: str
    requirements: str
    retrieved_candidates: List
    shortlisted_candidates: List
    ranked_candidates: str
    report: str
    feedback: str


def parse_jd_node(state):

    state["messages"].append(
        "JD Parsed"
    )

    return state


def extract_requirements_node(state):

    result = extract_requirements(
        state["job_description"]
    )

    state["requirements"] = result

    state["messages"].append(
        "Requirements Extracted"
    )

    return state


def search_resumes_node(state):

    from app_state import (
        vectordb,
        bm25,
        docs
    )

    candidates = hybrid_search(
        vectordb,
        bm25,
        docs,
        state["job_description"],
        top_k=100
    )

    state["retrieved_candidates"] = candidates

    state["shortlisted_candidates"] = candidates[:10]

    state["messages"].append(
        f"{len(candidates)} candidates retrieved"
    )

    return state


def rank_candidates_node(state):

    ranked = rank_candidates(
        state["job_description"],
        state["shortlisted_candidates"]
    )

    state["ranked_candidates"] = ranked

    state["messages"].append(
        "Candidates Ranked"
    )

    return state


def generate_report_node(state):

    report = f"""
    FINAL REPORT
    ======================

    {state["ranked_candidates"]}
    """

    state["report"] = report

    state["messages"].append(
        "Report Generated"
    )

    return state


def human_feedback_node(state):

    feedback = state.get(
        "feedback",
        ""
    )

    if feedback:

        state["messages"].append(
            f"Feedback Applied: {feedback}"
        )

    return state


def should_rerank(state):

    feedback = state.get(
        "feedback",
        ""
    )

    if feedback.strip():
        return "Rank Candidates"

    return END


graph = StateGraph(AgentState)

graph.add_node(
    "Parse JD",
    parse_jd_node
)

graph.add_node(
    "Extract Requirements",
    extract_requirements_node
)

graph.add_node(
    "Search Resumes",
    search_resumes_node
)

graph.add_node(
    "Rank Candidates",
    rank_candidates_node
)

graph.add_node(
    "Generate Report",
    generate_report_node
)

graph.add_node(
    "Human Feedback",
    human_feedback_node
)

graph.set_entry_point(
    "Parse JD"
)

graph.add_edge(
    "Parse JD",
    "Extract Requirements"
)

graph.add_edge(
    "Extract Requirements",
    "Search Resumes"
)

graph.add_edge(
    "Search Resumes",
    "Rank Candidates"
)

graph.add_edge(
    "Rank Candidates",
    "Generate Report"
)

graph.add_edge(
    "Generate Report",
    "Human Feedback"
)

graph.add_conditional_edges(
    "Human Feedback",
    should_rerank,
    {
        "Rank Candidates":
        "Rank Candidates",

        END: END
    }
)

memory = MemorySaver()

app = graph.compile(
    checkpointer=memory
)
