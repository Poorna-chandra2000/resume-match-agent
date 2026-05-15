import streamlit as st

from agent import app

from tools import (
    compare_candidates,
    generate_interview_questions
)

st.set_page_config(
    page_title="ATS Resume Agent",
    layout="wide"
)

st.title(
    "AI Resume Matching Agent"
)

job_description = st.text_area(
    "Enter Job Description",
    height=250
)

feedback = st.text_input(
    "Feedback / Re-ranking"
)

if st.button("Run Agent"):

    state = {

        "messages": [],

        "job_description":
        job_description,

        "requirements": "",

        "retrieved_candidates": [],

        "shortlisted_candidates": [],

        "ranked_candidates": "",

        "report": "",

        "feedback": feedback
    }

    config = {
        "configurable": {
            "thread_id":
            "resume-thread"
        }
    }

    st.subheader(
        "Workflow Streaming"
    )

    workflow_box = st.empty()

    logs = []

    for event in app.stream(
        state,
        config=config
    ):

        logs.append(str(event))

        workflow_box.code(
            "\n\n".join(logs)
        )

    result = app.invoke(
        state,
        config=config
    )

    st.subheader(
        "Final Report"
    )

    st.write(result["report"])

    st.subheader(
        "Candidate Comparison"
    )

    comparison = compare_candidates(
        result["ranked_candidates"]
    )

    st.write(comparison)

    st.subheader(
        "Interview Questions"
    )

    questions = (
        generate_interview_questions(
            result["ranked_candidates"]
        )
    )

    st.write(questions)
