from config import llm


def extract_requirements(jd: str):

    prompt = f"""
    Extract:

    1. Must-have skills
    2. Nice-to-have skills
    3. Experience required

    JOB DESCRIPTION:
    {jd}

    Return concise JSON.
    """

    response = llm.invoke(prompt)

    return response.content


def rank_candidates(jd, candidates):

    candidate_text = "\n\n".join([
        c.page_content
        for c in candidates
    ])

    prompt = f"""
    You are an expert ATS recruiter.

    Analyze and rank candidates.

    JOB DESCRIPTION:
    {jd}

    CANDIDATES:
    {candidate_text}

    Return:

    - Candidate Name
    - Match Score
    - Strengths
    - Gaps
    - Hire Recommendation
    """

    response = llm.invoke(prompt)

    return response.content


def compare_candidates(candidate_data):

    prompt = f"""
    Compare these candidates.

    Include:
    - Skills
    - Experience
    - Strengths
    - Weaknesses
    - Final Verdict

    CANDIDATES:
    {candidate_data}
    """

    response = llm.invoke(prompt)

    return response.content


def generate_interview_questions(candidate):

    prompt = f"""
    Generate interview questions.

    Candidate:
    {candidate}
    """

    response = llm.invoke(prompt)

    return response.content
