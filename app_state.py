from rag import build_rag

vectordb, bm25, docs = build_rag(
    resume_folder="resumes"
)
