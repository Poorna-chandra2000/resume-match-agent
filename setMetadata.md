Let's put everything into one single, complete, copy-and-paste script.

The trick is that we use **`text_splitter.create_documents()`**. This single command takes your raw text and your metadata, chops the text into pieces using your separators, and attaches the metadata to every single chunk automatically.

Here is the complete code:

```python
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Define your text splitter with your custom separators
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=250,
    separators=[
        "\nEducation",
        "\nExperience",
        "\nSkills",
        "\nProjects",
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

# Dummy load_file function just to make the code runnable for you
def load_file(file_path):
    # In your real code, this reads the PDF/Word/Txt file
    return f"This is the text content of {file_path}. Experience: Python Developer."


def build_rag(resume_folder="resumes"):
    docs = []

    # Find all files in the folder
    files = glob.glob(f"{resume_folder}/*")

    for file in files:
        lower = file.lower()

        # Check for valid file extensions
        if lower.endswith((".pdf", ".docx", ".txt")):
            
            # Read the text from the file
            text = load_file(file)

            if not text:
                continue

            # --- THIS IS THE EXACT LINE THAT DOES BOTH AT ONCE ---
            # It chunks the text AND assigns the metadata dictionary to every chunk.
            file_chunks = text_splitter.create_documents(
                texts=[text], 
                metadatas=[{"source": file}]
            )
            
            # Add these packaged chunks to our main master list
            docs.extend(file_chunks)
            
    return docs


# --- HOW TO RUN IT AND SEE THE RESULT ---
if __name__ == "__main__":
    # Create the data
    final_documents = build_rag(resume_folder="resumes")
    
    # Loop through the results to see your text and metadata together
    for chunk in final_documents:
        print("--- NEW CHUNK ---")
        print(f"TEXT CONTENT:\n{chunk.page_content}")
        print(f"METADATA ATTACHED:\n{chunk.metadata}")
        print("-" * 20)

```

### What happens when this runs:

Instead of just getting back plain strings, `final_documents` will contain LangChain Document objects. If you inspect any chunk inside that list, it will perfectly store both your split text and your filterable metadata like this:

* **`chunk.page_content`** $\rightarrow$ `"This is the text content of resumes/john_doe.pdf..."`
* **`chunk.metadata`** $\rightarrow$ `{"source": "resumes/john_doe.pdf"}`


Yes, **according to the official, latest LangChain documentation**, this is exactly how it is designed to work.

The `create_documents()` method is the official, built-in way to propagate custom metadata alongside text strings during the chunking phase.

Here is the exact method signature from the official LangChain API Reference:

```python
create_documents(
    texts: list[str], 
    metadatas: list[dict] | None = None
) -> list[Document]

```

### The 2 Rules of the Latest Doc for `create_documents`:

1. `texts` **must** be passed as a Python list of strings (e.g., `[text]`).
2. `metadatas` **must** be passed as a Python list of dictionaries (e.g., `[{"source": file}]`).

Because both parameters expect a **list**, LangChain matches them up by index. The first text in your list gets assigned the first metadata dictionary in your list.

When `text_splitter` slices up that text into 10 smaller pieces, it copies that exact metadata dictionary to all 10 created chunks. This ensures every individual piece is correctly tagged before it heads into your RAG database.