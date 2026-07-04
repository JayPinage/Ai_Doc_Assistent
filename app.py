import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
""",
        ),
    ]
)

summary_prompt = ChatPromptTemplate.from_template(
    """
You are an enterprise document intelligence assistant.

Summarize the following part of a document.

Rules:
- Keep only the important information.
- Remove repetition.
- Do not hallucinate.
- Maximum 10 bullet points.

Document Part:
{text}
"""
)

reduce_prompt = ChatPromptTemplate.from_template(
    """
You are an enterprise document intelligence assistant.

Below are summaries of different parts of the same document.

Merge them into ONE professional summary.

Format:

# 📄 Document Summary

## Overview

## Main Topics

## Key Points

## Important Findings

## Final Conclusion

Summaries:
{text}
"""
)

model1 = ChatMistralAI(model_name="mistral-large-latest", temperature=0.2)
model2 = ChatMistralAI(model_name="mistral-small-latest", temperature=0.2)

map_chain = summary_prompt | model2
reduce_chain = reduce_prompt | model2


# load files

def load_file(file_path: str):
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    elif ext in (".pptx", ".ppt"):
        loader = UnstructuredPowerPointLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return loader.load()


def build_pipeline(uploaded_files, persist_directory):
    """Runs the same load -> chunk -> embed -> store -> summarize pipeline."""

    all_docs = []
    tmp_dir = tempfile.mkdtemp()

    for uploaded_file in uploaded_files:
        tmp_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        all_docs.extend(load_file(tmp_path))

    # chunk data
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split = splitter.split_documents(all_docs)

    # embed data
    embeddings = MistralAIEmbeddings(model="mistral-embed")

    # store data
    vector_store = Chroma.from_documents(
        documents=split,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    # retrieve data
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3},
    )

    # ---------------- STEP 1: map ----------------
    batch_size = 10
    chunk_summaries = []

    progress = st.progress(0, text="Generating summaries...")
    total_batches = max(1, (len(split) + batch_size - 1) // batch_size)

    for idx, i in enumerate(range(0, len(split), batch_size)):
        batch = "\n\n".join(chunk.page_content for chunk in split[i:i + batch_size])
        result = map_chain.invoke({"text": batch})
        chunk_summaries.append(result.content)
        progress.progress((idx + 1) / total_batches, text=f"Summarizing batch {idx + 1}/{total_batches}")

    # ---------------- STEP 2: reduce ----------------
    while len(chunk_summaries) > 1:
        merged = []
        for i in range(0, len(chunk_summaries), batch_size):
            batch = "\n\n".join(chunk_summaries[i:i + batch_size])
            result = reduce_chain.invoke({"text": batch})
            merged.append(result.content)
        chunk_summaries = merged

    progress.empty()
    final_summary = chunk_summaries[0]

    return retriever, final_summary

# UI

st.set_page_config(page_title="Document Intelligence RAG", page_icon="📄", layout="wide")

st.title("📄 Document Intelligence & RAG Chat")
st.caption("Upload PDF, DOCX, TXT, or PPTX files — get a summary and chat with your documents.")

if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files (PDF, DOCX, TXT, PPTX)",
        type=["pdf", "docx", "txt", "pptx", "ppt"],
        accept_multiple_files=True,
    )

    process_btn = st.button("Process Documents", type="primary", disabled=not uploaded_files)

    if process_btn and uploaded_files:
        with st.spinner("Loading, chunking, and embedding documents..."):
            persist_directory = os.path.join(tempfile.mkdtemp(), "chroma-db")
            retriever, final_summary = build_pipeline(uploaded_files, persist_directory)
            st.session_state.retriever = retriever
            st.session_state.summary = final_summary
            st.session_state.messages = []
        st.success("Documents processed! Summary and chat are ready.")

    st.divider()
    if st.button("Reset session"):
        st.session_state.retriever = None
        st.session_state.summary = None
        st.session_state.messages = []
        st.rerun()

tab_summary, tab_chat = st.tabs(["📋 Summary", "💬 Chat with Documents"])

with tab_summary:
    if st.session_state.summary:
        st.markdown(st.session_state.summary)
    else:
        st.info("Upload document(s) and click **Process Documents** to generate a summary.")

with tab_chat:
    if not st.session_state.retriever:
        st.info("Upload document(s) and click **Process Documents** to start chatting.")
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        query = st.chat_input("Ask a question about your documents...")

        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    get_data = st.session_state.retriever.invoke(query)
                    context = "\n\n".join([doc.page_content for doc in get_data])

                    final_prompt = prompt.invoke({"context": context, "question": query})
                    response = model1.invoke(final_prompt)

                    st.markdown(response.content)

            st.session_state.messages.append({"role": "assistant", "content": response.content})