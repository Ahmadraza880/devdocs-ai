import os
import tempfile
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from retriever import (
    load_documents,
    build_vectorstore,
    retrieve,
    generate_answer,
)
from evaluator import run_evaluation, format_results

load_dotenv()

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="DevDocs AI",
    page_icon="📚",
    layout="wide"
)

# ── Session state init ─────────────────────────────────────
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 DevDocs AI")
    st.caption("Production RAG with Hybrid Search + Reranking")
    st.divider()

    st.subheader("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF docs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process Documents", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                tmp_paths = []
                for f in uploaded_files:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    tmp.write(f.read())
                    tmp.flush()
                    tmp.close() 
                    tmp_paths.append(tmp.name)

                chunks = load_documents(tmp_paths)
                vectorstore = build_vectorstore(chunks)

                st.session_state.chunks = chunks
                st.session_state.vectorstore = vectorstore
                st.session_state.chat_history = []
                st.session_state.eval_results = None

                # cleanup temp files
                for p in tmp_paths:
                    os.unlink(p)

            st.success(f"✅ {len(chunks)} chunks indexed from {len(uploaded_files)} file(s)")

    st.divider()

    # Eval panel
    if st.session_state.vectorstore:
        st.subheader("2. Run Evaluation")
        st.caption("Test your RAG pipeline quality")

        eval_question = st.text_input(
            "Eval question",
            placeholder="What is the purpose of this API?"
        )
        eval_ground_truth = st.text_input(
            "Ground truth (optional)",
            placeholder="Expected answer..."
        )

        if st.button("Run Ragas Eval", use_container_width=True):
            if eval_question:
                with st.spinner("Evaluating..."):
                    context_docs = retrieve(
                        eval_question,
                        st.session_state.vectorstore,
                        st.session_state.chunks
                    )
                    answer = generate_answer(eval_question, context_docs)
                    contexts = [[doc.page_content for doc in context_docs]]

                    ground_truths = [eval_ground_truth] if eval_ground_truth else None

                    results = run_evaluation(
                        questions=[eval_question],
                        answers=[answer],
                        contexts=contexts,
                        ground_truths=ground_truths
                    )
                    st.session_state.eval_results = results
            else:
                st.warning("Enter an eval question first.")

    st.divider()
    st.caption("Built by Ahmad Raza · github.com/Ahmadraza880")

# ── Main area ──────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Ask Your Docs")

    if not st.session_state.vectorstore:
        st.info("⬅️ Upload and process your PDF documents to get started.")
    else:
        # Chat history display
        for turn in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(turn["question"])
            with st.chat_message("assistant"):
                st.write(turn["answer"])
                with st.expander("📄 Sources used"):
                    for i, src in enumerate(turn["sources"], 1):
                        st.caption(
                            f"**[{i}]** {src['source']} — Page {src['page']}"
                        )
                        st.text(src["content"][:300] + "...")

        # Query input
        query = st.chat_input("Ask anything about your docs...")
        if query:
            with st.spinner("Retrieving and generating..."):
                context_docs = retrieve(
                    query,
                    st.session_state.vectorstore,
                    st.session_state.chunks
                )
                answer = generate_answer(query, context_docs)

                sources = [
                    {
                        "source": doc.metadata.get("source", "unknown"),
                        "page": doc.metadata.get("page", "?"),
                        "content": doc.page_content
                    }
                    for doc in context_docs
                ]

                st.session_state.chat_history.append({
                    "question": query,
                    "answer": answer,
                    "sources": sources
                })
                st.rerun()

with col2:
    st.header("📊 Eval Dashboard")

    if st.session_state.eval_results:
        results_dict = st.session_state.eval_results
        st.success("Evaluation complete")

        metrics_map = {
            "faithfulness": "Faithfulness",
            "answer_relevancy": "Answer Relevancy",
            "context_precision": "Context Precision",
            "context_recall": "Context Recall",
        }

        for key, label in metrics_map.items():
            if key in results_dict:
                score = results_dict[key]
                if score is None or (isinstance(score, float) and np.isnan(score)):
                    st.metric(label=label, value="N/A")
                    st.progress(0.0)
                else:
                    score = float(score)
                    st.metric(label=label, value=f"{score:.3f}")
                    st.progress(min(max(score, 0.0), 1.0))

        st.divider()
        st.code(format_results(results_dict), language="text")
    else:
        st.info("Run an evaluation from the sidebar to see scores here.")

        st.divider()
        st.subheader("What these metrics mean")
        st.markdown("""
**Faithfulness** — Is the answer grounded in the retrieved context? No hallucinations?

**Answer Relevancy** — Does the answer actually address the question?

**Context Precision** — Are the retrieved chunks relevant to the question?

**Context Recall** — Did we retrieve all the chunks needed to answer?

*Target: all scores above 0.7*
""")