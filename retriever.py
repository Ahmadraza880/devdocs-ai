import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.schema import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np

load_dotenv()

# ── Constants ──────────────────────────────────────────────
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHROMA_DIR = "./chroma_db"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K_RETRIEVE = 10   # fetch more, rerank down to TOP_K_FINAL
TOP_K_FINAL = 4

# ── Load & chunk PDFs ──────────────────────────────────────
def load_documents(pdf_paths: list[str]) -> list[Document]:
    docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)

# ── Build vector store ─────────────────────────────────────
def build_vectorstore(chunks: list[Document]) -> Chroma:
    # Remove empty chunks
    chunks = [c for c in chunks if c.page_content.strip()]
    if not chunks:
        raise ValueError("No valid chunks found in documents.")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    return vectorstore

# ── Load existing vector store ─────────────────────────────
def load_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

# ── BM25 keyword search ────────────────────────────────────
def bm25_search(query: str, chunks: list[Document], top_k: int) -> list[Document]:
    tokenized_corpus = [doc.page_content.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]

# ── Hybrid search (BM25 + vector) ─────────────────────────
def hybrid_search(
    query: str,
    vectorstore: Chroma,
    chunks: list[Document],
    top_k: int = TOP_K_RETRIEVE
) -> list[Document]:
    # Vector search
    vector_results = vectorstore.similarity_search(query, k=top_k)
    # BM25 search
    bm25_results = bm25_search(query, chunks, top_k)
    # Merge & deduplicate by page_content
    seen = set()
    merged = []
    for doc in vector_results + bm25_results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            merged.append(doc)
    return merged

# ── Reranker ───────────────────────────────────────────────
def rerank(query: str, docs: list[Document], top_k: int = TOP_K_FINAL) -> list[Document]:
    reranker = CrossEncoder(RERANK_MODEL)
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]

# ── Full retrieval pipeline ────────────────────────────────
def retrieve(query: str, vectorstore: Chroma, chunks: list[Document]) -> list[Document]:
    candidates = hybrid_search(query, vectorstore, chunks)
    return rerank(query, candidates)

# ── LLM answer generation ──────────────────────────────────
def generate_answer(query: str, context_docs: list[Document]) -> str:
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )
    context = "\n\n".join([
        f"[Source: {doc.metadata.get('source', 'unknown')} | Page: {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in context_docs
    ])
    prompt = f"""You are a technical documentation assistant.
Answer the question using ONLY the provided context.
If the answer is not in the context, say "I couldn't find this in the provided docs."

Context:
{context}

Question: {query}

Answer:"""
    response = llm.invoke(prompt)
    return response.content