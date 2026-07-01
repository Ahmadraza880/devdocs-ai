import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

# ── Ragas LLM + Embeddings setup ───────────────────────────
def get_ragas_llm():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )
    return LangchainLLMWrapper(llm)

def get_ragas_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return LangchainEmbeddingsWrapper(embeddings)

def build_eval_dataset(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str] = None
) -> Dataset:
    data = {
        "user_input": questions,        # ← renamed from "question"
        "response": answers,            # ← renamed from "answer"
        "retrieved_contexts": contexts, # ← renamed from "contexts"
    }
    if ground_truths:
        data["reference"] = ground_truths  # ← renamed from "ground_truth"

    return Dataset.from_dict(data)


def run_evaluation(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str] = None
) -> dict:

    dataset = build_eval_dataset(questions, answers, contexts, ground_truths)

    ragas_llm = get_ragas_llm()
    ragas_emb = get_ragas_embeddings()

    if ground_truths:
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
    else:
        metrics = [
            faithfulness,
            answer_relevancy,
        ]

    # Explicitly assign both llm and embeddings to every metric
    for metric in metrics:
        metric.llm = ragas_llm
        metric.embeddings = ragas_emb  # assign to ALL metrics not just hasattr check

    results = evaluate(dataset=dataset, metrics=metrics)
    return results

# ── Pretty print results ───────────────────────────────────
def format_results(results: dict) -> str:
    import math
    lines = ["── Ragas Evaluation Results ──────────────────"]
    for metric, score in results.items():
        if isinstance(score, float) and not math.isnan(score):
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"{metric:<22} {bar}  {score:.3f}")
        elif isinstance(score, float) and math.isnan(score):
            lines.append(f"{metric:<22} {'░' * 20}  N/A")
    lines.append("──────────────────────────────────────────────")
    return "\n".join(lines)