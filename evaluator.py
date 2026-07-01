import os
import math
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import numpy as np

load_dotenv_called = False

def get_llm():
    from dotenv import load_dotenv
    load_dotenv()
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# ── Faithfulness ────────────────────────────────────────────
def score_faithfulness(answer: str, contexts: list[str]) -> float:
    llm = get_llm()
    context_text = "\n\n".join(contexts)
    prompt = f"""You are an evaluation judge.

Given the CONTEXT and the ANSWER, rate how faithful the answer is to the context.
Faithful means every claim in the answer is supported by the context.

CONTEXT:
{context_text}

ANSWER:
{answer}

Return ONLY a number between 0.0 and 1.0 where:
1.0 = completely faithful, every claim supported
0.5 = partially faithful, some claims unsupported
0.0 = not faithful, answer contradicts or ignores context

Return ONLY the number, nothing else."""
    response = llm.invoke(prompt)
    try:
        score = float(response.content.strip())
        return min(max(score, 0.0), 1.0)
    except:
        return 0.0

# ── Answer Relevancy ────────────────────────────────────────
def score_answer_relevancy(question: str, answer: str) -> float:
    llm = get_llm()
    prompt = f"""You are an evaluation judge.

Rate how well the ANSWER addresses the QUESTION.

QUESTION: {question}
ANSWER: {answer}

Return ONLY a number between 0.0 and 1.0 where:
1.0 = answer directly and completely addresses the question
0.5 = answer partially addresses the question
0.0 = answer does not address the question at all

Return ONLY the number, nothing else."""
    response = llm.invoke(prompt)
    try:
        score = float(response.content.strip())
        return min(max(score, 0.0), 1.0)
    except:
        return 0.0

# ── Context Precision ───────────────────────────────────────
def score_context_precision(question: str, contexts: list[str]) -> float:
    llm = get_llm()
    relevant = 0
    for ctx in contexts:
        prompt = f"""Is the following CONTEXT relevant to answering the QUESTION?

QUESTION: {question}
CONTEXT: {ctx}

Return ONLY yes or no."""
        response = llm.invoke(prompt)
        if "yes" in response.content.lower():
            relevant += 1
    return relevant / len(contexts) if contexts else 0.0

# ── Context Recall ──────────────────────────────────────────
def score_context_recall(answer: str, contexts: list[str], ground_truth: str) -> float:
    llm = get_llm()
    context_text = "\n\n".join(contexts)
    prompt = f"""You are an evaluation judge.

Given the GROUND TRUTH answer and the RETRIEVED CONTEXT, rate how much of the
ground truth information is covered by the context.

GROUND TRUTH: {ground_truth}
RETRIEVED CONTEXT: {context_text}

Return ONLY a number between 0.0 and 1.0 where:
1.0 = all ground truth information is present in context
0.5 = some ground truth information is present
0.0 = ground truth information is not in context

Return ONLY the number, nothing else."""
    response = llm.invoke(prompt)
    try:
        score = float(response.content.strip())
        return min(max(score, 0.0), 1.0)
    except:
        return 0.0

# ── Run full evaluation ─────────────────────────────────────
def run_evaluation(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str] = None
) -> dict:
    results = {}

    # Average across all questions
    faithfulness_scores = []
    relevancy_scores = []
    precision_scores = []
    recall_scores = []

    for i, (q, a, ctx) in enumerate(zip(questions, answers, contexts)):
        faithfulness_scores.append(score_faithfulness(a, ctx))
        relevancy_scores.append(score_answer_relevancy(q, a))
        precision_scores.append(score_context_precision(q, ctx))
        if ground_truths and i < len(ground_truths):
            recall_scores.append(score_context_recall(a, ctx, ground_truths[i]))

    results["faithfulness"] = round(np.mean(faithfulness_scores), 3)
    results["answer_relevancy"] = round(np.mean(relevancy_scores), 3)
    results["context_precision"] = round(np.mean(precision_scores), 3)
    if recall_scores:
        results["context_recall"] = round(np.mean(recall_scores), 3)

    return results

# ── Pretty print results ────────────────────────────────────
def format_results(results: dict) -> str:
    import math
    lines = ["── RAG Evaluation Results ──────────────────"]
    for metric, score in results.items():
        if isinstance(score, float) and not math.isnan(score):
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"{metric:<22} {bar}  {score:.3f}")
        elif isinstance(score, float) and math.isnan(score):
            lines.append(f"{metric:<22} {'░' * 20}  N/A")
    lines.append("────────────────────────────────────────────")
    return "\n".join(lines)