"""
Module 6 Solution — Eval Suite Design
Exercise: 5 eval tasks + Cohen's kappa implementation.
"""
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def pass_at_k(p: float, k: int) -> float:
    return 1.0 - (1.0 - p) ** k

def pass_all_k(p: float, k: int) -> float:
    return p ** k


@dataclass
class EvalTask:
    task_id: str
    prompt: str
    reference: str
    grader: str   # "code" | "model" | "hybrid"
    partial_credit: str = ""  # criteria for scores between 0 and 1


def code_grader(prediction: str, reference: str) -> float:
    pred_lower = prediction.lower()
    keywords = [w for w in reference.lower().split() if len(w) > 4]
    if not keywords:
        return 0.0
    return sum(1 for k in keywords if k in pred_lower) / len(keywords)


def model_grader(prediction: str, reference: str, task_prompt: str) -> float:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": (
                "Score the prediction on faithfulness (does it match the reference?) "
                "and coverage (does it include all key facts?). "
                "Return JSON: {faithfulness: 0.0-1.0, coverage: 0.0-1.0, explanation: str}"
            )},
            {"role": "user", "content": f"Task: {task_prompt}\nReference: {reference}\nPrediction: {prediction}"}
        ],
        response_format={"type": "json_object"}
    )
    result = json.loads(response.choices[0].message.content)
    return (result["faithfulness"] + result["coverage"]) / 2.0


class EvalSuite:
    def __init__(self, tasks: list[EvalTask], agent_fn: Callable[[str], str]):
        self.tasks = tasks
        self.agent_fn = agent_fn

    def run_task(self, task: EvalTask, trial: int) -> dict:
        start = time.time()
        prediction = self.agent_fn(task.prompt)
        elapsed = time.time() - start

        if task.grader == "code":
            score = code_grader(prediction, task.reference)
        elif task.grader == "model":
            score = model_grader(prediction, task.reference, task.prompt)
        else:
            score = 0.4 * code_grader(prediction, task.reference) + \
                    0.6 * model_grader(prediction, task.reference, task.prompt)

        return {"task_id": task.task_id, "trial": trial,
                "score": score, "pass": score >= 0.7,
                "elapsed": elapsed, "prediction": prediction[:200]}

    def run(self, k: int = 3) -> dict:
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(self.run_task, t, i) for t in self.tasks for i in range(k)]
            all_results = [f.result() for f in futures]

        summary = {}
        for task in self.tasks:
            results = [r for r in all_results if r["task_id"] == task.task_id]
            passes = [r["pass"] for r in results]
            p_hat = sum(passes) / len(passes)
            summary[task.task_id] = {
                "p_hat": p_hat,
                "pass_at_k": pass_at_k(p_hat, k),
                "pass_all_k": pass_all_k(p_hat, k),
                "trials": passes,
            }
        return summary


# ── Exercise: 5 eval tasks for the document research agent ───────────────────

MY_TASKS = [
    EvalTask(
        task_id="t1",
        prompt="What is Byte-Pair Encoding and how is it used in GPT tokenizers?",
        reference="Byte-Pair Encoding BPE merges frequent character pairs iteratively to build vocabulary subword tokens GPT cl100k_base",
        grader="hybrid",
        partial_credit="Score 0.5 if BPE is defined correctly but GPT connection missing.",
    ),
    EvalTask(
        task_id="t2",
        prompt="How does Retrieval-Augmented Generation (RAG) reduce hallucinations?",
        reference="RAG retrieval augmented generation grounds answers in retrieved documents reduces hallucinations factual accuracy",
        grader="model",
        partial_credit="Score 0.5 if RAG is described but hallucination reduction mechanism not explained.",
    ),
    EvalTask(
        task_id="t3",
        prompt="What is the mathematical formula for scaled dot-product attention?",
        reference="Attention softmax query key value QKV sqrt dimension d_k scale dot product",
        grader="hybrid",
        partial_credit="Score 0.5 if QKV described without the sqrt(d_k) scaling.",
    ),
    EvalTask(
        task_id="t4",
        prompt="What optimizer and learning rate schedule is standard for BERT fine-tuning?",
        reference="AdamW optimizer learning rate 2e-5 linear warmup weight decay fine-tuning BERT",
        grader="code",
        partial_credit="Score 0.5 if AdamW named without lr or warmup details.",
    ),
    EvalTask(
        task_id="t5",
        prompt="Explain what multi-head attention adds over single-head attention.",
        reference="multi-head attention parallel heads different subspaces representation capture diverse patterns position",
        grader="model",
        partial_credit="Score 0.5 if multiple heads mentioned without explaining why different subspaces matter.",
    ),
]


# ── Cohen's kappa ─────────────────────────────────────────────────────────────

def cohens_kappa(grader_scores: list[float], human_scores: list[float], threshold: float = 0.7) -> float:
    """
    Measure inter-rater agreement between LLM grader and human labels.
    Both lists converted to binary (pass/fail) at threshold.

    κ = (P_o - P_e) / (1 - P_e)
    P_o = observed agreement, P_e = expected agreement by chance.
    κ < 0: worse than chance | 0.6-0.8: substantial | >0.8: almost perfect
    """
    assert len(grader_scores) == len(human_scores), "Lists must be same length"
    n = len(grader_scores)

    g_bin = [1 if s >= threshold else 0 for s in grader_scores]
    h_bin = [1 if s >= threshold else 0 for s in human_scores]

    # Observed agreement
    p_o = sum(g == h for g, h in zip(g_bin, h_bin)) / n

    # Expected agreement by chance
    p_g1 = sum(g_bin) / n   # grader positive rate
    p_h1 = sum(h_bin) / n   # human positive rate
    p_e = p_g1 * p_h1 + (1 - p_g1) * (1 - p_h1)

    if p_e == 1.0:
        return 1.0  # perfect agreement by definition
    return (p_o - p_e) / (1 - p_e)


if __name__ == "__main__":
    p, k = 0.8, 3
    print(f"For p={p}, k={k}:")
    print(f"  pass@{k} = {pass_at_k(p, k):.3f}")
    print(f"  pass^{k} = {pass_all_k(p, k):.3f}")
    print(f"  Interpretation: nearly certain to solve at least once ({pass_at_k(p,k)*100:.0f}%),")
    print(f"  but only {pass_all_k(p,k)*100:.0f}% chance all 3 trials pass (reliability signal)\n")

    print("Tasks defined:", [t.task_id for t in MY_TASKS])

    # Cohen's kappa demo
    grader = [0.9, 0.4, 0.8, 0.3, 0.75]
    human  = [0.85, 0.5, 0.9, 0.2, 0.8]
    kappa = cohens_kappa(grader, human)
    print(f"\nCohen's kappa (grader vs human): {kappa:.3f}")
    print("  > 0.6 = substantial agreement; > 0.8 = almost perfect")
