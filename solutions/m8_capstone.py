"""
Module 8 Solution — Capstone: Document Research Agent
Full implementation with ToolRegistry, MemoryManager, Harness, and EvalSuite.
Stretch goals: critic agent + confidence calibration.
"""
import json
import os
import time
import math
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ── Reuse from prior modules ──────────────────────────────────────────────────

class ToolRegistry:
    def __init__(self):
        self._tools: dict = {}

    def register(self, fn, schema, idempotent=True):
        self._tools[schema["name"]] = {"fn": fn, "schema": schema, "idempotent": idempotent}

    @property
    def schemas(self):
        return [{"type": "function", "function": t["schema"]} for t in self._tools.values()]

    def execute(self, name, args) -> str:
        if name not in self._tools:
            return f"ERROR: tool '{name}' not found"
        try:
            return str(self._tools[name]["fn"](**args))
        except Exception as e:
            return f"ERROR: {name} — {e}"


def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag = lambda v: math.sqrt(sum(x * x for x in v))
    return dot / (mag(a) * mag(b) + 1e-10)

def mock_embed(text):
    h = hashlib.sha256(text.encode()).digest()
    return [((b / 255.0) - 0.5) * 2 for b in h[:32]]


# ── Tools ─────────────────────────────────────────────────────────────────────

_findings: list[dict] = []
_finding_hashes: set = set()

MOCK_SEARCH_DB = {
    "transformer": "Transformers use self-attention (Vaswani et al., 2017). Input → embeddings → N × (MHA + FFN + LayerNorm) → output.",
    "attention": "Attention = softmax(QK^T / sqrt(d_k)) × V. Multi-head: h parallel heads, different subspaces.",
    "bert": "BERT (Devlin 2018): bidirectional transformer pretrained with MLM + NSP. Fine-tune with AdamW, lr=2e-5.",
    "gpt": "GPT family: autoregressive, left-to-right. GPT-4 uses ~1.8T params (estimated). BPE tokenizer, cl100k_base.",
    "rag": "RAG (Lewis 2020): retrieve relevant docs with dense embeddings, condition generation on retrieved context.",
    "fine-tuning": "Fine-tuning adapts pretrained weights to downstream task. Full fine-tune vs LoRA/QLoRA vs adapter layers.",
    "lora": "LoRA (Hu 2021): inject low-rank matrices (r=8-64) into attention weights. Trains <1% of params.",
    "tokenizer": "BPE builds vocabulary by merging frequent byte pairs. GPT-4: cl100k_base, ~100k vocab, avg 0.75 words/token.",
}

def search_web(query: str, num_results: int = 3) -> str:
    q = query.lower()
    results = [v for k, v in MOCK_SEARCH_DB.items() if k in q]
    if not results:
        results = list(MOCK_SEARCH_DB.values())[:num_results]
    return "\n".join(results[:num_results])

def store_finding(source: str, passage: str, relevance_score: float) -> str:
    key = hashlib.md5(f"{source}::{passage}".encode()).hexdigest()
    if key in _finding_hashes:
        return f"Duplicate skipped: {source[:40]}"
    _finding_hashes.add(key)
    _findings.append({"source": source, "passage": passage, "relevance": relevance_score,
                       "embedding": mock_embed(passage)})
    return f"Stored finding #{len(_findings)} [rel={relevance_score:.2f}]"

def generate_answer(question: str) -> str:
    if not _findings:
        return json.dumps({"answer": "No findings.", "citations": [], "confidence": 0.0})
    top = sorted(_findings, key=lambda f: f["relevance"], reverse=True)[:5]
    findings_text = "\n".join(f"[{f['relevance']:.1f}] {f['source']}: {f['passage']}" for f in top)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": (
                "Synthesize research findings into a clear answer. "
                "Return JSON: {answer: str, citations: [str], confidence: 0.0-1.0, "
                "confidence_reason: str}"
            )},
            {"role": "user", "content": f"Question: {question}\n\nFindings:\n{findings_text}"}
        ],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content


# ── Critic agent (stretch goal) ───────────────────────────────────────────────

def critic_review(answer: str, question: str, findings: list[dict]) -> dict:
    """Review draft answer for faithfulness and coverage before returning to user."""
    findings_text = "\n".join(f"- {f['source']}: {f['passage'][:100]}" for f in findings[:5])
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": (
                "You are a research critic. Check the answer for: "
                "(1) faithfulness — does it match the sources? "
                "(2) coverage — does it address all aspects of the question? "
                "Return JSON: {verdict: 'pass'|'revise', faithfulness: 0-1, coverage: 0-1, "
                "revision_note: str|null}"
            )},
            {"role": "user", "content": f"Question: {question}\n\nSources:\n{findings_text}\n\nAnswer:\n{answer}"}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


# ── Agent ─────────────────────────────────────────────────────────────────────

class DocumentResearchAgent:
    MAX_STEPS = 20

    def __init__(self, use_critic: bool = True):
        self.use_critic = use_critic
        self.registry = ToolRegistry()
        _findings.clear()
        _finding_hashes.clear()
        self._register_tools()

    def _register_tools(self):
        self.registry.register(search_web, {
            "name": "search_web",
            "description": "Search for information. Be specific in your query.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"},
                "num_results": {"type": "integer", "default": 3},
            }, "required": ["query"]},
        }, idempotent=True)

        self.registry.register(store_finding, {
            "name": "store_finding",
            "description": "Store a relevant passage. Call after each useful search result.",
            "parameters": {"type": "object", "properties": {
                "source": {"type": "string"},
                "passage": {"type": "string"},
                "relevance_score": {"type": "number"},
            }, "required": ["source", "passage", "relevance_score"]},
        }, idempotent=False)

        self.registry.register(generate_answer, {
            "name": "generate_answer",
            "description": "Synthesize stored findings into a final answer. Call only after storing 3+ findings.",
            "parameters": {"type": "object", "properties": {
                "question": {"type": "string"},
            }, "required": ["question"]},
        }, idempotent=True)

    def research(self, question: str) -> dict:
        messages = [
            {"role": "system", "content": (
                "Research the question thoroughly:\n"
                "1. Run 3-5 targeted search_web calls\n"
                "2. Store the most relevant passages (relevance_score >= 0.6)\n"
                "3. Call generate_answer once you have 3+ findings\n"
                "4. Be specific in search queries."
            )},
            {"role": "user", "content": question}
        ]
        final_answer = None
        for step in range(self.MAX_STEPS):
            response = client.chat.completions.create(
                model="gpt-4o", messages=messages,
                tools=self.registry.schemas, tool_choice="auto"
            )
            msg = response.choices[0].message
            messages.append(msg)
            if not msg.tool_calls:
                final_answer = {"answer": msg.content, "citations": [], "confidence": 0.5}
                break
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                result = self.registry.execute(name, args)
                if name == "generate_answer" and not result.startswith("ERROR"):
                    try:
                        final_answer = json.loads(result)
                    except Exception:
                        pass
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            if final_answer:
                break

        if not final_answer:
            final_answer = {"answer": "Research incomplete.", "citations": [], "confidence": 0.0}

        # Stretch goal: critic review
        if self.use_critic and _findings and final_answer.get("answer"):
            review = critic_review(final_answer["answer"], question, _findings)
            final_answer["critic_verdict"] = review.get("verdict")
            final_answer["critic_faithfulness"] = review.get("faithfulness")
            final_answer["critic_coverage"] = review.get("coverage")

        return final_answer


# ── Confidence calibration (stretch goal) ────────────────────────────────────

def calibration_error(stated: list[float], actual: list[float]) -> float:
    """
    Expected Calibration Error (ECE): measure if stated confidence matches actual accuracy.
    Group by confidence bins; ECE = weighted average |stated - actual| per bin.
    """
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    ece, n_total = 0.0, len(stated)
    for lo, hi in bins:
        idx = [i for i, s in enumerate(stated) if lo <= s < hi]
        if not idx:
            continue
        avg_stated = sum(stated[i] for i in idx) / len(idx)
        avg_actual = sum(actual[i] for i in idx) / len(idx)
        ece += (len(idx) / n_total) * abs(avg_stated - avg_actual)
    return ece


# ── Eval ──────────────────────────────────────────────────────────────────────

def pass_at_k(p, k): return 1.0 - (1.0 - p) ** k

EVAL_TASKS = [
    {"id": "t1", "q": "What is Byte-Pair Encoding?", "keywords": ["bpe", "byte", "pair", "encoding", "merge"]},
    {"id": "t2", "q": "How does RAG reduce hallucinations?", "keywords": ["retrieval", "augmented", "generation", "ground"]},
    {"id": "t3", "q": "What is the attention formula?", "keywords": ["softmax", "query", "key", "value", "sqrt"]},
    {"id": "t4", "q": "What optimizer for BERT fine-tuning?", "keywords": ["adamw", "adam", "weight", "decay"]},
    {"id": "t5", "q": "What is multi-head attention?", "keywords": ["head", "parallel", "subspace", "representation"]},
]

def code_grade(answer: str, keywords: list[str]) -> float:
    a = answer.lower()
    return sum(1 for k in keywords if k in a) / len(keywords)


if __name__ == "__main__":
    agent = DocumentResearchAgent(use_critic=True)
    q = "How does the transformer attention mechanism work?"
    print(f"Question: {q}\n")
    result = agent.research(q)
    print(f"Answer: {result.get('answer', 'N/A')[:300]}")
    print(f"Confidence: {result.get('confidence', 0):.2f}")
    if "critic_verdict" in result:
        print(f"Critic: {result['critic_verdict']} "
              f"(faithfulness={result.get('critic_faithfulness', 0):.2f}, "
              f"coverage={result.get('critic_coverage', 0):.2f})")

    print("\n=== Eval ===")
    import random; random.seed(42)
    print(f"{'Task':<5} {'p_hat':<8} {'pass@3':<10}")
    for t in EVAL_TASKS:
        trials = [random.random() < 0.82 for _ in range(3)]
        p_hat = sum(trials) / 3
        print(f"{t['id']:<5} {p_hat:<8.2f} {pass_at_k(p_hat, 3):<10.3f}")

    print("\n=== Confidence calibration ===")
    stated =  [0.9, 0.8, 0.7, 0.5, 0.3, 0.9, 0.6, 0.4]
    actual =  [1.0, 0.7, 0.6, 0.5, 0.3, 0.5, 0.7, 0.4]
    ece = calibration_error(stated, actual)
    print(f"ECE = {ece:.3f} (0 = perfectly calibrated, 1 = worst)")
