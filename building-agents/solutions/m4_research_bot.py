"""
Module 4 Solution — Research Bot
(a) Real search via Tavily  (b) relevance_score filtering.

pip install openai tavily-python
export OPENAI_API_KEY="sk-..."
export TAVILY_API_KEY="tvly-..."
"""
import json, os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MAX_STEPS = 15
_findings: list[dict] = []

# ── (a) Real search via Tavily ─────────────────────────────────────────────

def search_tavily(query: str) -> str:
    """Real web search. Falls back to mock if TAVILY_API_KEY not set."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return _mock_search(query)
    try:
        from tavily import TavilyClient
        tavily = TavilyClient(api_key=api_key)
        result = tavily.search(query=query, max_results=3)
        snippets = [r.get("content", "") for r in result.get("results", [])]
        return "\n\n".join(snippets[:3]) if snippets else "No results found."
    except Exception as e:
        return f"Search error: {e}"

MOCK_SEARCH_DB = {
    "transformer": "Transformers use self-attention (Vaswani 2017). Input → embeddings → N × (MHA + FFN) → output.",
    "attention": "Attention = softmax(QK^T / sqrt(d_k)) × V. Multi-head runs h parallel heads.",
    "bert": "BERT (Devlin 2018): bidirectional, pretrained with MLM. Fine-tune AdamW, lr=2e-5.",
    "gpt": "GPT family: autoregressive. Uses BPE tokenizer (cl100k_base for GPT-4).",
    "rag": "RAG (Lewis 2020): retrieve relevant docs, condition generation. Reduces hallucinations.",
}

def _mock_search(query: str) -> str:
    q = query.lower()
    hits = [v for k, v in MOCK_SEARCH_DB.items() if k in q]
    return hits[0] if hits else "No results found."

# ── Tools ──────────────────────────────────────────────────────────────────

TOOLS = [
    {"type": "function", "function": {
        "name": "search",
        "description": "Search the web for information. Be specific — 'BERT fine-tuning 2024' beats 'BERT'.",
        "parameters": {"type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]}
    }},
    {"type": "function", "function": {
        "name": "save_finding",
        "description": "Store a relevant passage. Call after each useful search result. Only save relevance_score >= 0.6.",
        "parameters": {"type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source name or URL."},
                "text": {"type": "string", "description": "Relevant passage to save."},
                # (b) relevance_score field added
                "relevance_score": {"type": "number", "description": "How relevant is this to the question? 0.0–1.0."},
            }, "required": ["source", "text", "relevance_score"]}
    }},
    {"type": "function", "function": {
        "name": "write_summary",
        "description": "Write a final answer from saved findings. Call only after saving 2+ findings.",
        "parameters": {"type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"]}
    }},
]

def run_tool(name: str, args: dict) -> str:
    if name == "search":
        return search_tavily(args["query"])
    if name == "save_finding":
        score = args.get("relevance_score", 0.0)
        # (b) Filter low-relevance findings
        if score < 0.6:
            return f"Finding skipped — relevance {score:.2f} below threshold 0.6."
        _findings.append(args)
        return f"Saved finding #{len(_findings)} [score={score:.2f}]"
    if name == "write_summary":
        if not _findings:
            return "No findings saved yet. Search and save first."
        context = "\n".join(f"[{f['source']}] {f['text']}" for f in _findings)
        r = client.chat.completions.create(model="gpt-4o", messages=[
            {"role": "system", "content": "Synthesize these findings into a clear, accurate answer."},
            {"role": "user", "content": f"Question: {args['question']}\n\nFindings:\n{context}"}
        ])
        return r.choices[0].message.content
    return f"Unknown tool: {name}"

def research(question: str) -> str:
    _findings.clear()
    messages = [
        {"role": "system", "content": "Research the question thoroughly: search 2-4 times, save relevant findings (score >= 0.6), then write_summary."},
        {"role": "user", "content": question}
    ]
    for _ in range(MAX_STEPS):
        r = client.chat.completions.create(model="gpt-4o", messages=messages, tools=TOOLS)
        msg = r.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            return msg.content
        for tc in msg.tool_calls:
            result = run_tool(tc.function.name, json.loads(tc.function.arguments))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    return "Max steps reached."

if __name__ == "__main__":
    answer = research("How does the transformer attention mechanism work?")
    print(answer)
    print(f"\nFindings stored: {len(_findings)}")
