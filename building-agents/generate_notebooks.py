"""Generate 4 beginner Colab notebooks for the Building Agents mini-course."""
import json, os

COLAB_BASE = "https://colab.research.google.com/github/rajeevraibhatia/agent-harness-evals/blob/main/building-agents/notebooks/"
COURSE_URL = "https://rajeevraibhatia.com/curriculum/building-agents"

SETUP_CODE = '''# ── Setup ─────────────────────────────────────────────────────────────────────
# Option A: OpenAI API (recommended for Colab)
!pip install openai --quiet

import os
from openai import OpenAI

# Set your OpenAI API key — in Colab: Secrets (🔑) → add OPENAI_API_KEY
# Then enable notebook access, or paste directly (don't commit keys to git)
# os.environ["OPENAI_API_KEY"] = "sk-..."   # ← uncomment and paste if not using Secrets

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4o"

# ── Option B: Ollama (local, no API key, no cost) ─────────────────────────────
# 1. Install Ollama: https://ollama.com/download
# 2. Run: ollama pull llama3.2
# 3. Uncomment the two lines below and comment out the OpenAI lines above:
#
# client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
# MODEL = "llama3.2"   # or: mistral, phi4, gemma3, qwen2.5, etc.
#
# Everything in this notebook works with either client — MODEL is passed through.
print(f"Client ready. Using model: {MODEL}")'''

def nb(cells):
    return {"nbformat": 4, "nbformat_minor": 4,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {
                    "name": "python", "version": "3.10.0",
                    "codemirror_mode": {"name": "ipython", "version": 3},
                    "file_extension": ".py", "mimetype": "text/x-python",
                    "nbconvert_exporter": "python", "pygments_lexer": "ipython3"
                },
                "colab": {"provenance": []}},
            "cells": cells}

def md(text):
    lines = text.strip().split("\n")
    return {"cell_type": "markdown", "metadata": {},
            "source": [l+"\n" for l in lines[:-1]] + [lines[-1]]}

def code(src, cell_id=None):
    lines = src.strip().split("\n")
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [l+"\n" for l in lines[:-1]] + [lines[-1]]}

def badge(filename, module_num):
    return md(f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_BASE}{filename}) [![Course](https://img.shields.io/badge/Course-rajeevraibhatia.com-7c3aed)]({COURSE_URL}#module-{module_num})")


# ── M1 ────────────────────────────────────────────────────────────────────────

M1_CODE = '''import json
import math
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

TOOLS = [{
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a math expression. Example: \'847 * 0.15\'",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]
        }
    }
}]

def run_tool(name: str, args: dict) -> str:
    if name == "calculator":
        try:
            return str(eval(args["expression"], {"__builtins__": {}}, vars(math)))
        except Exception as e:
            return f"Error: {e}"
    return f"Unknown tool: {name}"

def run_agent(question: str, max_steps: int = 10) -> str:
    messages = [{"role": "user", "content": question}]
    for step in range(max_steps):
        r = client.chat.completions.create(model="gpt-4o", messages=messages, tools=TOOLS)
        msg = r.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            return msg.content
        for tc in msg.tool_calls:
            result = run_tool(tc.function.name, json.loads(tc.function.arguments))
            print(f"  [{step}] {tc.function.name}({tc.function.arguments}) → {result}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    return "Max steps reached."

print(run_agent("What is 15% of 847?"))'''

M1_EXERCISE = '''# Exercise: add a string_length tool
# 1. Write the JSON schema and add it to TOOLS
# 2. Add the handler in run_tool()
# 3. Test: "How many characters are in \'supercalifragilistic\'?"
# 4. Bonus: ask a question that uses BOTH tools

# Your code here:'''

m1 = nb([
    badge("m1_first_agent.ipynb", 1),
    md(f"# Module 1 — Your First Agent\n\n**Level:** Beginner | **Time:** ~30 min | [Full reading →]({COURSE_URL}#module-1)\n\n### What you'll build\nA working agent with one tool in 25 lines. No frameworks.\n\n### Key concepts\n- **Chatbot vs agent**: tool schema + loop = agent\n- The model *requests* tools; your code *runs* them\n- Loop exits when no tool calls are returned"),
    code(SETUP_CODE),
    code(M1_CODE),
    md("## Exercise\n\nAdd a `string_length(text)` tool that returns the number of characters in a string."),
    code(M1_EXERCISE)
])


# ── M2 ────────────────────────────────────────────────────────────────────────

M2_CODE = '''import json
import math
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
_notes: list[str] = []

TOOLS = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate a math expression. Example: \'847 * 0.15\'",
        "parameters": {"type": "object",
            "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
    }},
    {"type": "function", "function": {
        "name": "save_note",
        "description": "Save a text note for later retrieval.",
        "parameters": {"type": "object",
            "properties": {"text": {"type": "string"}}, "required": ["text"]}
    }},
    {"type": "function", "function": {
        "name": "get_notes",
        "description": "Retrieve all saved notes.",
        "parameters": {"type": "object", "properties": {}}
    }},
]

def run_tool(name: str, args: dict) -> str:
    if name == "calculator":
        try:
            return str(eval(args["expression"], {"__builtins__": {}}, vars(math)))
        except Exception as e:
            return f"Error: {e}"
    if name == "save_note":
        _notes.append(args["text"])
        return f"Note saved (#{len(_notes)})."
    if name == "get_notes":
        return "\\n".join(f"{i+1}. {n}" for i, n in enumerate(_notes)) if _notes else "No notes yet."
    return f"Unknown tool: {name}"

def run_agent(user_input: str, messages: list) -> str:
    messages.append({"role": "user", "content": user_input})
    while True:
        r = client.chat.completions.create(model="gpt-4o", messages=messages, tools=TOOLS)
        msg = r.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            return msg.content
        for tc in msg.tool_calls:
            result = run_tool(tc.function.name, json.loads(tc.function.arguments))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

history = [{"role": "system", "content": "You are a helpful assistant."}]
print(run_agent("What is 15% of 847? Save the result as a note.", history))
print(run_agent("What notes have I saved so far?", history))'''

M2_EXERCISE = '''# Exercise: add delete_note(index) tool
# index is 1-based (first note = 1)
# Return an error string if index is out of range (don\'t raise!)
# Test: save 3 notes, delete the second, call get_notes

# Your code here:'''

m2 = nb([
    badge("m2_tools_memory.ipynb", 2),
    md(f"# Module 2 — Tools & Memory\n\n**Level:** Beginner | **Time:** ~45 min | [Full reading →]({COURSE_URL}#module-2)\n\n### Key concepts\n- Tool descriptions are the model's only docs — make them verbose\n- Conversation history = the `messages` list. Keep it between turns.\n- Return errors as strings, not exceptions"),
    code(SETUP_CODE),
    code(M2_CODE),
    md("## Exercise\n\nAdd a `delete_note(index)` tool. Handle out-of-range gracefully."),
    code(M2_EXERCISE)
])


# ── M3 ────────────────────────────────────────────────────────────────────────

M3_CODE = '''import json
import math
import os
import time
from openai import OpenAI
import openai

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MAX_STEPS = 10

TOOLS = [{"type": "function", "function": {
    "name": "calculator",
    "description": "Evaluate a math expression.",
    "parameters": {"type": "object",
        "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
}}]

TOOL_FNS = {
    "calculator": lambda args: str(eval(args["expression"], {"__builtins__": {}}, vars(math)))
}

def safe_run_tool(name: str, args: dict) -> str:
    """Always returns string — never raises."""
    if name not in TOOL_FNS:
        return f"Error: tool \'{name}\' not found. Available: {list(TOOL_FNS.keys())}"
    try:
        return TOOL_FNS[name](args)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

def run_agent(question: str) -> str:
    messages = [{"role": "user", "content": question}]
    recent_calls: list[str] = []
    for step in range(MAX_STEPS):
        r = client.chat.completions.create(model="gpt-4o", messages=messages, tools=TOOLS)
        msg = r.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            return msg.content
        for tc in msg.tool_calls:
            call_key = f"{tc.function.name}:{tc.function.arguments}"
            recent_calls.append(call_key)
            if recent_calls[-3:] == [call_key] * 3:
                return "Stopped: agent looping on same tool call."
            result = safe_run_tool(tc.function.name, json.loads(tc.function.arguments))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    return f"Stopped: {MAX_STEPS} steps reached."

print(run_agent("What is 15% of 847?"))
print(run_agent("What is the square root of -1?"))  # error case'''

M3_EXERCISE = '''# Exercise: add retry-with-exponential-backoff around the API call
# Retry on openai.RateLimitError and openai.APIConnectionError
# Wait 2s → 4s → 8s between attempts (exponential backoff)
# After 3 failures, re-raise the original error

def chat_with_retry(messages, tools, max_retries=3):
    # Your code here:
    raise NotImplementedError'''

m3 = nb([
    badge("m3_loops_reliability.ipynb", 3),
    md(f"# Module 3 — Loops & Reliability\n\n**Level:** Beginner | **Time:** ~45 min | [Full reading →]({COURSE_URL}#module-3)\n\n### Key concepts\n- Add `max_steps` — agents can loop forever without it\n- Circuit breaker: same tool call 3× in a row → stop\n- Error wrapping: always return errors as strings, never raise\n- Retry-with-backoff for rate limits and network errors"),
    code(SETUP_CODE),
    code(M3_CODE),
    md("## Exercise\n\nWrap the API call with exponential backoff."),
    code(M3_EXERCISE)
])


# ── M4 ────────────────────────────────────────────────────────────────────────

M4_CODE = '''import json
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MAX_STEPS = 15
_findings: list[dict] = []

MOCK_SEARCH_DB = {
    "transformer": "Transformers use self-attention (Vaswani 2017). Input → embeddings → N × (MHA + FFN) → output.",
    "attention": "Attention = softmax(QK^T / sqrt(d_k)) × V. Multi-head runs h parallel heads.",
    "bert": "BERT (Devlin 2018): bidirectional, pretrained with MLM. Fine-tune AdamW, lr=2e-5.",
    "rag": "RAG (Lewis 2020): retrieve docs, condition generation. Reduces hallucinations.",
}

TOOLS = [
    {"type": "function", "function": {
        "name": "search",
        "description": "Search for information. Be specific — \'BERT fine-tuning 2024\' beats \'BERT\'.",
        "parameters": {"type": "object",
            "properties": {"query": {"type": "string"}}, "required": ["query"]}
    }},
    {"type": "function", "function": {
        "name": "save_finding",
        "description": "Store a relevant passage. Call after each useful search.",
        "parameters": {"type": "object",
            "properties": {"source": {"type": "string"}, "text": {"type": "string"}},
            "required": ["source", "text"]}
    }},
    {"type": "function", "function": {
        "name": "write_summary",
        "description": "Write a final answer from saved findings. Call only after saving 2+ findings.",
        "parameters": {"type": "object",
            "properties": {"question": {"type": "string"}}, "required": ["question"]}
    }},
]

def run_tool(name, args):
    if name == "search":
        q = args["query"].lower()
        hits = [v for k, v in MOCK_SEARCH_DB.items() if k in q]
        return hits[0] if hits else "No results found."
    if name == "save_finding":
        _findings.append(args); return f"Saved finding #{len(_findings)}"
    if name == "write_summary":
        if not _findings: return "No findings yet. Search first."
        context = "\\n".join(f"[{f[\'source\']}] {f[\'text\']}" for f in _findings)
        r = client.chat.completions.create(model="gpt-4o", messages=[
            {"role": "system", "content": "Synthesize these findings into a clear answer."},
            {"role": "user", "content": f"Question: {args[\'question\']}\\n\\nFindings:\\n{context}"}
        ])
        return r.choices[0].message.content
    return f"Unknown tool: {name}"

def research(question: str) -> str:
    _findings.clear()
    messages = [
        {"role": "system", "content": "Research thoroughly: search 2-3 times, save findings, then write_summary."},
        {"role": "user", "content": question}
    ]
    for _ in range(MAX_STEPS):
        r = client.chat.completions.create(model="gpt-4o", messages=messages, tools=TOOLS)
        msg = r.choices[0].message
        messages.append(msg)
        if not msg.tool_calls: return msg.content
        for tc in msg.tool_calls:
            result = run_tool(tc.function.name, json.loads(tc.function.arguments))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    return "Max steps reached."

answer = research("How does the transformer attention mechanism work?")
print(answer)'''

M4_EXERCISE = '''# Capstone Exercise
# (a) Replace MOCK_SEARCH_DB with Tavily real search:
#     pip install tavily-python
#     from tavily import TavilyClient
#     client_tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
#
# (b) Add relevance_score: float to save_finding schema.
#     Only save findings with score >= 0.6.
#     Return "Skipped — score below threshold" for low-score findings.

# Your code here:'''

M4_NEXT = '''# 🎉 You built a working research agent!
#
# You now know:
# ✓ Tool calling and the agent loop
# ✓ Multi-tool agents with memory
# ✓ Reliability: max steps, error wrapping, circuit breaker, retries
# ✓ A real end-to-end research pipeline
#
# Ready for the next level?
# Building an Agent Harness with Evals:
# https://rajeevraibhatia.com/curriculum/agent-harness-evals
print("Course complete! Next: https://rajeevraibhatia.com/curriculum/agent-harness-evals")'''

m4 = nb([
    badge("m4_research_bot.ipynb", 4),
    md(f"# Module 4 — Mini-Project: Research Bot\n\n**Level:** Beginner | **Time:** ~1h | [Full reading →]({COURSE_URL}#module-4)\n\n### What you'll build\nA complete research bot: search → save findings → synthesize answer.\n\n### The three-tool research pattern\n1. `search(query)` — find information\n2. `save_finding(source, text)` — store what matters\n3. `write_summary(question)` — synthesize the answer\n\nThis exact pattern is in production at every major AI company."),
    code(SETUP_CODE + "\n\n# Optional: real web search\n# !pip install tavily-python --quiet"),
    code(M4_CODE),
    md("## Exercise\n\n(a) Replace mock search with Tavily. (b) Add relevance score filtering."),
    code(M4_EXERCISE),
    md("## What's Next?\n\nYou've completed the Building Agents mini-course. The advanced course goes deeper: production harnesses, multi-agent systems, eval suites, and safety.\n\n[**Building an Agent Harness with Evals →**](https://rajeevraibhatia.com/curriculum/agent-harness-evals)"),
    code(M4_NEXT)
])


# ── Write notebooks ────────────────────────────────────────────────────────────

NOTEBOOKS = [
    ("m1_first_agent.ipynb", m1),
    ("m2_tools_memory.ipynb", m2),
    ("m3_loops_reliability.ipynb", m3),
    ("m4_research_bot.ipynb", m4),
]

out_dir = os.path.join(os.path.dirname(__file__), "notebooks")
os.makedirs(out_dir, exist_ok=True)

for filename, notebook in NOTEBOOKS:
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        json.dump(notebook, f, indent=1)
    print(f"  ✓ {filename} ({len(notebook['cells'])} cells)")

print(f"\nAll {len(NOTEBOOKS)} notebooks written.")
