"""
Module 1 Solution — ReAct Loop
Exercise: architecture selection for 3 tasks.
"""
import json
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── Mock tools ────────────────────────────────────────────────────────────────

TOOLS = [
    {"type": "function", "function": {
        "name": "search",
        "description": "Search the web. Be precise — 'Paris metro area population 2024' beats 'Paris population'.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}
        }, "required": ["query"]}
    }},
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate a math expression. Examples: '20.1 / 12.3', 'sqrt(144)'.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string"}
        }, "required": ["expression"]}
    }},
]

def mock_tool_executor(name: str, args: dict) -> str:
    if name == "search":
        q = args.get("query", "").lower()
        if "paris" in q:
            return "Ile-de-France had ~12.3 million residents in 2024."
        if "new york" in q or "nyc" in q:
            return "Greater NYC metro area had ~20.1 million residents in 2024."
        return "No results found."
    if name == "calculator":
        import math
        try:
            result = eval(args["expression"], {"__builtins__": {}}, vars(math))
            return str(result)
        except Exception as e:
            return f"Error: {e}"
    return f"Unknown tool: {name}"


def react_loop(question: str, max_steps: int = 10) -> str:
    messages = [
        {"role": "system", "content": "Think step by step. Use tools for external data or math."},
        {"role": "user", "content": question},
    ]
    for step in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, tools=TOOLS, tool_choice="auto"
        )
        msg = response.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            return msg.content
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            observation = mock_tool_executor(name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": observation})
    return "Max steps reached."


# ── Exercise solution: architecture selection ─────────────────────────────────

SOLUTIONS = [
    {
        "task": "Extract name, date, and amount from a PDF invoice.",
        "architecture": "Single LLM call (structured output)",
        "reason": (
            "Fixed input/output — no branching needed. "
            "One prompt with JSON schema output handles it. "
            "No tool calls, no loops. Fast and cheap."
        ),
    },
    {
        "task": "Write a 5-page research report on climate policy, citing 10+ sources.",
        "architecture": "Agent loop (ReAct)",
        "reason": (
            "Requires iterative search + synthesis across unknown sources. "
            "Number of searches not known in advance — agent decides. "
            "Each search result informs the next query."
        ),
    },
    {
        "task": "Classify customer support tickets into 5 buckets.",
        "architecture": "Workflow (prompt chain with routing)",
        "reason": (
            "Deterministic output space (5 classes). "
            "Single LLM call per ticket with few-shot examples. "
            "For high volume: batch + smaller model (Haiku/Flash). "
            "No tool calls needed — classification is pure LLM."
        ),
    },
]

if __name__ == "__main__":
    print("=== Exercise: Architecture Selection ===\n")
    for s in SOLUTIONS:
        print(f"Task: {s['task']}")
        print(f"  Architecture: {s['architecture']}")
        print(f"  Reason: {s['reason']}\n")

    print("=== ReAct Loop Demo ===\n")
    answer = react_loop("How much larger is NYC metro than Paris? Express as a percentage.")
    print(f"Answer: {answer}")
