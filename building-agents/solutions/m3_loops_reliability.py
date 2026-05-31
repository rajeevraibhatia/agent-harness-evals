"""Module 3 Solution — Loops & Reliability + exponential backoff."""
import json, math, os, time
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
    """Always returns a string — never raises."""
    if name not in TOOL_FNS:
        return f"Error: tool '{name}' not found. Available: {list(TOOL_FNS.keys())}"
    try:
        return TOOL_FNS[name](args)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

def chat_with_retry(messages: list, tools: list, max_retries: int = 3):
    """
    Call the OpenAI API with exponential backoff.
    Retries on RateLimitError and APIConnectionError.
    After max_retries, raises the original error.
    """
    RETRYABLE = (openai.RateLimitError, openai.APIConnectionError)
    delay = 2.0
    last_error = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="gpt-4o", messages=messages, tools=tools
            )
        except RETRYABLE as e:
            last_error = e
            if attempt < max_retries - 1:
                print(f"  [retry {attempt+1}/{max_retries}] {type(e).__name__} — waiting {delay:.0f}s")
                time.sleep(delay)
                delay *= 2  # exponential backoff: 2s → 4s → 8s
        except Exception:
            raise  # non-retryable errors propagate immediately
    raise last_error  # all retries exhausted

def run_agent(question: str) -> str:
    messages = [{"role": "user", "content": question}]
    recent_calls: list[str] = []

    for step in range(MAX_STEPS):
        r = chat_with_retry(messages, TOOLS)
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

    return f"Stopped: reached {MAX_STEPS} steps."

if __name__ == "__main__":
    print(run_agent("What is 15% of 847?"))
    print(run_agent("What is the square root of -1?"))  # error case — agent reads error and adapts
    print(run_agent("Call a tool named nonexistent_tool."))  # unknown tool case
