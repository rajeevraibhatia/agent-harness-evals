"""Module 1 Solution — Your First Agent + string_length tool."""
import json, math, os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

TOOLS = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate a math expression. Example: '847 * 0.15'",
        "parameters": {"type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]}
    }},
    {"type": "function", "function": {
        "name": "string_length",
        "description": "Return the number of characters in a string. Example: string_length('hello') → 5",
        "parameters": {"type": "object",
            "properties": {"text": {"type": "string", "description": "The string to measure."}},
            "required": ["text"]}
    }},
]

def run_tool(name: str, args: dict) -> str:
    if name == "calculator":
        try:
            return str(eval(args["expression"], {"__builtins__": {}}, vars(math)))
        except Exception as e:
            return f"Error: {e}"
    if name == "string_length":
        return str(len(args["text"]))
    return f"Unknown tool: {name}"

def run_agent(question: str, max_steps: int = 10) -> str:
    messages = [{"role": "user", "content": question}]
    for _ in range(max_steps):
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
    print(run_agent("What is 15% of 847?"))
    print(run_agent("How many characters are in 'supercalifragilistic'?"))
    # Bonus: both tools in one session
    print(run_agent("If the word 'transformer' has N characters, what is N squared?"))
