"""Module 2 Solution — Tools & Memory + delete_note tool."""
import json, math, os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
_notes: list[str] = []

TOOLS = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate a math expression. Example: '847 * 0.15'",
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
    {"type": "function", "function": {
        "name": "delete_note",
        "description": "Delete a note by its 1-based position in the list. Example: delete_note(index=2) removes the second note.",
        "parameters": {"type": "object",
            "properties": {
                "index": {"type": "integer", "description": "1-based position of the note to delete."}
            }, "required": ["index"]}
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
        if not _notes:
            return "No notes yet."
        return "\n".join(f"{i+1}. {n}" for i, n in enumerate(_notes))
    if name == "delete_note":
        idx = args["index"] - 1  # convert 1-based to 0-based
        if idx < 0 or idx >= len(_notes):
            return f"Error: index {args['index']} out of range (have {len(_notes)} notes)."
        removed = _notes.pop(idx)
        return f"Deleted note #{args['index']}: '{removed[:50]}'"
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

if __name__ == "__main__":
    history = [{"role": "system", "content": "You are a helpful assistant with note-taking tools."}]
    print(run_agent("Save three notes: 'Transformers use self-attention', 'BERT is bidirectional', 'GPT is autoregressive'", history))
    print(run_agent("What notes do I have?", history))
    print(run_agent("Delete the second note.", history))
    print(run_agent("What notes are left?", history))
