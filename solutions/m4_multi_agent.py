"""
Module 4 Solution — Multi-Agent Orchestration
Exercise: 3-specialist topology + deadlock detection + "coder" agent.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def run_specialist(system_prompt: str, task: str, model: str = "gpt-4o") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": task}]
    )
    return response.choices[0].message.content


SPECIALISTS = {
    "researcher": "You are a research specialist. Find and synthesize factual information. Be concise.",
    "writer": "You are a writing specialist. Produce clear, well-structured prose for technical audiences.",
    "coder": (
        "You are a code specialist. Write clean, working Python code with type annotations. "
        "Include a brief docstring and at least one usage example in a main block."
    ),
    "critic": (
        "You are a quality critic. Review content for accuracy, clarity, and completeness. "
        "Return JSON: {strengths: [str], weaknesses: [str], verdict: 'pass' | 'revise', "
        "revised_task: str | null}"
    ),
}

ROUTING_SCHEMA = {
    "type": "function",
    "function": {
        "name": "route_task",
        "description": "Route a task to the appropriate specialist.",
        "parameters": {"type": "object", "properties": {
            "specialist": {
                "type": "string",
                "enum": ["researcher", "writer", "coder", "critic", "escalate_to_human"],
                "description": "Which specialist to invoke. Use 'escalate_to_human' when unsure.",
            },
            "task": {"type": "string"},
            "reason": {"type": "string"},
        }, "required": ["specialist", "task", "reason"]},
    }
}


# ── Deadlock detection ────────────────────────────────────────────────────────

class DeadlockDetector:
    def __init__(self, threshold: int = 3):
        self._history: list[str] = []
        self.threshold = threshold

    def record(self, specialist: str) -> bool:
        """Record a routing decision. Returns True if deadlock detected."""
        self._history.append(specialist)
        recent = self._history[-self.threshold:]
        if len(recent) == self.threshold and len(set(recent)) == 1:
            return True
        return False

    def reset(self):
        self._history = []


# ── Supervisor ────────────────────────────────────────────────────────────────

def supervisor(user_request: str, context: str = "") -> dict:
    msg = f"User request: {user_request}"
    if context:
        msg += f"\n\nContext:\n{context}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Route tasks to specialists. If unsure, use escalate_to_human."},
            {"role": "user", "content": msg}
        ],
        tools=[ROUTING_SCHEMA],
        tool_choice={"type": "function", "function": {"name": "route_task"}}
    )
    return json.loads(response.choices[0].message.tool_calls[0].function.arguments)


# ── Producer-critic loop ──────────────────────────────────────────────────────

def producer_critic_loop(producer_type: str, task: str, max_revisions: int = 2) -> str:
    draft = run_specialist(SPECIALISTS[producer_type], task)
    for _ in range(max_revisions):
        feedback_raw = run_specialist(SPECIALISTS["critic"], f"Review:\n\n{draft}")
        try:
            feedback = json.loads(feedback_raw)
            verdict = feedback.get("verdict", "pass")
            revised_task = feedback.get("revised_task")
        except json.JSONDecodeError:
            verdict = "pass" if "pass" in feedback_raw.lower() else "revise"
            revised_task = None

        if verdict == "pass":
            return draft
        if revised_task:
            draft = run_specialist(SPECIALISTS[producer_type], revised_task)
        else:
            draft = run_specialist(SPECIALISTS[producer_type],
                f"Revise based on feedback.\n\nOriginal task: {task}\n\nDraft:\n{draft}\n\nFeedback:\n{feedback_raw}")
    return draft


# ── Parallel fan-out ──────────────────────────────────────────────────────────

def parallel_research(queries: list[str]) -> list[str]:
    """Run multiple research tasks concurrently."""
    with ThreadPoolExecutor(max_workers=len(queries)) as ex:
        futures = [ex.submit(run_specialist, SPECIALISTS["researcher"], q) for q in queries]
        return [f.result() for f in futures]


# ── Full pipeline ─────────────────────────────────────────────────────────────

def multi_agent_pipeline(user_request: str) -> str:
    detector = DeadlockDetector(threshold=3)
    results = []
    context = ""

    # Step 1: route
    route = supervisor(user_request, context)
    specialist = route["specialist"]
    print(f"[supervisor] → {specialist}: {route['reason']}")

    if detector.record(specialist):
        print("[deadlock] Same specialist 3x — escalating to human.")
        return "Escalated: routing loop detected."

    if specialist == "escalate_to_human":
        return f"Escalated to human: {route['task']}"

    # Step 2: execute with producer-critic if writer or coder
    if specialist in ("writer", "coder"):
        result = producer_critic_loop(specialist, route["task"])
    else:
        result = run_specialist(SPECIALISTS[specialist], route["task"])

    results.append(result)
    return "\n\n".join(results)


# ── Exercise: routing state machine ──────────────────────────────────────────

"""
State machine for product requiring research + code + writing:

States: START → RESEARCH → CODE → WRITE → CRITIQUE → (PASS → END | REVISE → WRITE)
Deadlock mitigations:
  1. WRITE↔CRITIQUE loop: critic must provide revised_task or verdict=pass after 2 rounds
  2. RESEARCH→RESEARCH loop: deduplicate queries; if same query 2x, mark "no new info" and proceed
"""

ROUTING_LOGIC = {
    "factual_question": "researcher",
    "write_report": "writer",
    "generate_code": "coder",
    "review_output": "critic",
    "unclear_or_loop": "escalate_to_human",
}


if __name__ == "__main__":
    result = multi_agent_pipeline(
        "Explain the key differences between RAG and fine-tuning for LLM applications."
    )
    print(f"\nResult:\n{result[:500]}")
