"""
Module 3 Solution — Memory Manager
Exercise: retrieve_context() + context rot simulation.
"""
import math
import time
from dataclasses import dataclass


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag = lambda v: math.sqrt(sum(x * x for x in v))
    return dot / (mag(a) * mag(b) + 1e-10)

def mock_embed(text: str) -> list[float]:
    import hashlib
    h = hashlib.sha256(text.encode()).digest()
    return [((b / 255.0) - 0.5) * 2 for b in h[:32]]


@dataclass
class EpisodicMemory:
    user_id: str
    timestamp: float
    text: str
    embedding: list[float]

@dataclass
class SemanticFact:
    subject: str
    predicate: str
    object_: str
    confidence: float = 1.0


class MemoryManager:
    MAX_IN_CONTEXT_TURNS = 8
    MAX_EPISODIC_RESULTS = 3

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._turns: list[dict] = []
        self._episodic: list[EpisodicMemory] = []
        self._semantic: list[SemanticFact] = []
        self._summary: str = ""

    def add_turn(self, role: str, content: str):
        self._turns.append({"role": role, "content": content})
        if len(self._turns) > self.MAX_IN_CONTEXT_TURNS:
            self._compress_oldest()

    def _compress_oldest(self):
        oldest = self._turns[:4]
        fragment = "; ".join(f"{t['role']}: {t['content'][:60]}..." for t in oldest)
        self._summary = f"{self._summary} | {fragment}".strip(" |")
        self._turns = self._turns[4:]

    def store_episode(self, text: str):
        self._episodic.append(EpisodicMemory(
            user_id=self.user_id, timestamp=time.time(),
            text=text, embedding=mock_embed(text)
        ))

    def retrieve_episodes(self, query: str, max_tokens: int = 500) -> list[str]:
        if not self._episodic:
            return []
        q_emb = mock_embed(query)
        scored = sorted(self._episodic, key=lambda e: cosine_sim(q_emb, e.embedding), reverse=True)
        results, total = [], 0
        for ep in scored[:self.MAX_EPISODIC_RESULTS]:
            est = len(ep.text.split()) * 1.3
            if total + est > max_tokens:
                break
            results.append(ep.text)
            total += est
        return results

    def add_fact(self, subject: str, predicate: str, object_: str, confidence: float = 1.0):
        self._semantic.append(SemanticFact(subject, predicate, object_, confidence))

    def get_facts(self, subject: str) -> list[SemanticFact]:
        return [f for f in self._semantic if f.subject.lower() == subject.lower()]

    def get_context(self, query: str, max_tokens: int = 1000) -> str:
        parts = []
        if self._summary:
            parts.append(f"[History] {self._summary}")
        episodes = self.retrieve_episodes(query, max_tokens=400)
        if episodes:
            parts.append("[Past sessions]\n" + "\n".join(f"- {e}" for e in episodes))
        if self._semantic:
            parts.append("[Known facts]\n" + "\n".join(
                f"- {f.subject} {f.predicate} {f.object_}" for f in self._semantic[:10]
            ))
        return "\n\n".join(parts) if parts else ""


# ── Exercise solution: retrieve_context ──────────────────────────────────────

def retrieve_context(query: str, user_id: str, max_tokens: int, memory: MemoryManager) -> str:
    """
    Retrieval strategy for a multi-session coding agent:

    Tier 1 (verbatim): last N turns — kept as-is for coherence in current session.
    Tier 2 (episodic): past session digests — retrieved by semantic similarity to current query.
    Tier 3 (semantic): distilled facts about the user/project — always prepended (high signal, low tokens).

    Token budget allocation:
    - Facts:   ~15% (small, always useful)
    - Episodes: ~35% (retrieved, variable)
    - Turns:   ~50% (most recent, highest recency value)
    """
    budget = max_tokens
    parts = []

    # Tier 3: semantic facts (always include, small cost)
    facts = memory._semantic
    if facts:
        fact_text = "[Project facts]\n" + "\n".join(
            f"- {f.subject} {f.predicate} {f.object_}" for f in facts[:8]
        )
        parts.append(fact_text)
        budget -= len(fact_text.split()) * 1.3

    # Tier 2: episodic retrieval (query-relevant past sessions)
    episodes = memory.retrieve_episodes(query, max_tokens=int(budget * 0.4))
    if episodes:
        ep_text = "[Past sessions]\n" + "\n".join(f"- {e}" for e in episodes)
        parts.append(ep_text)
        budget -= len(ep_text.split()) * 1.3

    # Tier 1: rolling summary + recent turns (highest recency value)
    if memory._summary:
        parts.append(f"[Session history] {memory._summary}")
    recent = memory._turns[-min(6, len(memory._turns)):]
    if recent:
        turn_text = "\n".join(f"{t['role']}: {t['content']}" for t in recent)
        parts.append(turn_text)

    return "\n\n".join(parts) if parts else ""


# ── Context rot simulation ────────────────────────────────────────────────────

def simulate_context_rot():
    """Show quality signal before and after context rot prevention."""
    mem = MemoryManager("user_demo")

    print("Without compression (naive — dumps all turns):")
    naive_turns = []
    for i in range(20):
        naive_turns.append({"role": "user", "content": f"Question {i} about Python"})
        naive_turns.append({"role": "assistant", "content": f"Answer {i}: here is how..."})
    print(f"  Total tokens (est): {sum(len(t['content'].split()) for t in naive_turns) * 1.3:.0f}")
    print(f"  Relevant to current query: low (20 turns of noise)")

    print("\nWith MemoryManager compression:")
    for i in range(20):
        mem.add_turn("user", f"Question {i} about Python")
        mem.add_turn("assistant", f"Answer {i}: here is how...")
    ctx = mem.get_context("How do I use async/await?")
    print(f"  In-context turns: {len(mem._turns)} (compressed from 20)")
    print(f"  Summary preserved: {bool(mem._summary)}")
    print(f"  Context tokens (est): {len(ctx.split()) * 1.3:.0f}")


if __name__ == "__main__":
    mem = MemoryManager("user_123")
    mem.store_episode("User fine-tuned BERT on sentiment. Used AdamW, lr=2e-5, batch=32.")
    mem.store_episode("User built RAG with Pinecone. Chunk 512, overlap 64.")
    mem.add_fact("user", "prefers", "PyTorch")
    mem.add_fact("project", "uses", "Cloudflare Workers for deployment")

    ctx = retrieve_context("How do I fine-tune a language model?", "user_123", 800, mem)
    print("Retrieved context:\n", ctx)

    print("\n---")
    simulate_context_rot()
