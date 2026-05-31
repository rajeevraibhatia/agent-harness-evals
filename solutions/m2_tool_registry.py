"""
Module 2 Solution — Tool Registry
Exercise: 4 tool schemas for a document research agent, with deduplication.
"""
import math
from typing import Callable

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, dict] = {}

    def register(self, fn: Callable, schema: dict, idempotent: bool = True):
        self._tools[schema["name"]] = {"fn": fn, "schema": schema, "idempotent": idempotent}

    @property
    def schemas(self) -> list[dict]:
        return [{"type": "function", "function": t["schema"]} for t in self._tools.values()]

    def execute(self, name: str, args: dict) -> str:
        if name not in self._tools:
            return f"ERROR: tool '{name}' not found. Available: {list(self._tools.keys())}"
        try:
            return str(self._tools[name]["fn"](**args))
        except TypeError as e:
            return f"ERROR: bad arguments for '{name}': {e}"
        except Exception as e:
            return f"ERROR: {name} failed — {type(e).__name__}: {e}"

    def is_idempotent(self, name: str) -> bool:
        return self._tools.get(name, {}).get("idempotent", False)


# ── Tool implementations ──────────────────────────────────────────────────────

def search_web(query: str, num_results: int = 5) -> str:
    # Replace with real search API (Brave, Serper, Tavily)
    return f"[mock] Top results for: {query}"

def extract_pdf(url: str, max_chars: int = 4000) -> str:
    # Replace with PDF parsing (pypdf, pdfplumber)
    return f"[mock] Extracted {max_chars} chars from {url}"

# Deduplicated store: keyed by (source, passage) hash
_findings: dict[str, dict] = {}

def store_finding(source: str, passage: str, relevance_score: float) -> str:
    import hashlib
    key = hashlib.md5(f"{source}::{passage}".encode()).hexdigest()
    if key in _findings:
        return f"Duplicate skipped — already stored from {source[:40]}"
    _findings[key] = {"source": source, "passage": passage, "relevance": relevance_score}
    return f"Stored finding #{len(_findings)} [score={relevance_score:.2f}]"

def generate_citation(title: str, year: int, author: str = "", url: str = "", style: str = "apa") -> str:
    if style == "apa":
        base = f"{author + '. ' if author else ''}({year}). {title}."
        return f"{base} {url}" if url else base
    if style == "mla":
        return f"{author + '. ' if author else ''}\"{title}.\" {year}."
    return f"{title} ({year})"


# ── Register all 4 tools ──────────────────────────────────────────────────────

registry = ToolRegistry()

registry.register(
    fn=search_web,
    schema={
        "name": "search_web",
        "description": (
            "Search the web and return relevant text snippets. "
            "Be specific — 'BERT fine-tuning classification 2024' beats 'BERT training'. "
            "Use for current facts and verifiable information."
        ),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Specific search query. Max 200 chars."},
            "num_results": {"type": "integer", "description": "Number of results (1-10). Default: 5.", "default": 5},
        }, "required": ["query"]},
    },
    idempotent=True,  # same query → same results (deterministic search)
)

registry.register(
    fn=extract_pdf,
    schema={
        "name": "extract_pdf",
        "description": (
            "Extract text from a PDF URL. "
            "Use after search_web returns a PDF link. "
            "Example: extract_pdf(url='https://arxiv.org/pdf/2302.01318', max_chars=3000)"
        ),
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "Direct URL to a PDF file."},
            "max_chars": {"type": "integer", "description": "Max characters to extract. Default: 4000.", "default": 4000},
        }, "required": ["url"]},
    },
    idempotent=True,  # reading a file doesn't change it
)

registry.register(
    fn=store_finding,
    schema={
        "name": "store_finding",
        "description": (
            "Store a research finding for later synthesis. "
            "Call after extracting a relevant passage. "
            "Deduplicates by (source, passage) — safe to call multiple times."
        ),
        "parameters": {"type": "object", "properties": {
            "source": {"type": "string", "description": "Source name or URL."},
            "passage": {"type": "string", "description": "Relevant text excerpt."},
            "relevance_score": {"type": "number", "description": "Relevance to research question, 0.0–1.0."},
        }, "required": ["source", "passage", "relevance_score"]},
    },
    idempotent=False,  # writes to store (though deduplication makes re-calls safe in practice)
)

registry.register(
    fn=generate_citation,
    schema={
        "name": "generate_citation",
        "description": "Format a citation in APA, MLA, or Chicago style. Idempotent — safe to call multiple times.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"},
            "year": {"type": "integer"},
            "author": {"type": "string"},
            "url": {"type": "string"},
            "style": {"type": "string", "enum": ["apa", "mla", "chicago"], "default": "apa"},
        }, "required": ["title", "year"]},
    },
    idempotent=True,
)


if __name__ == "__main__":
    print("Schemas loaded:", [s["function"]["name"] for s in registry.schemas])

    # Deduplication test
    print(store_finding("arxiv.org/1234", "Attention is all you need.", 0.95))
    print(store_finding("arxiv.org/1234", "Attention is all you need.", 0.95))  # duplicate

    # Error wrapping
    print(registry.execute("search_web", {"bad_arg": "x"}))
    print(registry.execute("nonexistent", {}))

    # Citation
    print(generate_citation("Attention Is All You Need", 2017, "Vaswani et al.", style="apa"))
