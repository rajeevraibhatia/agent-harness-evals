"""
Module 5 Solution — Harness Architecture
Exercise: parallel_execute() + rollback_last().
"""
import json
import os
import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


@dataclass
class Feature:
    name: str
    steps: list[str]
    status: str = "failing"   # failing | passing | skipped

@dataclass
class ReplayStep:
    step_id: int
    prompt_hash: str
    response_summary: str
    tool_calls: list[dict]
    timestamp: float


class Harness:
    MAX_STEPS_PER_FEATURE = 15
    CIRCUIT_BREAKER_THRESHOLD = 3

    def __init__(self, task: str):
        self.task = task
        self.features: list[Feature] = []
        self.progress_log: list[str] = []
        self.replay_log: list[ReplayStep] = []
        self._step_counter = 0
        self._recent_calls: list[str] = []
        self._passing_history: list[str] = []  # for rollback

    def initialize(self) -> list[Feature]:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": (
                    "Decompose the task into 3-5 concrete, verifiable features. "
                    "Return JSON: {features: [{name: str, steps: [str], status: 'failing'}]}"
                )},
                {"role": "user", "content": f"Task: {self.task}"}
            ],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        self.features = [Feature(**f) for f in data["features"]]
        self._log(f"Initialized: {len(self.features)} features")
        return self.features

    def execute_next(self) -> Optional[Feature]:
        failing = [f for f in self.features if f.status == "failing"]
        if not failing:
            return None
        feature = failing[0]
        self._log(f"Executing: {feature.name}")
        success = self._implement(feature)
        if success:
            feature.status = "passing"
            self._passing_history.append(feature.name)
        self._log(f"  → {'PASS' if success else 'FAIL'}: {feature.name}")
        return feature

    def _implement(self, feature: Feature) -> bool:
        self._recent_calls = []
        for step in range(self.MAX_STEPS_PER_FEATURE):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Implement the feature. Verify it works. Return JSON: {done: bool, notes: str}"},
                    {"role": "user", "content": f"Feature: {feature.name}\nSteps: {feature.steps}"}
                ],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            self._record(f"Feature:{feature.name}:step{step}", content, [])
            result = json.loads(content)
            if result.get("done"):
                return True
        return False

    def _check_circuit_breaker(self, tool: str, args: dict) -> bool:
        key = f"{tool}:{json.dumps(args, sort_keys=True)}"
        self._recent_calls.append(key)
        recent = self._recent_calls[-self.CIRCUIT_BREAKER_THRESHOLD:]
        return len(recent) >= self.CIRCUIT_BREAKER_THRESHOLD and len(set(recent)) == 1

    def _log(self, msg: str):
        entry = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self.progress_log.append(entry)
        print(entry)

    def _record(self, prompt: str, response: str, tool_calls: list):
        self.replay_log.append(ReplayStep(
            step_id=self._step_counter,
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:8],
            response_summary=response[:100],
            tool_calls=tool_calls,
            timestamp=time.time()
        ))
        self._step_counter += 1

    def status(self) -> dict:
        total = len(self.features)
        passing = sum(1 for f in self.features if f.status == "passing")
        return {"total": total, "passing": passing, "failing": total - passing,
                "features": [{"name": f.name, "status": f.status} for f in self.features]}


# ── Exercise solutions ────────────────────────────────────────────────────────

def parallel_execute(harness: Harness, max_workers: int = 3) -> list[Feature]:
    """
    Run independent failing features concurrently.
    Fan-out: submit all failing features to thread pool.
    Fan-in: collect results, update statuses.
    Note: features with data dependencies should NOT run in parallel —
          detect by checking if feature.steps reference prior feature names.
    """
    failing = [f for f in harness.features if f.status == "failing"]
    if not failing:
        return []

    results = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(failing))) as ex:
        future_to_feature = {ex.submit(harness._implement, f): f for f in failing}
        for future in as_completed(future_to_feature):
            feature = future_to_feature[future]
            success = future.result()
            feature.status = "passing" if success else "failing"
            if success:
                harness._passing_history.append(feature.name)
            harness._log(f"[parallel] {'PASS' if success else 'FAIL'}: {feature.name}")
            results.append(feature)
    return results


def rollback_last(harness: Harness) -> Optional[Feature]:
    """
    Revert the last passing feature to failing.
    In a real harness: also run `git revert HEAD` to undo the commit.
    Use when e2e test fails after the last feature was marked passing.
    """
    if not harness._passing_history:
        print("Nothing to rollback.")
        return None

    last_name = harness._passing_history.pop()
    feature = next((f for f in harness.features if f.name == last_name), None)
    if feature:
        feature.status = "failing"
        harness._log(f"[rollback] Reverted '{last_name}' to failing")
        # In production: subprocess.run(["git", "revert", "HEAD", "--no-edit"])
    return feature


if __name__ == "__main__":
    harness = Harness(task="Build a URL shortener with click tracking")
    features = harness.initialize()

    print("\n=== Parallel execution ===")
    parallel_execute(harness, max_workers=3)

    print("\n=== Status ===")
    print(json.dumps(harness.status(), indent=2))

    print("\n=== Rollback last passing feature ===")
    rollback_last(harness)
    print(json.dumps(harness.status(), indent=2))
