"""Real end-to-end check against the live Gemini API.

Makes 4 small calls (normal, streaming, nested tags, untagged), then checks
that llmburn recorded each one with the right tags and token counts.

    set GEMINI_API_KEY=your-key        (PowerShell: $env:GEMINI_API_KEY="your-key")
    python examples/smoke_gemini.py

Cost: a few hundred tokens on 3.5-flash-lite, well under $0.01 (free on the free tier).
"""
import os
import sys
import tempfile

from google import genai

import llmburn
from llmburn.cli import main as cli_main

MODEL = "gemini-3.5-flash-lite"
PROMPT = "Reply with exactly one word: ok"

if not os.environ.get("GEMINI_API_KEY"):
    sys.exit("Set GEMINI_API_KEY first.")

db = os.path.join(tempfile.mkdtemp(), "smoke.db")
store = llmburn.init(f"sqlite:///{db}")
print("instrumented:", llmburn.instrument())

client = genai.Client()
failures = []


def check(name, condition, detail=""):
    print(("PASS " if condition else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(name)


# 1. Normal call, tagged
with llmburn.track(user_id="alice", plan="pro", feature="smoke"):
    resp = client.models.generate_content(model=MODEL, contents=PROMPT)
check("response still works", bool(resp.text), repr(resp.text.strip()))

# Compare against Google's own token counter (free endpoint)
expected_in = client.models.count_tokens(model=MODEL, contents=PROMPT).total_tokens
alice = {t.key: t for t in store.totals(by="user_id")}.get("alice")
check("normal call recorded", alice is not None and alice.calls == 1)
check("input tokens match Google's count", alice and alice.input_tokens == expected_in,
      f"llmburn={alice and alice.input_tokens}, google={expected_in}")
check("output tokens recorded", alice and alice.output_tokens > 0,
      f"{alice and alice.output_tokens}")
check("cost is positive", alice and alice.cost_usd > 0, f"${alice and alice.cost_usd}")

# 2. Streaming call, recorded once
with llmburn.track(user_id="bob"):
    chunks = list(client.models.generate_content_stream(model=MODEL, contents=PROMPT))
bob = {t.key: t for t in store.totals(by="user_id")}.get("bob")
check("stream recorded exactly once", bob is not None and bob.calls == 1,
      f"{len(chunks)} chunks, {bob and bob.calls} rows")

# 3. Nested tags
with llmburn.track(user_id="carol", plan="free"):
    with llmburn.track(feature="nested"):
        client.models.generate_content(model=MODEL, contents=PROMPT)
features = {t.key for t in store.totals(by="feature")}
check("nested feature tag saved", "nested" in features)

# 4. Untagged call
client.models.generate_content(model=MODEL, contents=PROMPT)
keys = {t.key for t in store.totals(by="user_id")}
check("untagged call saved under no user", None in keys)

print()
cli_main(["report", "--db", f"sqlite:///{db}", "--last", "all"])
print()
if failures:
    sys.exit(f"{len(failures)} check(s) failed: {failures}")
print("All checks passed. llmburn works against the real Gemini API.")
