"""Real end-to-end check against the live Anthropic API.

Makes 3 small calls (messages.create, create with stream=True, and the
messages.stream() helper), then checks that llmburn recorded each one with
the right tags and token counts.

    set ANTHROPIC_API_KEY=your-key     (PowerShell: $env:ANTHROPIC_API_KEY="your-key")
    python examples/smoke_anthropic.py

Cost: a few hundred tokens on claude-haiku-5-5, well under $0.01.
"""
import os
import sys
import tempfile

from anthropic import Anthropic

import llmburn
from llmburn.cli import main as cli_main

MODEL = "claude-haiku-5-5"
MESSAGES = [{"role": "user", "content": "Reply with exactly one word: ok"}]

if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("Set ANTHROPIC_API_KEY first.")

db = os.path.join(tempfile.mkdtemp(), "smoke.db")
store = llmburn.init(f"sqlite:///{db}")
print("instrumented:", llmburn.instrument())

client = Anthropic()
failures = []


def check(name, condition, detail=""):
    print(("PASS " if condition else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(name)


def user(name):
    return {t.key: t for t in store.totals(by="user_id")}.get(name)


# 1. messages.create, tagged; compare with Anthropic's own token counter (free)
with llmburn.track(user_id="alice", plan="pro", feature="smoke"):
    r = client.messages.create(model=MODEL, max_tokens=10, messages=MESSAGES)
check("response still works", bool(r.content and r.content[0].text), repr(r.content[0].text))
expected_in = client.messages.count_tokens(model=MODEL, messages=MESSAGES).input_tokens
alice = user("alice")
check("create recorded", alice is not None and alice.calls == 1)
check("input tokens match Anthropic's count", alice and alice.input_tokens == expected_in,
      f"llmburn={alice and alice.input_tokens}, anthropic={expected_in}")
check("output tokens match r.usage", alice and alice.output_tokens == r.usage.output_tokens,
      f"llmburn={alice and alice.output_tokens}, anthropic={r.usage.output_tokens}")
check("cost is positive", alice and alice.cost_usd > 0, f"${alice and alice.cost_usd}")

# 2. create(stream=True): one row with the final output count
with llmburn.track(user_id="bob"):
    events = list(client.messages.create(model=MODEL, max_tokens=10, messages=MESSAGES,
                                         stream=True))
bob = user("bob")
check("raw stream recorded exactly once", bob is not None and bob.calls == 1,
      f"{len(events)} events, {bob and bob.calls} rows")
check("raw stream tokens recorded", bob and bob.input_tokens > 0 and bob.output_tokens > 0)

# 3. messages.stream() helper with text_stream
with llmburn.track(user_id="carol"):
    with client.messages.stream(model=MODEL, max_tokens=10, messages=MESSAGES) as stream:
        text = "".join(stream.text_stream)
        final = stream.get_final_message()
carol = user("carol")
check("stream helper text works", bool(text), repr(text))
check("stream helper recorded exactly once", carol is not None and carol.calls == 1)
check("stream helper output matches final message",
      carol and carol.output_tokens == final.usage.output_tokens,
      f"llmburn={carol and carol.output_tokens}, anthropic={final.usage.output_tokens}")

print()
cli_main(["report", "--db", f"sqlite:///{db}", "--last", "all"])
print()
if failures:
    sys.exit(f"{len(failures)} check(s) failed: {failures}")
print("All checks passed. llmburn works against the real Anthropic API.")
