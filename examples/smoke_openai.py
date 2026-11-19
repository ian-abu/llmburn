"""Real end-to-end check against the live OpenAI API.

Makes 4 small calls (chat, chat stream, responses, responses stream), then
checks that llmburn recorded each one with the right tags and token counts.

    set OPENAI_API_KEY=your-key        (PowerShell: $env:OPENAI_API_KEY="your-key")
    python examples/smoke_openai.py

Cost: a few hundred tokens on gpt-6-luna, well under $0.01.
"""
import os
import sys
import tempfile

from openai import OpenAI

import llmburn
from llmburn.cli import main as cli_main

MODEL = "gpt-6-luna"
PROMPT = "Reply with exactly one word: ok"
MESSAGES = [{"role": "user", "content": PROMPT}]

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY first.")

db = os.path.join(tempfile.mkdtemp(), "smoke.db")
store = llmburn.init(f"sqlite:///{db}")
print("instrumented:", llmburn.instrument())

client = OpenAI()
failures = []


def check(name, condition, detail=""):
    print(("PASS " if condition else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(name)


def user(name):
    return {t.key: t for t in store.totals(by="user_id")}.get(name)


# 1. Chat call, tagged; compare with the usage OpenAI returned
with llmburn.track(user_id="alice", plan="pro", feature="smoke"):
    r = client.chat.completions.create(model=MODEL, messages=MESSAGES)
check("chat response still works", bool(r.choices[0].message.content),
      repr(r.choices[0].message.content))
alice = user("alice")
check("chat call recorded", alice is not None and alice.calls == 1)
check("input tokens match r.usage", alice and alice.input_tokens == r.usage.prompt_tokens,
      f"llmburn={alice and alice.input_tokens}, openai={r.usage.prompt_tokens}")
check("output tokens match r.usage", alice and alice.output_tokens == r.usage.completion_tokens,
      f"llmburn={alice and alice.output_tokens}, openai={r.usage.completion_tokens}")
check("cost is positive", alice and alice.cost_usd > 0, f"${alice and alice.cost_usd}")

# 2. Chat stream: one row, and no usage-only chunk reaches the caller
with llmburn.track(user_id="bob"):
    chunks = list(client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True))
bob = user("bob")
check("chat stream recorded exactly once", bob is not None and bob.calls == 1,
      f"{len(chunks)} chunks, {bob and bob.calls} rows")
check("chat stream tokens recorded", bob and bob.input_tokens > 0 and bob.output_tokens > 0)
check("no chunk without choices reached the caller", all(c.choices for c in chunks))

# 3. Responses API
with llmburn.track(user_id="carol"):
    r = client.responses.create(model=MODEL, input=PROMPT)
carol = user("carol")
check("responses call recorded", carol is not None and carol.calls == 1)
check("responses tokens match r.usage",
      carol and (carol.input_tokens, carol.output_tokens)
      == (r.usage.input_tokens, r.usage.output_tokens),
      f"llmburn={carol and (carol.input_tokens, carol.output_tokens)}, "
      f"openai={(r.usage.input_tokens, r.usage.output_tokens)}")

# 4. Responses stream
with llmburn.track(user_id="dave"):
    events = list(client.responses.create(model=MODEL, input=PROMPT, stream=True))
dave = user("dave")
check("responses stream recorded exactly once", dave is not None and dave.calls == 1,
      f"{len(events)} events, {dave and dave.calls} rows")

print()
cli_main(["report", "--db", f"sqlite:///{db}", "--last", "all"])
print()
if failures:
    sys.exit(f"{len(failures)} check(s) failed: {failures}")
print("All checks passed. llmburn works against the real OpenAI API.")
