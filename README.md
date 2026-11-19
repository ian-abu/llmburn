# llmburn

Track what each of your users costs you in OpenAI, Anthropic and Gemini calls, and cut them off before they cost too much.

```bash
pip install "llmburn[openai]"      # or [anthropic], [gemini], [all]
```

```python
import llmburn

llmburn.init()        # writes to ./llmburn.db
llmburn.instrument()  # from here on, every OpenAI, Anthropic and Gemini call is recorded
```

Then tag the calls with the user who made them:

```python
with llmburn.track(user_id=user.id, plan=user.plan, feature="summarize"):
    resp = client.chat.completions.create(
        model="gpt-6-luna",
        messages=[{"role": "user", "content": prompt}],
    )
```

And see who's spending what:

```console
$ llmburn report --by user --last 30d
LLM spend by user, the last 30 days

USER    CALLS  INPUT TOK  OUTPUT TOK    COST
--------------------------------------------
u_4821  9,412  9,412,000   3,100,000  $41.20
u_0193  2,201  2,201,000     600,000   $6.75
u_7730    310    150,000      40,000   $0.42
--------------------------------------------
TOTAL  11,923                         $48.37
```

(Example data.) If you charge a flat monthly price, the top row is the one to look at.

## How it works

`instrument()` wraps the create methods of whichever official SDKs are installed: `openai`, `anthropic` and `google-genai`. Each call goes to the provider exactly as before. When the response comes back, llmburn reads the token counts the provider already includes in it, prices them, and writes one row to SQLite. Your code and the response are untouched.

The token counts are the provider's own numbers, not an estimate. Prices come from a bundled table (`prices.json`) that handles reasoning tokens, caching, long-prompt tiers and price changes a provider has announced with a date.

If anything inside llmburn fails (unknown model, locked database), it logs a warning and your request carries on. The one exception is a budget you set to `block`, below. Errors from the provider itself (a bad key, a model name the provider doesn't accept) reach your code unchanged.

## Tagging

`track()` works as a `with` block or a decorator, and tags nest:

```python
with llmburn.track(user_id=user.id, plan="pro"):
    write_cover_letter(job)            # inherits user_id and plan

@llmburn.track(feature="cover_letter")
def write_cover_letter(job): ...       # adds feature
```

Tags are per thread and per async task, so concurrent requests in Flask or FastAPI don't mix up users. Calls with no tags are still recorded, as `(untagged)`.

A Flask route looks like this:

```python
@app.post("/rewrite")
def rewrite():
    with llmburn.track(user_id=g.user.id, plan=g.user.plan, feature="rewrite"):
        resp = client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": request.json["text"]}],
        )
    return {"text": resp.content[0].text}
```

## Budgets

You can cap spending per user, for the whole app, or both. The check runs before the request is sent, so a blocked call never reaches the provider and isn't billed. One budget covers all three SDKs: a user who spent their limit on Claude is blocked on GPT too.

```python
llmburn.init(
    budget_per_user="$2/day",
    budget_total="$50/day",
    on_limit="block",
)
```

```python
try:
    with llmburn.track(user_id=g.user.id):
        resp = client.chat.completions.create(...)
except llmburn.BudgetExceeded as e:
    if e.is_total:
        return {"error": "AI features are paused for today."}, 503
    return {"error": "You've reached today's limit."}, 429
```

`on_limit` can be:

- `"block"`: raise `llmburn.BudgetExceeded`
- `"downgrade"`: send the call to a cheaper model instead (`downgrade_to="gpt-6-luna"`). Pick a model from the same provider as your calls. A Gemini call can't be sent to an OpenAI model.
- `"warn"`: log it and let the call through

Periods are `/hour`, `/day`, `/week` or `/month` (30 days), measured as a rolling window, so `$2/day` means the last 24 hours. Untagged calls only count toward `budget_total`.

## Reports

```bash
llmburn report                           # by user, last 7 days
llmburn report --by feature --last 30d
llmburn report --by model --last 24h
llmburn report --by plan --last all
llmburn report --db sqlite:///path/to/llmburn.db --limit 50
```

Or from Python:

```python
store = llmburn.init()
for row in store.totals(by="user_id"):
    print(row.key, row.calls, row.cost_usd)
```

## What's stored

One row per call: timestamp (UTC), `user_id`, `plan`, `feature`, model, input/output/cached tokens, and cost in USD. Prompts and responses are never stored.

## Supported

| SDK | Methods | Streaming |
| --- | --- | --- |
| `openai` | `chat.completions.create`, `responses.create` (and the `.stream()` helpers, which use them) | yes |
| `anthropic` | `messages.create`, `messages.stream` | yes |
| `google-genai` | `models.generate_content`, `models.generate_content_stream` | yes |

Async clients work the same way. A stream is recorded once, as one row, when it ends: when you read it to the end, close it, or leave its `with` block. For OpenAI chat streams, llmburn asks for the token count with `include_usage` and hides the extra chunk that comes back, so your code sees the same chunks as before.

## Limitations

- Prices ship with the package. When a provider changes them, you need a new llmburn release until automatic updates land.
- Budgets can overshoot by a call or two when several requests for the same user arrive at once.
- Standard-tier prices only. Batch, Flex, priority and regional pricing aren't applied.
- Text generation only. Image, audio and embedding calls aren't tracked, and models without a price are skipped with a warning.
- OpenAI's `.parse()` helpers and `.with_raw_response` aren't tracked yet.

## Planned

- Flask, FastAPI and Django middleware that sets `user_id` for you
- Automatic price-table updates
- A hosted dashboard with revenue per customer next to cost

Python 3.9+. No required dependencies. MIT license.
