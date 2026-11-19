"""Turn token counts into US dollars using the bundled price table."""
import json
import re
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

_PRICES_FILE = Path(__file__).with_name("prices.json")
# What may follow "<key>-" in a versioned model name: -001, -20251001,
# -2024-08-06, -latest, -preview-05-20, -exp-03-25.
_VERSION_SUFFIX = re.compile(r"(\d{3}|\d{8}|\d{4}-\d{2}-\d{2}|latest|(preview|exp)(-\d{2,4})*)")


class UnknownModelError(KeyError):
    """The model isn't in the price table."""


@lru_cache(maxsize=1)
def _load_models():
    with open(_PRICES_FILE, encoding="utf-8") as f:
        return json.load(f)["models"]


def _resolve_model(model):
    """Map an SDK model name to a price-table key.

    Handles "models/gemini-2.5-flash" and versioned names like
    "gemini-2.5-flash-001" or "claude-haiku-4-5-20251001". A longer name only
    matches a key when the rest is a version suffix, so "gemini-2.5-flash-image"
    is NOT priced as "gemini-2.5-flash". The longest valid match wins.
    """
    models = _load_models()
    name = model.removeprefix("models/")
    if name in models:
        return name
    matches = [
        key for key in models
        if name.startswith(key + "-") and _VERSION_SUFFIX.fullmatch(name[len(key) + 1:])
    ]
    if not matches:
        raise UnknownModelError(model)
    return max(matches, key=len)


def _price_on(periods, day):
    """Pick the latest price period that has started by `day`."""
    active = None
    for period in periods:
        start = period.get("effective_from")
        if start is None or date.fromisoformat(start) <= day:
            if active is None or (start or "") >= (active.get("effective_from") or ""):
                active = period
    return active


def cost(model, input_tokens, output_tokens, cached_tokens=0, at=None,
         cache_write_tokens=0, cache_write_1h_tokens=0, cache_write_12h_tokens=0):
    """Return the USD cost of one LLM call.

    input_tokens          -- total prompt tokens, INCLUDING cached and
                             cache-written ones
    output_tokens         -- response tokens plus thinking/reasoning tokens
    cached_tokens         -- the part of input_tokens read from the cache
    at                    -- when the call happened (date or datetime); default today
    cache_write_tokens    -- the part of input_tokens written to the cache
                             (Anthropic: 5-minute cache, OpenAI: 30-minute)
    cache_write_1h_tokens -- the part written to Anthropic's 1-hour cache
    cache_write_12h_tokens -- the part written to OpenAI's 12-hour cache
    """
    key = _resolve_model(model)
    day = at.date() if isinstance(at, datetime) else (at or date.today())
    price = _price_on(_load_models()[key]["prices"], day)

    long_context = price.get("long_context")
    if long_context and input_tokens > long_context["above"]:
        price = {**price, **long_context}

    # Split the prompt into parts; clamp so fresh input never goes negative.
    cached = min(cached_tokens, input_tokens)
    writes_5m = min(cache_write_tokens, input_tokens - cached)
    writes_1h = min(cache_write_1h_tokens, input_tokens - cached - writes_5m)
    writes_12h = min(cache_write_12h_tokens, input_tokens - cached - writes_5m - writes_1h)
    fresh = input_tokens - cached - writes_5m - writes_1h - writes_12h

    usd = (
        fresh * price["input"]
        + cached * price["cached_input"]
        + writes_5m * price.get("cache_write", price["input"])
        + writes_1h * price.get("cache_write_1h", price["input"])
        + writes_12h * price.get("cache_write_12h", price["input"])
        + output_tokens * price["output"]
    ) / 1_000_000
    return round(usd, 8)
