"""Auto-track every call made through the openai SDK.

Wraps chat.completions.create and responses.create (sync and async). Each
wrapper checks the budget, calls the real SDK method, reads the usage the
API returns, records the cost, and hands back exactly what the SDK returned.
The SDK's own .stream() helpers go through create, so they are tracked too.
"""
import functools
import logging

from .. import core
from ._common import TrackedStream, apply_budget, safe_record

log = logging.getLogger("llmburn")

_RAW_HEADER = "X-Stainless-Raw-Response"
_RESPONSE_DONE = ("response.completed", "response.incomplete", "response.failed")
_originals = {}


def usage_from(usage):
    """Token counts for core.record() from a Chat Completions or Responses usage object.

    Input includes cached and cache-written tokens; output already includes
    reasoning tokens. Cache writes are 30-minute unless the Responses API
    reports 12-hour ones (Chat Completions doesn't split them). Returns None
    if there is no usage.
    """
    if usage is None:
        return None
    if hasattr(usage, "prompt_tokens"):  # Chat Completions
        details = getattr(usage, "prompt_tokens_details", None)
        input_tokens, output_tokens = usage.prompt_tokens, usage.completion_tokens
    else:  # Responses
        details = getattr(usage, "input_tokens_details", None)
        input_tokens, output_tokens = usage.input_tokens, usage.output_tokens
    written = getattr(details, "cache_write_tokens", None) or 0
    writes_30m, writes_12h = written, 0
    ttl = getattr(details, "cache_write_tokens_details", None)
    if ttl is not None:
        t30 = getattr(ttl, "ttl_30m_tokens", None) or 0
        t12 = getattr(ttl, "ttl_12h_tokens", None) or 0
        if t30 + t12 == written:
            writes_30m, writes_12h = t30, t12
    return {
        "input_tokens": input_tokens or 0,
        "output_tokens": output_tokens or 0,
        "cached_tokens": getattr(details, "cached_tokens", None) or 0,
        "cache_write_tokens": writes_30m,
        "cache_write_12h_tokens": writes_12h,
    }


def _usage_of(obj):
    try:
        return usage_from(getattr(obj, "usage", None))
    except Exception:
        log.exception("llmburn: failed to read OpenAI usage")
        return None


def _is_raw(kwargs):
    # .with_raw_response / .with_streaming_response return the HTTP response,
    # not the parsed result; leave those alone.
    headers = kwargs.get("extra_headers")
    return isinstance(headers, dict) and _RAW_HEADER in headers


def _prepare_chat_stream(kwargs):
    """Ask for usage on chat streams. True if llmburn added include_usage."""
    try:
        if kwargs.get("stream") is not True or _is_raw(kwargs):
            return False
        options = kwargs.get("stream_options")
        if isinstance(options, dict) and options.get("include_usage"):
            return False  # the caller asked for usage themselves
        base = options if isinstance(options, dict) else {}
        kwargs["stream_options"] = {**base, "include_usage": True}
        return True
    except Exception:
        log.exception("llmburn: could not request stream usage")
        return False


def _track_chat(result, model, tags, kwargs, injected):
    if _is_raw(kwargs):
        return result
    if kwargs.get("stream") is not True:
        safe_record(model, _usage_of(result), tags)
        return result

    seen = {}

    def on_item(chunk):
        if getattr(chunk, "usage", None) is not None:
            seen["usage"] = _usage_of(chunk)

    def skip(chunk):
        # The usage-only final chunk exists only because llmburn asked for it.
        return injected and not chunk.choices and chunk.usage is not None

    return TrackedStream(result, on_item,
                         lambda: safe_record(model, seen.get("usage"), tags), skip)


def _track_responses(result, model, tags, kwargs):
    if _is_raw(kwargs):
        return result
    if kwargs.get("stream") is not True:
        safe_record(model, _usage_of(result), tags)
        return result

    seen = {}

    def on_item(event):
        if getattr(event, "type", None) in _RESPONSE_DONE:
            seen["usage"] = _usage_of(event.response)

    return TrackedStream(result, on_item, lambda: safe_record(model, seen.get("usage"), tags))


def _wrap_chat(orig):
    @functools.wraps(orig)
    def create(self, *args, **kwargs):
        # Captured at call time, so a stream consumed after the track()
        # block still gets the right user.
        tags = core.current_tags()
        model = apply_budget(kwargs, tags)
        injected = _prepare_chat_stream(kwargs)
        result = orig(self, *args, **kwargs)
        return _track_chat(result, model, tags, kwargs, injected)
    return create


def _wrap_chat_async(orig):
    @functools.wraps(orig)
    async def create(self, *args, **kwargs):
        tags = core.current_tags()
        model = apply_budget(kwargs, tags)
        injected = _prepare_chat_stream(kwargs)
        result = await orig(self, *args, **kwargs)
        return _track_chat(result, model, tags, kwargs, injected)
    return create


def _wrap_responses(orig):
    @functools.wraps(orig)
    def create(self, *args, **kwargs):
        tags = core.current_tags()
        model = apply_budget(kwargs, tags)
        result = orig(self, *args, **kwargs)
        return _track_responses(result, model, tags, kwargs)
    return create


def _wrap_responses_async(orig):
    @functools.wraps(orig)
    async def create(self, *args, **kwargs):
        tags = core.current_tags()
        model = apply_budget(kwargs, tags)
        result = await orig(self, *args, **kwargs)
        return _track_responses(result, model, tags, kwargs)
    return create


def instrument():
    """Patch the openai SDK. Safe to call more than once. False if not installed."""
    try:
        from openai.resources import responses
        from openai.resources.chat import completions
    except ImportError:
        return False

    wrappers = {
        (completions.Completions, "create"): _wrap_chat,
        (completions.AsyncCompletions, "create"): _wrap_chat_async,
        (responses.Responses, "create"): _wrap_responses,
        (responses.AsyncResponses, "create"): _wrap_responses_async,
    }
    for (cls, name), wrap in wrappers.items():
        if (cls, name) in _originals:
            continue  # already patched
        orig = getattr(cls, name)
        _originals[(cls, name)] = orig
        setattr(cls, name, wrap(orig))
    return True


def uninstrument():
    """Restore the original SDK methods."""
    for (cls, name), orig in _originals.items():
        setattr(cls, name, orig)
    _originals.clear()
