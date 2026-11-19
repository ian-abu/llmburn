"""Auto-track every call made through the google-genai SDK.

Wraps Models.generate_content / generate_content_stream and their async
versions. Each wrapper calls the real SDK method, reads usage_metadata from
the response, records the cost, and returns the response unchanged.
"""
import functools
import logging

from .. import core

log = logging.getLogger("llmburn")

_METHODS = ("generate_content", "generate_content_stream")
_originals = {}


def _usage(response):
    """(input, output, cached) tokens from a response, or None if missing."""
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return None
    # Tool-use prompt tokens are billed as input; thinking tokens as output.
    input_tokens = (meta.prompt_token_count or 0) + (meta.tool_use_prompt_token_count or 0)
    output_tokens = (meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0)
    cached_tokens = meta.cached_content_token_count or 0
    return input_tokens, output_tokens, cached_tokens


def _record(model, response, tags=None):
    try:
        usage = _usage(response)
        if usage is not None:
            core.record(model, *usage, tags=tags)
    except Exception:
        log.exception("llmburn: failed to read Gemini usage")


def _apply_budget(kwargs, tags=None):
    """Run the budget check; may swap kwargs["model"] or raise BudgetExceeded."""
    model = kwargs.get("model")
    use = core.before_call(model, tags)
    if use != model:
        kwargs["model"] = use
    return use


def _wrap_sync(orig):
    @functools.wraps(orig)
    def generate_content(self, *args, **kwargs):
        model = _apply_budget(kwargs)
        response = orig(self, *args, **kwargs)
        _record(model, response)
        return response
    return generate_content


def _wrap_sync_stream(orig):
    @functools.wraps(orig)
    def generate_content_stream(self, *args, **kwargs):
        # A plain function, not a generator: this body runs at call time, so
        # the tags are captured inside the track() block even if the stream
        # is consumed later (e.g. by a Flask streaming response).
        tags = core.current_tags()
        model = _apply_budget(kwargs, tags)
        stream = orig(self, *args, **kwargs)

        def tracked():
            last = None
            try:
                for chunk in stream:
                    if getattr(chunk, "usage_metadata", None) is not None:
                        last = chunk
                    yield chunk
            finally:
                # Runs on normal end, early break, or error: record what we saw.
                if last is not None:
                    _record(model, last, tags)

        return tracked()
    return generate_content_stream


def _wrap_async(orig):
    @functools.wraps(orig)
    async def generate_content(self, *args, **kwargs):
        model = _apply_budget(kwargs)
        response = await orig(self, *args, **kwargs)
        _record(model, response)
        return response
    return generate_content


def _wrap_async_stream(orig):
    @functools.wraps(orig)
    async def generate_content_stream(self, *args, **kwargs):
        tags = core.current_tags()
        model = _apply_budget(kwargs, tags)
        stream = await orig(self, *args, **kwargs)

        async def tracked():
            last = None
            try:
                async for chunk in stream:
                    if getattr(chunk, "usage_metadata", None) is not None:
                        last = chunk
                    yield chunk
            finally:
                if last is not None:
                    _record(model, last, tags)

        return tracked()
    return generate_content_stream


def instrument():
    """Patch google-genai. Safe to call more than once. False if not installed."""
    try:
        from google.genai import models
    except ImportError:
        return False

    wrappers = {
        (models.Models, "generate_content"): _wrap_sync,
        (models.Models, "generate_content_stream"): _wrap_sync_stream,
        (models.AsyncModels, "generate_content"): _wrap_async,
        (models.AsyncModels, "generate_content_stream"): _wrap_async_stream,
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
