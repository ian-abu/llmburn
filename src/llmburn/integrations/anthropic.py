"""Auto-track every call made through the anthropic SDK.

Wraps messages.create and messages.stream (sync and async). messages.stream
does not go through create (it sends its own request when its with-block is
entered), so it is wrapped separately. Each wrapper checks the budget, calls
the real SDK method, records the cost from the usage Anthropic returns, and
hands back exactly what the SDK returned.
"""
import functools
import logging
from types import SimpleNamespace

from .. import core
from ._common import TrackedStream, apply_budget, safe_record

log = logging.getLogger("llmburn")

_RAW_HEADER = "X-Stainless-Raw-Response"
_USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_read_input_tokens",
                 "cache_creation_input_tokens", "cache_creation")
_originals = {}


def usage_from(usage):
    """Token counts for core.record() from an Anthropic usage object.

    Anthropic's input_tokens leaves out cache reads and writes; llmburn's
    input_tokens is the whole prompt, so they are added back. Cache writes are
    split into 5-minute and 1-hour when the breakdown adds up, otherwise all
    are counted as 5-minute. Returns None if there is no usage.
    """
    if usage is None:
        return None
    read = getattr(usage, "cache_read_input_tokens", None) or 0
    written = getattr(usage, "cache_creation_input_tokens", None) or 0
    writes_5m, writes_1h = written, 0
    breakdown = getattr(usage, "cache_creation", None)
    if breakdown is not None:
        b5 = getattr(breakdown, "ephemeral_5m_input_tokens", None) or 0
        b1 = getattr(breakdown, "ephemeral_1h_input_tokens", None) or 0
        if b5 + b1 == written:
            writes_5m, writes_1h = b5, b1
    return {
        "input_tokens": (usage.input_tokens or 0) + read + written,
        "output_tokens": usage.output_tokens or 0,
        "cached_tokens": read,
        "cache_write_tokens": writes_5m,
        "cache_write_1h_tokens": writes_1h,
    }


def _usage_safe(usage):
    try:
        return usage_from(usage)
    except Exception:
        log.exception("llmburn: failed to read Anthropic usage")
        return None


def _is_raw(kwargs):
    # .with_raw_response / .with_streaming_response: leave those alone.
    headers = kwargs.get("extra_headers")
    return isinstance(headers, dict) and _RAW_HEADER in headers


def _track_create(result, model, tags, kwargs):
    if _is_raw(kwargs):
        return result
    if kwargs.get("stream") is not True:
        safe_record(model, _usage_safe(getattr(result, "usage", None)), tags)
        return result

    seen = {}

    def on_item(event):
        kind = getattr(event, "type", None)
        if kind == "message_start":
            usage = event.message.usage
            seen.update({f: getattr(usage, f, None) for f in _USAGE_FIELDS})
        elif kind == "message_delta" and seen:
            usage = event.usage
            seen["output_tokens"] = usage.output_tokens  # cumulative
            # Newer APIs may also send updated input counts here.
            for f in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                if getattr(usage, f, None) is not None:
                    seen[f] = getattr(usage, f)

    def on_done():
        if seen:
            safe_record(model, _usage_safe(SimpleNamespace(**seen)), tags)

    return TrackedStream(result, on_item, on_done)


class _TrackedManager:
    """Proxy for the manager returned by messages.stream().

    Entering it returns the real MessageStream; leaving it records the usage
    from the stream's final snapshot, then lets the real manager clean up.
    """

    def __init__(self, manager, model, tags):
        self._manager = manager
        self._model = model
        self._tags = tags
        self._stream = None
        self._recorded = False

    def _record(self):
        if self._stream is None or self._recorded:
            return
        self._recorded = True
        try:
            usage = self._stream.current_message_snapshot.usage
        except Exception:
            return  # nothing was received
        safe_record(self._model, _usage_safe(usage), self._tags)

    def __enter__(self):
        self._stream = self._manager.__enter__()
        return self._stream

    def __exit__(self, *exc):
        self._record()
        return self._manager.__exit__(*exc)

    async def __aenter__(self):
        self._stream = await self._manager.__aenter__()
        return self._stream

    async def __aexit__(self, *exc):
        self._record()
        return await self._manager.__aexit__(*exc)

    def __getattr__(self, name):
        manager = self.__dict__.get("_manager")
        if manager is None:
            raise AttributeError(name)
        return getattr(manager, name)


def _wrap_create(orig):
    @functools.wraps(orig)
    def create(self, *args, **kwargs):
        # Captured at call time, so a stream consumed after the track()
        # block still gets the right user.
        tags = core.current_tags()
        model = apply_budget(kwargs, tags)
        result = orig(self, *args, **kwargs)
        return _track_create(result, model, tags, kwargs)
    return create


def _wrap_create_async(orig):
    @functools.wraps(orig)
    async def create(self, *args, **kwargs):
        tags = core.current_tags()
        model = apply_budget(kwargs, tags)
        result = await orig(self, *args, **kwargs)
        return _track_create(result, model, tags, kwargs)
    return create


def _wrap_stream(orig):
    # Sync and async alike: messages.stream() is a plain method in both.
    @functools.wraps(orig)
    def stream(self, *args, **kwargs):
        tags = core.current_tags()
        model = apply_budget(kwargs, tags)
        return _TrackedManager(orig(self, *args, **kwargs), model, tags)
    return stream


def instrument():
    """Patch the anthropic SDK. Safe to call more than once. False if not installed."""
    try:
        from anthropic.resources import messages
    except ImportError:
        return False

    wrappers = {
        (messages.Messages, "create"): _wrap_create,
        (messages.Messages, "stream"): _wrap_stream,
        (messages.AsyncMessages, "create"): _wrap_create_async,
        (messages.AsyncMessages, "stream"): _wrap_stream,
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
