"""Helpers shared by the OpenAI and Anthropic integrations."""
import logging

from .. import core

log = logging.getLogger("llmburn")


def apply_budget(kwargs, tags=None):
    """Run the budget check and return the model actually used.

    May swap kwargs["model"] for a cheaper one, or raise BudgetExceeded
    (on purpose: that's how on_limit="block" stops the call).
    """
    model = kwargs.get("model")
    use = core.before_call(model, tags)
    if use != model:
        kwargs["model"] = use
    return use


def safe_record(model, usage, tags=None):
    """Record one call from a usage dict. Never raises; no-op if anything is missing."""
    if not model or not usage:
        return
    try:
        core.record(model, tags=tags, **usage)
    except Exception:
        log.exception("llmburn: failed to record call")


def _safely(func, *args):
    try:
        return func(*args)
    except Exception:
        log.exception("llmburn: stream tracking failed")
        return None


class TrackedStream:
    """Proxy around an SDK stream that watches each item and records at the end.

    on_item(item) sees every item; items for which skip(item) is true are
    hidden from the caller. on_done() runs exactly once: when the stream is
    used up, closed, or its with-block ends. Everything else (e.g. .response)
    is passed through to the real stream.
    """

    def __init__(self, inner, on_item, on_done, skip=None):
        self._inner = inner
        self._on_item = on_item
        self._on_done = on_done
        self._skip = skip
        self._finished = False
        self._iter = None
        self._aiter = None

    def _finish(self):
        if not self._finished:
            self._finished = True
            _safely(self._on_done)

    def _show(self, item):
        """Watch one item; True if the caller should see it."""
        _safely(self._on_item, item)
        return not (self._skip is not None and _safely(self._skip, item))

    # sync iteration
    def __iter__(self):
        return self

    def __next__(self):
        if self._iter is None:
            self._iter = iter(self._inner)
        while True:
            try:
                item = next(self._iter)
            except StopIteration:
                self._finish()
                raise
            if self._show(item):
                return item

    # async iteration
    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._aiter is None:
            self._aiter = self._inner.__aiter__()
        while True:
            try:
                item = await self._aiter.__anext__()
            except StopAsyncIteration:
                self._finish()
                raise
            if self._show(item):
                return item

    # context managers
    def __enter__(self):
        if hasattr(self._inner, "__enter__"):
            self._inner.__enter__()
        return self

    def __exit__(self, *exc):
        self._finish()
        if hasattr(self._inner, "__exit__"):
            return self._inner.__exit__(*exc)
        return None

    async def __aenter__(self):
        if hasattr(self._inner, "__aenter__"):
            await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc):
        self._finish()
        if hasattr(self._inner, "__aexit__"):
            return await self._inner.__aexit__(*exc)
        return None

    def close(self):
        """Close the real stream. For async streams, await the result."""
        self._finish()
        return self._inner.close()

    def __getattr__(self, name):
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)
