import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("openai")
import openai  # noqa: E402
from openai.resources import responses  # noqa: E402
from openai.resources.chat import completions  # noqa: E402
from openai.types import CompletionUsage  # noqa: E402
from openai.types.completion_usage import PromptTokensDetails  # noqa: E402
from openai.types.responses import ResponseUsage  # noqa: E402
from openai.types.responses.response_usage import (  # noqa: E402
    InputTokensDetails, OutputTokensDetails)

import llmburn  # noqa: E402
from llmburn.integrations import openai as oa  # noqa: E402

MODEL = "gpt-6.1-terra"  # $1.00 in / $0.05 cached / $5.00 out


def chat_usage(prompt=1000, out=500, cached=0, written=0):
    return CompletionUsage(
        prompt_tokens=prompt, completion_tokens=out, total_tokens=prompt + out,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=cached, cache_write_tokens=written))


def resp_usage(prompt=1000, out=500, cached=0):
    return ResponseUsage(
        input_tokens=prompt, output_tokens=out, total_tokens=prompt + out,
        input_tokens_details=InputTokensDetails(cached_tokens=cached, cache_write_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0))


def chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))],
                           usage=None)


def usage_chunk():
    return SimpleNamespace(choices=[], usage=chat_usage(out=300))


class FakeStream:
    """Stands in for openai.Stream: iterable, closable, a context manager, has .response."""

    def __init__(self, items):
        self._items = list(items)
        self.response = SimpleNamespace(status_code=200)
        self.closed = False

    def __iter__(self):
        return iter(self._items)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.closed = True


class FakeAsyncStream(FakeStream):
    def __aiter__(self):
        async def gen():
            for item in self._items:
                yield item
        return gen()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True


def chat_stream_items(include_usage):
    items = [chunk("Hel"), chunk("lo")]
    if include_usage:
        items.append(usage_chunk())
    return items


def response_events():
    done = SimpleNamespace(type="response.completed",
                           response=SimpleNamespace(usage=resp_usage(out=40)))
    return [SimpleNamespace(type="response.output_text.delta", delta="hi"), done]


@pytest.fixture
def sent():
    return []


@pytest.fixture
def fake_sdk(monkeypatch, sent):
    """Replace the network methods with fakes, then instrument them."""
    def fake_chat(self, **kw):
        sent.append(kw)
        if kw.get("stream"):
            opts = kw.get("stream_options") or {}
            return FakeStream(chat_stream_items(opts.get("include_usage")))
        return SimpleNamespace(id="c1", usage=chat_usage(cached=200, written=100),
                               choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))])

    async def fake_achat(self, **kw):
        sent.append(kw)
        if kw.get("stream"):
            opts = kw.get("stream_options") or {}
            return FakeAsyncStream(chat_stream_items(opts.get("include_usage")))
        return SimpleNamespace(id="c1", usage=chat_usage())

    def fake_resp(self, **kw):
        sent.append(kw)
        if kw.get("stream"):
            return FakeStream(response_events())
        return SimpleNamespace(id="r1", output_text="hi", usage=resp_usage(cached=100))

    async def fake_aresp(self, **kw):
        sent.append(kw)
        if kw.get("stream"):
            return FakeAsyncStream(response_events())
        return SimpleNamespace(id="r1", output_text="hi", usage=resp_usage())

    monkeypatch.setattr(completions.Completions, "create", fake_chat)
    monkeypatch.setattr(completions.AsyncCompletions, "create", fake_achat)
    monkeypatch.setattr(responses.Responses, "create", fake_resp)
    monkeypatch.setattr(responses.AsyncResponses, "create", fake_aresp)

    store = llmburn.init(":memory:")
    assert "openai" in llmburn.instrument()
    yield store
    store.close()
    llmburn.init(":memory:")


@pytest.fixture
def client():
    return openai.OpenAI(api_key="x")


@pytest.fixture
def aclient():
    return openai.AsyncOpenAI(api_key="x")


def msgs():
    return [{"role": "user", "content": "hi"}]


# --- Chat Completions --------------------------------------------------------

def test_chat_call_is_tracked_with_exact_cost(fake_sdk, client):
    with llmburn.track(user_id="alice", feature="chat"):
        r = client.chat.completions.create(model=MODEL, messages=msgs())

    assert r.id == "c1"
    total = fake_sdk.totals()[0]
    assert (total.key, total.calls, total.input_tokens, total.output_tokens) == ("alice", 1, 1000, 500)
    # 700 fresh at $1, 100 written (30-minute) at $1.25, 200 cached at $0.05, 500 out at $5
    assert total.cost_usd == pytest.approx(
        (700 * 1.00 + 100 * 1.25 + 200 * 0.05 + 500 * 5.00) / 1e6)
    assert fake_sdk.totals(by="feature")[0].key == "chat"


def test_chat_stream_records_once_and_hides_injected_usage_chunk(fake_sdk, client, sent):
    with llmburn.track(user_id="bob"):
        chunks = list(client.chat.completions.create(model=MODEL, messages=msgs(), stream=True))

    assert sent[0]["stream_options"] == {"include_usage": True}
    assert len(chunks) == 2 and all(c.choices for c in chunks)
    total = fake_sdk.totals()[0]
    assert (total.calls, total.output_tokens) == (1, 300)


def test_chat_stream_keeps_usage_chunk_when_caller_asked(fake_sdk, client, sent):
    stream = client.chat.completions.create(
        model=MODEL, messages=msgs(), stream=True,
        stream_options={"include_usage": True, "include_obfuscation": False})
    chunks = list(stream)

    assert sent[0]["stream_options"] == {"include_usage": True, "include_obfuscation": False}
    assert len(chunks) == 3 and chunks[-1].usage is not None
    assert fake_sdk.totals()[0].calls == 1


def test_chat_stream_merges_existing_stream_options(fake_sdk, client, sent):
    list(client.chat.completions.create(model=MODEL, messages=msgs(), stream=True,
                                        stream_options={"include_obfuscation": False}))
    assert sent[0]["stream_options"] == {"include_obfuscation": False, "include_usage": True}


def test_stream_consumed_after_track_block_keeps_tags(fake_sdk, client):
    with llmburn.track(user_id="carol"):
        stream = client.chat.completions.create(model=MODEL, messages=msgs(), stream=True)
    list(stream)  # e.g. a Flask streaming response
    assert fake_sdk.totals()[0].key == "carol"


def test_stream_as_context_manager_passes_attributes_through(fake_sdk, client):
    with client.chat.completions.create(model=MODEL, messages=msgs(), stream=True) as stream:
        assert stream.response.status_code == 200
        for _ in stream:
            pass
    assert stream.closed
    assert fake_sdk.totals()[0].calls == 1


def test_stream_closed_early_records_once(fake_sdk, client):
    stream = client.chat.completions.create(model=MODEL, messages=msgs(), stream=True)
    next(stream)
    stream.close()
    stream.close()
    assert fake_sdk.totals() == []  # usage hadn't arrived yet: nothing to record


# --- Responses API -----------------------------------------------------------

def test_responses_call(fake_sdk, client):
    with llmburn.track(user_id="dave"):
        r = client.responses.create(model=MODEL, input="hi")
    assert r.output_text == "hi"
    total = fake_sdk.totals()[0]
    assert (total.key, total.input_tokens, total.output_tokens) == ("dave", 1000, 500)
    assert total.cost_usd == pytest.approx((900 * 1.00 + 100 * 0.05 + 500 * 5.00) / 1e6)


def test_responses_stream(fake_sdk, client):
    with llmburn.track(user_id="erin"):
        events = list(client.responses.create(model=MODEL, input="hi", stream=True))
    assert [e.type for e in events] == ["response.output_text.delta", "response.completed"]
    total = fake_sdk.totals()[0]
    assert (total.key, total.calls, total.output_tokens) == ("erin", 1, 40)


# --- async -------------------------------------------------------------------

def test_async_chat(fake_sdk, aclient):
    async def run():
        with llmburn.track(user_id="fay"):
            return await aclient.chat.completions.create(model=MODEL, messages=msgs())
    assert asyncio.run(run()).id == "c1"
    assert fake_sdk.totals()[0].key == "fay"


def test_async_chat_stream(fake_sdk, aclient):
    async def run():
        with llmburn.track(user_id="gus"):
            stream = await aclient.chat.completions.create(
                model=MODEL, messages=msgs(), stream=True)
        async with stream:
            return [c async for c in stream]
    chunks = asyncio.run(run())
    assert len(chunks) == 2
    total = fake_sdk.totals()[0]
    assert (total.key, total.calls, total.output_tokens) == ("gus", 1, 300)


def test_async_responses_stream(fake_sdk, aclient):
    async def run():
        stream = await aclient.responses.create(model=MODEL, input="hi", stream=True)
        return [e async for e in stream]
    assert len(asyncio.run(run())) == 2
    assert fake_sdk.totals()[0].output_tokens == 40


# --- robustness and budgets --------------------------------------------------

def test_instrument_twice_does_not_double_count(fake_sdk, client):
    llmburn.instrument()
    client.chat.completions.create(model=MODEL, messages=msgs())
    assert fake_sdk.totals()[0].calls == 1


def test_unknown_model_does_not_break_the_call(fake_sdk, client):
    r = client.chat.completions.create(model="gpt-99-ultra", messages=msgs())
    assert r.id == "c1"
    assert fake_sdk.totals() == []


def test_sdk_errors_still_reach_the_app(monkeypatch, fake_sdk, client):
    oa.uninstrument()

    def failing(self, **kw):
        raise RuntimeError("404 model not found")

    monkeypatch.setattr(completions.Completions, "create", failing)
    llmburn.instrument()
    with pytest.raises(RuntimeError, match="404"):
        client.chat.completions.create(model=MODEL, messages=msgs())


def _overspend(store, user):
    from llmburn.storage import Call
    store.save(Call(model=MODEL, input_tokens=1, output_tokens=1, cost_usd=5.0, user_id=user))


def test_block_stops_call_before_it_reaches_openai(fake_sdk, client, sent):
    store = llmburn.init(":memory:", budget_per_user="$1/day", on_limit="block")
    _overspend(store, "alice")
    with llmburn.track(user_id="alice"):
        with pytest.raises(llmburn.BudgetExceeded):
            client.chat.completions.create(model=MODEL, messages=msgs())
        with pytest.raises(llmburn.BudgetExceeded):
            client.responses.create(model=MODEL, input="hi", stream=True)
    assert sent == []


def test_downgrade_sends_cheaper_model(fake_sdk, client, sent):
    store = llmburn.init(":memory:", budget_per_user="$1/day", on_limit="downgrade",
                         downgrade_to="gpt-6-luna")
    _overspend(store, "alice")
    with llmburn.track(user_id="alice"):
        client.chat.completions.create(model="gpt-6-astra", messages=msgs())
    assert sent[0]["model"] == "gpt-6-luna"
    assert "gpt-6-luna" in {t.key for t in store.totals(by="model")}


def test_usage_from_treats_none_as_zero():
    u = CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    assert oa.usage_from(u) == {"input_tokens": 10, "output_tokens": 5, "cached_tokens": 0,
                                "cache_write_tokens": 0, "cache_write_12h_tokens": 0}
    assert oa.usage_from(None) is None


def test_responses_usage_splits_30m_and_12h_writes():
    from openai.types.responses.response_usage import InputTokensDetailsCacheWriteTokensDetails
    u = ResponseUsage(
        input_tokens=1000, output_tokens=5, total_tokens=1005,
        input_tokens_details=InputTokensDetails(
            cached_tokens=0, cache_write_tokens=300,
            cache_write_tokens_details=InputTokensDetailsCacheWriteTokensDetails(
                ttl_30m_tokens=100, ttl_12h_tokens=200)),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0))
    got = oa.usage_from(u)
    assert (got["cache_write_tokens"], got["cache_write_12h_tokens"]) == (100, 200)
