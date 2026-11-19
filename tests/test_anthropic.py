import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("anthropic")
import anthropic  # noqa: E402
from anthropic.resources import messages  # noqa: E402
from anthropic.types import CacheCreation, MessageDeltaUsage, Usage  # noqa: E402

import llmburn  # noqa: E402
from llmburn.integrations import anthropic as an  # noqa: E402

MODEL = "claude-sonnet-5-5"  # $2 in / $2.50 5m write / $4 1h write / $0.10 read / $10 out


def usage(inp=100, out=50, read=1000, w5=300, w1=200):
    return Usage(
        input_tokens=inp, output_tokens=out,
        cache_read_input_tokens=read, cache_creation_input_tokens=w5 + w1,
        cache_creation=CacheCreation(ephemeral_5m_input_tokens=w5, ephemeral_1h_input_tokens=w1))


# 100 fresh*$2 + 1000 read*$0.10 + 300 5m*$2.50 + 200 1h*$4 + 50 out*$10
EXPECTED_USD = (100 * 2 + 1000 * 0.10 + 300 * 2.50 + 200 * 4 + 50 * 10) / 1e6


def raw_events():
    return [
        SimpleNamespace(type="message_start",
                        message=SimpleNamespace(usage=usage(out=1))),
        SimpleNamespace(type="content_block_delta",
                        delta=SimpleNamespace(type="text_delta", text="hi")),
        SimpleNamespace(type="message_delta", usage=MessageDeltaUsage(output_tokens=20)),
        SimpleNamespace(type="message_delta", usage=MessageDeltaUsage(output_tokens=50)),
        SimpleNamespace(type="message_stop"),
    ]


class FakeStream:
    def __init__(self, items):
        self._items = list(items)
        self.response = SimpleNamespace(status_code=200)

    def __iter__(self):
        return iter(self._items)

    def __aiter__(self):
        async def gen():
            for item in self._items:
                yield item
        return gen()

    def close(self):
        pass


class FakeMessageStream:
    """Stands in for anthropic's MessageStream (what the helper's with-block gives you)."""

    def __init__(self, final_usage):
        self._final = final_usage
        self._received = False
        self.text_stream = self._text()

    def _text(self):
        self._received = True
        yield "Hel"
        yield "lo"

    async def _atext(self):
        for t in self._text():
            yield t

    @property
    def current_message_snapshot(self):
        assert self._received  # the real SDK asserts too before anything arrives
        return SimpleNamespace(usage=self._final)


class FakeManager:
    def __init__(self, final_usage, sent, kw):
        self._final, self._sent, self._kw = final_usage, sent, kw
        self.exited = False
        self.stream = None

    def __enter__(self):
        self._sent.append(self._kw)  # the real manager sends the request here
        self.stream = FakeMessageStream(self._final)
        return self.stream

    def __exit__(self, *exc):
        self.exited = True

    async def __aenter__(self):
        s = self.__enter__()
        s.text_stream = s._atext()
        return s

    async def __aexit__(self, *exc):
        self.exited = True


@pytest.fixture
def sent():
    return []


@pytest.fixture
def fake_sdk(monkeypatch, sent):
    def fake_create(self, **kw):
        sent.append(kw)
        if kw.get("stream"):
            return FakeStream(raw_events())
        return SimpleNamespace(id="m1", content=[SimpleNamespace(text="hi")], usage=usage())

    async def fake_acreate(self, **kw):
        return fake_create(self, **kw)

    def fake_stream(self, **kw):
        return FakeManager(usage(), sent, kw)  # nothing sent until __enter__

    monkeypatch.setattr(messages.Messages, "create", fake_create)
    monkeypatch.setattr(messages.AsyncMessages, "create", fake_acreate)
    monkeypatch.setattr(messages.Messages, "stream", fake_stream)
    monkeypatch.setattr(messages.AsyncMessages, "stream", fake_stream)

    store = llmburn.init(":memory:")
    assert "anthropic" in llmburn.instrument()
    yield store
    store.close()
    llmburn.init(":memory:")


@pytest.fixture
def client():
    return anthropic.Anthropic(api_key="x")


@pytest.fixture
def aclient():
    return anthropic.AsyncAnthropic(api_key="x")


def call_kw(**extra):
    return dict(model=MODEL, max_tokens=10, messages=[{"role": "user", "content": "hi"}], **extra)


def test_create_is_tracked_with_cache_reads_and_writes(fake_sdk, client):
    with llmburn.track(user_id="alice", plan="pro"):
        r = client.messages.create(**call_kw())

    assert r.id == "m1"
    total = fake_sdk.totals()[0]
    assert (total.key, total.calls, total.input_tokens, total.output_tokens) == ("alice", 1, 1600, 50)
    assert total.cost_usd == pytest.approx(EXPECTED_USD)


def test_create_stream_records_once_with_final_output(fake_sdk, client):
    with llmburn.track(user_id="bob"):
        events = list(client.messages.create(**call_kw(stream=True)))

    assert len(events) == 5  # nothing hidden
    total = fake_sdk.totals()[0]
    assert (total.calls, total.input_tokens, total.output_tokens) == (1, 1600, 50)
    assert total.cost_usd == pytest.approx(EXPECTED_USD)


def test_create_stream_consumed_after_track_block_keeps_tags(fake_sdk, client):
    with llmburn.track(user_id="carol"):
        stream = client.messages.create(**call_kw(stream=True))
    with stream:
        assert stream.response.status_code == 200
        for _ in stream:
            pass
    total = fake_sdk.totals()[0]
    assert (total.key, total.calls) == ("carol", 1)


def test_stream_helper_records_on_exit(fake_sdk, client, sent):
    with llmburn.track(user_id="dave"):
        manager = client.messages.stream(**call_kw())
        assert sent == []  # the helper only sends on enter
        with manager as stream:
            text = "".join(stream.text_stream)

    assert text == "Hello"
    total = fake_sdk.totals()[0]
    assert (total.key, total.calls, total.input_tokens) == ("dave", 1, 1600)


def test_stream_helper_with_nothing_received_records_nothing(fake_sdk, client):
    with client.messages.stream(**call_kw()):
        pass
    assert fake_sdk.totals() == []


def test_stream_helper_attribute_passthrough(fake_sdk, client):
    manager = client.messages.stream(**call_kw())
    assert manager.exited is False  # from the real (fake) manager


def test_async_create_and_stream(fake_sdk, aclient):
    async def run():
        with llmburn.track(user_id="erin"):
            await aclient.messages.create(**call_kw())
            stream = await aclient.messages.create(**call_kw(stream=True))
        return [e async for e in stream]

    assert len(asyncio.run(run())) == 5
    total = fake_sdk.totals()[0]
    assert (total.key, total.calls) == ("erin", 2)


def test_async_stream_helper(fake_sdk, aclient):
    async def run():
        with llmburn.track(user_id="fay"):
            async with aclient.messages.stream(**call_kw()) as stream:
                return "".join([t async for t in stream.text_stream])

    assert asyncio.run(run()) == "Hello"
    assert fake_sdk.totals()[0].key == "fay"


def test_usage_without_breakdown_counts_writes_as_5m():
    u = Usage(input_tokens=10, output_tokens=5, cache_read_input_tokens=0,
              cache_creation_input_tokens=100)
    assert an.usage_from(u) == {"input_tokens": 110, "output_tokens": 5, "cached_tokens": 0,
                                "cache_write_tokens": 100, "cache_write_1h_tokens": 0}


def test_usage_with_mismatched_breakdown_counts_writes_as_5m():
    u = usage(w5=10, w1=10)
    u.cache_creation_input_tokens = 50
    assert an.usage_from(u)["cache_write_tokens"] == 50
    assert an.usage_from(u)["cache_write_1h_tokens"] == 0


def test_unknown_model_does_not_break_the_call(fake_sdk, client):
    r = client.messages.create(**{**call_kw(), "model": "claude-ultra-99"})
    assert r.id == "m1"
    assert fake_sdk.totals() == []


def _overspend(store, user):
    from llmburn.storage import Call
    store.save(Call(model=MODEL, input_tokens=1, output_tokens=1, cost_usd=5.0, user_id=user))


def test_block_stops_call_before_it_reaches_anthropic(fake_sdk, client, sent):
    store = llmburn.init(":memory:", budget_per_user="$1/day", on_limit="block")
    _overspend(store, "alice")
    with llmburn.track(user_id="alice"):
        with pytest.raises(llmburn.BudgetExceeded):
            client.messages.create(**call_kw())
        with pytest.raises(llmburn.BudgetExceeded):
            client.messages.stream(**call_kw())
    assert sent == []


def test_downgrade_sends_cheaper_model(fake_sdk, client, sent):
    store = llmburn.init(":memory:", budget_per_user="$1/day", on_limit="downgrade",
                         downgrade_to="claude-haiku-5-5")
    _overspend(store, "alice")
    with llmburn.track(user_id="alice"):
        client.messages.create(**call_kw())
        with client.messages.stream(**call_kw()) as stream:
            list(stream.text_stream)
    assert [kw["model"] for kw in sent] == ["claude-haiku-5-5", "claude-haiku-5-5"]
    assert "claude-haiku-5-5" in {t.key for t in store.totals(by="model")}


# --- one budget across providers --------------------------------------------

def test_spending_on_anthropic_blocks_the_user_on_openai(monkeypatch, fake_sdk, client):
    openai = pytest.importorskip("openai")
    from openai.resources.chat import completions

    from llmburn.integrations import openai as oa

    openai_sent = []

    def fake_chat(self, **kw):
        openai_sent.append(kw)
        return SimpleNamespace(id="c1", usage=None)

    # Unpatch first, so the fakes (not llmburn's wrappers) are what gets wrapped.
    an.uninstrument()
    oa.uninstrument()
    big = SimpleNamespace(id="m2", usage=Usage(input_tokens=1_000_000, output_tokens=0))
    monkeypatch.setattr(messages.Messages, "create", lambda self, **kw: big)
    monkeypatch.setattr(completions.Completions, "create", fake_chat)
    store = llmburn.init(":memory:", budget_per_user="$1/day", on_limit="block")
    llmburn.instrument()

    with llmburn.track(user_id="alice"):
        client.messages.create(**call_kw())  # $2.00 on Claude, allowed (was at $0)
        with pytest.raises(llmburn.BudgetExceeded) as e:
            openai.OpenAI(api_key="x").chat.completions.create(
                model="gpt-6-luna", messages=[{"role": "user", "content": "hi"}])

    assert e.value.spent == pytest.approx(2.00)
    assert openai_sent == []
    assert {t.key for t in store.totals(by="model")} == {MODEL}
