import asyncio

import pytest

pytest.importorskip("google.genai")
from google.genai import models, types  # noqa: E402

import llmburn  # noqa: E402
from llmburn.integrations import gemini  # noqa: E402


def make_response(prompt=1000, out=500, thoughts=0, cached=0, text="hi"):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(
            role="model", parts=[types.Part(text=text)]))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt,
            candidates_token_count=out,
            thoughts_token_count=thoughts,
            cached_content_token_count=cached,
        ),
    )


def stream_chunks():
    # Gemini sends usage on chunks; the last one has the final totals.
    return [
        types.GenerateContentResponse(usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=1000, candidates_token_count=10)),
        types.GenerateContentResponse(usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=1000, candidates_token_count=300)),
    ]


@pytest.fixture
def fake_sdk(monkeypatch):
    """Replace the real network methods with fakes, then instrument them."""
    def fake_generate(self, *, model, contents, config=None):
        return make_response(thoughts=200)

    def fake_stream(self, *, model, contents, config=None):
        yield from stream_chunks()

    async def fake_agenerate(self, *, model, contents, config=None):
        return make_response()

    async def fake_astream(self, *, model, contents, config=None):
        async def gen():
            for c in stream_chunks():
                yield c
        return gen()

    monkeypatch.setattr(models.Models, "generate_content", fake_generate)
    monkeypatch.setattr(models.Models, "generate_content_stream", fake_stream)
    monkeypatch.setattr(models.AsyncModels, "generate_content", fake_agenerate)
    monkeypatch.setattr(models.AsyncModels, "generate_content_stream", fake_astream)

    store = llmburn.init(":memory:")
    assert "gemini" in llmburn.instrument()
    yield store
    gemini.uninstrument()
    store.close()


def test_sync_call_is_tracked_and_response_unchanged(fake_sdk):
    with llmburn.track(user_id="alice", feature="resume"):
        resp = models.Models.generate_content(None, model="gemini-2.5-flash", contents="x")

    assert resp.text == "hi"
    total = fake_sdk.totals()[0]
    assert total.key == "alice"
    assert total.input_tokens == 1000
    assert total.output_tokens == 700  # 500 answer + 200 thinking
    assert total.cost_usd == pytest.approx((1000 * 0.30 + 700 * 2.50) / 1e6)


def test_stream_records_once_with_final_usage(fake_sdk):
    with llmburn.track(user_id="bob"):
        chunks = list(models.Models.generate_content_stream(
            None, model="gemini-2.5-flash", contents="x"))

    assert len(chunks) == 2
    totals = fake_sdk.totals()
    assert totals[0].calls == 1
    assert totals[0].output_tokens == 300


def test_stream_iterated_after_track_block_keeps_tags(fake_sdk):
    with llmburn.track(user_id="carol"):
        stream = models.Models.generate_content_stream(
            None, model="gemini-2.5-flash", contents="x")
    list(stream)  # consumed outside the block, e.g. Flask streaming response
    assert fake_sdk.totals()[0].key == "carol"


def test_async_call(fake_sdk):
    async def run():
        with llmburn.track(user_id="dave"):
            return await models.AsyncModels.generate_content(
                None, model="gemini-2.5-flash", contents="x")

    assert asyncio.run(run()).text == "hi"
    assert fake_sdk.totals()[0].key == "dave"


def test_async_stream(fake_sdk):
    async def run():
        with llmburn.track(user_id="erin"):
            stream = await models.AsyncModels.generate_content_stream(
                None, model="gemini-2.5-flash", contents="x")
            return [c async for c in stream]

    assert len(asyncio.run(run())) == 2
    assert fake_sdk.totals()[0].output_tokens == 300


def test_instrument_twice_does_not_double_count(fake_sdk):
    llmburn.instrument()
    models.Models.generate_content(None, model="gemini-2.5-flash", contents="x")
    assert fake_sdk.totals()[0].calls == 1


def test_unknown_model_does_not_break_the_call(fake_sdk):
    resp = models.Models.generate_content(None, model="gemini-99-ultra", contents="x")
    assert resp.text == "hi"
    assert fake_sdk.totals() == []


def test_sdk_errors_still_reach_the_app(monkeypatch, fake_sdk):
    gemini.uninstrument()

    def failing(self, **kw):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(models.Models, "generate_content", failing)
    llmburn.instrument()
    with pytest.raises(RuntimeError, match="quota exceeded"):
        models.Models.generate_content(None, model="gemini-2.5-flash", contents="x")


# --- budgets through the real SDK wrapper -----------------------------------

def _overspend(store, user):
    from llmburn.storage import Call
    store.save(Call(model="gemini-2.5-flash", input_tokens=1, output_tokens=1,
                    cost_usd=5.0, user_id=user))


def test_block_stops_call_before_it_reaches_google(monkeypatch, fake_sdk):
    calls = []

    def counting(self, *, model, contents, config=None):
        calls.append(model)
        return make_response()

    gemini.uninstrument()
    monkeypatch.setattr(models.Models, "generate_content", counting)
    llmburn.init(":memory:", budget_per_user="$1/day", on_limit="block")
    llmburn.instrument()
    _overspend(llmburn.core._storage, "alice")

    with llmburn.track(user_id="alice"):
        with pytest.raises(llmburn.BudgetExceeded):
            models.Models.generate_content(None, model="gemini-2.5-flash", contents="x")
    assert calls == []  # never sent, never billed
    llmburn.init(":memory:")


def test_downgrade_sends_cheaper_model_and_records_it(monkeypatch, fake_sdk):
    sent = []

    def capturing(self, *, model, contents, config=None):
        sent.append(model)
        return make_response()

    gemini.uninstrument()
    monkeypatch.setattr(models.Models, "generate_content", capturing)
    store = llmburn.init(":memory:", budget_per_user="$1/day", on_limit="downgrade",
                         downgrade_to="gemini-3.5-flash-lite")
    llmburn.instrument()
    _overspend(store, "alice")

    with llmburn.track(user_id="alice"):
        models.Models.generate_content(None, model="gemini-2.5-pro", contents="x")

    assert sent == ["gemini-3.5-flash-lite"]
    assert "gemini-3.5-flash-lite" in {t.key for t in store.totals(by="model")}
    llmburn.init(":memory:")


def test_block_on_stream_raises_at_call_time(fake_sdk):
    llmburn.init(":memory:", budget_per_user="$1/day", on_limit="block")
    _overspend(llmburn.core._storage, "alice")
    with llmburn.track(user_id="alice"):
        with pytest.raises(llmburn.BudgetExceeded):
            models.Models.generate_content_stream(None, model="gemini-2.5-flash", contents="x")
    llmburn.init(":memory:")
