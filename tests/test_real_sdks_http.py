"""The real openai and anthropic clients, with only the network faked.

Each handler answers like the real API would (JSON or server-sent events),
so these tests check llmburn against the SDKs' own parsing, stream classes
and .stream() helpers.
"""
import asyncio
import json

import pytest

httpx2 = pytest.importorskip("httpx2")
openai = pytest.importorskip("openai")
anthropic = pytest.importorskip("anthropic")

import llmburn  # noqa: E402


@pytest.fixture
def store():
    s = llmburn.init(":memory:")
    llmburn.instrument()
    yield s
    s.close()
    llmburn.init(":memory:")


def sse(events):
    """Encode (event_name or None, payload) pairs as a server-sent-events body."""
    out = []
    for name, data in events:
        if name:
            out.append(f"event: {name}\n")
        out.append(f"data: {data if isinstance(data, str) else json.dumps(data)}\n\n")
    return "".join(out).encode()


def sse_response(events):
    return httpx2.Response(200, headers={"content-type": "text/event-stream"},
                           content=sse(events))


def by_user(store):
    return {t.key: t for t in store.totals(by="user_id")}


# --- OpenAI ------------------------------------------------------------------

CHAT_USAGE = {"prompt_tokens": 12000, "completion_tokens": 3000, "total_tokens": 15000,
              "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}}


def chat_chunk(content=None, usage=None, finish=None):
    choices = [] if content is None and finish is None else [
        {"index": 0, "delta": {"content": content} if content else {},
         "finish_reason": finish}]
    return {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1,
            "model": "gpt-6-luna", "choices": choices, "usage": usage}


RESPONSE = {
    "id": "resp_1", "object": "response", "created_at": 1, "model": "gpt-6-luna",
    "status": "completed", "output": [{
        "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": "ok", "annotations": []}]}],
    "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
    "usage": {"input_tokens": 20, "output_tokens": 4, "total_tokens": 24,
              "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
              "output_tokens_details": {"reasoning_tokens": 0}},
}


def openai_handler(seen):
    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if request.url.path.endswith("/chat/completions"):
            if not body.get("stream"):
                return httpx2.Response(200, json={
                    "id": "chatcmpl-1", "object": "chat.completion", "created": 1,
                    "model": body["model"], "usage": CHAT_USAGE,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": "ok"}}]})
            events = [(None, chat_chunk("o")), (None, chat_chunk("k", finish="stop"))]
            if (body.get("stream_options") or {}).get("include_usage"):
                events.append((None, chat_chunk(usage=CHAT_USAGE)))  # choices=[]
            return sse_response(events + [(None, "[DONE]")])
        if request.url.path.endswith("/responses"):
            if not body.get("stream"):
                return httpx2.Response(200, json=RESPONSE)
            return sse_response([
                ("response.created", {"type": "response.created", "sequence_number": 0,
                                      "response": {**RESPONSE, "status": "in_progress",
                                                   "usage": None, "output": []}}),
                ("response.output_text.delta", {
                    "type": "response.output_text.delta", "sequence_number": 1,
                    "item_id": "msg_1", "output_index": 0, "content_index": 0,
                    "delta": "ok", "logprobs": []}),
                ("response.completed", {"type": "response.completed", "sequence_number": 2,
                                        "response": RESPONSE}),
            ])
        return httpx2.Response(404, json={"error": {"message": "not found"}})
    return handler


@pytest.fixture
def oa_seen():
    return []


@pytest.fixture
def oa_client(oa_seen):
    return openai.OpenAI(api_key="x", max_retries=0, http_client=httpx2.Client(
        transport=httpx2.MockTransport(openai_handler(oa_seen))))


def test_openai_chat(store, oa_client):
    with llmburn.track(user_id="alice"):
        r = oa_client.chat.completions.create(
            model="gpt-6-luna", messages=[{"role": "user", "content": "hi"}])
    assert isinstance(r, openai.types.chat.ChatCompletion)
    assert r.choices[0].message.content == "ok"
    row = by_user(store)["alice"]
    assert (row.input_tokens, row.output_tokens) == (12000, 3000)
    assert row.cost_usd == pytest.approx((12000 * 0.10 + 3000 * 0.50) / 1e6)


def test_openai_chat_stream_hides_injected_usage_chunk(store, oa_client, oa_seen):
    with llmburn.track(user_id="bob"):
        chunks = list(oa_client.chat.completions.create(
            model="gpt-6-luna", messages=[{"role": "user", "content": "hi"}], stream=True))
    assert oa_seen[0]["stream_options"] == {"include_usage": True}
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "ok"
    assert all(c.choices for c in chunks)
    row = by_user(store)["bob"]
    assert (row.calls, row.input_tokens, row.output_tokens) == (1, 12000, 3000)


def test_openai_chat_stream_helper_is_tracked(store, oa_client):
    with llmburn.track(user_id="carol"):
        with oa_client.chat.completions.stream(
                model="gpt-6-luna", messages=[{"role": "user", "content": "hi"}]) as stream:
            final = stream.get_final_completion()
    assert final.choices[0].message.content == "ok"
    assert by_user(store)["carol"].calls == 1


def test_openai_responses(store, oa_client):
    with llmburn.track(user_id="dave"):
        r = oa_client.responses.create(model="gpt-6-luna", input="hi")
        events = list(oa_client.responses.create(model="gpt-6-luna", input="hi", stream=True))
    assert r.output_text == "ok"
    assert events[-1].type == "response.completed"
    row = by_user(store)["dave"]
    assert (row.calls, row.input_tokens, row.output_tokens) == (2, 40, 8)


def test_openai_async_chat_stream(store, oa_seen):
    client = openai.AsyncOpenAI(api_key="x", max_retries=0, http_client=httpx2.AsyncClient(
        transport=httpx2.MockTransport(openai_handler(oa_seen))))

    async def run():
        with llmburn.track(user_id="erin"):
            stream = await client.chat.completions.create(
                model="gpt-6-luna", messages=[{"role": "user", "content": "hi"}], stream=True)
        return [c async for c in stream]

    assert len(asyncio.run(run())) == 2
    assert by_user(store)["erin"].output_tokens == 3000


def test_openai_http_errors_reach_the_caller_unchanged(store):
    def failing(request):
        return httpx2.Response(404, json={"error": {
            "message": "The model `gpt-nope` does not exist", "type": "invalid_request_error",
            "code": "model_not_found"}})

    client = openai.OpenAI(api_key="x", max_retries=0, http_client=httpx2.Client(
        transport=httpx2.MockTransport(failing)))
    with pytest.raises(openai.NotFoundError):
        client.chat.completions.create(model="gpt-nope", messages=[{"role": "user", "content": "hi"}])
    assert store.totals() == []


# --- Anthropic ---------------------------------------------------------------

ANT_USAGE = {"input_tokens": 12000, "output_tokens": 3000, "cache_read_input_tokens": 100000,
             "cache_creation_input_tokens": 50000,
             "cache_creation": {"ephemeral_5m_input_tokens": 50000, "ephemeral_1h_input_tokens": 0}}

MESSAGE = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
           "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
           "stop_sequence": None, "usage": ANT_USAGE}


def anthropic_events():
    start = {**MESSAGE, "content": [], "stop_reason": None, "usage": {**ANT_USAGE, "output_tokens": 1}}
    return [
        ("message_start", {"type": "message_start", "message": start}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": "ok"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta",
                           "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                           "usage": {"output_tokens": 3000}}),
        ("message_stop", {"type": "message_stop"}),
    ]


def anthropic_handler(request):
    body = json.loads(request.content)
    if body.get("stream"):
        return sse_response(anthropic_events())
    return httpx2.Response(200, json=MESSAGE)


@pytest.fixture
def ant_client():
    return anthropic.Anthropic(api_key="x", max_retries=0, http_client=httpx2.Client(
        transport=httpx2.MockTransport(anthropic_handler)))


def call_kw(**extra):
    return dict(model="claude-haiku-5-5", max_tokens=10,
                messages=[{"role": "user", "content": "hi"}], **extra)


# 12k fresh*$0.10 + 100k read*$0.01 + 50k 5m-write*$0.125 + 3k out*$0.50
# (162k prompt: over the 100k long_context line, so use those prices)
ANT_USD = (12000 * 0.50 + 100000 * 0.05 + 50000 * 0.625 + 3000 * 2.50) / 1e6


def test_anthropic_create(store, ant_client):
    with llmburn.track(user_id="fay"):
        r = ant_client.messages.create(**call_kw())
    assert isinstance(r, anthropic.types.Message) and r.content[0].text == "ok"
    row = by_user(store)["fay"]
    assert (row.input_tokens, row.output_tokens) == (162000, 3000)
    assert row.cost_usd == pytest.approx(ANT_USD)


def test_anthropic_create_stream(store, ant_client):
    with llmburn.track(user_id="gus"):
        events = list(ant_client.messages.create(**call_kw(stream=True)))
    assert [e.type for e in events][0] == "message_start"
    assert len(events) == 6
    row = by_user(store)["gus"]
    assert (row.calls, row.input_tokens, row.output_tokens) == (1, 162000, 3000)
    assert row.cost_usd == pytest.approx(ANT_USD)


def test_anthropic_stream_helper(store, ant_client):
    with llmburn.track(user_id="hal"):
        with ant_client.messages.stream(**call_kw()) as stream:
            text = "".join(stream.text_stream)
            final = stream.get_final_message()
    assert text == "ok" and final.usage.output_tokens == 3000
    row = by_user(store)["hal"]
    assert (row.calls, row.input_tokens, row.output_tokens) == (1, 162000, 3000)


def test_anthropic_async_stream_helper(store):
    client = anthropic.AsyncAnthropic(api_key="x", max_retries=0, http_client=httpx2.AsyncClient(
        transport=httpx2.MockTransport(anthropic_handler)))

    async def run():
        with llmburn.track(user_id="ivy"):
            async with client.messages.stream(**call_kw()) as stream:
                return "".join([t async for t in stream.text_stream])

    assert asyncio.run(run()) == "ok"
    assert by_user(store)["ivy"].calls == 1


def test_anthropic_api_error_reaches_the_caller(store):
    def failing(request):
        return httpx2.Response(404, json={"type": "error", "error": {
            "type": "not_found_error", "message": "model: claude-nope"}})

    client = anthropic.Anthropic(api_key="x", max_retries=0, http_client=httpx2.Client(
        transport=httpx2.MockTransport(failing)))
    with pytest.raises(anthropic.NotFoundError):
        client.messages.create(**{**call_kw(), "model": "claude-nope"})
    assert store.totals() == []
