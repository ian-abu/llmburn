import logging
import threading

import pytest

import llmburn
from llmburn import core


@pytest.fixture
def store():
    s = llmburn.init(":memory:")
    core._warned_models.clear()
    yield s
    s.close()


def test_record_saves_with_tags(store):
    with llmburn.track(user_id="alice", plan="pro", feature="resume"):
        usd = llmburn.record("gemini-2.5-flash", 1_000_000, 0)

    assert usd == pytest.approx(0.30)
    total = store.totals(by="user_id")[0]
    assert total.key == "alice"
    assert total.cost_usd == pytest.approx(0.30)
    assert store.totals(by="plan")[0].key == "pro"
    assert store.totals(by="feature")[0].key == "resume"


def test_tags_cleared_after_block(store):
    with llmburn.track(user_id="alice"):
        pass
    assert llmburn.current_tags() == {}


def test_nested_tracks_merge(store):
    with llmburn.track(user_id="alice", plan="pro"):
        with llmburn.track(feature="cover_letter"):
            assert llmburn.current_tags() == {
                "user_id": "alice", "plan": "pro", "feature": "cover_letter"}
        assert llmburn.current_tags() == {"user_id": "alice", "plan": "pro"}


def test_user_id_is_stored_as_string(store):
    with llmburn.track(user_id=42):
        llmburn.record("gemini-2.5-flash", 10, 10)
    assert store.totals()[0].key == "42"


def test_decorator(store):
    @llmburn.track(feature="summary")
    def summarize():
        llmburn.record("gemini-2.5-flash", 10, 10)
        return "done"

    assert summarize() == "done"
    assert summarize.__name__ == "summarize"
    assert store.totals(by="feature")[0].key == "summary"


def test_threads_keep_their_own_tags(store):
    seen = {}

    def worker(name):
        with llmburn.track(user_id=name):
            seen[name] = llmburn.current_tags()["user_id"]

    threads = [threading.Thread(target=worker, args=(f"u{i}",)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert seen == {f"u{i}": f"u{i}" for i in range(10)}


def test_unknown_model_never_raises_and_warns_once(store, caplog):
    with caplog.at_level(logging.WARNING, logger="llmburn"):
        assert llmburn.record("gpt-9000", 10, 10) is None
        assert llmburn.record("gpt-9000", 10, 10) is None
    assert len([r for r in caplog.records if "gpt-9000" in r.message]) == 1
    assert store.totals() == []


def test_storage_failure_never_raises(store):
    store.close()  # writes will now fail
    assert llmburn.record("gemini-2.5-flash", 10, 10) is not None


def test_exception_inside_track_still_resets_tags(store):
    with pytest.raises(RuntimeError):
        with llmburn.track(user_id="alice"):
            raise RuntimeError("boom")
    assert llmburn.current_tags() == {}