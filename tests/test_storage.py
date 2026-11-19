import threading
from datetime import datetime, timedelta, timezone

import pytest

from llmburn.storage import Call, Storage


@pytest.fixture
def store(tmp_path):
    s = Storage(f"sqlite:///{tmp_path / 'test.db'}")
    yield s
    s.close()


def make_call(user="u1", cost=0.01, **kw):
    return Call(model="gemini-2.5-flash", input_tokens=100, output_tokens=50,
                cost_usd=cost, user_id=user, **kw)


def test_save_and_total_by_user(store):
    store.save(make_call("alice", 0.30))
    store.save(make_call("alice", 0.20))
    store.save(make_call("bob", 0.10))

    totals = store.totals(by="user_id")

    assert [t.key for t in totals] == ["alice", "bob"]  # most expensive first
    assert totals[0].calls == 2
    assert totals[0].cost_usd == pytest.approx(0.50)
    assert totals[0].input_tokens == 200


def test_group_by_feature(store):
    store.save(make_call(feature="resume_rewrite", cost=0.4))
    store.save(make_call(feature="cover_letter", cost=0.1))
    keys = [t.key for t in store.totals(by="feature")]
    assert keys == ["resume_rewrite", "cover_letter"]


def test_since_filters_old_calls(store):
    now = datetime.now(timezone.utc)
    store.save(make_call(cost=1.00, ts=now - timedelta(days=10)))
    store.save(make_call(cost=0.25, ts=now - timedelta(hours=1)))

    recent = store.totals(since=now - timedelta(days=7))

    assert recent[0].cost_usd == pytest.approx(0.25)


def test_untagged_calls_group_under_none(store):
    store.save(Call(model="gemini-2.5-flash", input_tokens=1,
                    output_tokens=1, cost_usd=0.01))
    assert store.totals()[0].key is None


def test_rejects_unknown_group_column(store):
    with pytest.raises(ValueError):
        store.totals(by="cost_usd; DROP TABLE calls")


def test_data_survives_reopen(tmp_path):
    path = f"sqlite:///{tmp_path / 'keep.db'}"
    s = Storage(path)
    s.save(make_call("alice", 0.5))
    s.close()

    s2 = Storage(path)
    assert s2.totals()[0].cost_usd == pytest.approx(0.5)
    s2.close()


def test_parallel_writes_from_threads(store):
    def worker():
        for _ in range(50):
            store.save(make_call("alice", 0.01))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert store.totals()[0].calls == 400


def test_in_memory_db():
    s = Storage(":memory:")
    s.save(make_call())
    assert s.totals()[0].calls == 1