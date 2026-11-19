from datetime import datetime, timedelta, timezone

import pytest

import llmburn
from llmburn import core
from llmburn.budget import BudgetExceeded, parse_budget
from llmburn.storage import Call


# --- parsing ---------------------------------------------------------------

@pytest.mark.parametrize("text,usd,period", [
    ("$2/day", 2.0, timedelta(days=1)),
    ("2/day", 2.0, timedelta(days=1)),
    ("$0.50/hour", 0.5, timedelta(hours=1)),
    ("$10 / week", 10.0, timedelta(weeks=1)),
    ("$30/month", 30.0, timedelta(days=30)),
])
def test_parse_budget(text, usd, period):
    b = parse_budget(text)
    assert b.usd == usd and b.period == period


@pytest.mark.parametrize("bad", ["2 dollars", "$2/year", "$0/day", "/day", ""])
def test_parse_budget_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        parse_budget(bad)


# --- init validation ---------------------------------------------------------

def test_init_rejects_bad_settings():
    with pytest.raises(ValueError):
        llmburn.init(":memory:", budget_per_user="lots")
    with pytest.raises(ValueError):
        llmburn.init(":memory:", budget_per_user="$1/day", on_limit="explode")
    with pytest.raises(ValueError):
        llmburn.init(":memory:", budget_per_user="$1/day", on_limit="downgrade")
    with pytest.raises(Exception):
        llmburn.init(":memory:", budget_per_user="$1/day", on_limit="downgrade",
                     downgrade_to="not-a-model")


# --- behaviour ---------------------------------------------------------------

def spend(store, user, usd, ago=timedelta(minutes=5)):
    store.save(Call(model="gemini-2.5-flash", input_tokens=1, output_tokens=1,
                    cost_usd=usd, user_id=user, ts=datetime.now(timezone.utc) - ago))


@pytest.fixture
def block_store():
    s = llmburn.init(":memory:", budget_per_user="$1/day", on_limit="block")
    yield s
    llmburn.init(":memory:")  # reset budget for other tests
    s.close()


def test_under_budget_passes(block_store):
    spend(block_store, "alice", 0.50)
    with llmburn.track(user_id="alice"):
        assert core.before_call("gemini-2.5-flash") == "gemini-2.5-flash"


def test_block_raises_when_over(block_store):
    spend(block_store, "alice", 1.00)
    with llmburn.track(user_id="alice"):
        with pytest.raises(BudgetExceeded) as e:
            core.before_call("gemini-2.5-flash")
    assert e.value.user_id == "alice"
    assert e.value.spent == pytest.approx(1.00)
    assert str(e.value.limit) == "$1/day"


def test_other_users_not_affected(block_store):
    spend(block_store, "alice", 5.00)
    with llmburn.track(user_id="bob"):
        assert core.before_call("gemini-2.5-flash") == "gemini-2.5-flash"


def test_old_spending_outside_window_ignored(block_store):
    spend(block_store, "alice", 5.00, ago=timedelta(days=2))
    with llmburn.track(user_id="alice"):
        assert core.before_call("gemini-2.5-flash") == "gemini-2.5-flash"


def test_untagged_calls_never_limited(block_store):
    spend(block_store, None, 100.0)
    assert core.before_call("gemini-2.5-flash") == "gemini-2.5-flash"


def test_downgrade_swaps_model():
    s = llmburn.init(":memory:", budget_per_user="$1/day", on_limit="downgrade",
                     downgrade_to="gemini-3.5-flash-lite")
    spend(s, "alice", 2.00)
    with llmburn.track(user_id="alice"):
        assert core.before_call("gemini-2.5-pro") == "gemini-3.5-flash-lite"
    llmburn.init(":memory:")


def test_warn_logs_and_allows(caplog):
    s = llmburn.init(":memory:", budget_per_user="$1/day", on_limit="warn")
    spend(s, "alice", 2.00)
    with llmburn.track(user_id="alice"):
        assert core.before_call("gemini-2.5-flash") == "gemini-2.5-flash"
    assert any("alice" in r.message for r in caplog.records)
    llmburn.init(":memory:")


def test_db_failure_lets_call_through(block_store):
    block_store.close()
    with llmburn.track(user_id="alice"):
        assert core.before_call("gemini-2.5-flash") == "gemini-2.5-flash"


def test_no_budget_configured_is_a_noop():
    llmburn.init(":memory:")
    with llmburn.track(user_id="alice"):
        assert core.before_call("gemini-2.5-flash") == "gemini-2.5-flash"


# --- app-wide total budget --------------------------------------------------

@pytest.fixture
def total_store():
    s = llmburn.init(":memory:", budget_total="$10/day", on_limit="block")
    yield s
    llmburn.init(":memory:")
    s.close()


def test_total_under_limit_passes(total_store):
    spend(total_store, "alice", 4.0)
    spend(total_store, "bob", 5.0)
    with llmburn.track(user_id="carol"):
        assert core.before_call("gemini-2.5-flash") == "gemini-2.5-flash"


def test_total_counts_all_users_together(total_store):
    spend(total_store, "alice", 6.0)
    spend(total_store, "bob", 4.0)
    with llmburn.track(user_id="carol"):  # carol spent nothing herself
        with pytest.raises(BudgetExceeded) as e:
            core.before_call("gemini-2.5-flash")
    assert e.value.is_total
    assert e.value.user_id is None
    assert e.value.spent == pytest.approx(10.0)
    assert "app (total)" in str(e.value)


def test_total_blocks_untagged_calls_too(total_store):
    spend(total_store, None, 10.0)
    with pytest.raises(BudgetExceeded):
        core.before_call("gemini-2.5-flash")


def test_total_ignores_spending_outside_window(total_store):
    spend(total_store, "alice", 50.0, ago=timedelta(days=2))
    assert core.before_call("gemini-2.5-flash") == "gemini-2.5-flash"


def test_both_budgets_together():
    s = llmburn.init(":memory:", budget_per_user="$2/day", budget_total="$10/day",
                     on_limit="block")
    spend(s, "alice", 3.0)  # alice over her own limit, app total fine
    with llmburn.track(user_id="alice"):
        with pytest.raises(BudgetExceeded) as e:
            core.before_call("gemini-2.5-flash")
    assert e.value.user_id == "alice" and not e.value.is_total
    with llmburn.track(user_id="bob"):  # bob is fine
        assert core.before_call("gemini-2.5-flash") == "gemini-2.5-flash"

    spend(s, "dave", 7.0)  # app total now 10.0
    with llmburn.track(user_id="bob"):
        with pytest.raises(BudgetExceeded) as e:
            core.before_call("gemini-2.5-flash")
    assert e.value.is_total
    llmburn.init(":memory:")


def test_total_downgrade():
    s = llmburn.init(":memory:", budget_total="$1/day", on_limit="downgrade",
                     downgrade_to="gemini-3.5-flash-lite")
    spend(s, None, 1.0)
    assert core.before_call("gemini-2.5-pro") == "gemini-3.5-flash-lite"
    llmburn.init(":memory:")


def test_init_rejects_bad_total():
    with pytest.raises(ValueError):
        llmburn.init(":memory:", budget_total="a lot")
