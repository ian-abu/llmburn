import argparse
from datetime import datetime, timedelta, timezone

import pytest

from llmburn.cli import main, parse_last
from llmburn.storage import Call, Storage


@pytest.fixture
def db(tmp_path):
    url = f"sqlite:///{tmp_path / 'r.db'}"
    s = Storage(url)
    now = datetime.now(timezone.utc)
    s.save(Call(model="gemini-2.5-flash", input_tokens=9000, output_tokens=3000,
                cost_usd=41.20, user_id="u_4821", plan="pro", feature="resume"))
    s.save(Call(model="gemini-2.5-flash", input_tokens=1000, output_tokens=500,
                cost_usd=0.0021, user_id="u_0193", plan="free", feature="cover_letter"))
    s.save(Call(model="gemini-2.5-pro", input_tokens=10, output_tokens=10,
                cost_usd=5.00, user_id="old_user", ts=now - timedelta(days=30)))
    s.save(Call(model="gemini-2.5-flash", input_tokens=10, output_tokens=10,
                cost_usd=0.01))
    s.close()
    return url


def test_parse_last():
    assert parse_last("24h") == timedelta(hours=24)
    assert parse_last("7d") == timedelta(days=7)
    assert parse_last("2w") == timedelta(weeks=2)
    assert parse_last("all") is None
    with pytest.raises(argparse.ArgumentTypeError):
        parse_last("7 days")


def test_report_by_user_default_week(db, capsys):
    assert main(["report", "--db", db]) == 0
    out = capsys.readouterr().out
    assert "last 7 days" in out
    lines = out.splitlines()
    first_row = next(l for l in lines if l.startswith("u_"))
    assert first_row.startswith("u_4821")  # most expensive first
    assert "$41.20" in out
    assert "$0.0021" in out
    assert "(untagged)" in out
    assert "old_user" not in out  # 30 days ago, outside the window


def test_report_all_time_includes_old(db, capsys):
    main(["report", "--db", db, "--last", "all"])
    out = capsys.readouterr().out
    assert "old_user" in out
    assert "all time" in out


def test_report_by_feature(db, capsys):
    main(["report", "--db", db, "--by", "feature"])
    out = capsys.readouterr().out
    assert "FEATURE" in out and "resume" in out and "cover_letter" in out


def test_report_by_model_shows_total(db, capsys):
    main(["report", "--db", db, "--by", "model", "--last", "all"])
    out = capsys.readouterr().out
    assert "gemini-2.5-pro" in out
    assert "TOTAL" in out
    assert "$46.21" in out


def test_empty_db(tmp_path, capsys):
    main(["report", "--db", f"sqlite:///{tmp_path / 'empty.db'}"])
    assert "No tracked calls" in capsys.readouterr().out


def test_bad_by_value_exits(db):
    with pytest.raises(SystemExit):
        main(["report", "--db", db, "--by", "price"])
