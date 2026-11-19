import json
from datetime import date, datetime

import pytest

from llmburn.pricing import _PRICES_FILE, UnknownModelError, cost

PRICE_KEYS = ("input", "output", "cached_input", "cache_write", "cache_write_1h",
              "cache_write_12h")


def test_basic_flash_call():
    # 1M in at $0.30 + 1M out at $2.50
    assert cost("gemini-2.5-flash", 1_000_000, 1_000_000) == pytest.approx(2.80)


def test_small_call():
    # 1,000 in + 500 out on flash-lite: 1000*0.10/1e6 + 500*0.40/1e6
    assert cost("gemini-2.5-flash-lite", 1_000, 500) == pytest.approx(0.0003)


def test_cached_tokens_are_cheaper():
    full = cost("gemini-2.5-flash", 100_000, 0)
    half_cached = cost("gemini-2.5-flash", 100_000, 0, cached_tokens=50_000)
    # 50k at $0.30 + 50k at $0.03
    assert half_cached == pytest.approx(0.0165)
    assert half_cached < full


def test_price_change_on_effective_date():
    before = cost("gemini-4-argon", 1_000_000, 0, at=date(2026, 11, 30))
    after = cost("gemini-4-argon", 1_000_000, 0, at=date(2026, 12, 2))
    assert before == pytest.approx(2.00)
    assert after == pytest.approx(4.00)


def test_price_change_starts_on_the_day_itself():
    assert cost("gemini-4-argon", 1_000_000, 0, at=date(2026, 12, 1)) == pytest.approx(4.00)


def test_accepts_datetime():
    assert cost("gemini-4-argon", 1_000_000, 0, at=datetime(2026, 12, 5, 14, 30)) == pytest.approx(4.00)


def test_long_context_pricing_on_pro():
    short = cost("gemini-2.5-pro", 200_000, 1_000_000)
    long = cost("gemini-2.5-pro", 200_001, 1_000_000)
    assert short == pytest.approx(0.25 + 10.00)
    assert long == pytest.approx(200_001 * 2.50 / 1e6 + 15.00)


def test_model_name_variants():
    plain = cost("gemini-2.5-flash", 1000, 1000)
    assert cost("models/gemini-2.5-flash", 1000, 1000) == plain
    assert cost("gemini-2.5-flash-001", 1000, 1000) == plain


def test_versioned_name_picks_longest_match():
    # must resolve to flash-lite, not flash
    assert cost("gemini-2.5-flash-lite-001", 1_000_000, 0) == pytest.approx(0.10)


def test_unknown_model_raises():
    with pytest.raises(UnknownModelError):
        cost("gpt-9000", 10, 10)

# --- model-name suffix rule ----------------------------------------------------

@pytest.mark.parametrize("name", [
    "gemini-2.5-flash-image",
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-flash-native-audio-preview-12-2025",
    "gemini-2.5-flash-lite-tts",
])
def test_other_models_with_a_known_prefix_are_not_mispriced(name):
    with pytest.raises(UnknownModelError):
        cost(name, 10, 10)


@pytest.mark.parametrize("name", [
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-flash-exp-03-25",
    "gemini-2.5-flash-latest",
])
def test_version_suffixes_still_match(name):
    assert cost(name, 1000, 1000) == cost("gemini-2.5-flash", 1000, 1000)


# --- OpenAI ----------------------------------------------------------------------

def test_openai_basic_cost():
    # 100k in at $1.00 + 100k out at $5.00
    assert cost("gpt-6.1-terra", 100_000, 100_000) == pytest.approx(0.60)


def test_openai_dated_snapshot():
    assert cost("gpt-6.1-terra-2026-10-01", 100_000, 0) == pytest.approx(0.10)


def test_openai_cached_input():
    # 100k prompt, 60k of it cached: 40k at $2.00 + 60k at $0.10
    assert cost("gpt-6.1-sol", 100_000, 0, cached_tokens=60_000) == pytest.approx(0.086)


def test_openai_long_context_above_272k():
    assert cost("gpt-6-luna", 272_000, 0) == pytest.approx(272_000 * 0.10 / 1e6)
    assert cost("gpt-6-luna", 272_001, 1_000_000) == pytest.approx(272_001 * 0.20 / 1e6 + 0.75)


def test_openai_30m_and_12h_cache_writes():
    # 100k prompt: 10k read ($0.10) + 20k 30m write ($2.50) + 30k 12h write ($5) + 40k fresh ($2)
    usd = cost("gpt-6.1-sol", 100_000, 0, cached_tokens=10_000,
               cache_write_tokens=20_000, cache_write_12h_tokens=30_000)
    assert usd == pytest.approx((10_000 * 0.10 + 20_000 * 2.50 + 30_000 * 5.0
                                 + 40_000 * 2.0) / 1e6)


def test_openai_models_without_cache_writes_bill_them_as_input():
    assert cost("chat-latest", 1000, 0, cache_write_tokens=1000) == cost("chat-latest", 1000, 0)


# --- Anthropic -------------------------------------------------------------------

def test_anthropic_reads_and_both_cache_writes():
    # 4M prompt: 1M read ($0.10) + 1M 5m write ($2.50) + 1M 1h write ($4)
    # + 1M fresh ($2), then 1M output ($10)
    usd = cost("claude-sonnet-5-5", 4_000_000, 1_000_000, cached_tokens=1_000_000,
               cache_write_tokens=1_000_000, cache_write_1h_tokens=1_000_000)
    assert usd == pytest.approx(18.60)


def test_anthropic_dated_id():
    assert cost("claude-haiku-4-5-20251001", 1_000_000, 0) == pytest.approx(1.00)


def test_haiku_5_5_long_context_tier():
    assert cost("claude-haiku-5-5", 100_000, 0) == pytest.approx(100_000 * 0.10 / 1e6)
    assert cost("claude-haiku-5-5", 100_001, 0) == pytest.approx(100_001 * 0.50 / 1e6)


@pytest.mark.parametrize("name,usd", [
    ("claude-opus-4-20250514", 15.0),
    ("claude-sonnet-4-20250514", 3.0),
    ("claude-opus-4-1-20250805", 15.0),
    ("claude-opus-4-5-20251101", 5.0),
    ("claude-mythos-5-2", 8.0),
    ("gemini-3.1-pro-preview-customtools", 2.0),
    ("gemini-4-hydrogen", 1.5),
])
def test_more_models_resolve(name, usd):
    assert cost(name, 100_000, 0) == pytest.approx(usd / 10)


def test_gemini_2_5_pro_billed_as_3_8_flash_from_december():
    assert cost("gemini-2.5-pro", 100_000, 0, at=date(2026, 11, 30)) == pytest.approx(0.125)
    assert cost("gemini-2.5-pro", 100_000, 0, at=date(2026, 12, 1)) == pytest.approx(0.075)
    assert cost("gemini-2.5-pro", 300_000, 0, at=date(2026, 12, 1)) == pytest.approx(0.225)


def test_clamping_never_goes_negative():
    # Parts bigger than the prompt are clamped to it, never below zero.
    usd = cost("claude-sonnet-5-5", 100, 0, cached_tokens=80,
               cache_write_tokens=80, cache_write_1h_tokens=80)
    assert usd == pytest.approx((80 * 0.10 + 20 * 2.50) / 1e6)
    assert cost("claude-sonnet-5-5", 0, 0, cached_tokens=50, cache_write_tokens=50) == 0


# --- price table sanity ----------------------------------------------------------

def test_price_table_has_no_gaps():
    """Every model and price period is complete; lists all problems at once."""
    with open(_PRICES_FILE, encoding="utf-8") as f:
        models = json.load(f)["models"]

    problems = []
    for name, model in models.items():
        for n, period in enumerate(model["prices"]):
            where = f"{name} (period {n})"
            for key in ("input", "output", "cached_input"):
                if key not in period:
                    problems.append(f"{where}: missing {key}")
            if period.get("cached_input", 0) > period.get("input", float("inf")):
                problems.append(f"{where}: cached_input is higher than input")
            long_context = period.get("long_context")
            if long_context is None:
                continue
            for key in PRICE_KEYS:
                if key in period and key not in long_context:
                    problems.append(f"{where}: long_context is missing {key}")
            if long_context.get("cached_input", 0) > long_context.get("input", float("inf")):
                problems.append(f"{where}: long_context cached_input is higher than input")

    assert not problems, "price table gaps:\n" + "\n".join(problems)


def test_gpt_6_cyber_long_prompt_uses_long_context_write_rate():
    # 300k prompt (over 272k): 200k fresh at $40 + 100k 30-minute writes at $50
    usd = cost("gpt-6-cyber", 300_000, 0, cache_write_tokens=100_000)
    assert usd == pytest.approx(13.00)
