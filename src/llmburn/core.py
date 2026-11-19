"""Tag calls with user/plan/feature, price them and save them.

Golden rule: nothing in here may ever raise into the host app.
"""
import contextvars
import functools
import logging
from contextlib import contextmanager

from datetime import datetime, timezone

from .budget import ACTIONS, BudgetExceeded, parse_budget
from .pricing import UnknownModelError, _resolve_model, cost
from .storage import Call, Storage

log = logging.getLogger("llmburn")

_tags = contextvars.ContextVar("llmburn_tags", default={})
_storage = None
_warned_models = set()
_budget = None
_budget_total = None
_on_limit = "block"
_downgrade_to = None


def init(db="sqlite:///llmburn.db", budget_per_user=None, budget_total=None,
         on_limit="block", downgrade_to=None):
    """Set up llmburn. Call once at app startup.

    db              -- where calls are saved
    budget_per_user -- e.g. "$2/day" for each user; None means no limit
    budget_total    -- e.g. "$50/day" for the whole app, all calls counted
    on_limit        -- what happens when a budget is exceeded:
                       "block"     raise llmburn.BudgetExceeded before the call
                       "downgrade" switch the call to `downgrade_to` (a cheaper model)
                       "warn"      log a warning and let the call through
    downgrade_to    -- model to use when on_limit="downgrade"

    Raises ValueError for bad settings, so mistakes show up at startup.
    """
    global _storage, _budget, _budget_total, _on_limit, _downgrade_to
    budget = parse_budget(budget_per_user) if budget_per_user else None
    budget_total = parse_budget(budget_total) if budget_total else None
    if on_limit not in ACTIONS:
        raise ValueError(f"on_limit must be one of {ACTIONS}")
    if on_limit == "downgrade":
        if not downgrade_to:
            raise ValueError('on_limit="downgrade" needs downgrade_to="<cheaper model>"')
        _resolve_model(downgrade_to)  # raises if it's not in the price table
    _storage = Storage(db)
    _budget, _budget_total = budget, budget_total
    _on_limit, _downgrade_to = on_limit, downgrade_to
    return _storage


def _get_storage():
    global _storage
    if _storage is None:
        _storage = Storage()  # default: ./llmburn.db
    return _storage


@contextmanager
def _tag_block(tags):
    # Nested blocks inherit outer tags; inner values win.
    token = _tags.set({**_tags.get(), **tags})
    try:
        yield
    finally:
        _tags.reset(token)


class track:
    """Attach user_id / plan / feature to every LLM call inside.

    As a context manager:
        with llmburn.track(user_id=user.id, feature="resume_rewrite"):
            client.models.generate_content(...)

    As a decorator:
        @llmburn.track(feature="cover_letter")
        def write_cover_letter(...): ...
    """

    def __init__(self, user_id=None, plan=None, feature=None):
        self.tags = {
            k: str(v)
            for k, v in dict(user_id=user_id, plan=plan, feature=feature).items()
            if v is not None
        }
        self._cm = None

    def __enter__(self):
        self._cm = _tag_block(self.tags)
        return self._cm.__enter__()

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with _tag_block(self.tags):
                return func(*args, **kwargs)
        return wrapper


def current_tags():
    return dict(_tags.get())


def before_call(model, tags=None):
    """Check the budgets. Returns the model to actually use.

    The app-wide budget_total is checked first and covers every call; then
    budget_per_user for calls tagged with a user_id.
    Raises BudgetExceeded when on_limit="block" and a budget is exceeded.
    Any other problem (e.g. database error) lets the call through unchanged.
    """
    if _budget is None and _budget_total is None:
        return model
    tags = _tags.get() if tags is None else tags
    user_id = tags.get("user_id")
    now = datetime.now(timezone.utc)

    try:
        over = None  # (user_id or None, spent, budget)
        if _budget_total is not None:
            spent = _get_storage().spent_total(now - _budget_total.period)
            if spent >= _budget_total.usd:
                over = (None, spent, _budget_total)
        if over is None and _budget is not None and user_id is not None:
            spent = _get_storage().spent(user_id, now - _budget.period)
            if spent >= _budget.usd:
                over = (user_id, spent, _budget)
    except Exception:
        log.exception("llmburn: budget check failed, allowing the call")
        return model

    if over is None:
        return model
    if _on_limit == "block":
        raise BudgetExceeded(*over)
    if _on_limit == "downgrade":
        return _downgrade_to
    who = "app total" if over[0] is None else f"user {over[0]!r}"
    log.warning("llmburn: %s spent $%.4f, over %s", who, over[1], over[2])
    return model


def record(model, input_tokens, output_tokens, cached_tokens=0, tags=None,
           cache_write_tokens=0, cache_write_1h_tokens=0, cache_write_12h_tokens=0):
    """Price one call and save it. Returns the cost, or None on failure.

    input_tokens includes cached and cache-written tokens (see pricing.cost).
    tags -- override the current track() tags (used by streams, which can
            finish after the track() block has already ended)
    """
    try:
        usd = cost(model, input_tokens, output_tokens, cached_tokens,
                   cache_write_tokens=cache_write_tokens,
                   cache_write_1h_tokens=cache_write_1h_tokens,
                   cache_write_12h_tokens=cache_write_12h_tokens)
    except UnknownModelError:
        if model not in _warned_models:
            _warned_models.add(model)
            log.warning("llmburn: no price for model %r, call not tracked", model)
        return None
    except Exception:
        log.exception("llmburn: pricing failed")
        return None

    try:
        tags = _tags.get() if tags is None else tags
        _get_storage().save(Call(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost_usd=usd,
            user_id=tags.get("user_id"),
            plan=tags.get("plan"),
            feature=tags.get("feature"),
        ))
    except Exception:
        log.exception("llmburn: failed to save call")
    return usd
