"""Per-user spending limits, checked before each LLM call."""
import re
from dataclasses import dataclass
from datetime import timedelta

ACTIONS = ("block", "downgrade", "warn")
_PERIODS = {
    "hour": timedelta(hours=1),
    "day": timedelta(days=1),
    "week": timedelta(weeks=1),
    "month": timedelta(days=30),
}


class BudgetExceeded(Exception):
    """Raised before the LLM call when on_limit="block" and a user is over budget.

    Catch it in your app to show a "limit reached" message:

        except llmburn.BudgetExceeded as e:
            return {"error": "Daily AI limit reached"}, 429
    """

    def __init__(self, user_id, spent, limit):
        self.user_id = user_id  # None when the app-wide budget_total was hit
        self.spent = spent
        self.limit = limit
        who = "the app (total)" if user_id is None else f"user {user_id!r}"
        super().__init__(f"{who} spent ${spent:.4f}, over the limit of {limit}")

    @property
    def is_total(self):
        """True if the whole app's budget_total was hit, not one user's."""
        return self.user_id is None


@dataclass(frozen=True)
class Budget:
    usd: float
    period: timedelta
    label: str

    def __str__(self):
        return self.label


def parse_budget(text):
    """'$2/day' -> Budget(2.0, 1 day). Also hour, week, month (30 days)."""
    m = re.fullmatch(r"\s*\$?\s*(\d+(?:\.\d+)?)\s*/\s*(hour|day|week|month)\s*", text)
    if not m:
        raise ValueError(
            f"budget {text!r} not understood; use e.g. '$2/day', '$0.50/hour', '$30/month'"
        )
    usd = float(m.group(1))
    if usd <= 0:
        raise ValueError("budget must be more than $0")
    return Budget(usd, _PERIODS[m.group(2)], f"${usd:g}/{m.group(2)}")
