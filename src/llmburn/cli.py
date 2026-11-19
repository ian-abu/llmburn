"""The `llmburn` terminal command.

    llmburn report                         # top users, last 7 days
    llmburn report --by feature --last 30d
    llmburn report --by model --last all --db sqlite:///prod.db
"""
import argparse
import re
import sys
from datetime import datetime, timedelta, timezone

from .storage import Storage

BY_CHOICES = {"user": "user_id", "plan": "plan", "feature": "feature", "model": "model"}
_UNITS = {"h": "hours", "d": "days", "w": "weeks"}


def parse_last(value):
    """'24h' / '7d' / '4w' -> timedelta; 'all' -> None."""
    if value == "all":
        return None
    m = re.fullmatch(r"(\d+)([hdw])", value)
    if not m:
        raise argparse.ArgumentTypeError("use a number + h/d/w (e.g. 24h, 7d, 4w) or 'all'")
    return timedelta(**{_UNITS[m.group(2)]: int(m.group(1))})


def _money(usd):
    # Cents for normal amounts, more digits for sub-cent ones.
    return f"${usd:,.4f}" if 0 < usd < 0.01 else f"${usd:,.2f}"


def format_report(totals, by, last_label):
    if not totals:
        return f"No tracked calls in {last_label}."

    header = (by.upper(), "CALLS", "INPUT TOK", "OUTPUT TOK", "COST")
    rows = [
        (t.key if t.key is not None else "(untagged)",
         f"{t.calls:,}", f"{t.input_tokens:,}", f"{t.output_tokens:,}", _money(t.cost_usd))
        for t in totals
    ]
    total_cost = sum(t.cost_usd for t in totals)
    total_calls = sum(t.calls for t in totals)
    footer = ("TOTAL", f"{total_calls:,}", "", "", _money(total_cost))

    widths = [max(len(r[i]) for r in [header, *rows, footer]) for i in range(5)]

    def line(cells):
        first = cells[0].ljust(widths[0])
        rest = [c.rjust(w) for c, w in zip(cells[1:], widths[1:])]
        return "  ".join([first, *rest])

    rule = "-" * len(line(header))
    title = f"LLM spend by {by}, {last_label}"
    return "\n".join([title, "", line(header), rule, *map(line, rows), rule, line(footer)])


def main(argv=None):
    parser = argparse.ArgumentParser(prog="llmburn", description="See who's burning your AI budget.")
    sub = parser.add_subparsers(dest="command", required=True)

    rep = sub.add_parser("report", help="show LLM spend grouped by user, plan, feature or model")
    rep.add_argument("--by", choices=BY_CHOICES, default="user")
    rep.add_argument("--last", default="7d",
                     help="time window: 24h, 7d, 4w or all (default 7d)")
    rep.add_argument("--db", default="sqlite:///llmburn.db", help="default: ./llmburn.db")
    rep.add_argument("--limit", type=int, default=20, help="rows to show (default 20)")

    args = parser.parse_args(argv)
    try:
        last = parse_last(args.last)
    except argparse.ArgumentTypeError as e:
        parser.error(f"--last: {e}")
    last_label = "all time" if last is None else "the last " + _label(args.last)

    since = None if last is None else datetime.now(timezone.utc) - last
    store = Storage(args.db)
    try:
        totals = store.totals(by=BY_CHOICES[args.by], since=since, limit=args.limit)
    finally:
        store.close()

    print(format_report(totals, args.by, last_label))
    return 0


def _label(value):
    """'7d' -> '7 days', '1w' -> '1 week'."""
    n, unit = int(value[:-1]), _UNITS[value[-1]]
    return f"{n} {unit[:-1] if n == 1 else unit}"


if __name__ == "__main__":
    sys.exit(main())
