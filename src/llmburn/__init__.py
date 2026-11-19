"""llmburn: see who's burning your AI budget."""
from .budget import BudgetExceeded
from .core import current_tags, init, record, track

__version__ = "0.2.0"
__all__ = ["init", "instrument", "track", "record", "current_tags", "BudgetExceeded"]


def instrument():
    """Auto-track every LLM call from the SDKs that are installed.

    Patches google-genai, openai and anthropic, whichever are importable, and
    returns the names that were patched, e.g. ["gemini", "openai"]. Safe to
    call more than once.
    """
    from .integrations import anthropic, gemini, openai

    patched = []
    for name, module in (("gemini", gemini), ("openai", openai), ("anthropic", anthropic)):
        if module.instrument():
            patched.append(name)
    return patched
