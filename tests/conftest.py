import pytest

from llmburn.integrations import anthropic, gemini, openai


@pytest.fixture(autouse=True)
def _uninstrument_after_each_test(monkeypatch):
    """Undo llmburn's patches after every test.

    Requesting monkeypatch makes this fixture tear down BEFORE monkeypatch
    restores the real SDK methods. Otherwise a fake saved in _originals would
    be put back on top of the real method and leak into later tests.
    """
    yield
    gemini.uninstrument()
    openai.uninstrument()
    anthropic.uninstrument()
