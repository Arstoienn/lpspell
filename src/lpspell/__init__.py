"""A spell checker that reads the whole sentence: catches "form" when you meant "from"."""

from .checker import Checker, Config, Issue, check

__all__ = ["Checker", "Config", "Issue", "check"]
