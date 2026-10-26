"""Deprecated alias: use `ll3m` (LL3M). Kept so old imports keep working."""
from ll3m import *  # noqa: F403
from ll3m import __version__

__all__ = list(getattr(__import__("ll3m"), "__all__", []))
