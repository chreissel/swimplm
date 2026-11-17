"""NPLM test methods.

All share the interface ``fit(data, ref, w_data=None, w_ref=None) -> NPLMResult``.
The ``falkon`` package is imported only when a Falkon test is actually fitted.
"""

from .falkon import Falkon
from .sparker import SparKer
from .swim import Swim

__all__ = ["Falkon", "SparKer", "Swim"]
