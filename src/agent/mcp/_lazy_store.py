"""Lazy global singleton — birinchi get()'da factory bilan quriladi.

ponytail: draft_store/flow_tests/flow_versions/workspace'da bir xil global `_store` +
lazy `get_X()` + `set_X()` boilerplate 4 marta takror edi — bitta class'ga yig'ildi.
`reset()` — test hook (eski `set_X_store` o'rnini bosadi).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class LazyStore(Generic[T]):
    def __init__(self, factory: Callable[[], T]) -> None:
        self._factory = factory
        self._value: T | None = None

    def get(self) -> T:
        if self._value is None:
            self._value = self._factory()
        return self._value

    def reset(self, value: T | None = None) -> None:
        self._value = value
