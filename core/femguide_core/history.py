"""Undo/redo as plan history (docs/ARCHITECTURE.md section 5.1, D4).

Plans are immutable snapshots, so undo is just "restore the previous plan";
geometry is regenerated from whatever plan is current. Nothing here knows
what a plan contains.
"""

from __future__ import annotations

from typing import Generic, List, TypeVar

P = TypeVar("P")


class PlanHistory(Generic[P]):
    """Linear undo/redo stack of immutable plan snapshots.

    Committing a new plan after undoing discards the redo branch, like every
    editor. Committing a plan equal to the current one is a no-op, so UI code
    can commit on every widget change without flooding the history.
    """

    def __init__(self, initial: P, limit: int = 500) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        self._states: List[P] = [initial]
        self._index = 0
        self._limit = limit

    @property
    def current(self) -> P:
        return self._states[self._index]

    @property
    def can_undo(self) -> bool:
        return self._index > 0

    @property
    def can_redo(self) -> bool:
        return self._index < len(self._states) - 1

    def commit(self, plan: P) -> P:
        if plan == self.current:
            return plan
        del self._states[self._index + 1:]
        self._states.append(plan)
        if len(self._states) > self._limit:
            del self._states[0]
        self._index = len(self._states) - 1
        return plan

    def undo(self) -> P:
        if not self.can_undo:
            raise IndexError("nothing to undo")
        self._index -= 1
        return self.current

    def redo(self) -> P:
        if not self.can_redo:
            raise IndexError("nothing to redo")
        self._index += 1
        return self.current
