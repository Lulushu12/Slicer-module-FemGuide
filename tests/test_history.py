"""Undo/redo as plan history (core, plan-agnostic)."""

import pytest

from femguide_core.history import PlanHistory


def test_commit_undo_redo():
    h = PlanHistory("a")
    h.commit("b")
    h.commit("c")
    assert h.current == "c"
    assert h.undo() == "b"
    assert h.undo() == "a"
    assert not h.can_undo
    assert h.redo() == "b"
    assert h.redo() == "c"
    assert not h.can_redo


def test_commit_after_undo_discards_redo_branch():
    h = PlanHistory("a")
    h.commit("b")
    h.undo()
    h.commit("x")
    assert h.current == "x"
    assert not h.can_redo
    assert h.undo() == "a"


def test_commit_of_unchanged_plan_is_noop():
    h = PlanHistory("a")
    h.commit("a")
    assert not h.can_undo


def test_limit_drops_oldest():
    h = PlanHistory(0, limit=3)
    for i in range(1, 6):
        h.commit(i)
    assert h.current == 5
    h.undo()
    h.undo()
    assert h.current == 3
    assert not h.can_undo


def test_undo_redo_past_ends_raise():
    h = PlanHistory("a")
    with pytest.raises(IndexError):
        h.undo()
    with pytest.raises(IndexError):
        h.redo()
