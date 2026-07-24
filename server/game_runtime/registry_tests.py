"""Black-box tests for the owned-value game registry."""

from server.game_runtime.registry import GameRegistry


def test_create_returns_unique_opaque_ids() -> None:
    registry = GameRegistry[str]()

    first = registry.create("first")
    second = registry.create("second")

    assert first != second
    assert first
    assert second
    assert registry.list_ids() == (first, second)


def test_get_refreshes_last_access() -> None:
    now = 100.0

    def clock() -> float:
        return now

    registry = GameRegistry[str](clock)
    value_id = registry.create("value")
    now = 108.0
    assert registry.get(value_id) == "value"
    now = 115.0

    assert registry.expire(max_idle_seconds=10) == ()
    assert registry.get(value_id) == "value"


def test_get_missing_does_not_create_an_entry() -> None:
    registry = GameRegistry[str]()

    assert registry.get("missing") is None
    assert registry.list_ids() == ()


def test_delete_returns_owned_value_and_is_idempotent() -> None:
    registry = GameRegistry[str]()
    value_id = registry.create("value")

    assert registry.delete(value_id) == "value"
    assert registry.delete(value_id) is None
    assert registry.list_ids() == ()


def test_expire_returns_removed_values_for_cleanup() -> None:
    now = 100.0

    def clock() -> float:
        return now

    registry = GameRegistry[str](clock)
    expired_id = registry.create("expired")
    now = 120.0
    live_id = registry.create("live")

    removed = registry.expire(max_idle_seconds=10)

    assert removed == ("expired",)
    assert registry.get(expired_id) is None
    assert registry.get(live_id) == "live"


def test_expire_uses_strict_idle_boundary() -> None:
    now = 100.0

    def clock() -> float:
        return now

    registry = GameRegistry[str](clock)
    value_id = registry.create("value")
    now = 110.0

    assert registry.expire(max_idle_seconds=10) == ()
    assert registry.get(value_id) == "value"
