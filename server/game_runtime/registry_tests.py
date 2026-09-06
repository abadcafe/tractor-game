"""Black-box tests for the owned-value game registry."""

from server.game_runtime.registry import GameId, GameRegistry


def test_create_returns_unique_opaque_ids() -> None:
    registry = GameRegistry[str]()

    first = registry.create(lambda _game_id: "first")
    second = registry.create(lambda _game_id: "second")

    assert first != second
    assert GameId.parse(first.value) == first
    assert GameId.parse(second.value) == second
    assert registry.list_ids() == (first, second)


def test_get_refreshes_last_access() -> None:
    now = 100.0

    def clock() -> float:
        return now

    registry = GameRegistry[str](clock)
    value_id = registry.create(lambda _game_id: "value")
    now = 108.0
    assert registry.get(value_id) == "value"
    now = 115.0

    assert registry.expire(max_idle_seconds=10) == ()
    assert registry.get(value_id) == "value"


def test_get_missing_does_not_create_an_entry() -> None:
    registry = GameRegistry[str]()

    missing = GameId.parse("0" * 32)
    assert missing is not None
    assert registry.get(missing) is None
    assert registry.list_ids() == ()


def test_delete_returns_owned_value_and_is_idempotent() -> None:
    registry = GameRegistry[str]()
    value_id = registry.create(lambda _game_id: "value")

    assert registry.delete(value_id) == "value"
    assert registry.delete(value_id) is None
    assert registry.list_ids() == ()


def test_expire_returns_removed_values_for_cleanup() -> None:
    now = 100.0

    def clock() -> float:
        return now

    registry = GameRegistry[str](clock)
    expired_id = registry.create(lambda _game_id: "expired")
    now = 120.0
    live_id = registry.create(lambda _game_id: "live")

    removed = registry.expire(max_idle_seconds=10)

    assert removed == ("expired",)
    assert registry.get(expired_id) is None
    assert registry.get(live_id) == "live"


def test_expire_uses_strict_idle_boundary() -> None:
    now = 100.0

    def clock() -> float:
        return now

    registry = GameRegistry[str](clock)
    value_id = registry.create(lambda _game_id: "value")
    now = 110.0

    assert registry.expire(max_idle_seconds=10) == ()
    assert registry.get(value_id) == "value"


def test_create_supplies_identity_before_value_construction() -> None:
    registry = GameRegistry[GameId]()

    game_id = registry.create(lambda identity: identity)

    assert registry.get(game_id) == game_id


def test_parse_rejects_noncanonical_identity() -> None:
    assert GameId.parse(None) is None
    assert GameId.parse("") is None
    assert GameId.parse("not-a-game") is None
    assert GameId.parse("A" * 32) is None
