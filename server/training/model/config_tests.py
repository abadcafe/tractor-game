"""Black-box tests for persisted model shape configuration."""

from server.training.model import ModelConfig


def test_model_config_round_trips_exact_json_schema() -> None:
    config = ModelConfig(d_model=64, layers=2, heads=4)

    assert ModelConfig.from_json(config.to_json()) == config
