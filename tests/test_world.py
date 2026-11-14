"""world.py 테스트: 설정 파일을 읽고 1일차 마을을 만드는지 확인한다."""

import copy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from pondvillage.world import Beliefs, ConfigError, World, config_from_dict, load_config

BASE_YAML = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"

RAW = {
    "pond": {
        "capacity": 100,
        "initial_stock": 100,
        "regen_multiplier": 2,
        "collapse_below": 5,
        "estimate_noise": [0.8, 1.2],
    },
    "economy": {
        "start_money": 20,
        "fish_price": 1,
        "daily_cost": 5,
        "max_catch_per_day": 20,
        "inspect_catch_factor": 0.5,
        "bankrupt_after_days": 3,
    },
    "run": {"days": 30, "seed": 1},
    "agents": [
        {"id": "a1", "name": "가", "persona": "어부 가"},
        {"id": "a2", "name": "나", "persona": "어부 나"},
    ],
}


def test_base_yaml_matches_spec_numbers():
    config = load_config(BASE_YAML)

    assert (config.pond.capacity, config.pond.initial_stock) == (100, 100)
    assert (config.pond.regen_multiplier, config.pond.collapse_below) == (2, 5)
    assert config.pond.estimate_noise == (0.8, 1.2)
    assert (config.economy.start_money, config.economy.fish_price) == (20, 1)
    assert (config.economy.daily_cost, config.economy.max_catch_per_day) == (5, 20)
    assert config.economy.inspect_catch_factor == 0.5
    assert config.economy.bankrupt_after_days == 3
    assert (config.days, config.seed) == (30, 1)
    assert [a.id for a in config.agents] == [f"agent_0{i}" for i in range(1, 6)]
    assert [a.name for a in config.agents] == ["만석", "순자", "영철", "미란", "동수"]


def test_world_starts_at_day_one_with_full_pond():
    world = World.from_config(load_config(BASE_YAML))

    assert world.day == 1
    assert (world.pond.stock, world.pond.capacity, world.pond.collapsed) == (100, 100, False)
    assert len(world.agents) == 5


def test_agents_start_with_spec_defaults():
    world = World.from_config(load_config(BASE_YAML))

    for agent in world.agents:
        assert agent.money == 20
        assert agent.alive is True
        assert agent.debt_days == 0
        assert agent.memory == []
        assert agent.beliefs is None
        assert agent.trust == {}
        assert agent.persona


def test_agents_do_not_share_memory_or_trust():
    world = World.from_config(config_from_dict(RAW))
    first, second = world.agents

    first.trust["a2"] = 0.9
    first.beliefs = Beliefs(estimated_stock=80, confidence=0.5, sustainable_guess=10)

    assert second.trust == {}
    assert second.beliefs is None


def test_alive_agents_and_get_agent():
    world = World.from_config(config_from_dict(RAW))
    world.get_agent("a1").alive = False

    assert [a.id for a in world.alive_agents()] == ["a2"]
    assert world.get_agent("a2").name == "나"
    with pytest.raises(KeyError):
        world.get_agent("없는사람")


def test_config_cannot_be_changed_during_run():
    config = config_from_dict(RAW)

    with pytest.raises(FrozenInstanceError):
        config.pond.capacity = 999


def test_missing_item_raises_config_error():
    raw = copy.deepcopy(RAW)
    del raw["economy"]["daily_cost"]

    with pytest.raises(ConfigError):
        config_from_dict(raw)


def test_misspelled_item_raises_config_error():
    raw = copy.deepcopy(RAW)
    raw["pond"]["capasity"] = raw["pond"].pop("capacity")

    with pytest.raises(ConfigError):
        config_from_dict(raw)


def test_duplicate_agent_id_raises_config_error():
    raw = copy.deepcopy(RAW)
    raw["agents"][1]["id"] = "a1"

    with pytest.raises(ConfigError):
        config_from_dict(raw)
