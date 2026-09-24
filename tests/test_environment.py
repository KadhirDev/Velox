import numpy as np
import pytest

from ai_engine.environment.cloud_env import VeloxEnv
from ai_engine.environment.action_decoder import ActionDecoder

CONFIG = {
    "max_episode_steps": 50,
    "pricing_normalization": 10.0,
    "carbon_normalization": 600.0,
    "latency_normalization": 1000.0,
    "pricing_fallback_path": "data/pricing/aws_pricing.json",
}


@pytest.fixture
def env():
    e = VeloxEnv(CONFIG)
    yield e
    e.close()


def test_observation_shape(env):
    obs, _ = env.reset(seed=0)
    assert obs.shape == (45,)
    assert obs.dtype == np.float32
    assert not np.any(np.isnan(obs))
    assert not np.any(np.isinf(obs))


def test_action_space_product(env):
    assert list(env.action_space.nvec) == [4, 10, 10, 4, 6, 6]
    assert int(np.prod(env.action_space.nvec)) == 57_600


def test_step_output_types(env):
    env.reset(seed=1)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (45,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert "reward_components" in info
    assert "total" in info["reward_components"]


def test_reward_clipped(env):
    env.reset(seed=2)
    for _ in range(20):
        _, reward, _, _, _ = env.step(env.action_space.sample())
        assert -10.0 <= reward <= 10.0


def test_episode_terminates(env):
    env.reset(seed=3)
    done, steps = False, 0
    while not done:
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        done = term or trunc
        steps += 1
        assert steps <= CONFIG["max_episode_steps"] + 1


def test_action_decoder_coverage():
    dec = ActionDecoder()
    for _ in range(200):
        action = np.array([
            np.random.randint(0, 4),
            np.random.randint(0, 10),
            np.random.randint(0, 10),
            np.random.randint(0, 4),
            np.random.randint(0, 6),
            np.random.randint(0, 6),
        ])
        d = dec.decode(action)
        assert d["cloud"] in dec.CLOUDS
        assert d["instance_type"] in dec.INSTANCE_TYPES
        assert d["purchase_option"] in dec.PURCHASE_OPTIONS
        assert d["scaling_level"] in dec.SCALING_LEVELS
