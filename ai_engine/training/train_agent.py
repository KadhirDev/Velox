import logging
import yaml
from pathlib import Path

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from ai_engine.environment.cloud_env import VeloxEnv
from ai_engine.training.callbacks import MetricsCallback

logger = logging.getLogger(__name__)


def _make_env(config: dict, rank: int):
    def _init():
        env = VeloxEnv(config)
        env.reset(seed=rank)
        return Monitor(env)
    return _init


def train(config_path: str = "config/settings.yaml"):
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    t_cfg     = cfg.get("training", {})
    n_envs    = t_cfg.get("n_envs", 4)
    timesteps = t_cfg.get("total_timesteps", 2_000_000)
    save_dir  = Path(t_cfg.get("model_save_dir", "models"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── vectorised training env ─────────────────────────────────────────────
    train_env = SubprocVecEnv([_make_env(cfg, i) for i in range(n_envs)])
    train_env = VecNormalize(
        train_env, norm_obs=True, norm_reward=True,
        clip_obs=10.0, gamma=t_cfg.get("gamma", 0.99),
    )

    # ── evaluation env ──────────────────────────────────────────────────────
    eval_env = VecNormalize(
        SubprocVecEnv([_make_env(cfg, 99)]),
        norm_obs=True, norm_reward=False,
        clip_obs=10.0, training=False,
    )

    # ── model ───────────────────────────────────────────────────────────────
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=t_cfg.get("learning_rate", 3e-4),
        n_steps=t_cfg.get("n_steps", 2048),
        batch_size=t_cfg.get("batch_size", 512),
        n_epochs=t_cfg.get("n_epochs", 10),
        gamma=t_cfg.get("gamma", 0.99),
        gae_lambda=t_cfg.get("gae_lambda", 0.95),
        clip_range=t_cfg.get("clip_range", 0.2),
        ent_coef=t_cfg.get("ent_coef", 0.01),
        vf_coef=t_cfg.get("vf_coef", 0.5),
        max_grad_norm=t_cfg.get("max_grad_norm", 0.5),
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
            activation_fn=nn.Tanh,
        ),
        verbose=1,
        tensorboard_log=str(save_dir / "tensorboard"),
        device="auto",
    )

    # ── callbacks ───────────────────────────────────────────────────────────
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=max(50_000 // n_envs, 1),
            save_path=str(save_dir / "checkpoints"),
            name_prefix="velox_rl",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(save_dir / "best"),
            log_path=str(save_dir / "eval_logs"),
            eval_freq=max(20_000 // n_envs, 1),
            n_eval_episodes=20,
            deterministic=True,
        ),
        MetricsCallback(log_dir=str(save_dir / "metrics")),
    ])

    logger.info("Training PPO for %d timesteps on %d envs", timesteps, n_envs)
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)

    model.save(str(save_dir / "velox_rl_final"))
    train_env.save(str(save_dir / "vec_normalize.pkl"))
    logger.info("Training complete.")
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    train()