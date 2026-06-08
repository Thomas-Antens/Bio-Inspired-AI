"""
Training entry point for SAC (default) or PPO.

The agent is trained on a randomised strength_scale each episode so it
generalises across thermal strengths (required for the MacCready sweep).
Models are saved to results/models/.
"""

import json
import os
import sys
import time

import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from soaring.env.soaring_env import SoaringEnv


SAC_KWARGS = dict(
    policy          = "MlpPolicy",
    learning_rate   = 3e-4,
    buffer_size     = 500_000,
    learning_starts = 5_000,
    batch_size      = 256,
    tau             = 0.005,
    gamma           = 0.995,
    train_freq      = 1,
    gradient_steps  = 1,
    ent_coef        = "auto",
    policy_kwargs   = dict(net_arch=[64, 64]),
    verbose         = 0,
)

PPO_KWARGS = dict(
    policy          = "MlpPolicy",
    learning_rate   = 3e-4,
    n_steps         = 2048,
    batch_size      = 64,
    n_epochs        = 10,
    gamma           = 0.995,
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    ent_coef        = 0.01,
    policy_kwargs   = dict(net_arch=[64, 64]),
    verbose         = 0,
)

FIELD_KW = dict(
    n_thermals     = 40,             # scaled to preserve density on 15000×7500 domain
    wind           = (-5.0, 0.0),
    W_range        = (1.0, 3.0),
    R_range        = (200.0, 350.0),
    lifetime_range = (500, 1200),
    turbulence_amp = 0.05,
)

STRENGTH_RANGE = (0.3, 4.0)   # extended to reach Mc > 3 m/s for MacCready sweep



class RewardLogger(BaseCallback):
    """Collects episode returns and writes them to a .npy file."""

    def __init__(self, out_path, verbose=0):
        super().__init__(verbose)
        self.out_path = out_path
        self.ep_rewards: list[float] = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
        return True

    def _on_training_end(self):
        np.save(self.out_path, np.array(self.ep_rewards))



def make_env(seed, algo="sac"):
    def _inner():
        env = SoaringEnv(
            field_kw=dict(FIELD_KW),
            strength_scale_range=STRENGTH_RANGE,
        )
        env = Monitor(env)
        return env
    return _inner


def train_one(algo, seed, timesteps, results_dir, extra_kw=None, obs_mask=None):
    os.makedirs(results_dir, exist_ok=True)
    model_dir = os.path.join(results_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    env_fns = [make_env(seed + i, algo) for i in range(4 if algo == "ppo" else 1)]
    _mask = obs_mask
    vec_env = make_vec_env(lambda: SoaringEnv(
        field_kw=dict(FIELD_KW), strength_scale_range=STRENGTH_RANGE,
        near_thermal_prob=0.4, obs_mask=_mask),
        n_envs=4 if algo == "ppo" else 1, seed=seed)

    tb_log = os.path.join(results_dir, "tensorboard")
    if algo == "sac":
        kw = dict(SAC_KWARGS)
        if extra_kw:
            kw.update(extra_kw)
        model = SAC(env=vec_env, seed=seed, tensorboard_log=tb_log, **kw)
    else:
        kw = dict(PPO_KWARGS)
        if extra_kw:
            kw.update(extra_kw)
        model = PPO(env=vec_env, seed=seed, tensorboard_log=tb_log, **kw)

    reward_path = os.path.join(results_dir, f"rewards_{algo}_seed{seed}.npy")
    logger_cb   = RewardLogger(reward_path)

    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=logger_cb,
                tb_log_name=f"{algo}_seed{seed}")
    elapsed = time.time() - t0

    model_path = os.path.join(model_dir, f"{algo}_seed{seed}")
    model.save(model_path)
    print(f"  Saved model to {model_path}.zip  ({elapsed:.0f} s)")

    # save config
    cfg = dict(
        algo=algo, seed=seed, timesteps=timesteps,
        hyperparams=kw, field_kw=FIELD_KW, strength_range=STRENGTH_RANGE,
    )
    cfg_path = os.path.join(model_dir, f"{algo}_seed{seed}_config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    return model, reward_path



def eval_model(model, n_episodes=20, strength_scale=1.0, seed=0):
    """Return mean episode reward over n_episodes at a fixed strength_scale."""
    env = SoaringEnv(field_kw={**FIELD_KW, "strength_scale": strength_scale})
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            total += r
            done = terminated or truncated
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))



def main(
    algo        = "sac",
    seeds       = (0, 1, 2, 3, 4),
    timesteps   = 1_000_000,
    results_dir = "results",
):
    print(f"Training {algo.upper()} for {timesteps:,} timesteps "
          f"on seeds {list(seeds)}")

    all_rewards = {}
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        model, rp = train_one(algo, seed, timesteps, results_dir)
        rewards = np.load(rp)
        all_rewards[seed] = rewards
        if len(rewards) > 0:
            last_n = min(100, len(rewards))
            print(f"  Mean last {last_n} episode rewards: "
                  f"{rewards[-last_n:].mean():.1f} +/- {rewards[-last_n:].std():.1f}")

    print("\nTraining complete. Models saved to results/models/")


if __name__ == "__main__":
    main(algo = "sac", seeds = (0,), timesteps = 1_500_000, results_dir = "results")
