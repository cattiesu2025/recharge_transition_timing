"""Seed-matched Double DQN training and checkpoint provenance."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_baselines3
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from torch.nn import functional as F

from .config import canonical_hash, file_hash
from .env import ACTION_COUNT, MASK_SLICE, RechargeEnv, Scenario


class DoubleDQN(DQN):
    @staticmethod
    def _random_valid_actions(observation: np.ndarray) -> np.ndarray:
        values = np.asarray(observation)
        masks = values[..., MASK_SLICE] > 0.5
        if masks.ndim == 1:
            masks = masks[None, :]
        actions = []
        for mask in masks:
            valid = np.flatnonzero(mask)
            if len(valid) == 0:
                raise ValueError("Observation has no valid actions")
            actions.append(int(np.random.choice(valid)))
        return np.asarray(actions, dtype=np.int64)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """Epsilon-greedy prediction restricted to the observation's action mask."""
        obs_tensor, vectorized = self.policy.obs_to_tensor(observation)
        with th.no_grad():
            q_values = self.q_net(obs_tensor)
            masks = obs_tensor[..., MASK_SLICE] > 0.5
            if masks.shape[-1] != ACTION_COUNT or not th.all(masks.any(dim=1)):
                raise ValueError("Observation contains an invalid action mask")
            actions = q_values.masked_fill(~masks, -th.inf).argmax(dim=1).cpu().numpy()
        if not deterministic:
            random_actions = self._random_valid_actions(np.asarray(observation))
            if random_actions.shape != actions.shape:
                random_actions = random_actions.reshape(actions.shape)
            explore = np.random.random(len(actions)) < self.exploration_rate
            actions = np.where(explore, random_actions, actions)
        if not vectorized:
            actions = actions.squeeze(axis=0)
        return actions, state

    def _sample_action(self, learning_starts: int, action_noise=None,
                       n_envs: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Use valid actions during replay-buffer warmup as well as epsilon exploration."""
        assert self._last_obs is not None, "Last observation is required for action masking"
        if self.num_timesteps < learning_starts:
            action = self._random_valid_actions(self._last_obs)
        else:
            action, _ = self.predict(self._last_obs, deterministic=False)
            action = np.asarray(action, dtype=np.int64).reshape(n_envs)
        return action, action.copy()

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = data.discounts if getattr(data, "discounts", None) is not None else self.gamma
            with th.no_grad():
                online_values = self.q_net(data.next_observations)
                masks = data.next_observations[..., MASK_SLICE] > 0.5
                actions = online_values.masked_fill(~masks, -th.inf).argmax(dim=1, keepdim=True)
                values = self.q_net_target(data.next_observations).gather(1, actions)
                target = data.rewards + (1 - data.dones) * discounts * values
            current = self.q_net(data.observations).gather(1, data.actions.long())
            loss = F.smooth_l1_loss(current, target)
            losses.append(float(loss.item()))
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", float(np.mean(losses)))


def sample_training_scenario(config: dict[str, Any], seed: int, episode: int) -> Scenario:
    bounds = config["environment"]["training_distribution"]
    rng = np.random.default_rng(np.random.SeedSequence([seed, episode, 0x524543]))
    return Scenario(f"train_s{seed}_e{episode:06d}",
                    int(rng.choice(bounds["distances"])),
                    float(rng.choice(bounds["batteries"])),
                    int(rng.choice(bounds["quotas"])))


class TrainingScenarioEnv(gym.Wrapper):
    def __init__(self, env: RechargeEnv, config: dict[str, Any], seed: int) -> None:
        super().__init__(env)
        self.config = config
        self.training_seed = seed
        self.episode = 0
        self.episode_return = 0.0
        self.episode_length = 0
        self.current: Scenario | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.current = sample_training_scenario(self.config, self.training_seed, self.episode)
        self.episode += 1
        self.episode_return = 0.0
        self.episode_length = 0
        return self.env.reset(seed=seed, options={"scenario": self.current})

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_return += reward
        self.episode_length += 1
        if terminated or truncated:
            info = dict(info)
            info["training_episode"] = {"episode": self.episode - 1,
                "return": self.episode_return, "length": self.episode_length,
                "outcome": info["outcome"], "work": info["total_work"],
                "scenario": asdict(self.current)}
        return obs, reward, terminated, truncated, info


class MetricsCallback(BaseCallback):
    def __init__(self, path: Path) -> None:
        super().__init__(verbose=0)
        self.path = path
        self.handle = None
        self.completed = 0

    def _on_training_start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("training_episode")
            if episode is not None:
                self.handle.write(json.dumps({"steps": self.num_timesteps, **episode},
                                             sort_keys=True) + "\n")
                self.handle.flush()
                self.completed += 1
        return True

    def _on_training_end(self) -> None:
        if self.handle:
            self.handle.close()


def train_condition(config: dict[str, Any], condition: str, seed: int,
                    output: str | Path, device: str = "cpu", steps: int | None = None) -> Path:
    total = int(steps if steps is not None else config["experiment"]["train_steps"])
    if total <= 0:
        raise ValueError("Training steps must be positive")
    agent = config["agent"]
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=True)
    env = TrainingScenarioEnv(RechargeEnv(config, condition), config, seed)
    callback = MetricsCallback(dest / "training_metrics.jsonl")
    model = DoubleDQN("MlpPolicy", env,
        learning_rate=float(agent["learning_rate"]), buffer_size=int(agent["replay_capacity"]),
        learning_starts=int(agent["learning_starts"]), batch_size=int(agent["batch_size"]),
        gamma=float(agent["gamma"]), train_freq=int(agent["train_frequency"]),
        gradient_steps=int(agent["gradient_steps"]),
        replay_buffer_kwargs={"handle_timeout_termination": False},
        target_update_interval=int(agent["target_update_interval"]),
        exploration_fraction=min(1.0, float(agent["epsilon_decay_steps"]) / total),
        exploration_initial_eps=float(agent["epsilon_start"]),
        exploration_final_eps=float(agent["epsilon_end"]),
        max_grad_norm=float(agent["gradient_clip"]),
        policy_kwargs={"net_arch": [int(x) for x in agent["hidden_sizes"]]},
        seed=seed, device=device, verbose=0)
    model.learn(total_timesteps=total, callback=callback, log_interval=None)
    checkpoint = dest / "final.zip"
    model.save(checkpoint)
    metadata = {"algorithm": "MaskedDoubleDQN_overridden_SB3_DQN_train_and_predict",
        "stable_baselines3_version": stable_baselines3.__version__,
        "condition": condition, "seed": seed, "training_steps": model.num_timesteps,
        "episodes": callback.completed, "config_hash": canonical_hash(config),
        "checkpoint_sha256": file_hash(checkpoint), "checkpoint_role": "final_fixed_budget"}
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    model.get_env().close()
    return checkpoint
