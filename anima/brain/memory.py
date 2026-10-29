"""Hippocampus: one-shot episodic memory with prioritised offline replay.

Cortex learns slowly; the hippocampus stores single experiences at once.
Each waking moment that matters (it brought reward or punishment, or it was
surprising) is snapshotted: what the senses reported, what the body was
feeling and what action was taken.  During slow-wave sleep the brain replays
these episodes back into the network together with the dopamine they earned,
so the slow synapses get many more learning trials than waking life provided
("complementary learning systems", McClelland, McNaughton & O'Reilly 1995;
reward-weighted replay, Gomperts et al. 2015).

This module is an abstract (non-spiking) model of the hippocampal store; the
replay itself is delivered as spiking activity into the network by
:class:`~anima.brain.brain.Brain`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Episode:
    t_s: float
    inputs: dict[str, np.ndarray]  # input-port intensities at the time
    actions: dict[str, int]  # output group -> chosen channel
    reward: float
    novelty: float
    salience: float = 0.0
    replays: int = 0


@dataclass
class HippocampusConfig:
    capacity: int = 300
    threshold: float = 0.08  # minimum salience to store
    decay_per_hour: float = 0.5  # salience decay per simulated hour
    replay_discount: float = 0.7  # salience multiplier after each replay


@dataclass
class Hippocampus:
    cfg: HippocampusConfig = field(default_factory=HippocampusConfig)
    episodes: list[Episode] = field(default_factory=list)
    stored_total: int = 0

    def encode(self, t_s: float, inputs: dict[str, np.ndarray], actions: dict[str, int],
               reward: float, novelty: float) -> bool:
        salience = abs(reward) + 0.5 * novelty
        if salience < self.cfg.threshold:
            return False
        ep = Episode(t_s, {k: np.asarray(v, dtype=np.float32).copy() for k, v in inputs.items()},
                     dict(actions), float(reward), float(novelty), float(salience))
        if len(self.episodes) >= self.cfg.capacity:
            weakest = int(np.argmin([e.salience for e in self.episodes]))
            if self.episodes[weakest].salience >= salience:
                return False
            self.episodes.pop(weakest)
        self.episodes.append(ep)
        self.stored_total += 1
        return True

    def decay(self, dt_s: float) -> None:
        if not self.episodes:
            return
        f = float(np.exp(np.log(1.0 - self.cfg.decay_per_hour) * dt_s / 3600.0)) \
            if self.cfg.decay_per_hour < 1 else 0.0
        for e in self.episodes:
            e.salience *= f

    def sample(self, rng: np.random.Generator) -> Episode | None:
        """Pick an episode to replay, proportional to salience."""
        if not self.episodes:
            return None
        s = np.array([e.salience for e in self.episodes])
        if s.sum() <= 0:
            return None
        ep = self.episodes[int(rng.choice(len(s), p=s / s.sum()))]
        ep.salience *= self.cfg.replay_discount
        ep.replays += 1
        return ep

    def __len__(self) -> int:
        return len(self.episodes)

    def state_dict(self) -> dict:
        return {
            "stored_total": self.stored_total,
            "episodes": [
                {"t_s": e.t_s, "inputs": {k: v.tolist() for k, v in e.inputs.items()},
                 "actions": e.actions, "reward": e.reward, "novelty": e.novelty,
                 "salience": e.salience, "replays": e.replays}
                for e in self.episodes
            ],
        }

    def load_state_dict(self, s: dict) -> None:
        self.stored_total = int(s["stored_total"])
        self.episodes = [
            Episode(float(e["t_s"]), {k: np.array(v, dtype=np.float32) for k, v in e["inputs"].items()},
                    {k: int(v) for k, v in e["actions"].items()}, float(e["reward"]),
                    float(e["novelty"]), float(e["salience"]), int(e["replays"]))
            for e in s["episodes"]
        ]
