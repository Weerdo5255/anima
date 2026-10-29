"""The body: physiology, sense organs and effectors.

The body sits between a :class:`~anima.world.base.World` and a
:class:`~anima.brain.brain.Brain`.  Every brain tick it

1. reads the world's sensors and writes them into the brain's input ports
   (with hunger sharpening the sense of smell, as ghrelin does);
2. plays any queued sound (your words, its own voice) into the ears;
3. decodes the motor and vocal populations into movements and syllables;
4. applies movements to the world and turns the consequences into primary
   rewards: food is pleasant (more so when hungry), food smell getting
   stronger is mildly pleasant when hungry, bumping into walls hurts;
5. burns energy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .motor import MotorConfig, MotorDecoder
from .voice import Ear, Voice, VoiceConfig

if TYPE_CHECKING:  # pragma: no cover
    from ..brain.brain import Brain
    from ..world.base import World


@dataclass
class BodyConfig:
    energy: float = 0.75
    basal_cost_per_s: float = 0.002  # energy burnt per second awake
    sleep_cost_factor: float = 0.35
    move_cost: float = 0.0004  # per movement
    speak_cost: float = 0.0002  # per syllable
    food_reward: float = 1.2  # x (0.5 + hunger) x energy gained / 0.25
    approach_reward: float = 1.0  # x hunger x relative change of the smell ahead
    bump_reward: float = -0.25
    smell_gain_sated: float = 0.35  # smell sensitivity when sated (1.0 when starving)
    speech_contact: float = 1.0  # social contact strength of being spoken to
    hand_cost: float = 0.0005  # energy per use of a hand (creatures with hands)
    #: sensor name in the world -> input port in the brain
    sensor_ports: dict[str, str] = field(default_factory=lambda: {
        "bearing": "sense.bearing", "wall": "sense.wall", "taste": "sense.taste"})
    motor: MotorConfig = field(default_factory=MotorConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    # hands reach rarely and deliberately: higher threshold, a few seconds' rest
    hands: MotorConfig = field(default_factory=lambda: MotorConfig(
        threshold=0.42, margin=1.3, duration_ms=250.0, refractory_ms=5000.0))


class Body:
    def __init__(self, brain: "Brain", world: "World", config: BodyConfig | None = None) -> None:
        self.brain = brain
        self.world = world
        self.cfg = config or BodyConfig()
        self.energy = self.cfg.energy
        self.motor = MotorDecoder(brain, "motor", self.cfg.motor) if "motor" in brain.outputs else None
        self.hands = MotorDecoder(brain, "hands", self.cfg.hands) if "hands" in brain.outputs else None
        self.ear = None
        if "audio" in brain.inputs:
            from ..brain.presets import CONSONANTS, VOWELS
            self.ear = Ear(brain.inputs["audio"].labels, CONSONANTS, VOWELS)
        self.voice = None
        if all(k in brain.outputs for k in ("vocal.gate", "vocal.C", "vocal.V")):
            self.voice = Voice(brain, self.ear, self.cfg.voice)
        self._smell_at_start = 0.0
        self._was_hearing = False
        self.stats = {"ate": 0, "bumps": 0, "moves": 0, "syllables": 0, "utterances": 0,
                      "heard": 0}
        if self.hands is not None:
            self.stats["reaches"] = 0

    # ------------------------------------------------------------ senses
    def hear(self, text: str, valence: float = 0.0) -> None:
        """Someone speaks to us (``valence`` from smiles / frowns)."""
        n = self.ear.hear(text) if self.ear is not None else 0
        if n or valence:
            self.brain.social_contact(self.cfg.speech_contact if n else 0.6, valence)
            if not self.brain.awake:
                self.brain.sleep.stimulate(0.08 * n)
            self.stats["heard"] += 1

    def hear_sound(self, text: str) -> int:
        """A sound that is not someone talking to it (e.g. a program's
        output): heard phoneme by phoneme like speech, but with no social
        contact.  Returns the number of phonemes queued."""
        return self.ear.hear(text) if self.ear is not None else 0

    def sense(self, dt_ms: float) -> None:
        b = self.brain
        obs = self.world.observe()
        smell_gain = self.cfg.smell_gain_sated + (1.0 - self.cfg.smell_gain_sated) * b.drives.hunger
        for key, port in self.cfg.sensor_ports.items():
            if key in obs and port in b.inputs:
                val = np.asarray(obs[key], dtype=float)
                if key == "bearing":
                    val = val * smell_gain
                b.set_input(port, np.clip(val, 0.0, 1.0))
        if self.ear is not None and "audio" in b.inputs:
            b.set_input("audio", self.ear.step(dt_ms))
            if self.ear.hearing_external and not b.awake:
                b.sleep.stimulate(0.004 * dt_ms / 10.0)

    # ------------------------------------------------------------ actions
    def act(self, t_ms: float) -> list[tuple[str, dict[str, Any]]]:
        events: list[tuple[str, dict[str, Any]]] = []
        b, cfg = self.brain, self.cfg
        if self.motor is not None:
            res = self.motor.update(t_ms)
            if res is not None:
                kind, k = res
                label = self.motor.labels[k]
                if kind == "start":
                    # The movement is being executed; its sensory consequences
                    # arrive when it completes.  (Changing the world right away
                    # would let dopamine credit the *new* sensory state with the
                    # decision that was taken in the old one.)
                    self._smell_at_start = self.world.smell_ahead()
                    self.energy -= cfg.move_cost
                    self.stats["moves"] += 1
                    events.append(("action", {"action": label, "t": t_ms / 1000.0}))
                else:  # movement completed: apply it and feel the consequences
                    outcome = self.world.apply(label)
                    for ev in outcome:
                        events.append(ev)
                        if ev[0] == "ate":
                            gained = min(1.0 - self.energy, ev[1].get("energy", 0.25))
                            hunger = b.drives.hunger
                            self.energy += gained
                            b.deliver_reward(cfg.food_reward * (0.5 + hunger) * gained / 0.25, "food")
                            self.stats["ate"] += 1
                        elif ev[0] == "bump":
                            b.deliver_reward(cfg.bump_reward, "pain")
                            self.stats["bumps"] += 1
                    # did the food smell ahead get stronger?  Relative change
                    # (Weber's law): turning toward food and stepping closer
                    # are judged on the same scale at any distance.
                    after = self.world.smell_ahead()
                    before = self._smell_at_start
                    rel = (after - before) / (0.5 * (after + before) + 0.02)
                    b.deliver_reward(cfg.approach_reward * b.drives.hunger * rel, "smell")
        if self.hands is not None:
            res = self.hands.update(t_ms)
            if res is not None and res[0] == "start":
                k = res[1]
                self.energy -= cfg.hand_cost
                self.stats["reaches"] = self.stats.get("reaches", 0) + 1
                events.append(("hand", {"hand": k, "label": self.hands.labels[k], "t": t_ms / 1000.0}))
        if self.voice is not None:
            for ev in self.voice.update(t_ms):
                events.append(ev)
                if ev[0] == "syllable":
                    self.energy -= cfg.speak_cost
                    self.stats["syllables"] += 1
                elif ev[0] == "utterance":
                    self.stats["utterances"] += 1
        return events

    def metabolism(self, dt_s: float) -> None:
        cost = self.cfg.basal_cost_per_s * dt_s
        if not self.brain.awake:
            cost *= self.cfg.sleep_cost_factor
        self.energy = float(np.clip(self.energy - cost, 0.0, 1.0))

    # ------------------------------------------------------------ persistence
    def state_dict(self) -> dict:
        return {"energy": self.energy, "stats": dict(self.stats),
                "voice": self.voice.state_dict() if self.voice else None,
                "ear": self.ear.state_dict() if self.ear else None,
                "motor": self.motor.state_dict() if self.motor else None,
                "hands": self.hands.state_dict() if self.hands else None,
                "smell_at_start": self._smell_at_start,
                "cfg": asdict(self.cfg)}

    def load_state_dict(self, s: dict) -> None:
        self.energy = float(s["energy"])
        self.stats.update(s["stats"])
        if self.voice is not None and s.get("voice"):
            self.voice.load_state_dict(s["voice"])
        if self.ear is not None and s.get("ear"):
            self.ear.load_state_dict(s["ear"])
        if self.motor is not None and s.get("motor"):
            self.motor.load_state_dict(s["motor"])
        if self.hands is not None and s.get("hands"):
            self.hands.load_state_dict(s["hands"])
        self._smell_at_start = float(s.get("smell_at_start", 0.0))
