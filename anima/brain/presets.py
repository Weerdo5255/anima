"""Ready-made brain specs.

``infant_spec()`` is the default creature: roughly 1,700 spiking neurons in
15 populations.  It is deliberately infant-like — it has drives, reflex-like
priors and plastic sensorimotor maps, but no knowledge.  Everything it ends
up doing well (finding food, babbling the syllables you reward, echoing the
sounds it hears) is learned with local rules while it lives.

            interoception (hunger, fatigue, boredom, loneliness)
                 |                 \\
   senses ---> CORTEX (800E/200I, recurrent, STDP) ---> MOTOR channels (WTA)
   (smell bearing, walls, taste)   |  \\                  forward / left / right
                 \\_______________ |___\\ (reward-STDP, dopamine)
   audio (17 phonemes) --> listening interneurons --| VOCAL gate --> consonant x vowel
          \\______ mirror (Hebbian) ________________________/        (WTA, babbling)
"""

from __future__ import annotations

from typing import Any

from .drives import DRIVE_NAMES

CONSONANTS = list("bdgklmnpstwh")
VOWELS = list("aeiou")
PHONEMES = CONSONANTS + VOWELS
BEARING_LABELS = [str(int(a)) for a in range(-180, 180, 45)]
MOTOR_LABELS = ["forward", "left", "right"]


def infant_spec(cortex_size: int = 1000, seed: int = 7, day_length_s: float = 1200.0,
                **brain_overrides: Any) -> dict[str, Any]:
    """The default brain.

    cortex_size: total cortical neurons (80% excitatory, 20% inhibitory).
    day_length_s: length of one simulated day-night cycle in seconds.
    brain_overrides: e.g. ``effects={"social_reward": 2.0}``.
    """
    nE = int(round(cortex_size * 0.8))
    nI = cortex_size - nE
    ncons, nvow = len(CONSONANTS), len(VOWELS)
    # Dopamine-gated sensorimotor learning.  Movement rewards arrive as the
    # movement completes, so a short eligibility trace (~0.3 s) gives sharp
    # credit; social rewards come seconds after a babble, hence 2 s for voice.
    rstdp_motor = {"type": "reward_stdp", "a_plus": 1.0, "a_minus": 0.0, "tau_e": 300.0,
                   "lr": 0.05, "normalize": "both"}
    rstdp_vocal = {"type": "reward_stdp", "a_plus": 1.0, "a_minus": 0.0, "tau_e": 2000.0,
                   "lr": 0.06, "normalize": "pre"}
    pops: list[dict[str, Any]] = [
        # ---------------------------------------------------- inputs
        {"name": "intero", "size": 60, "channels": 4, "kind": "RS", "noise": 2.0, "role": "intero",
         "input": {"max_current": 9.0, "labels": list(DRIVE_NAMES), "gated": False}},
        {"name": "sense.bearing", "size": 120, "channels": 8, "kind": "RS", "noise": 2.0,
         "role": "sensory", "input": {"max_current": 12.0, "labels": BEARING_LABELS}},
        {"name": "sense.wall", "size": 30, "channels": 3, "kind": "RS", "noise": 2.0,
         "role": "sensory", "input": {"max_current": 12.0, "labels": ["left", "front", "right"]}},
        {"name": "sense.taste", "size": 10, "channels": 1, "kind": "RS", "noise": 2.0,
         "role": "sensory", "input": {"max_current": 12.0, "labels": ["food"]}},
        {"name": "audio", "size": 8 * len(PHONEMES), "channels": len(PHONEMES), "kind": "RS",
         "noise": 2.0, "role": "audio", "input": {"max_current": 14.0, "labels": PHONEMES}},
        # ---------------------------------------------------- cortex
        {"name": "cortex.E", "size": nE, "kind": "RS", "noise": 4.0, "jitter": 1.0,
         "nmda_ratio": 0.35, "target_rate": 4.0, "role": "cortex"},
        {"name": "cortex.I", "size": nI, "kind": "FS", "noise": 3.5, "jitter": 1.0,
         "target_rate": 12.0, "role": "cortex"},
        # ---------------------------------------------------- action selection
        {"name": "motor", "size": 60, "channels": 3, "kind": "RS", "noise": 3.5, "role": "motor",
         "output": {"labels": MOTOR_LABELS}},
        {"name": "motor.inh", "size": 30, "kind": "FS", "noise": 2.0, "role": "motor"},
        # ---------------------------------------------------- voice
        {"name": "vocal.gate", "size": 20, "channels": 1, "kind": "RS", "noise": 3.0, "bias": -1.0,
         "nmda_ratio": 0.5, "role": "vocal", "output": {"labels": ["speak"]}},
        {"name": "vocal.C", "size": 8 * ncons, "channels": ncons, "kind": "RS", "noise": 3.5,
         "role": "vocal", "output": {"labels": CONSONANTS}},
        {"name": "vocal.V", "size": 10 * nvow, "channels": nvow, "kind": "RS", "noise": 3.5,
         "role": "vocal", "output": {"labels": VOWELS}},
        {"name": "vocal.Cinh", "size": 20, "kind": "FS", "noise": 2.0, "role": "vocal"},
        {"name": "vocal.Vinh", "size": 15, "kind": "FS", "noise": 2.0, "role": "vocal"},
        {"name": "listen.inh", "size": 20, "kind": "FS", "noise": 1.0, "role": "listen"},
    ]
    cons_gain = [1.0] * ncons + [0.0] * nvow
    vow_gain = [0.0] * ncons + [1.0] * nvow
    projs: list[dict[str, Any]] = [
        # ---- cortex: balanced recurrent E/I network with STDP + homeostasis
        {"pre": "cortex.E", "post": "cortex.E", "p": 0.1, "weight": 0.015, "w_max": 0.04,
         "scaling": True, "rule": {"type": "stdp", "a_plus": 0.004, "a_minus": 0.005,
                                   "gate": "acetylcholine"}},
        {"pre": "cortex.E", "post": "cortex.I", "p": 0.2, "weight": 0.015},
        {"pre": "cortex.I", "post": "cortex.E", "p": 0.3, "weight": 0.08},
        {"pre": "cortex.I", "post": "cortex.I", "p": 0.2, "weight": 0.04},
        # ---- thalamo-cortical input
        {"pre": "intero", "post": "cortex.E", "p": 0.1, "weight": 0.02},
        {"pre": "sense.bearing", "post": "cortex.E", "p": 0.1, "weight": 0.025},
        {"pre": "sense.wall", "post": "cortex.E", "p": 0.1, "weight": 0.025},
        {"pre": "sense.taste", "post": "cortex.E", "p": 0.1, "weight": 0.025},
        {"pre": "audio", "post": "cortex.E", "p": 0.1, "weight": 0.025},
        # ---- action selection: plastic sensorimotor maps (dopamine-gated)
        {"pre": "sense.bearing", "post": "motor", "p": 0.5, "weight": 0.008, "w_max": 0.05,
         "fixed_indegree": True, "rule": dict(rstdp_motor)},
        {"pre": "sense.wall", "post": "motor", "p": 0.5, "weight": 0.004, "w_max": 0.04,
         "fixed_indegree": True, "rule": dict(rstdp_motor)},
        {"pre": "cortex.E", "post": "motor", "p": 0.15, "weight": 0.002, "w_max": 0.02,
         "fixed_indegree": True, "rule": dict(rstdp_motor)},
        # hunger and boredom energise locomotion (innate)
        {"pre": "intero", "post": "motor", "p": 0.5, "weight": 0.012, "fixed_indegree": True,
         "pre_channel_gain": [1.0, 0.0, 0.8, 0.0]},
        {"pre": "motor", "post": "motor", "topology": "channels", "p": 0.3, "weight": 0.01,
         "fixed_indegree": True},
        {"pre": "motor", "post": "motor.inh", "p": 0.5, "weight": 0.02, "fixed_indegree": True},
        {"pre": "motor.inh", "post": "motor", "p": 0.5, "weight": 0.06, "fixed_indegree": True},
        # ---- voice: when to speak
        {"pre": "intero", "post": "vocal.gate", "p": 0.6, "weight": 0.024, "fixed_indegree": True,
         "pre_channel_gain": [0.0, 0.0, 0.5, 1.0]},
        {"pre": "audio", "post": "vocal.gate", "p": 0.3, "weight": 0.006},
        {"pre": "audio", "post": "listen.inh", "p": 0.4, "weight": 0.02},
        {"pre": "listen.inh", "post": "vocal.gate", "p": 0.6, "weight": 0.1},
        {"pre": "cortex.E", "post": "vocal.gate", "p": 0.1, "weight": 0.002, "w_max": 0.02,
         "rule": dict(rstdp_vocal)},
        # ---- voice: what to say (consonant x vowel competitions)
        {"pre": "vocal.gate", "post": "vocal.C", "p": 0.3, "weight": 0.008, "fixed_indegree": True},
        {"pre": "vocal.gate", "post": "vocal.V", "p": 0.3, "weight": 0.008, "fixed_indegree": True},
        {"pre": "vocal.C", "post": "vocal.C", "topology": "channels", "p": 0.4, "weight": 0.012,
         "fixed_indegree": True},
        {"pre": "vocal.V", "post": "vocal.V", "topology": "channels", "p": 0.4, "weight": 0.012,
         "fixed_indegree": True},
        {"pre": "vocal.C", "post": "vocal.Cinh", "p": 0.5, "weight": 0.02, "fixed_indegree": True},
        {"pre": "vocal.Cinh", "post": "vocal.C", "p": 0.5, "weight": 0.06, "fixed_indegree": True},
        {"pre": "vocal.V", "post": "vocal.Vinh", "p": 0.5, "weight": 0.02, "fixed_indegree": True},
        {"pre": "vocal.Vinh", "post": "vocal.V", "p": 0.5, "weight": 0.06, "fixed_indegree": True},
        # mirror mapping hearing -> articulation, learned from self-hearing
        {"pre": "audio", "post": "vocal.C", "p": 0.6, "weight": 0.008, "w_max": 0.06,
         "fixed_indegree": True, "pre_channel_gain": cons_gain,
         "rule": {"type": "hebbian", "rate": 0.005}},
        {"pre": "audio", "post": "vocal.V", "p": 0.6, "weight": 0.008, "w_max": 0.06,
         "fixed_indegree": True, "pre_channel_gain": vow_gain,
         "rule": {"type": "hebbian", "rate": 0.005}},
        # which sounds are worth making (shaped by social reward)
        {"pre": "intero", "post": "vocal.C", "p": 0.5, "weight": 0.003, "w_max": 0.03,
         "fixed_indegree": True, "rule": dict(rstdp_vocal)},
        {"pre": "intero", "post": "vocal.V", "p": 0.5, "weight": 0.003, "w_max": 0.03,
         "fixed_indegree": True, "rule": dict(rstdp_vocal)},
        {"pre": "cortex.E", "post": "vocal.C", "p": 0.1, "weight": 0.002, "w_max": 0.02,
         "fixed_indegree": True, "rule": dict(rstdp_vocal)},
        {"pre": "cortex.E", "post": "vocal.V", "p": 0.1, "weight": 0.002, "w_max": 0.02,
         "fixed_indegree": True, "rule": dict(rstdp_vocal)},
        # ---- efference copies back to cortex
        {"pre": "motor", "post": "cortex.E", "p": 0.05, "weight": 0.02},
        {"pre": "vocal.C", "post": "cortex.E", "p": 0.05, "weight": 0.015},
        {"pre": "vocal.V", "post": "cortex.E", "p": 0.05, "weight": 0.015},
    ]
    brain: dict[str, Any] = {"drives": {"day_length_s": day_length_s}}
    for k, v in brain_overrides.items():
        brain.setdefault(k, {}).update(v)
    return {"name": "infant", "seed": seed, "population": pops, "projection": projs, "brain": brain}


def minimal_spec(seed: int = 1) -> dict[str, Any]:
    """A small free-running cortex with drives and one motor group; handy for tests."""
    return {
        "name": "minimal", "seed": seed,
        "population": [
            {"name": "intero", "size": 20, "channels": 4, "noise": 2.0, "role": "intero",
             "input": {"labels": list(DRIVE_NAMES), "gated": False}},
            {"name": "cortex.E", "size": 160, "noise": 4.0, "jitter": 1.0, "role": "cortex",
             "target_rate": 4.0},
            {"name": "cortex.I", "size": 40, "kind": "FS", "noise": 3.5, "jitter": 1.0,
             "role": "cortex"},
            {"name": "motor", "size": 30, "channels": 3, "noise": 3.5, "role": "motor",
             "output": {"labels": MOTOR_LABELS}},
            {"name": "motor.inh", "size": 10, "kind": "FS", "noise": 2.0, "role": "motor"},
        ],
        "projection": [
            {"pre": "cortex.E", "post": "cortex.E", "p": 0.2, "weight": 0.015, "rule": {"type": "stdp"}},
            {"pre": "cortex.E", "post": "cortex.I", "p": 0.3, "weight": 0.015},
            {"pre": "cortex.I", "post": "cortex.E", "p": 0.4, "weight": 0.08},
            {"pre": "intero", "post": "cortex.E", "p": 0.2, "weight": 0.02},
            {"pre": "cortex.E", "post": "motor", "p": 0.2, "weight": 0.004,
             "rule": {"type": "reward_stdp", "a_minus": 0.0}},
            {"pre": "intero", "post": "motor", "p": 0.5, "weight": 0.012,
             "pre_channel_gain": [1.0, 0.0, 0.8, 0.0]},
            {"pre": "motor", "post": "motor.inh", "p": 0.5, "weight": 0.02},
            {"pre": "motor.inh", "post": "motor", "p": 0.5, "weight": 0.06},
        ],
        "brain": {},
    }


PRESETS = {"infant": infant_spec, "minimal": minimal_spec}
