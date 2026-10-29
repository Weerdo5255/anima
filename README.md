# anima

**A local framework for brain-like agents that live on their own clock.**

anima does not build a language model.  It builds a *spiking neural network*
— neurons with membrane potentials, synapses with conductances, learning by
local plasticity — and embeds it in what real brains are embedded in: drives
that build up, neuromodulators that change how the whole network computes,
a sleep/wake cycle, memory replay, and a body in a world.

Then it lets it run.  The network fires whether or not anyone is there.  The
creature gets hungry, bored, lonely and sleepy; it wanders, forages,
babbles, falls asleep, replays its day and wakes up.  **It never waits for a
prompt.**  Your text is not a request — it is turned into sound and heard,
phoneme by phoneme, while everything else keeps going.  What it does next is
whatever its dynamics produce.

```
$ anima chat
anima is alive (1667 spiking neurons, 07:11 awake). It does not wait for you: it will
babble, wander, get hungry, bored, lonely and sleepy on its own.
Talk to it whenever you like. /help for commands, /quit to leave.

anima › liwoli su   [07:15]
anima › tu   [07:18]
anima › le mutu   [07:23]
you › hello
anima › mu no sa lapi   [07:31]
anima › so   [07:38]
you › mama
anima › kahe ti ta   [07:45]
anima › gi   [07:50]
you › :)
anima › he   [07:54]
anima › gi   [07:58]
anima › tela hi   [08:06]
you › baba
anima › hihi   [08:09]
      (anima found food)
anima › lanu   [08:19]
you › /status
[08:20 awake] hunger 0.18 fatigue 0.22 boredom 0.36 loneliness 0.32 | DA 0.20 NE 0.35 ACh 0.80 5HT 0.51
anima › ba   [08:24]
      (anima found food)
anima › bu   [08:27]
you › /quit
saved anima_brain.npz — run again to continue its life.
```

That is a real session with a newborn creature; the only edits are removing
the terminal's prompt redraws, wrapping the first line and shortening the
save path.  The first three utterances happened before anyone said
anything.  The bracketed times are its own clock — one simulated day lasts
20 real minutes by default, so it gets sleepy about 13 minutes in.

Everything runs locally on a CPU with **NumPy as the only dependency**.

---

## Quick start

```bash
cd anima
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .                                       # just numpy
anima chat                                             # talk to it (any terminal)
anima live                                             # live dashboard (macOS/Linux; Windows: pip install windows-curses)
```

`anima` and `python -m anima` are the same thing.  The creature is saved to
`anima_brain.npz` when you quit and resumed next time — same synapses, same
memories, same time of day.  `--new` starts a fresh one.

In the chat, type anything at any time.  `:)` or `*smile*` is approval and
`:(` disapproval (non-verbal, no sound).  `/status`, `/world`, `/stats`,
`/sleep`, `/wake`, `/save`, `/quit`.

The live dashboard shows the world, its drives and neuromodulators, the
activity of each region, a live raster of cortical spikes and the
conversation:

```
 anima  07:42   AWAKE     age    25.3s  1.0x realtime  1667 neurons  memories 14  replays 0

 world                                          drives
 +------------------------------------------+   hunger     ███████████······················· 0.33
 |                                          |   fatigue    ██████···························· 0.18
 |                                          |   boredom    ███████████······················· 0.31
 |                                          |   loneliness ████████████······················ 0.35
 |                                          |
 |                     .. .           *     |   neuromodulators
 |                     .. . . ..            |   dopamine   ████████·························· 0.22
 |                        *     .. .        |     phasic   █████████████████················· +0.02
 |                                  ..v     |   noradren.  ████████████████████·············· 0.59
 |                                     *    |   acetylch.  ████████████████████████████······ 0.82
 |                                          |   serotonin  █████████████████················· 0.51
 +------------------------------------------+
 doing: —        energy ███████···              activity (Hz)
 cortex spikes (10 neurons, last 6 s)           cortex   █▆▆▆▆▅▅▄▅▄▄▄▄▄▃▃▂▃▂▃▃▂▃▂▂▂▂▁▁▁▁    ▁   5.7
 |   | ||   |  || | | | ||| | |    |  |  |  |   senses   ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▆▆▆▆▆   9.8
 |||  |    | | |  | |  | ||  |  |||  | |   ||   hearing  █▇▇▇▆▆▅▅▄▄▃▃▃▃▂▂▁▁▁▁▁▁             ▁   0.6
 ||| ||| |  || ||  | | || ||  | | ||  |   |||   motor    ▇▇▇▇▇▇▆▆▆▆▇▇▆█▆▆▆▆▆▆▆▆▆▆▆▆▆▇▆▆▇▆▆▆▆▆   8.9
    ||  ||  |  |  || | | ||  | ||   |  | ||||   voice    █▆▆▆▅▄▃▃▂▂▁▁▁      ▂▂▂▂▃▄▄▄▄▃▃▃▄▄▄▆▄   1.0
 |||| ||  || | | | | |||| || | | |    | |  |    interoc. ▆▆▆▅▆▅▅▅▅▅▅▅▅▆▅▅▆▆▅▅▅▆▆▆▆▆▆▇▇▇█▇▇▇▇▇   4.1
 || || |  | || |  | |  |  | |   | | | | |  ||
 || ||    | | |   | || || | |||| |||| | | ||
 |   |||    |||| | |  |  || |    | |  | | |||
 ||  | ||   ||  |   |  | |||    || | |   |  |
 || |||  ||    |  || | || |||  | | | || || ||

 conversation
     9.5  anima › le mutu
    14.8  anima › mu no sa
    17.4  anima › we
    20.2  you   › hello mama
    23.1  anima › ti welu toto
          anima is saying: si…
you ›                                                                 Enter: say  /save  /sleep  /wake
```

## Raising it with a local LLM (Ollama)

The brain is not a language model — but a language model can raise it.
`anima train` gives a local LLM, served by [Ollama](https://ollama.com), the
job of a parent or animal trainer, and takes the creature through four
lessons until it can look after itself:

```bash
ollama pull llama3.2      # once; any chat model works (qwen3, gemma3, mistral, ...)
anima tutor-check         # is Ollama up? does the model give usable decisions? how fast?
anima train               # up to 30 creature-minutes of lessons in real time; Ctrl-C any time
anima train               # later: the lessons continue where they stopped
anima live                # meet the result
```

No Ollama?  `anima train --teacher scripted` uses the built-in teacher.
Other machines: `--host gpu-box:11434` (or `OLLAMA_HOST`).  Reasoning models
are faster with `--think off`.  Once it has graduated, nobody needs to be
there: it forages, babbles and sleeps on its own.  In `anima chat` you can
still talk to it, smile at it, and `/feed` it.

A real session with the built-in scripted teacher, in two sittings (paths
shortened; long stretches cut at the "…"):

```
$ anima train --teacher scripted --minutes 8.5 --fast
teacher: scripted teacher (no LLM)
lesson 1 of 4: Find food — turn toward the smell of food and walk to it, without help
training for up to 8.5 creature-minutes as fast as the CPU allows; Ctrl-C stops (progress is saved)

   [07:11 forage] scripted: Let it practise; the guide keeps food within reach. → wait
      anima › susupe sape
      anima › ke
   [07:21 forage] scripted: Let it practise; the guide keeps food within reach. → wait
      anima › do ku
   …
   ✎ [13:34] exam 'forage': score 0.68 (chance 0.33; correct moves left 18/20, ahead 13/20, right 10/20)
   …
   ✎ [16:15] exam 'forage': score 0.67 (chance 0.33; correct moves left 16/20, ahead 12/20, right 12/20)
   …
$ anima train --teacher scripted --minutes 2 --fast      # later: it carries on where it stopped
resumed anima_brain.npz (age 510s of simulated life)
  schooling: in lesson 1 of 4 (forage)  [anima_brain.tutor.json]
   …
   ✎ [18:59] exam 'forage': score 0.77 (chance 0.33; correct moves left 19/20, ahead 11/20, right 16/20)
★ [18:59] graduated 'Find food': exam 0.77 (need 0.60; chance 0.33), food 1.8/min (need 1.0)
   ── lesson 2: Copy me — start its reply with the consonant it just heard
   [19:08 echo] scripted: Offer a new sound (n) to copy; the reflex praises a copy. → say 'no no'
      anima › si to wuwo
   [19:18 echo] scripted: Offer a new sound (t) to copy; the reflex praises a copy. → say 'te te'
      anima › ke tutu
```

With an LLM teacher the decision lines carry the model's name, its latency
and its own one-line thought instead of the script's.

**How it teaches.** Every ~8 seconds the model gets a short JSON report —
drives, energy, what the creature just said and did, its measured skills,
the lesson's goal, progress and tips — and answers with up to three actions:
say a syllable or two, smile, frown, give food (ahead/left/right/near),
switch a reflex rule on or off, go back to review an earlier lesson, or
leave itself a note.  It never touches a weight; all learning is still the
creature's own dopamine, spike timing and sleep.

Rewards only teach if they come within about a second of the behaviour (the
width of the eligibility traces), and a model takes seconds to think.  So
teaching runs at two speeds, as a human trainer's does with a clicker: the
LLM picks the strategy and the moments, and **reflex rules** running on the
creature's own 10 ms clock deliver the contingent part — a smile half a
second after an *m* syllable, food in front of it when it calls while
hungry, food within reach while it learns to forage, a smile when it copies
a sound.  The creature never waits for the model: a slow answer just means
fewer decisions.

| lesson | goal | reflexes | conditions | passed when (measured) |
|---|---|---|---|---|
| 1 forage | find food alone | `guide_food` | kept mildly hungry (energy ≤ 0.5) | two exams in a row: ≥ 60% correct moves with food held left / ahead / right (chance 33%); ≥ 1 meal/min |
| 2 echo | copy the sound it hears | `reward_echo` | a little hungry (≤ 0.6) | starts its reply with your consonant ≥ 1.8× as often as chance (and ≥ 15%) |
| 3 babble | make the target sound | `reward_syllable` | a little hungry (≤ 0.6) | ≥ 30% of its syllables start with *m* |
| 4 independence | live on its own | none | — | keeps its energy up (mean ≥ 0.35, never < 0.08) for 8 awake minutes, no help |

Each lesson also needs a minimum of awake practice, and its test has to keep
passing for a minute.  Exams are taken by a **clone** of the creature — same
synapses, plasticity off — so testing never interrupts or trains it; passed
skills get spot checks later, and the teacher can send it back to review.
A lesson still not passed after 45 awake minutes is left "not mastered" and
the curriculum moves on, so training never gets stuck.  `--with-ask` adds an
experimental fifth lesson, *ask for food* (say *m* when hungry: no food
grows, calls are answered with food and calls made when full with a frown) —
see the honest limits for why it is not in the default set.
Graduation is decided by these measurements, never by the model, and every
decision is validated first: unknown tools and rules are dropped, rules
from other lessons and free food in the "ask" lesson are refused (unless it
is starving), lessons can't be skipped, speech is reduced to letters.  If
the model is unreachable or answers nonsense, the scripted teacher covers
that turn (after three failures in a row the model gets a minute's rest).
Progress — lesson, measurements, exams, rules, the teacher's journal — is
saved to `anima_brain.tutor.json` next to the brain, so skills build up over
as many sessions as it takes.  The whole curriculum takes about an hour of
the creature's life, nights included.

From Python:

```python
from anima import Agent
from anima.tutor import OllamaTeacher, Tutor

agent = Agent.create()
tutor = Tutor(agent, OllamaTeacher(model="llama3.2"))
print(tutor.train(minutes=30))       # creature-minutes, in real time
```

`examples/04_llm_tutor.py` is a complete version; `anima/tutor/` has the
client (standard library only), reflexes, measurements, exams, curriculum
and teachers.  Lessons are data: pass `Tutor(..., curriculum=Curriculum([Lesson(...), ...]))`
to teach your own, and add reflexes to `anima.tutor.RULES`.  Results are in [section 5](#5-what-a-teacher-adds).

## What is in the default creature

About 1,700 Izhikevich spiking neurons in 15 populations, 38 projections,
~234,000 synapses, simulated at 1 ms resolution.  The whole creature (brain,
body, world) runs about 1.7–1.9x faster than real time, single-threaded, on
the small cloud VM it was developed on; most laptops are faster.  If yours
can't keep up, `--cortex 600` makes a smaller brain.

| system | implementation | what it gives the creature |
|---|---|---|
| cortex | 800 excitatory + 200 inhibitory neurons, recurrent, conductance synapses (AMPA/NMDA/GABA), STDP, homeostasis | ongoing, self-sustained, oscillating activity; mixes senses, body state and its own actions |
| senses | smell (8 bearing channels), whiskers (3), taste, hearing (17 phonemes) | a world it can learn about; your words as sound |
| interoception | hunger, fatigue, boredom, loneliness as a spiking population | motivation that exists with nobody around |
| motor | 3 competing channels (forward/left/right), dopamine-gated plasticity | wandering, then learned foraging |
| voice | a "when to speak" gate + consonant × vowel competitions, breath, self-hearing, echoic memory, turn-taking reflex | spontaneous babbling in bouts that you can shape with approval; answering and echoing |
| neuromodulators | dopamine, noradrenaline, acetylcholine, serotonin | learning, arousal, sleep oscillations, impulsivity |
| ventral striatum | TD critic | dopamine = reward prediction error |
| cerebellum | forward model | surprise, curiosity, telling its own voice from yours |
| hippocampus | episodic store + prioritised replay | consolidation during slow-wave sleep |
| sleep | two-process model (sleep pressure x circadian clock), NREM/REM cycles | a day it lives through (default day = 20 min) |

Nothing is trained by back-propagation, and nothing is pre-trained.  Every
weight change is local: spike timing, the current weight, and a broadcast
neuromodulator.  See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the
full design and the neuroscience behind each piece.

## Results

Every number here was produced by the shipped code: two seeds for each
learning experiment, one simulated day for the first.  The first four take
roughly 1.5 hours on the 2-core development VM, the tutoring comparison
about 2.5 hours.  `anima experiment <name>` reruns any of them, and the raw
numbers are in [`results/`](results/README.md).  Nothing was tuned per seed.

### 1. Left completely alone, it still lives a day

One simulated day (20 min) in an empty world with no input of any kind (`anima experiment spontaneous`):

* it produced **173 babbled utterances (662 syllables)** and **1728 movements** without a single prompt, growing steadily hungrier and lonelier (there is no food and nobody comes);
* it slept 33% of the day, falling asleep after 13.2 min and waking at 19.8 min, cycling through NREM and REM;
* awake, its cortex fired densely at 4.1 Hz with a ~21 Hz rhythm; in slow-wave sleep it dropped to 0.3 Hz of sparse firing that waxes and wanes slowly (power below 2 Hz: 0.7% of the spectrum awake, 17% in NREM); REM brought back faster rhythms (~14 Hz) with the senses and muscles still switched off.

![a day alone](docs/figures/day.png)

### 2. It learns to forage

The smell→movement synapses start weak and random.  Every 2.5 minutes the food was held 90° left, straight ahead or 90° right and 40 decisions were counted (plasticity paused while probing; chance = 0.33).  Mildly food-restricted, as in animal learning studies (`anima experiment foraging`).

| | 0.0 min | 2.5 min | 5.0 min | 7.5 min | 10.0 min | food found in 10 min |
|---|---|---|---|---|---|---|
| plastic, seed 1 | 0.38 | 0.72 | 0.71 | 0.62 | 0.75 | 16 |
| plastic, seed 2 | 0.33 | 0.57 | 0.74 | 0.77 | 0.77 | 24 |
| frozen, seed 1 | 0.38 | 0.33 | 0.28 | 0.38 | 0.33 | 3 |
| frozen, seed 2 | 0.33 | 0.28 | 0.31 | 0.28 | 0.31 | 7 |

With plasticity the creature ends at **0.76** correct (from 0.35); the same brain with plasticity frozen stays at 0.32.  Food found: 20 vs 5.

![foraging](docs/figures/foraging.png)

### 3. You can shape what it babbles by smiling at it

A scripted caregiver smiles (non-verbal `:)`, so hearing plays no part) half a second after any syllable starting with **m**, from minute 2 on.  Controls: the same number of smiles at random moments (*yoked*), and plasticity frozen (`anima experiment babble`).  Share of syllables starting with *m*:

| | 0–2 min | 2–4 min | 4–6 min | 6–8 min | 8–10 min | smiles |
|---|---|---|---|---|---|---|
| contingent, seed 1 | 0.07 | 0.20 | 0.22 | 0.24 | 0.60 | 61 |
| contingent, seed 2 | 0.08 | 0.08 | 0.14 | 0.37 | 0.47 | 56 |
| yoked, seed 1 | 0.07 | 0.21 | 0.07 | 0.17 | 0.07 | 67 |
| yoked, seed 2 | 0.08 | 0.04 | 0.05 | 0.12 | 0.04 | 65 |
| frozen, seed 1 | 0.02 | 0.08 | 0.09 | 0.16 | 0.13 | 30 |
| frozen, seed 2 | 0.10 | 0.11 | 0.06 | 0.10 | 0.12 | 28 |

Contingent smiles raise *m* from 0.07 to **0.53**; random smiles end at 0.06 and a frozen brain at 0.13 (smiles still change its drives, which nudges the competition a little, but nothing is learned).  It is the *contingency* that teaches it, as with human infants (Goldstein et al. 2003).

![babble shaping](docs/figures/babble.png)

### 4. It learns to echo you by listening to itself

Five minutes of babbling alone, hearing its own voice, then a partner says random syllable pairs ("dodo", "mimi", ...) every 4–8 s (`anima experiment imitation`).

| | hearing→speaking map, consonants (diagonal / off-diagonal) | vowels | answered | echoes heard consonant (chance) | echoes heard vowel (chance) |
|---|---|---|---|---|---|
| mirror learning on, seed 1 | 0.98 → 2.97 | 0.98 → 3.49 | 48/48 | 0.17 (0.08) | 0.29 (0.20) |
| mirror learning on, seed 2 | 1.00 → 3.06 | 0.98 → 3.73 | 48/48 | 0.25 (0.08) | 0.27 (0.20) |
| mirror learning frozen, seed 1 | 0.98 → 0.98 | 0.98 → 0.98 | 48/48 | 0.06 (0.08) | 0.25 (0.20) |
| mirror learning frozen, seed 2 | 1.00 → 1.00 | 0.98 → 0.98 | 48/48 | 0.08 (0.09) | 0.29 (0.20) |

Babbling alone is enough to learn which sound each articulation makes (the map becomes strongly diagonal).  With that map it starts its answer with the consonant it just heard **20 times in 96**, against ~8 expected by chance (binomial p ≈ 1e-04); with mirror learning frozen it stays at chance.  Vowel echoes were not distinguishable from chance in either condition.  It answered every utterance, thanks to the turn-taking reflex.

![imitation](docs/figures/imitation.png)

### 5. What a teacher adds

Two creatures per seed, with the same brain and world seeds.  One is raised
by the tutor through the four lessons — by the **scripted** teacher, whose
decisions go through exactly the same tools and validation as an LLM's (no
real model could be run here; see the honest limits).  The other lives
alone for the same length of time in the same world: natural food, nobody
to talk to.  It keeps its plasticity and learns whatever its life teaches
it.  Then both live on alone until 9 am the next morning and sit the same
exam, with plasticity off (`anima experiment tutoring`, about 2.5 hours on
the 2-core VM):

| | lived | finds food (chance 0.33) | share of syllables starting with *m* | *m* share hungry / sated | copies the consonant it hears (chance) |
|---|---|---|---|---|---|
| tutored, seed 1 | 62 min | 0.81 | **0.49** | 0.62 / 0.47 | **6 of 29** (1.5 expected, p = 0.003) |
| tutored, seed 2 | 53 min | 0.73 | **0.72** | 0.79 / 0.75 | 2 of 30 (0.75 expected, p = 0.17) |
| alone, seed 1 | 62 min | 0.91 | 0.05 | 0.00 / 0.01 | 3 of 30 (1.8 expected, p = 0.26) |
| alone, seed 2 | 53 min | 0.94 | 0.12 | 0.09 / 0.04 | 4 of 30 (2.0 expected, p = 0.13) |

Both tutored creatures passed all four lessons at the first attempt:
foraging after 8 minutes (exams 0.75 then 0.77, and 0.65 then 0.77),
copying after 21–22 minutes, the target sound after 42–45 minutes, and
living alone — ten awake minutes with no reflexes and no help, energy 0.88
and 0.92 on average and never below 0.3 — after 53–62 minutes, two nights
included.  The teacher put food down 10 and 9 times in all.

What the teacher added is the voice: half to three quarters of the tutored
creatures' syllables start with *m*, against 5% and 12% for the creatures
that lived alone (about 1 in 12 by chance).  Copying is the weak result:
only tutored seed 1 copied the consonant it heard clearly more often than
chance.  Teaching did **not** make better foragers in the long run: a
creature living alone learns to forage from food by itself, and both did
better in the exam (0.91 and 0.94 against 0.81 and 0.73), probably because
the smiles that shape a tutored creature's voice also reach, through global
dopamine, the synapses that steer it.  The foraging lesson is scaffolding
for the first minutes, not a lasting edge.  And the trained *m* is not a
request yet: the tutored creatures say it about as often when sated as when
hungry (the optional *ask* lesson tries to change that — see the honest
limits).

![tutoring](docs/figures/tutoring.png)

## Build your own brain

A brain is data.  Dump the default and edit it:

```bash
anima spec --out mybrain.toml          # populations, projections, rules, system settings
anima live --spec mybrain.toml --new
```

```toml
[[population]]
name = "cortex.E"
size = 800
kind = "RS"            # Izhikevich cell type: RS IB CH FS LTS TC RZ
noise = 4.0            # spontaneous background drive
nmda_ratio = 0.35
target_rate = 4.0      # homeostatic set point (Hz)
role = "cortex"        # how neuromodulators act on it

[[projection]]
pre = "sense.bearing"
post = "motor"
p = 0.5
weight = 0.008
fixed_indegree = true
[projection.rule]
type = "reward_stdp"   # or "stdp", "hebbian"
tau_e = 300.0
normalize = "both"     # heterosynaptic competition for a fixed synaptic budget

[brain.drives]
day_length_s = 1200.0
```

Or in Python, with your own world (five methods):

```python
from anima import Agent, BrainBuilder, BodyConfig, World

b = BrainBuilder("my creature", seed=1)
b.input("eyes", 40, channels=2, labels=["left", "right"])
b.population("cortex.E", 200, kind="RS", noise=4.0, role="cortex")
b.population("cortex.I", 50, kind="FS", noise=3.5, role="cortex")
b.output("motor", 40, channels=2, labels=["veer left", "veer right"], noise=3.5)
b.connect("eyes", "motor", p=0.5, weight=0.008,
          rule={"type": "reward_stdp", "a_minus": 0.0, "normalize": "both"})
...
agent = Agent(b.build(), MyWorld(), BodyConfig(sensor_ports={"eyes": "eyes"}))
agent.on("action", print)
agent.start()          # lives in a background thread, in real time
agent.hear("hello")    # from any thread, whenever you like
```

`examples/03_custom_brain.py` is a complete phototaxis creature built this
way.  The lower level (`anima.core.Network`) is a general spiking simulator
you can use on its own.

## Project layout

```
anima/
  core/        spiking simulator: neurons, network, plasticity rules, monitors
  brain/       drives, neuromodulators, sleep, hippocampus, cerebellum, critic,
               Brain orchestrator, spec builder, presets
  body/        physiology, senses, motor decoding, voice and hearing
  world/       World interface, foraging arena, empty world
  runtime/     the never-blocking life loop, save/load
  tutor/       teaching: Ollama client, reflex rules, skill meters, curriculum,
               scripted and LLM teachers, the tutor loop
  ui/          terminal chat, curses dashboard
  experiments/ the experiments below, runnable with `anima experiment ...`
tests/         pytest suite (64 tests, about a minute), incl. a mock Ollama server
examples/      01 alone · 02 talk from code · 03 build your own brain + world · 04 LLM tutor
configs/       infant.toml — the default brain as an editable spec
docs/          ARCHITECTURE.md (how it works), NOTES.md (design notes, lessons learned), figures
results/       the raw numbers behind every table and figure, with a guide
```

## Honest limits

* **Scale.** ~1,700 neurons. A human brain has ~86 billion. This is closer to
  a very simple animal than to a person, and it behaves like one.
* **Language.** It perceives 17 phonemes and produces consonant-vowel
  syllables. It can be shaped and can echo; it does not understand words.
  Getting from babbling to meaning is the open problem this framework is
  meant to help explore, not something it has solved.
* **Abstractions.** Spiking: cortex, senses, interoception, motor, voice.
  Rate-level/abstract: neuromodulator levels, drives, the critic, the
  cerebellar forward model and the hippocampal store (their outputs act on
  the spiking network). Each is a few hundred lines and can be swapped for a
  spiking version.
* **Speed.** Pure NumPy, one core. On the development VM a 2,000-neuron
  cortex runs at about real time and a 4,000-neuron one at half real time;
  the dense weight matrix grows as N².
* **Reproducibility.** Runs are deterministic for a given seed and sequence
  of inputs; anything typed live makes a run unique, as it should.
* **The LLM teacher has not met a real model here.** The development VM
  could not download models, so the Ollama side (client, prompt, structured
  outputs, fallbacks, the never-waiting loop) is tested against an
  in-process mock of the Ollama API, and the teaching results in section 5
  use the scripted teacher, which goes through exactly the same tools and
  validation. How good a real model's strategy is depends on the model; the
  reflexes, the validation and the measured graduation are what keep a weak
  or erratic one from doing harm. `anima tutor-check` tells you in a few
  seconds whether yours produces usable decisions.
* **A narrow curriculum, and asking is not solved.** Four lessons about food
  and sounds. The optional *ask* lesson (`--with-ask`: no food grows, calls
  made when hungry are answered with food, calls made when full with a
  frown) shows a real limit of this brain. The creature learns to call "m"
  for food, but after 20 minutes of the lesson its synapses onto the "m"
  articulator from hunger, boredom and loneliness were all about equally
  strong (1.05, 1.09 and 1.05 in seed 1): dopamine is global, and a sound it
  makes every few seconds gets credited for any reward nearby. Frowning at
  calls made when full did not change that. In six exams during those runs
  it said "m" in 47–77% of its syllables when hungry and 55–82% when full.
  Making the call hunger-specific probably needs a stronger interoceptive
  pathway into the voice, or a more specific credit signal — an open
  problem, not a solved one.

## References

The design leans on: Izhikevich 2003 & 2007; Bi & Poo 1998; Turrigiano 1998;
Schultz, Dayan & Montague 1997; Doya 2002; Aston-Jones & Cohen 2005;
Hasselmo 2006; Borbély 1982; Tononi & Cirelli 2014; Wilson & McNaughton
1994; Gomperts et al. 2015; Wolpert, Miall & Kawato 1998; Keramati & Gutkin
2014; Goldstein, King & West 2003; Heyes 2010.  Full list in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

MIT licensed.
