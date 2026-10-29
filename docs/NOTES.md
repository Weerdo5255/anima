# anima — brain-like agent framework: design notes and results

*Built 28 Oct 2026 (Python ≥3.10, NumPy only). v0.2 adds a local-LLM
(Ollama) tutor, v0.3 programs it can use, a thought log and a bridge to
other programs — see the last two sections.*

## Goal
A local framework that builds **not an LLM but a brain-like neural network**,
and an agent that **does not wait for prompts**: it runs continuously and
produces behaviour from its own dynamics.

## What was built
* **Spiking core** (`anima/core`): Izhikevich neurons; conductance synapses
  (AMPA, NMDA with Mg²⁺ block, GABA-A) integrated with exponential Euler;
  Dale's law; event-driven propagation through one global weight matrix
  (each projection is a block view); homeostasis (intrinsic plasticity,
  synaptic scaling); a slow K⁺ adaptation current. Local rules only: STDP
  (sparse path for bursty recurrent cortex), dopamine-gated reward-STDP with
  eligibility traces and optional heterosynaptic budget normalisation,
  symmetric Hebbian with normalisation.
* **Brain systems** (`anima/brain`): hypothalamic drives (hunger, fatigue,
  boredom, loneliness), neuromodulators after Doya 2002 (DA, NE, ACh, 5-HT),
  TD critic (ventral striatum) → phasic dopamine, cerebellar forward model
  (surprise/novelty), hippocampal episodic store with reward-weighted replay
  in NREM, two-process sleep model with NREM/REM cycles, declarative specs
  (TOML/JSON) + fluent `BrainBuilder`.
* **Default creature** (`infant_spec`): ~1,700 neurons, 15 populations, 38
  projections; senses (8-channel smell bearing, whiskers, taste, 17-phoneme
  hearing), 3 motor channels, a babbling vocal system (gate + 12 consonants ×
  5 vowels, breath, self-hearing, echoic memory, innate turn-taking reflex).
* **Runtime** (`anima/runtime`): 10 ms life loop that never blocks; inbox is
  drained if non-empty; thread-safe `hear()`; events; real-time pacing; exact
  save/load (spike-for-spike continuation); `anima chat`, `anima live`
  (curses dashboard), `anima run`, `anima experiment`, `anima spec`.

## Design lessons (each was a real failure first)
1. **Plain Euler + conductance synapses is unstable** under strong
   inhibition (v overshoots E_GABA → neurons fire every ms). Fixed with
   exponential Euler for the synaptic part.
2. **Random connectivity biases competitions** (one motor channel never
   won). Fixed with fixed in-degree wiring.
3. **Commitment feedback causes perseveration** → added a refractory reset
   after each movement.
4. **Credit assignment needs the right timing**: movement consequences must
   arrive when the movement *completes*, otherwise dopamine credits the
   post-movement state.
5. **Only-the-chosen-action learning → one action takes over** (rich get
   richer). Fixed with heterosynaptic competition for a fixed synaptic
   budget (`normalize="both"` / `"pre"`).
6. **Global average-reward baseline is too noisy** (after food, everything
   looks bad for ~8 s). Fixed with a learned state-value critic: dopamine =
   TD error.
7. **Hunger gates smell**, so policy probes must be run at fixed hunger.
8. **Slow-wave sleep needs slow adaptation + reduced tonic drive** to get
   slow-wave-like dynamics; the Izhikevich recovery variable is too fast.
9. **Sleep paralysis must act downstream** (in the decoders), or replay
   cannot reactivate motor populations — replay was silently inert until
   this was fixed (now covered by a test).
10. **Fast homeostasis must pause in sleep.** Synaptic scaling "corrected"
   the low sleep firing rate by driving recurrent weights to their maximum
   within ~20 s (near-silent NREM with giant bursts, over-excitable cortex
   on waking). Intrinsic plasticity similarly tunes cortex to waking inputs
   (movement feedback, self-hearing) that vanish in sleep, so the sleep-onset
   drop in drive must be gentle (noise × 0.8 plus slow adaptation). Final
   NREM is sparse (~0.3 Hz), slowly waxing/waning firing — a simplified
   slow-wave state, not crisp ~1 Hz up/down states (an open item).

## Results (final code; 2 seeds per learning experiment, 1 simulated day)
* **Autonomy (1 simulated day, empty world, zero input):** 173 utterances
  (662 syllables), 1,728 movements; slept 33% (23:00→07:00 on its clock)
  with NREM/REM cycles; cortex ~4.1 Hz with a ~21 Hz rhythm awake, ~0.3 Hz
  sparse slowly fluctuating firing in NREM (power <2 Hz: 0.7% → 17%).
* **Foraging (dopamine-gated sensorimotor learning):** correct move with food
  left/ahead/right rose from 0.35 to 0.76 in 10 min (chance 0.33); frozen
  control 0.32. Food found 20 vs 5 (mean of 2 seeds).
* **Babble shaping (contingent smiles after "m" syllables):** share of
  "m" 0.07 → 0.53; yoked random smiles 0.06; frozen 0.13.
* **Imitation:** 5 min of babbling alone made the hearing→speaking map
  diagonal (consonants 1.0 → 3.0×, vowels → 3.6×); consonant echo 20/96 vs
  ~8 expected (p ≈ 1e-4); frozen mirror at chance; vowel echo not
  significant. Turn-taking reflex: answered 48/48.
* **Replay:** a rewarded episode replayed in NREM strengthens the matching
  sensorimotor synapses (test; inert before the atonia fix).
* **Speed:** default creature 1.7–1.9× real time on a 2-core VM;
  2,000-neuron cortex ≈ real time; 4,000 ≈ 0.5×.

## Honest limits
~1,700 neurons; infant-level phonology (no word meaning); several systems
are rate-level abstractions acting on the spiking network (neuromodulator
levels, drives, critic, cerebellum, hippocampal store); pure NumPy, dense
N×N weight matrix (fine to ~5k neurons).

## Natural next steps
* Sparse/GPU backend (CuPy drop-in) to scale cortex 10–100×.
* Topographic multi-area cortex; spiking critic and hippocampus.
* Vision + object permanence → grounding words (associating heard syllable
  patterns with objects), the step from babbling to meaning.
* Behavioural test of sleep consolidation (policy before/after a night with
  replay on vs off).
* Crisper slow-wave sleep (thalamocortical up/down states).
* Run the tutoring comparison with a real local LLM as the teacher
  (`anima experiment tutoring` accepts an `OllamaTeacher`).
* A hunger-specific "ask" call (stronger interoceptive input to the voice or
  a more specific credit signal).

## v0.2 — a local LLM (Ollama) as the creature's teacher

*Added 28 Oct 2026. Raw numbers: [`results/`](../results/README.md).*

**Request:** let a local LLM running on Ollama act as an agent that trains the
brain-like network, slowly building up its skills until it can act as an
independent agent.

### Design
* **The LLM is the teacher, not the brain.** It never writes weights. Every
  ~8 s of creature time it reads a compact JSON report (drives, energy, what
  the creature said/did/heard, passive skill measures, exam results, lesson
  goal / progress / tips / allowed rules) and answers `{"thought", "actions"}`
  with ≤3 tool calls: `wait, say, praise, scold, give_food(ahead|left|right|near),
  set_rule, clear_rule, switch_lesson (review), note`.
* **Two-speed teaching.** Eligibility traces are ~0.3 s (motor) / 2 s
  (voice) and an LLM takes seconds, so contingent rewards come from *reflex
  rules* running in the creature's 10 ms tick hooks (`guide_food`,
  `reward_echo`, `reward_syllable`, `echo_back`, `feed_on_call`); the LLM
  switches them and chooses what/when to say and feed (the clicker-training
  split).
* **Never waits.** Creature thread and tutor thread are separate; a slow
  model just means fewer decisions (tested with a mock taking 1 s per reply:
  the creature kept ≥70% of real time). After 3 failures in a row the model
  rests 60 s while the scripted teacher covers.
* **Guard rails:** `validate()` drops unknown tools/rules, refuses rules
  outside the lesson and free food where the lesson forbids it (unless
  starving), blocks skipping lessons, sanitises speech; graduation is
  measured, never the model's call.
* **Ollama client:** stdlib only; `/api/version`, `/api/tags`, `/api/chat`
  with `format` = JSON schema (falls back to `"json"` on old servers),
  `keep_alive`, optional `think`; defensive JSON extraction; explicit errors
  ("ollama serve / --host", "ollama pull <model>" or a similar installed tag).
* **Curriculum (4 lessons + optional ask):** forage (2 exams in a row ≥0.60,
  ≥1 meal/min; kept keen energy ≤0.5; guide_food) → echo (copy rate ≥
  max(0.15, 1.8×chance), energy ≤0.6) → babble ("m" share ≥0.30, energy ≤0.6)
  → independence (no reflexes, no help; energy mean ≥0.35, min ≥0.08 over 8
  awake min). Tests must keep passing for 60 s; each lesson starts its
  measures afresh; lessons not passed after 45 awake min are left "not
  mastered"; spot checks + reviews. `--with-ask` adds *ask for food* (no wild
  food; feed_on_call answers hungry calls, frowns at full ones; exam on
  hungry vs sated "m" share).
* **Exams on clones:** `Agent.clone()` (deep snapshot on the agent thread,
  exact continuation); exams run with plasticity off, so testing never
  interrupts or trains the creature.
* **Persistence:** `<brain>.tutor.json` (lessons, meters, exams, rules,
  journal, milestones). Training conditions are owned by the tutor and *not*
  saved in the brain file, so `anima live` after training has a normal world.
* **CLI:** `anima train [--teacher ollama|scripted] [--model] [--host]
  [--think auto|on|off] [--minutes] [--fast] [--think-every] [--target]
  [--with-ask] [--quiet] [--no-fallback]`, `anima tutor-check`,
  `anima experiment tutoring`; `/feed` in chat and dashboard;
  `examples/04_llm_tutor.py`. 64 tests incl. a mock Ollama server.

### Lessons learned (each was a real failure first)
1. **A sated creature doesn't learn to forage** (hunger gates smell and scales
   the approach reward) → keep it keen during training, as with lab animals.
2. **Teacher food that regrows accumulates** → gifts are eaten once.
3. **Passive foraging scores are useless in a multi-food arena** (smell
   resultant length median 0.2–0.3): passive score ≈ chance while the probe
   said 0.70–0.78 → exams on clones.
4. **Skills interfere through global dopamine**: social rewards blurred the
   foraging map while sated (probe 0.60 → 0.50) → stay a little hungry,
   spot checks, reviews.
5. **Order matters**: with "make the sound" before "copy me", one dominant
   sound blurred the hearing→speaking map (echo rate 0.05–0.10 instead of
   ~0.2) → copying first.
6. **Measurements must start fresh per lesson** (old trials inflated the echo
   rate) and **tests must hold** (single lucky passes) → fresh windows,
   60 s hold, two exams in a row.
7. **Asking is not solved**: hunger→"m", boredom→"m", loneliness→"m"
   synapses all ended ≈1.05 after 20 min of the ask lesson (global dopamine,
   always-on drives); frowns at full calls didn't help (hungry 47–77% vs
   sated 55–82% "m" in six exams) → optional, documented as open.
8. **Training conditions must not persist in the brain file.**

### Results (scripted teacher, `anima experiment tutoring`)
Same brain/world seeds; tutored (scripted teacher, same tools/validation as
an LLM) vs living alone for the same time (62 / 53 min); both examined at
9 am the next morning with plasticity off.

| | finds food (chance .33) | "m" share | "m" hungry/sated | copies consonant (chance) |
|---|---|---|---|---|
| tutored s1 | 0.81 | **0.49** | 0.62 / 0.47 | **6/29** (1.5 exp., p=.003) |
| tutored s2 | 0.73 | **0.72** | 0.79 / 0.75 | 2/30 (0.75 exp., p=.17) |
| alone s1 | 0.91 | 0.05 | 0.00 / 0.01 | 3/30 (1.8 exp., p=.26) |
| alone s2 | 0.94 | 0.12 | 0.09 / 0.04 | 4/30 (2.0 exp., p=.13) |

* Both tutored creatures passed all 4 lessons first time: forage 8 min,
  echo 21–22, babble 42–45, independence 53–62 min (nights included); 9–10
  food placements in total; independence energy mean 0.88/0.92, min ≥0.30.
* The teacher's lasting contribution is the **voice** (target sound 0.49–0.72
  vs 0.05–0.12). Copying: only tutored s1 clearly above chance.
* **Foraging needs no teacher** in the long run: the lone creatures learned it
  from food and scored higher (0.91/0.94 vs 0.81/0.73), probably because
  voice-shaping smiles also reach the motor synapses via global dopamine.
* The trained "m" is not hunger-specific (not a request).
* Fresh-venv install of the zip: 64 tests pass; CLI checked against a mock
  Ollama (tutor-check, train, resume, missing-server message).

### Honest limits (v0.2)
* No real LLM was available in the build environment (model downloads
  blocked): the Ollama path is tested against an in-process mock of the API;
  the results use the scripted teacher (same tools and validation).
* The teacher's lasting contribution is the voice; foraging is learned from
  food alone. "Asking" is not hunger-specific with this brain.

## v0.3 — programs it can use, a log of its thoughts, a bridge to other programs

*Added 29 Oct 2026. Raw numbers: [`results/programs/`](../results/README.md).*

**Request:** let the creature interact with external programs and keep a log
of its own thoughts, towards a human-like thinking agent on par with LLM
agents.  Chosen approach (asked first): **spiking brain only** — no LLM
inside the creature; it learns by reward which programs to use, and the
thought log is read from its own activity.  Being on par with LLM agents is
not claimed (README: "How far is this from an LLM agent?").

### Design
* **Programs are bound to acts it can produce** (`anima/tools`): a syllable
  (`say = "la"` exact, `say = "b"` prefix) or a hand (`hand = k`).  It cannot
  type or compose arguments.  Calls carry its state (drives, energy, clock,
  stage) as JSON; answers are text (heard) or JSON
  `{"say", "reward" (clipped to ±1), "food", "ok"}`.  Kinds: `command`
  (argv list, never a shell, `{python}`/`{hunger}`… placeholders, stdin =
  the call), `url` (POST to a local service), `function`, `remote` (answered
  through the bridge).  `ToolBelt` runs them in worker threads and applies
  answers on the creature thread at the next tick; cooldowns, timeouts,
  output caps, a global calls-per-minute limit; inline deterministic
  `FunctionTool`s for experiments.  `anima tools FILE --try NAME`.
* **Hands** (`infant_spec(hands=4)`, opt-in): `hands` (15/channel) +
  `hands.inh` + `sense.hands`; cortex.E→hands and intero→hands reward-STDP
  (tau_e 2 s, lr 0.06, pre-normalised), channel-recurrent excitation, shared
  inhibition; decoder threshold 0.42, margin 1.3, 250 ms reach, 5 s
  refractory (≈5–6 reaches/min untrained).  `hands=0` leaves the default
  brain bit-identical (weights checked), so published results reproduce.
* **Thought log** (`anima/mind`): readouts every 100 ms (drive bands; smell
  population vector; synaptic drive into the motor and hand channels; vocal
  channels while listening; phasic DA with its cause; cerebellar surprise;
  hippocampal storage/replay; reflections over recent episodes; events).
  Readouts compete for a global workspace: the most salient candidate gets
  in, then a 2.5 s blink (6 s asleep); habituation per family (τ 2–20 s);
  losers fade after 2 s ("unconscious").  Always written: state changes,
  speech heard, eating, programs, reflections.  JSONL with rotation,
  `anima thoughts [--follow --kinds --json]`, `/thoughts` and `/mind` in the
  chat, thoughts in the dashboard, `its_thoughts` in the LLM teacher's
  report.
* **Bridge** (`anima/bridge`): `anima serve` = real-time creature + JSON
  HTTP on 127.0.0.1 (token required elsewhere): `GET / /state /thoughts
  /events (long poll) /stream (SSE) /tools /pending`, `POST /hear /reward
  /feed /tool_result`; `BridgeClient` (stdlib) with `follow()` and
  `serve({"bell": fn})`; `examples/05_external_program.py`.

### Lessons learned (each was a real failure first)
1. **The first evaluation readout said "worse than I expected (−0.33)"
   every few seconds**: it subtracted tonic dopamine from the phasic signal,
   which is already a separate, signed variable.  Now it reads phasic DA and
   names its cause (the reward of the last 0.5 s) or says "things are
   looking up/worse" when only the critic's forecast moved (a TD error).
2. **Logging everything is unreadable** — 127 lines a minute — and a token
   bucket drops whatever comes when it is empty, not the least important.
   A workspace competition with a blink and habituation gives ~30 lines a
   minute, keeps the salient ones, and records each line's salience.
3. **Intentions read from the motor activity trend only restated what it
   was doing**: in a pilot every logged intention named the movement it had
   just made, and the next movement matched 0.35 of the time (chance 0.33).
   Decisions are made by fast (30 ms) fluctuations and the decoder's
   commitment current dominates the activity trace.  The *synaptic drive*
   into each channel (presynaptic rates × weights, own pools excluded) is
   what biases the race; after foraging practice it predicted the majority
   movement of the next 3 s 0.51–0.60 of the time against 0.44 for the most
   common class and 0.41–0.42 for "same as the last 3 s".  For the hands the
   same readout follows learning closely (results 7).
4. **Rewarding the hand it already preferred made learning look trivial**
   (0.55 → 1.00 in five minutes).  The experiment pays the hand it used
   least in the first two minutes, and the frozen control is identical
   until that moment.
5. **Heard answers are a confound**: the creature echoes what it hears, so
   a program saying "nyam" would bias its babbling.  Program answers in the
   experiments are silent (`speak = False`).
6. **A free-running creature rings "b" programs by itself** (it babbles),
   which broke bridge tests → test programs are bound to "qx", a syllable it
   cannot produce.
7. **A call and its answer as two lines doubled the program noise** → one
   line when the answer comes within a second.

### Results (`anima experiment programs --workers 2`)
* **Hands** (paying program = the hand used least in the first 2 min, +0.6
  from minute 2, no sound; 2-hour day so it stays awake): share of reaches
  running it 0.00/0.18 → **1.00/1.00** (minutes 10–20) and reach rate
  5→11/min (the 5 s refractory allows ~11.4); frozen at minute 2:
  0.28/0.30 (chance 0.25).  The log's first "I feel like using program0"
  came 38/63 s after paying began, before the habit (3 of the last 6
  reaches) at 42/103 s; frozen creatures never logged an urge (drive margin
  never ≥ 0.06).  A first run with the default 20-minute day slept through
  minutes 13–19 (same outcome: 0.09/0.17 → 1.00, frozen 0.29/0.29); it was
  rerun with the long day for clean bins.  Motor inclinations in the four
  hands runs came true 0.30–0.54 (chance 0.31–0.42).
* **Voice-bound programs** ("b" pays +0.5 from minute 2, "d" nothing; well
  fed, void world): "b" share 0.14/0.07 → **0.41/0.40** (last 4 min) while
  "d" stayed 0.05–0.06; frozen at minute 2: "b" 0.06/0.07.
* **Thought log checks** (after 10 min of foraging practice, then spoken to
  every ~12 s for 8 min): inclinations came true 0.55/0.46 (n 29/26; chance
  0.35/0.34, persistence 0.41/0.42); the first syllable it silently mouthed
  matched the heard consonant 14/26 and 11/21 (chance 0.05; any syllable
  0.69/0.71 vs 0.08/0.09) — far above its overt echo rate (0.17–0.25);
  30 lines a minute, ~320 candidates per 8 minutes faded unwritten.
* 101 tests; fresh-venv install of the zip checked (tests, `anima tools`,
  `run --tools --thoughts`, `thoughts`, `serve` with curl, example 05).

### Honest limits (v0.3)
* Programs are buttons: it learns which of a few bound acts pays, from
  consequences; it cannot read answers, pass arguments or generalise to new
  programs.  Nothing here approaches an LLM agent's tool use.
* The thought log is a readout in our words; the predictive readouts are
  modest (inclinations ~0.5 vs ~0.35 chance) and the mouthing readout is
  right about half the time.
