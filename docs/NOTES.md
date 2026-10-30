# anima — brain-like agent framework: design notes and results

*Built 28 Oct 2026 (Python ≥3.10, NumPy only). v0.2 adds a local-LLM
(Ollama) tutor, v0.3 programs it can use, a thought log and a bridge to
other programs, v0.4 a colony — creatures that compete, cooperate and are
selected by a genetic algorithm — v0.5 a GPU engine, and v0.6 pain and
pleasure, a reward for repeating the teacher's words and a better LLM
teacher; see the last five sections.*

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

## v0.4 — a colony: competing, cooperating, and a genetic algorithm

*Added 29 Oct 2026. Raw numbers: [`results/colony/`](../results/README.md).*

**Request:** let creatures compete against one another, or collaborate if
they find it appropriate; at the end of a day, if one shows a noticeable
performance gain, use a genetic algorithm so that it and variations of it
run the next day; a flag to delete the old creatures.

### Design
* **One arena, many creatures** (`SharedArena`, per-creature `World` views):
  shared food (first come; rivals within 2.5 are told they *missed*), feasts
  that need ≥2 creatures within 1.6 at once (0.4 energy each vs 0.25 for a
  meal; stag hunt), hearing within 7 units (loudness scales the social
  contact), and an opt-in `sense.kin` bearing code (80 neurons → cortex, and
  → motor with reward-STDP).  Food and feasts nobody eats wither after
  60–120 s and grow back elsewhere.  Competition and cooperation are not
  choices the code makes for them; they follow from what each brain learns.
* **Rounds** (`anima/evolve/hosting.py`): every creature senses the arena as
  the round began, lives 10 ms, reports completed movements; the colony
  applies them in a shuffled order and delivers consequences next round
  (`Body.receive`, incl. the approach reward).  This makes worker processes
  exact: one process vs two give identical lives (tested).
* **Fitness** = food eaten in the day (energy).  **Stands out** = best ≥
  1.25 × median(others) and ≥ median + 0.5.  **Next day** = champion
  (elitism) + mutated variations; `inherit="brain"` (default: copy of the
  champion's state, new random streams, 5% multiplicative noise on plastic
  non-recurrent synapses — exploit-and-explore as in PBT) or `"genes"`
  (fresh brain from the mutated spec, aligned to the colony's time of day);
  optional uniform crossover with the runner-up; children are born within 2
  units of the parent; replaced creatures archived (`retired/`) or deleted
  (`delete_old`).  `selection="random"` (a control for experiments): same
  turnover rule, random parent.  11 genes, log-normal mutation
  with bounds, stored in the creature's state (spec / brain settings / body
  settings), so they survive save/load.
* **CLI** `anima arena` (`--days --new --from --population --cortex --day
  --margin --min-gain --mutation --mutation-rate --inherit --crossover
  --delete-old --workers --watch --thoughts --status`), Ctrl-C/SIGTERM save
  cleanly; `anima experiment colony` (`--policies` to run it in parts,
  `--collect` to put the parts together); `examples/06_colony.py`.
* Thought log: other creatures by name ("Bo says …", "Bo got to the food
  first", "a feast, shared with Bo", "someone nearby, to my left", "a feast
  — but it is too big to eat alone"); other creatures' voices compete for
  the workspace instead of always getting in.

### Lessons learned
1. **Ending a day by a clock crossing could fire twice** (found in a test):
   a fresh child's clock starts exactly at dawn, so the next tick looked like
   another dawn.  Days now also need half a day to have passed.
2. **The first colony was sequential**: each creature saw the arena after
   the creatures before it had moved, so worker processes would have changed
   the results.  Synchronous rounds with consequences delivered next round
   make one and N processes identical (a test compares them).
3. **Spawned workers re-import `__main__`** and failed when the colony was
   started from stdin → workers are forked on Linux; elsewhere scripts need
   the usual `if __name__ == "__main__":` guard (`ANIMA_MP_START` overrides).
4. **Every voice was an "important" thought**: with neighbours talking,
   "someone says" filled 40% of a creature's log (65 of 164 lines in a test
   colony).  Other creatures' voices now compete for the workspace.
5. Two things done by design rather than learned the hard way: the order in
   which movements are applied is shuffled every round (otherwise the first
   creature would always win a contested meal), and a copied brain gets new
   random streams (otherwise it would replay its parent's noise).
6. **Food that grew back only when eaten was a trap.**  In the first 4-day
   run, the colony that was never replaced starved: mean food per creature
   2.54 → 1.07 → 0.64 on days 1–3, mean energy ~0.05 on day 3.  Its brains
   were not broken.  Put alone in the arena, its day-3 creatures still found
   food (2–4 meals in 2 minutes each, against 2–5 for the same creatures a
   day or two earlier), and their synapses steered *more* strongly toward
   food than a day or two earlier (steering index,
   `anima.experiments.colony.steering`, mean 0.12 → 0.30).  Their steering toward each other had grown too, in 3
   of 4 (mean 0.08 → 0.12) — probably because being near others means
   hearing them, hearing relieves loneliness and relief is rewarding.  They
   moved as a tight herd in the middle (mean distance 2.2; near another 70%
   of their waking time, against 22% for four creatures placed at random), all
   four pieces of food ended up near the edges where the herd never went,
   and since food only grew back after being eaten, nothing new grew.  In
   60 s from that state the herd ate nothing; with their sense of each other
   silenced it spread out (mean distance 5.2) and ate 2 meals and a feast;
   with the food moved to random spots, 2 meals; scattered to random spots
   themselves, they re-formed the herd within seconds (mean distance 2.9,
   1 meal).  The evolving colony herded as much (near another 66–80% of the
   time) but kept eating.  Fix: food nobody eats withers and grows back
   elsewhere.
7. **That first comparison was also confounded.**  The children of a new
   generation were put at random spots — which by itself breaks up a herd
   and stumbles onto stranded food — and nothing separated selection from
   turnover as such.  Children are now born next to their parent, and the
   experiment has a third colony with the same turnover but a random parent.

### Results (`anima experiment colony`)
Run in three parts (`--policies evolving`, `random`, `static`; each
`--days 4 --seeds 1,2 --workers 2 --keep keep`, ~70 min on 2 cores) and put
together with `--collect keep`.  Per seed, three colonies of four young
creatures (600-neuron cortex, genes varied around the defaults), four
10-minute days, one and the same colony until the first evening with a
standout: *evolving* (best + 3 variations born next to it), *random parent*
(same rule, random parent), *static* (nobody replaced).

| food per creature, seed 1 / 2 | day 1 | day 2 | day 3 | day 4 |
|---|---|---|---|---|
| evolving | 2.40 / 2.74 | 2.83 / 1.83 | 2.55 / 2.75 | 2.67 / 2.23 |
| random parent | 2.40 / 2.74 | 2.83 / 1.83 | 1.77 / 2.75 | 2.09 / 2.42 |
| static | 2.40 / 2.74 | 2.83 / 1.83 | 2.27 / 2.75 | 2.85 / 3.01 |

* **First standouts:** seed 1 after day 2 (Bo 3.85 vs the others' median
  2.30; the random parent was Ada), seed 2 after day 3 (and the random draw
  picked the best as well, so evolving and random differ only by mutations
  and chance: 2.23 vs 2.42 on day 4).  On the three colony-days that
  differed, per creature: static 2.71, evolving 2.48, random 2.09.
* **Selection copies luck at this scale.**  In the static colonies a
  creature's food on one day did not predict the next (r = −0.05, 24
  pairs); the day's best out-ate the others the next day on 3 of 6 days.
  `anima.experiments.colony.repeatability` computes this; it is the check to
  run before trusting any selection scheme here.
* Seed 1's selected lineage steered least toward the others (kin steering
  0.03 on day 3, against 0.13 in the random parent's lineage) and ate more
  than the random lineage, like the first run's champion (0.006).  Over all
  56 creature-days, though, steering toward others only weakly predicted
  eating less (Spearman −0.23, p = 0.09, and the days are not independent).
* **Competition:** a rival within reach for 34–45% of meals on day 1,
  53–87% on day 4 — 77–87% in the evolving colonies (near-copies born side
  by side).  **Cooperation:** feasts were 11–42% of the food (1–6 per
  colony-day).  **Herding:** near another creature 57–70% of waking time on
  day 1 (22% for random placement), 79–84% (static) and 85–95% (colonies
  with turnover) on day 4.  13–24 pieces of food per colony-day withered
  uneaten, mostly at night.
* **Determinism:** the three colonies of a seed, run in separate processes
  and at different times, were identical meal for meal until they first
  differed.
* 128 tests; fresh-venv install of the zip checked (tests, `anima arena`
  with two worker processes and `--delete-old`, `--status`).

### Honest limits (v0.4)
* At four creatures and ten-minute days the genetic algorithm mostly
  copies the luckiest; selecting ability needs longer judging (longer days,
  larger colonies, or several days per verdict — not built).  Two seeds ×
  four days is a pilot.
* Inheritance is Lamarckian (a copy of the champion's synapses); children
  with fresh brains (`inherit="genes"`) were not evaluated.
* Nothing decides to compete or cooperate: they want the same food, and a
  feast is shared when two arrive together.  They learn to stay together,
  which raises competition; whether shared feasts make up for it was not
  measured.
* An ecological detail (food that never moved unless eaten) dominated the
  first result completely (lesson 6); conclusions here are about this arena.

## v0.5 — the spiking network on a GPU

*Added 29 Oct 2026. Raw numbers: [`results/gpu/`](../results/README.md).*

**Request:** restructure the program so that it can run on a local GPU
instead of the CPU, for speed.

### Design
* **One switch, everywhere:** `Network(device=...)`, `build_brain(...,
  device=)`, `Agent.create/load(device=)`, `Colony.create/load(device=)`,
  `--device` on every command that runs a creature, and `ANIMA_DEVICE` for
  the default.  `cpu` is the unchanged NumPy reference engine (all earlier
  results reproduce bit for bit); `auto`, `cuda[:k]`, `mps`, `torch:...`,
  `cupy:...` pick a GPU; `emulate` runs the device engine on the CPU.
* **A device engine written once** (`anima/core/engine.py`) against a small
  NumPy-style interface (`anima/core/device.py`) with three backends: CuPy
  (NVIDIA; AMD with ROCm builds), PyTorch (CUDA, ROCm, Apple `mps`, CPU) and
  NumPy (`emulate`).  With CuPy the per-neuron update is one fused
  elementwise CUDA kernel, verified against the array path at start-up and
  dropped (with a warning) if they disagree.
* **Only the spiking network moves.**  Neuron state, synapses, rules' state
  and noise live on the device; the slow systems, body, world and thought
  log stay on the host.  Per millisecond: at most one wait for the device
  (to learn how many neurons fired, for event-driven delivery); networks of
  ≤2,500 neurons skip it with one dense `[exc; inh]·fired @ W` product.  Per
  10 ms: the brain's per-neuron settings go up once, the readouts (activity
  traces, rates) come down once.
* **Plasticity per 10 ms window**, from recorded spike trains and traces,
  with small matrix products restricted to the neurons that fired (one wait
  per window): exact for RewardSTDP eligibility; Hebbian/STDP changes take
  effect at the window's end; STDP's soft bound uses the window-start weight.
* **float32** on the device; **noise drawn on the device** (`noise="host"`
  draws the reference stream, for checks); **files are device-independent**
  (`state_dict` returns the CPU format).
* `anima gpu-check` (finds the GPU, compares it with the CPU engine, times
  both) and `anima bench` (speed by brain size).

### Lessons learned
1. **Neither CuPy nor PyTorch could be installed** on the development VM
   (no GPU, and package downloads were blocked), so the engine was written
   against an interface NumPy also satisfies and tested that way, against
   the reference engine.  The checks that matter were made exact on
   purpose: in float64 with the reference noise stream and plasticity off,
   the device engine's arithmetic follows the reference's operation by
   operation, so any slip shows up as a difference in the spike trains.
2. **A float32 engine cannot load a float64 state exactly**: a creature
   saved on the CPU and loaded on a device differs in the 7th digit of its
   membrane potentials (found by the save/load test, which now allows it).
3. **`python -m anima` re-ran the whole CLI in spawned worker processes**
   (`anima/__main__.py` had no `__main__` guard).  Linux forked its workers,
   so it never showed; GPU colonies spawn them (a GPU library cannot be used
   in a forked process), so it would have.
4. (By design rather than learned the hard way.)  **On a GPU, waiting is
   the cost, not arithmetic.**  The default creature
   needs ~40 array operations per neuron update per millisecond; each is a
   kernel launch, and every host round trip stalls the pipeline.  Hence the
   fused kernel, dense delivery for small networks, per-window plasticity
   (about 10 kernels per plastic projection per window instead of a few per
   spike), one parameter upload and one readout download per tick.

### Results (CPU only — there was no GPU)
All on the 2-core development VM; `--device emulate` is the device engine
with NumPy standing in for the GPU library.

* **Exactness** (`anima gpu-check --device emulate`, `tests/test_device.py`):
  float64, the reference noise stream, plasticity off — spike counts,
  potentials and rates identical to the CPU engine for 2,000 ms (11,929
  spikes in a brain with a 200-neuron cortex; intrinsic plasticity and two
  rounds of synaptic scaling included).  Plasticity on, identical spikes for the first
  150 ms: RewardSTDP eligibility and weights and Hebbian weights equal (to
  the last bit, as it happens, at this scale), STDP within 6.8e-5 relative.
  Single controlled pairings: STDP (soft and hard bounds), Hebbian and
  RewardSTDP (with budget normalisation, with a weight floor) within 2e-5.
* **Learning in the GPU configuration** (float32, engine-drawn noise,
  plasticity per window; `anima experiment foraging`'s protocol, 10 min):
  correct choices 0.31 → 0.65 → 0.64 → 0.67 → **0.77** (seed 1, 19 meals)
  and 0.25 → 0.58 → 0.66 → 0.69 → **0.78** (seed 2, 40 meals); frozen
  0.33 and 0.36 (9 and 8 meals).  The CPU engine's published runs: 0.38 →
  0.75 and 0.33 → 0.77 (16 and 24 meals), frozen 0.33 and 0.31.
* **CPU engine speed by size** (whole creature): 2.39x real time at 1,667
  neurons, 0.42x at 4,667, 0.046x at 10,667 (21.8 s per simulated second;
  1.7 GB of memory).  Emulated device engine: 2.26x, 0.15x, 0.010x — its
  STDP update gathers whole columns of the weight matrix, cheap on a GPU and
  slow in NumPy.
* 142 tests (14 for the device engine).

### Honest limits (v0.5)
* **Not run on a GPU.**  The CuPy and PyTorch backends are thin and use only
  operations the NumPy path also uses, but they were never executed; `anima
  gpu-check` is the first thing to run.  How much faster a GPU is was not
  measured.
* **Small brains gain little.**  The default creature's millisecond is
  ~10–40 small operations however fast the GPU, so expect it to run about
  as fast as on a good CPU core; the gain is for large brains.
* **Memory is dense.**  The weight matrix is N×N float32 on the device (1.7 GB
  at 20,000 neurons) and the connection mask N×N bytes on the host;
  building a large brain also needs a few GB of host memory for a moment.
* Continuation after save/load on a device is deterministic but not the
  noise stream an uninterrupted run would have drawn; spike-for-spike
  equality with the CPU engine holds only with `noise="host"`, float64 and
  plasticity off.

## v0.6 — pain and pleasure; learning to repeat words

*Added 30 Oct 2026. Raw numbers: [`results/teaching/`](../results/README.md).*

**Request:** the GPU code works, so keep it.  The LLM teacher needs
improvement: the creatures are not learning.  Add pain and pleasure to the
creature; when it repeats a word or lesson from the LLM trainer, reward it;
in the chat, add a way to inflict pleasure or pain, to train it in real time
how it should react.

### Why it was not learning
Measured before anything was changed (the v0.5 brain):
1. **The rewarded synapses knew nothing about what it had heard.**  At the
   moment it started a reply, the consonant it had just heard got an extra
   push of +0.024 (in synaptic-drive units) from the unsupervised
   hearing→speaking mirror, but only +0.0004 from interoception and +0.0002
   from cortex — the two dopamine-gated inputs of the voice.  Rewarding a
   copy could teach "say *b* more often", not "say what you hear".
2. **Listening blurred the mirror.**  Its Hebbian synapses learned all the
   time, so a heard sound became linked to whatever it said next: a teacher
   repeating "ba ba" 40 times *flattened* the "b" row of the map
   (diagonal/off-diagonal 1.8 → 1.3).
3. **The map is what limits copying.**  Made diagonal by hand (same synaptic
   budget per articulator neuron), the creature repeated the first consonant
   of a word 80% of the time and its first syllable 32% (from 16% and 2%);
   half-diagonal: 50% and 18% (one v0.5 creature).  With the final code and
   two seeds (`--parts ceiling`): 84–91% and 41–64% (from 14–23% and 0–4%),
   half-diagonal 50% and 14–18%.
4. **Rewards from a person or an LLM come seconds late**, after the
   eligibility traces (0.3 s for movement, 2 s for the voice) have faded.
5. **A smile is a confounded reward**: it relieves loneliness, so a frown
   from a teacher of a lonely creature can be a net reward.

### Design
* **Pain and pleasure** (`brain/hedonic.py`, `Brain.feel`): a reward of
  ±(0.4 at once + 3/s for 0.2–0.7 s) × intensity through the critic (a
  dopamine integral of ~1×, 2× and 4× a smile at 0.3, 0.6, 1.0); a felt
  level (τ 1.5 s) in the snapshot, the dashboard and the thought log; pain
  freezes new acts for up to 0.4 s, raises noradrenaline and wakes a
  sleeper.  Bumps and food keep their old rewards.
* **Recall** (awake replay, reverse order): the brain keeps every committed
  act with the input snapshot taken when it was decided (`Brain.acts`); a
  feeling is *for* something — `auto` (its last words if it spoke in the
  last 8 s, else its last movement), `speech`, `movement`, `now`, or act
  ids — and those acts are replayed 80 ms each while the dopamine lasts:
  gated inputs set to the snapshot (the echo of the sound it had heard is
  back), the chosen articulator or muscle channel driven, the voice's "go"
  gate never; the body starts nothing new meanwhile.  The earliest recalled
  act becomes a hippocampal memory (replayed in sleep).
* **The mirror learns while it speaks** (and while it relives speaking in a
  recall or a dream), at a rate × (1 + 10 × phasic dopamine), from 0 up
  (`Brain._mirror_gate`, the modulator `mirror` that the Hebbian rule
  already reads as its gate — so the GPU engine needed no change).  A
  rewarded recall binds heard sound → produced sound ~5–8× faster than
  babbling; a painful one binds nothing.  **Echo masking**: when it starts a
  syllable, the echo of earlier sounds drops to 30% (the suffix effect), so
  what it hears while speaking is its own voice.
* **Reward for repeating the teacher** (`reward_imitation`): each word the
  teacher says opens a `WordTrial`; its reply (the syllables after the
  teacher stops, until it pauses 0.45 s or has said as many syllables as the
  word) is scored — first consonant 0.3, first syllable 0.5, the whole word
  1.0 pleasure — with credit for exactly the reply's syllables.  The reflexes
  reward with pleasure by default (`reinforcer="smile"` restores v0.5).
* **The LLM teacher:** new tools `pleasure` / `pain` (`amount`, `for`:
  reply — its answer to the teacher's last word, credited by the ids of
  exactly those syllables — speech | movement; pleasure 0.1–1, pain 0.1–0.6,
  at most one per turn, ≥2 s / ≥8 s apart, refused when there is no reply in
  the last 30 s or, "for speech", no words in the last 8 s); `say` is cut to one short word (≤2 words, ≤4 syllables, smileys
  removed).  The report is pre-digested: `last_reply` and `recent_replies`
  (you said / it replied / how much matched / what the reflexes gave),
  `practice` (the word being taught, said / repeated how often),
  `suggestion` (what a sound policy would do next), `its_last_words` (and
  how long ago), `feelings`, and the copy rates with chance levels.  The
  prompt was rewritten around how to teach (contingency, one word at a
  time, the same word until it is repeated, pleasure and pain *for* its last
  words, no random rewards, waiting is often best) with new examples.
  `anima tutor-check` also asks for a decision in a word-teaching situation.
* **Curriculum:** "Copy me" uses `reward_imitation`, the teacher keeps
  saying a word until it is repeated twice (or 8 tries), and the lesson is
  graduated by an **imitation exam** on a clone: 33 words, every consonant
  but "h" three times; it must repeat the first sound ≥30% and ≥2.5× chance,
  in two exams in a row.
* **Real-time controls:** `/pleasure` `/pain` (amount, for), `+ ++ +++` and
  `- -- ---` in the chat, single `+` / `-` keys and feeling bars in the
  dashboard — by default for its answer to what you last said (a
  `ReplyTracker` follows it, so the credit lands on that answer even if it
  has babbled on since; after 15 s, its last words or movement), `POST /pleasure` / `POST /pain` on the bridge (and the
  client's `pleasure()` / `pain()`), `Agent.pleasure()` / `Agent.pain()` /
  `Agent.feel()`; `pleasure` / `pain` events; thoughts like *that feels good
  — I think of saying "ba ba"*.

### Lessons learned (each was a real failure first)
1. **A dopamine-gated auditory area did not make copying teachable.**  The
   first fix was a new population (`phon`: one channel per phoneme,
   relaying the ears) with reward-modulated synapses onto the voice.  In five
   settings of learning rate, eligibility and recall length, 40 tries of "ba
   ba" with pleasure for copies left its "b" row at 1.5–2.4 (diagonal/off-
   diagonal) whatever it started from (1.3–5.0), and copies did not rise:
   food, company and novelty credited whatever it said right after hearing a
   sound about as much as the teacher's pleasure did.  Removed; the mirror
   itself became teachable instead.
2. **The mirror must not learn while it only listens** (diagnosis 2 above);
   gating it to its own speech turned the flattening into sharpening.  The
   final check (`--parts practice`, "ba ba" 40 times, two seeds): the "b"
   row went 1.8 → 3.1 and 1.5 → 2.1 without reward, 1.8 → 5.8 and 1.5 → 3.3
   with pleasure for copies, and 1.8 → 1.3 and 1.4 → 1.2 with the old rule
   (with or without pleasure).
3. **Masking the echo alone did little** (no teacher: diagonal index 4.8 vs
   4.4 after 10 minutes of babbling); together with the speaking gate it
   keeps the echo of the teacher's word out of what the reply's syllable
   learns.
4. **A painful recall would have taught the very association it punished**:
   replaying "heard *b*, said *d*" drives both, and a Hebbian synapse cannot
   tell pleasure from pain.  The mirror is paused (gate 0) while it replays
   a moment that hurt, in waking recall and in sleep.
5. **A sustained teaching signal reached its dopamine too slowly**: with only
   a per-second reward the burst built up over half a second (0.07 → 0.6),
   while a movement's eligibility lasts 0.3 s.  The signal now starts with an
   onset (0.4 × intensity), so the burst is there at once (0.9 at intensity
   1).
6. **The creature's night spoils an experiment**: in a 15-minute pilot every
   variant dropped at the last exam, taken from a clone woken at 2 am (it
   falls asleep ~13 minutes after birth with the default 20-minute day).
   Teaching experiments now stay in its morning (10 minutes).
7. **Counting replies through a capped log stops counting**: the first
   "focus" teacher counted its successes from the toolkit's last-20 trials,
   which stopped growing once full; successes are now counted as they come.
8. **Tests that read the thought workspace are fragile**: a reward's
   evaluation now competed with the memory of the rewarded moment; that test
   lives in an empty world and gives the reward up to three times.

### Results
*(final code.  `anima experiment teaching --seeds 1,2,3 --workers 2` (about
2 hours), `--parts practice,ceiling`, `anima experiment tutoring --workers
2`, `anima experiment imitation`.)*

**What makes it learn to repeat a word.**  Ten minutes of teaching in the
creature's morning, after a minute alone: the teacher says short words
("ba ba", "ki ki", every consonant but *h*) and repeats each one until its
first sound has been repeated twice (or 8 times in vain).  A clone sits a
44-word imitation exam every 5 minutes.  Three seeds; "first sound" chance
≈ 0.09, syllable ≈ 0.02:

| what follows its replies | map rule | first sound: before → 5 → 10 min | syllable at 10 | pleasure / pain |
|---|---|---|---|---|
| nothing (the teacher only talks) | 0.6 | 0.09 → 0.17 → **0.42** | 0.18 | — |
| a smile for the right first sound (0.5's reflex) | 0.6 | 0.09 → 0.27 → **0.36** | 0.16 | smiles |
| graded pleasure, with recall (0.6's reflex) | 0.6 | 0.09 → 0.19 → **0.44** | 0.16 | 23 / 0 |
| graded pleasure, no recall | 0.6 | 0.09 → 0.12 → **0.37** | 0.15 | 22 / 0 |
| a person 2.5–4.5 s late, for its last words | 0.6 | 0.09 → 0.23 → **0.43** | 0.19 | 20 / 80 |
| the same, for its answer (chat `+` / `-`) | 0.6 | 0.09 → 0.26 → **0.38** | 0.13 | 22 / 78 |
| the same, no recall | 0.6 | 0.09 → 0.21 → **0.34** | 0.15 | 22 / 78 |
| graded pleasure, with recall | 0.5 | 0.09 → 0.12 → **0.12** | 0.03 | 14 / 0 |
| nothing | 0.5 | 0.09 → 0.12 → **0.15** | 0.02 | — |

* **The mirror rule is the effect.**  With the 0.6 rule every condition
  reached 0.34–0.44 (syllable 0.13–0.19); with the 0.5 rule 0.12–0.15
  (0.02–0.03).  That is the answer to "the creatures are not learning".
* **What follows the replies matters much less over ten minutes.**  The
  teacher who only talks (0.42) did as well as graded pleasure (0.44): every
  word makes it answer (turn-taking), and every answer is practice for a map
  that learns from its own speech.  The reward conditions differ by less
  than the seed-to-seed spread (±0.07).  Recall gave a small edge both times
  it was tested (0.44 vs 0.37; 0.43 vs 0.34), no more.
* **Late is fine with recall**: the person's feedback, 2.5–4.5 s after the
  reply ended, taught as well as the reflex's immediate pleasure.
* **Practising one word** is where pleasure clearly helps ("ba ba" 40 times,
  two seeds): "b" row 1.8 → 5.8 and 1.5 → 3.3 with pleasure for copies, 1.8
  → 3.1 and 1.5 → 2.1 without; copies of *b* in the last ten tries 5 and 4,
  against 1 and 2.  The 0.5 rule flattened the row (1.8 → 1.3, 1.4 → 1.2).
* **Ceiling**: a hand-made perfect map gives 84–91% first sound and 41–64%
  syllable (whole word 4–7%); its own map after babbling, 14–23% and 0–4%.
* **Late shaping of babbling did not work**: pleasure 2.5–4.5 s after each
  utterance containing *m* (20–36 times in ten minutes) left the *m* share
  at 0.03–0.11, with recall or without.

**Tutoring (scripted teacher, two seeds; anima 0.5 in brackets).**  Both
tutored creatures passed all four lessons at the first attempt: foraging
after 8 minutes, "Copy me" after 29 (14 awake minutes; 36% and 48% in the
last exam), the target sound after 42–43 (6–7 awake minutes of pleasure,
[~20 of smiles]), independence after 53–54.  Final exams, tutored vs alone:
foraging 0.86 / 0.77 vs 0.89 / 0.90; *m* share 0.26 / 0.56 vs 0.02 / 0.18
[0.49–0.72 vs 0.05–0.12]; copies the consonant it hears 9 and 9 of 30 vs
11 and 19 [6 and 2 vs 3 and 4]; repeats a word's first sound 0.40 / 0.55
vs 0.32 / 0.52, its syllable 0.14 / 0.20 vs 0.20 / 0.23.  The lone
creatures now copy too — babbling to itself trains the map — and the
tutored ones copy a little less than their twins, because after "make the
sound" many of their answers start with *m*.

**Imitation from listening to itself** (5 minutes alone, then a partner):
consonant map diagonal/off-diagonal 0.98 → 3.94 and 1.00 → 4.10, vowels →
14.5 and 13.2 [3.5–3.7]; it starts its answer with the heard consonant 24
times in 96 (~8 by chance, p ≈ 6e-07) [20], with the heard vowel 51 in 96
(~20, p ≈ 1e-12) [chance]; mirror frozen: 13 (p = 0.06), vowels at chance.

**The chat example** (`examples/07_pain_and_pleasure.py`, one seed, four
minutes, feedback 2–4 s late — a demonstration, not a controlled test):
first sound 18% → 27%, syllable 3% → 21%, with 4 pleasures and 45 pains.

165 tests (23 more than v0.5: pain and pleasure, the recall, the mirror
gate, the chat's credit, word trials and the imitation exam), about 5
minutes.

### Honest limits (v0.6)
* **No real LLM was run** (no Ollama here).  The scripted teacher goes
  through the same tools, validation and report, and the mock-server tests
  check the plumbing, but how well a real model teaches with the new
  prompt is untested.  Likewise **no GPU**: v0.6 changed no engine code
  (the mirror gate is the existing `mirror` modulator), and the device
  engine's tests pass in its NumPy emulation only (neither CuPy nor
  PyTorch could be installed here).
* Over ten minutes, **practice did most of the work**; the reward
  conditions cannot be told apart with three seeds.  There was no
  pain-only condition, so what pain contributes on its own is unmeasured.
* It repeats **first sounds, not words**: syllable copies 13–20%, whole
  words a few percent even with a perfect map.  The echo of a heard word
  fades and is masked once it starts to speak, so nothing holds the rest
  of the word; it has no phonological loop and no meanings.
* Late pleasure **shaped replies but not babbling**: a late "good" for an
  utterance credits every sound in it (section 9 of the README).
* Recall only happens awake; pleasure or pain during sleep gives the
  signal (and wakes it) but replays nothing.
* Sections 1–3 and 6–8 of the README were not rerun with 0.6; its voice
  changed, so their babbling numbers may shift a little.
* The chat's `+` / `-` credit "its answer to what you last said": the
  syllables it said after you finished, as many as your word had, within
  15 s — not necessarily the whole utterance you see (`bibipa` → "bi pa").
