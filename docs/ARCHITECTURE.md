# How anima works

anima is built on one idea: **a brain is a dynamical system that is always
running, not a function that is evaluated on request.**  Everything below
follows from taking that literally.

```
                ┌───────────────────────── Brain ──────────────────────────┐
                │                                                          │
   World ──► Body: senses ──► input populations ──► CORTEX (E/I, STDP) ───┤
     ▲            │               (gated by thalamus     │   ▲            │
     │            │                in sleep)             ▼   │ efference  │
     │            │                              MOTOR / VOCAL channels   │
     │            │                              (winner-take-all,        │
     │            │                               reward-STDP)            │
     │            ▼                                      │                │
     └──── Body: movements, syllables ◄──────────────────┘                │
                  │                                                       │
                  │   ┌── slow, non-spiking systems (every 10 ms) ──┐     │
                  └──►│ hypothalamus: hunger, fatigue, boredom,     │     │
                      │               loneliness  ──► interoception │     │
                      │ neuromodulators: DA, NE, ACh, 5-HT ──► gain, │     │
                      │               adaptation, plasticity gating  │     │
                      │ ventral striatum (critic) ──► dopamine (TD)  │     │
                      │ cerebellum (forward model) ──► surprise      │     │
                      │ hippocampus: episodes ──► replay in NREM     │     │
                      │ sleep switch: two-process model              │     │
                      └──────────────────────────────────────────────┘     │
                └──────────────────────────────────────────────────────────┘
```

## 1. Why it never waits

| source of activity | where | what it does |
|---|---|---|
| background noise | every neuron (`noise`) | the barrage of unmodelled input + channel noise every real neuron gets; neurons fire a few Hz with no input |
| recurrent E/I dynamics | cortex | turns noise into structured, oscillating, self-sustained activity |
| drives | hypothalamus → `intero` population | hunger, fatigue, boredom and loneliness grow with time; they *are* input |
| competitions | motor & vocal channels | noise + drive make channels race; the winner becomes a movement or a syllable |
| clock | sleep controller | the circadian rhythm and sleep pressure change the whole brain's mode |

So with an empty world and nobody talking, the creature still wanders,
babbles, gets bored and lonely, gets sleepy, falls asleep, replays its day
and wakes up.  Your text is not a prompt: it is converted into a stream of
phonemes and played into the auditory population over time, exactly like any
other sensation, while everything else keeps running.

The life loop (`anima.runtime.agent.Agent.tick`, every 10 ms):

1. drain the inbox (whatever was typed since last tick; empty is fine)
2. body reads the world sensors → input populations
3. 10 × 1 ms of spiking dynamics
4. drives, critic, neuromodulators, sleep, memory updated
5. body decodes motor/vocal populations → movements, syllables → events

## 2. The spiking core (`anima/core`)

* **Neurons** — Izhikevich (2003) model, `v' = 0.04v² + 5v + 140 − u + I`,
  `u' = a(bv − u)`; presets RS, IB, CH, FS, LTS, TC, RZ with optional
  heterogeneity.
* **Synapses** — conductance based: AMPA (τ 5 ms), NMDA with Mg²⁺ block
  (τ 100 ms), GABA-A (τ 7 ms, E = −75 mV).  The synaptic part is integrated
  with exponential Euler, which is stable for any conductance (plain Euler
  overshoots E_GABA under strong inhibition and makes neurons fire every
  millisecond — one of the first bugs this project hit).
* **Dale's law** — each neuron is excitatory or inhibitory.
* **Event-driven propagation** — all neurons live in flat arrays, every
  projection is a block view of one global weight matrix; only rows of
  neurons that fired are summed.  ~75 NumPy calls per millisecond regardless
  of how many populations a brain has.
* **Homeostasis** — intrinsic plasticity (each neuron's excitability drifts
  toward a target rate) and multiplicative synaptic scaling
  (Turrigiano 1998) keep activity from dying out or exploding as synapses
  learn.

### Learning rules (`anima/core/plasticity.py`) — all local

| rule | used for | mechanism |
|---|---|---|
| `STDP` | cortex → cortex | pair-based STDP with soft bounds, gated by acetylcholine (learning in attentive waking) |
| `RewardSTDP` | senses/cortex/drives → motor & vocal | three-factor rule: spike coincidences charge an eligibility trace, dopamine converts it into weight change (Izhikevich 2007); optional heterosynaptic competition for a fixed synaptic budget |
| `Hebbian` | hearing → articulation (mirror map) | symmetric coincidence learning with postsynaptic normalisation (von der Malsburg 1973) |
| delta rule | cerebellum | error × presynaptic activity (Marr–Albus) |
| TD(λ) | ventral striatum (critic) | reward prediction error drives value learning |

No gradient ever flows backwards through the network.

Two details turned out to matter a lot, and both have neuroscience behind them:

1. **Synaptic competition.** With only the chosen action's synapses
   changing, an action that was rewarded often gains drive from *every*
   input and takes over (the agent learned to always turn left).  Real
   synapses compete for limited resources (heterosynaptic plasticity,
   synaptic scaling), so `RewardSTDP(normalize=...)` conserves each input's
   total output weight and/or each neuron's total input weight; reward can
   only *redistribute* a budget.
2. **A learned, state-specific prediction for dopamine.** With dopamine =
   reward minus a global running average, a big food reward made the next
   ~8 seconds of actions look bad.  A critic that learns how good each
   situation is (`brain/critic.py`) gives dopamine its classic meaning —
   the temporal-difference error (Schultz, Dayan & Montague 1997) — and
   made sensorimotor learning far more reliable.

## 3. Brain systems (`anima/brain`)

### Drives (hypothalamus)
`hunger = 1 − energy` (energy burnt by living, moving and speaking; restored
by food), `fatigue` = sleep pressure (process S, faster with high cortical
activity), `boredom` (rises when the world is predictable, relieved by
novelty = cerebellar prediction error), `loneliness` (rises without contact,
relieved when someone talks to it or smiles).  Drives are fed to the brain
as the `intero` population; *reductions* in hunger and loneliness are
rewarding (drive-reduction / homeostatic RL, Keramati & Gutkin 2014).

### Neuromodulators (following Doya 2002)
| | signal | effect in anima |
|---|---|---|
| dopamine | phasic: TD error from the critic; tonic: average reward | third factor of `RewardSTDP`; tonic level sets motor vigor |
| noradrenaline | arousal + surprise; ~0 in REM | neural gain of cortex, motor, vocal populations |
| acetylcholine | high awake/REM, low NREM | low ACh → less tonic drive to cortex and a slow (τ 500 ms) K⁺ adaptation current, so slow-wave sleep shows silent "down" states punctuated by synchronous bursts; gates cortical STDP; scales sensory gain |
| serotonin | slow mood from reward/punishment balance | action-release threshold (low 5-HT → impulsive) |

### Sleep
Two-process model (Borbély 1982), fitted so that with the default 20-minute
day the creature sleeps roughly 23:00–07:00, with NREM/REM cycles.  In
sleep the thalamic gate closes (sensory input × 0.12 in NREM, × 0.04 in REM),
motor and vocal output are paralysed, the hippocampus replays salient
episodes into the network with the dopamine they earned (offline
reinforcement; Wilson & McNaughton 1994, Gomperts et al. 2015), cortical STDP
synapses are gently downscaled (Tononi & Cirelli 2014), and fast intrinsic
homeostasis is paused so it does not fight the sleep state.  Loud enough
input wakes it up.

What the cortex does in slow-wave sleep is emergent, not scripted: falling
acetylcholine removes some tonic drive and unmasks a slow K⁺ current
(Compte et al. 2003; Bazhenov et al. 2002) on top of recurrent NMDA
excitation.  Over a simulated day the cortex fires densely at ~4 Hz with a
~21 Hz rhythm while awake and drops to ~0.3 Hz of sparse firing that waxes
and wanes slowly in NREM (power below 2 Hz rises from <1% to ~17% of the
spectrum).  It is a simplified slow-wave state, not the crisp ~1 Hz up/down
alternation of a full thalamocortical model.  REM has high acetylcholine and
therefore faster, wake-like rhythms with the senses and muscles switched off
("paradoxical sleep").

Two homeostatic mechanisms are deliberately paused during sleep: intrinsic
plasticity and synaptic scaling.  Both are fast set-point controllers here;
left on, scaling "corrected" the low sleep firing rate by driving recurrent
weights to their maximum within ~20 s.

### Hippocampus
One-shot store of salient moments (reward, punishment, surprise): the
sensory/interoceptive state at the time of the last decision and the action
taken.  Prioritised by salience, decays over hours, discounted after each
replay.  (Abstract store; replay itself is spiking activity.)

### Pain and pleasure (`brain/hedonic.py`, `Brain.feel`)
Two innate feelings, given from outside — by a person in the chat, a teacher's
reflexes or tools, another program over the bridge — and the most direct way
to teach (bumping into walls and food keep their own, older rewards).

* **A teaching signal.**  A reward of `0.4 × intensity` at once, then
  `3 × intensity` per second for `0.2 + 0.5 × intensity` s (negative for pain),
  through the critic like any reward, so the phasic dopamine is a prediction
  error.  With the default dopamine system (impulse gain 2, τ 0.2 s, clipped
  at ±1) the dopamine integral is about that of a smile at intensity 0.3,
  twice at 0.6 and four times at 1.0; above ~0.8 the burst is saturated and
  intensity mostly sets its duration.
* **Recall** (awake replay; Foster & Wilson 2006, Ambrose et al. 2016).  The
  brain keeps the acts it committed to in the last ~minute (`Brain.acts`:
  each syllable or movement with the input snapshot taken when it was
  decided).  A feeling names what it is for — `auto` (its last words if it
  spoke in the last 8 s, else its last movement), `speech`, `movement`, `now`
  (nothing recalled) or explicit act ids (the `act` of `syllable` events) —
  and those acts are replayed, most recent first, 80 ms each: the gated
  input ports are set to the snapshot (so the echo of the sound it had just
  heard is back), the decided consonant/vowel or motor channel is driven
  (+8, the others −2), as in sleep replay; the voice's "go" gate is never
  driven, and the body starts nothing new until the recall is over (an
  act in progress may finish first, ≤ 0.4 s).  The teaching signal is made
  to last at least as long as the recall.  The earliest recalled act is
  stored in the hippocampus with reward ±intensity, so sleep replays it too.
* **Pain** also freezes new acts for `0.4 × min(1, 0.4 + intensity)` s
  (a flinch), raises phasic noradrenaline to `0.6 × intensity`, and wakes a
  sleeper three times more easily than pleasure.
* **Felt levels** jump to the intensity and decay with τ 1.5 s
  (`snapshot()["feelings"]`); `pleasure` / `pain` events carry the
  intensity, the source, what was recalled and how long ago it happened.

Why recall rather than longer eligibility traces: longer traces would credit
everything the creature did in the meantime too; a person's `+` three
seconds after a reply is about the reply, and the recall puts the credit
there — and, for the voice, re-creates the pairing that matters for
imitation (heard sound → produced sound), which the trace of the moment
itself carries only weakly.

### Cerebellum
Linear forward model predicting exteroception ~100 ms ahead from the current
input, bodily state and efference copies of the commands just issued.  Its
error is *surprise* (→ noradrenaline, memory salience) and *novelty*
(→ relieves boredom).  Because its own voice is predicted from the
efference copy, hearing itself is less surprising than hearing you
(von Holst's reafference principle).

## 4. Body (`anima/body`)

* **Smell**: eight egocentric bearing channels (population code with von
  Mises tuning), compressive intensity coding, sensitivity scaled by hunger
  (as ghrelin does).  **Whiskers**: three wall-proximity channels.
  **Taste**: fires on eating.
* **Movement**: three motor channels in a winner-take-all competition.  A
  channel that crosses threshold and beats the runner-up is committed for
  200 ms (commitment current to the winner, suppression of the losers — the
  thalamocortical loop the basal ganglia disinhibit), then a 120 ms
  refractory reset.  A movement's consequences arrive when it completes.
* **Voice**: a gate population decides *when* to vocalise; consonant (12)
  and vowel (5) competitions decide *what*.  Speech uses breath, so babbling
  comes in bouts.  Its own voice is played back into its ears 20 ms later.
* **Hearing**: letters are mapped onto the 17 phonemes it can perceive
  (c→k, v→b, r→l, …) and played at ~60–100 ms per phoneme.  Each phoneme
  leaves a ~1 s echoic trace (auditory sensory memory; Darwin, Turvey &
  Crowder 1972), so what it just heard can still shape what it says next.
  A new sound overwrites it: when it starts a syllable, the echo of what it
  heard before drops to 30% (the "suffix effect"; Crowder & Morton 1969), so
  while it speaks it hears mostly its own voice (v0.6).
  Emoticons and gestures (`:)`, `*smile*`, `:(`) are non-verbal social
  signals: they relieve loneliness and add approval or disapproval without
  any sound.
* **Turn-taking**: while it hears someone, interneurons inhibit its vocal
  gate (it does not talk over you).  When you stop, an innate reflex briefly
  excites the gate — infants vocalise in response to caregiver speech — so it
  tends to answer.  The reflex biases *when* it vocalises, never *what*.
* **Mirror map**: nothing tells it which sound each articulation makes.
  Hearing its own voice 20 ms after each syllable lets Hebbian synapses from
  hearing to articulation learn that mapping while it babbles alone, which
  is what later lets it echo you (Heyes 2010).  Since v0.6 these synapses
  learn *only while it speaks* (or relives speaking, in a recall or a dream),
  at a rate set by dopamine — `× (1 + 10 × phasic dopamine)`, from 0 up
  (`Brain._mirror_gate`; dopamine gates cortical LTP, and pairing a sound
  with dopamine expands its map: Bao, Chan & Merzenich 2001).  Learning
  while merely listening made every sound it heard a little more linked to
  whatever it said next: a teacher repeating "ba ba" 40 times *flattened*
  the "b" row of the map (diagonal/off-diagonal 1.8 → 1.3 and 1.4 → 1.2 in
  two creatures).  With the v0.6 rule it sharpened instead (1.8 → 3.1 and
  1.5 → 2.1), and more when its copies were rewarded with pleasure (1.8 →
  5.8 and 1.5 → 3.3).  How well it copies is set by this map: made perfectly
  diagonal by hand (same synaptic budget), the creature repeats the first
  consonant of a word 84–91% of the time and the whole first syllable
  41–64% (from 14–23% and 0–4% with the map it had learned).

## 5. Teaching: a local LLM as the teacher (`anima/tutor`)

The creature learns from consequences; a teacher arranges the consequences.
`anima.tutor` lets a language model served locally by Ollama play the part
of a parent or animal trainer — without becoming part of the brain.

```
 creature thread ── 10 ms ticks, never waits ───────────────────────────────────►
     ▲  events: action, syllable, utterance, ate          ▲ tick hooks (every 10 ms)
     │                                                    │
     │      SkillMeter (passive measurement)   TeachingToolkit
     │        target share, echo rate, copies     reflex rules, e.g.
     │        of words, meals, energy upkeep      reward_imitation: pleasure the
     │      Exams on a clone (plasticity off)     moment its reply repeats the word
     │        forage probe, imitation (33 words), reward_syllable: pleasure 0.5 s
     │        hungry/sated calls                  after each "m…" syllable
     │                │                           feed_on_call: food when it calls
     │                ▼
 tutor thread ── every ~8 s: observe ─► teacher.decide ─► validate ─► execute ──►
                                       (Ollama /api/chat,   (lesson     say, pleasure,
                                        JSON schema;        rules,       pain, praise,
                                        seconds)            welfare)     food, rules
```

**Why two speeds.**  Reward-modulated STDP links a reward to whatever the
synapses did within the eligibility window — about 0.3 s for movements and
2 s for the voice here.  A model that needs several seconds per decision
cannot hit that window, and attention given at random teaches random things
(the yoked control in the babble experiment).  So the LLM chooses the
*strategy* — which reflex rules are on, what to say, when to give food, when
to go back over an old lesson — and the reflexes, running on the creature's
own clock, deliver the contingent part within half a second.  This is the
same split an animal trainer makes with a clicker (Skinner 1951; Pryor
1984): the click is instant, the plan is not.

**What the LLM sees and can do.**  Each turn it gets a compact JSON report:
time of day, drives, energy and felt pleasure/pain, the syllables it said
and the moves it made since the last report, what it heard, **its last
reply** to the teacher's word (`you_said`, `it_replied`, `match`: nothing /
the vowel / the first sound / the first syllable / the whole word, and what
the reflexes already gave for it) and the few before, the word being
practised (`said_times`, `repeated_times`) and a `suggestion` (in "Copy me":
say it again, or move on to another consonant), what it said last and how
many seconds ago, the measured skills with chance levels, the lesson (goal,
progress, tips, which rules and whether food are allowed), the active rules
and its own last actions and notes.  It answers `{"thought": …, "actions":
[…]}` with at most three of: `wait`, `say` (one short word — the creature
hears phonemes, not words), `pleasure` / `pain` (`amount`, `for`: `reply` —
its answer to the teacher's last word, by the ids of exactly those
syllables, even if it has said other things since — `speech`, whatever it
said last, or `movement`; the creature recalls that act while it feels it,
so a decision that takes seconds still lands on the right act),
`praise`, `scold`, `give_food` (ahead / left / right / near), `set_rule` /
`clear_rule`, `switch_lesson` (review a passed lesson for three minutes) and
`note`.  The system prompt is about *how to teach*: reward exactly what you
want; one word at a time, the same word until it is repeated; pleasure and
pain are for its last words; never reward at random (talking to it is
already rewarding); waiting is often best — with worked examples.  Requests use Ollama's structured
outputs (a JSON schema in `format`); replies are still parsed defensively
(prose, code fences and `<think>` blocks are skipped), and older servers that
reject schemas get plain JSON mode.

**Guard rails.**  Every decision goes through `validate()`: unknown tools and
rules are dropped, rules outside the current lesson are refused, food is
refused when the lesson forbids it (in "ask" a free meal would make the call
meaningless) unless the creature is starving, lessons cannot be skipped,
speech is cut to one short word (≤ 2 words, ≤ 4 syllables, letters only),
pleasure is capped at 1 and pain at 0.6, only one of them per turn, the
tutor spaces them (≥ 2 s between pleasures, ≥ 8 s between pains), and
neither is accepted "for speech" when it has said nothing for 8 s.  If the model is unreachable,
slow or answers nonsense, the scripted teacher covers that turn; the creature
never waits either way.  Graduation is decided by the measurements, never by
the model's opinion.

**Conditions.**  A lesson also sets the conditions the tutor maintains, as
a trainer controls the training room: in *forage* the creature is kept
keen — mildly food-restricted (energy ≤ 0.5), as laboratory animals are
during training (Toth & Gardiner 2000) — because hunger sharpens smell and
scales the reward of approaching food; in *babble* and *echo* it stays a
little hungry (≤ 0.6), because dopamine is global and the smiles that shape
its voice otherwise blur a foraging map it no longer practises (a pilot's
probe fell from 0.60 to 0.50 while it was sated); in the optional *ask*
lesson no food grows in the arena, so calling is the way to eat.  Food a
teacher gives is eaten once and does not grow back.  Conditions belong to
the tutor and are not saved in the brain file, so a trained creature always
wakes up in a normal world.

**Curriculum** (`anima/tutor/curriculum.py`): forage → echo → babble →
independence (no reflexes, no help), plus an optional, experimental *ask for
food* lesson before independence (`--with-ask`).  In *echo* ("Copy me") the
reflex `reward_imitation` follows each word the teacher says (a `WordTrial`:
its reply is the syllables it says once the teacher has stopped, until it
pauses 0.45 s or has said as many as the word has) and gives pleasure for
the reply's syllables — 0.3 if it repeats the first consonant, 0.5 the first
syllable, 1.0 the whole word; the teacher says the same word until it has
been repeated twice (or 8 times), then another consonant; and the lesson is
graduated by an imitation exam on a clone (33 words; the first sound
repeated ≥ 30% and ≥ 2.5× chance, twice in a row).  The reflexes reward with
pleasure since v0.6 (`reinforcer="smile"` gives anima 0.5's smiles).  Copying comes before the
target sound because once one sound dominates its babbling, the echoic trace
of a sound it hears keeps being paired with the dominant sound it makes
next, which blurs the hearing-to-speaking map (with the other order the echo
rate fell from about 0.2 to 0.05–0.1).  Each lesson has a minimum amount of
*awake* practice and an objective test.  Babbling, echoing and keeping fed
are measured passively from the creature's own events.  Foraging and asking
are not: with several pieces of food in the arena the smell usually comes
from several directions at once (the resultant length of its smell
population vector has a median of 0.2–0.3), so natural moves say little
about whether it can follow its nose — a passive score sat near chance while
the probe showed 0.7; and a creature fed whenever it calls is rarely hungry
for long.  For those two the tutor gives an **exam** (`anima/tutor/exams.py`)
to a *clone* of the creature (`Agent.clone()`: same synapses and state,
plasticity off): food held 90° left / ahead / 90° right, as in the foraging
experiment; or alternating 15 s blocks held hungry and sated.  The creature
itself is not interrupted, moved or trained by being tested.  An exam must
be passed twice in a row and any test has to keep passing for a minute (no
graduating on one lucky moment); each lesson starts its measurements afresh;
passed skills get spot checks later on; the teacher can send it back to
review; and a lesson not passed after 45 awake minutes is left "not
mastered" so the curriculum never gets stuck.

**Why "ask" is optional.**  In that lesson no food grows; a call ("m…") made
while hungry is answered with food, one made when full with a frown.  The
creature does learn to call for food, but not to call *only* when hungry:
after 20 minutes of the lesson its synapses onto the "m" articulator from
hunger, boredom and loneliness were about equally strong (1.05, 1.09 and
1.05 in seed 1).  Boredom and loneliness are active in every state, the sound
is made every few seconds and dopamine is global, so any nearby reward — the
walk to the food, eating, novelty — credits it whatever the stomach says.
Frowns at full-stomach calls did not tip the balance: in six exams during
those runs the share of "m" was 47–77% held hungry and 55–82% held sated.
A hunger-specific call probably needs a stronger interoceptive input to the
voice, or a more specific credit signal.

Tutor state (lesson, measurements, exams, rules, journal, milestones) is
saved as `<brain>.tutor.json` next to the brain, so skills build up over
sessions.

## 6. Programs, thoughts and other programs

### Using programs (`anima/tools`)

The creature can run external programs, but it cannot type: it has no words
and no way to compose arguments.  So each program you allow is bound to an
act it can actually produce — the way a lever in an operant chamber is bound
to a press (Thorndike 1898; Skinner 1938):

* **a syllable** (`say = "la"`, or `say = "b"` for any syllable starting with
  b) — works with every creature; its voice is how it acts on the world of
  programs;
* **a hand** (`hand = 0..3`) — an optional extra motor area (`hands=4`):
  a competition of four channels driven by cortex and by interoception
  (hunger, boredom, loneliness), with dopamine-gated plasticity (tau_e 2 s)
  and its own decoder (threshold 0.42, a reach commits for 250 ms and then a
  5 s refractory, so a few reaches a minute when nothing is learned yet).
  `sense.hands` tells it, for half a second, which hand's program answered.
  Hands are opt-in: the default brain is unchanged, so the published results
  still reproduce.

A call carries who and when (its drives, energy, clock, stage) — as JSON on
a command's stdin, a web service's POST body, or a function's argument — and
the answer comes back as consequences the creature can feel: a sound it
hears (without the social reward of being talked to), food placed near it,
a reward or punishment in [−1, 1].  What it learns about programs it learns
from these, with the same eligibility traces and dopamine as everything else.
Program kinds: `command` (argv list, never a shell, placeholders such as
`{python}` and `{hunger}`), `url` (a local web service), `function` (Python)
and `remote` (nobody is launched; the call is published and another program
answers it through the bridge).  The `ToolBelt` never blocks the creature:
programs run in worker threads and their answers are applied on the
creature's own thread at its next tick; deterministic in-line functions
exist for experiments.  Limits: only listed programs, a timeout and an output
cap per call, a cooldown per program, a global limit on calls per minute,
and every call and answer is an event.

### The thought log (`anima/mind`)

What is written down is not composed by the creature.  It is decoded from the
activity and state that drive its behaviour — a readout in the sense of a
brain–computer interface (Georgopoulos et al. 1986; Hochberg et al. 2006):

| kind | readout |
|---|---|
| feeling | a drive crossing into a new band (0.30 / 0.55 / 0.75) |
| perception | the population vector of the 8 smell channels: side and strength; whiskers |
| intention | the synaptic drive into each movement (and hand) channel from the rest of the brain — presynaptic rates (1 s average) times the current weights, the channel's own pools left out — when one leads by ≥ 6%: what biases the fast, noisy race between them |
| inner | which consonant and vowel channels lead *while it listens* — the mirror map moving its articulators without the voice gate |
| evaluation | phasic dopamine (the critic's prediction error) above 0.2, attributed to the reward that came with it, or to the forecast itself ("things are looking up") |
| surprise | the cerebellar prediction error |
| memory / dream | an episode stored by the hippocampus / replayed in slow-wave sleep, with the actions it held |
| reflection | every 3 awake minutes: what the episodes stored since then add up to, per action |
| speech, action, state, social, program | its utterances, eating, bumps, reaches, sleep stages, what it heard, the programs it used |

Which of these get written follows global-workspace accounts of
consciousness (Baars 1988; Dehaene & Changeux 2011): every 100 ms the
readouts offer candidates with a salience, the most salient enters a single
workspace, which then stays closed for 2.5 s awake (6 s asleep), an
"attentional blink" (Raymond et al. 1992).  Losers fade within two seconds
(counted as "unconscious").  A family of thoughts that just got in
habituates (Thompson & Spencer 1966) and recovers over 5–20 s, so a smell
flipping sides or a repeated babble does not crowd out everything else.
Waking, falling asleep, being spoken to, eating, programs and reflections
always get through.  The result is 25–35 lines a minute awake and ~10 asleep.
Every line keeps its numbers and its salience, and `anima experiment
programs` checks two readouts against what happened (README, results 7).

### The bridge (`anima/bridge`)

`anima serve` runs the creature in real time and opens a small JSON web
service on 127.0.0.1 (a token is required to listen anywhere else):
state, thoughts, an event log with sequence numbers (long polling, or
server-sent events), and POSTs to talk to it, reward it (clipped to ±1),
give it pleasure or pain for what it just did (`/pleasure`, `/pain`), feed
it, and answer remote programs' calls.  `BridgeClient` is the other end
in the standard library.  Requests are served on their own threads and act
on the creature at its next tick: the creature never waits for a client.

## 7. A colony: living together, and evolving (`anima/world/arena.py`, `anima/evolve`)

### The shared arena

`SharedArena` is the foraging world with several creatures in it.  Each sees
it through its own view (a `World`), so neither bodies nor brains change.
Nobody is told to compete or to cooperate; the arena only makes both
possible:

* **competition** — food is eaten by whoever gets there first; a creature
  within 2.5 units when another eats is told it *missed* the meal.  Food (and
  a feast) that nobody eats withers after 60–120 s and grows back elsewhere —
  without this, food in a corner nobody visits stays there for ever, nothing
  new grows, and a colony that keeps to the middle starves (see results);
* **cooperation** — now and then a *feast* appears: it smells twice as
  strong and is worth 0.4 energy *to each* eater (a meal is 0.25), but it
  can only be eaten when two or more creatures are within 1.6 units of it at
  the same time.  Going for it pays only if someone else comes too — the
  stag hunt (Skyrms 2004).  A creature that arrives alone feels *a feast too
  big to eat alone*;
* **company** — what one says is heard by the others within 7 units, more
  faintly further away, and relieves their loneliness just as your words
  do; turn-taking and the answering reflex make them babble back and forth;
* **sensing each other** — an opt-in `sense.kin` port (`infant_spec(kin=True)`,
  80 neurons) carries a bearing code of where the others are, to cortex and
  — through dopamine-gated synapses — to the movement channels, so
  approaching or avoiding the others is learned.

### Rounds, and worker processes

A colony advances in rounds of 10 ms (`anima/evolve/hosting.py`): each
creature senses the arena *as the round began*, lives 10 ms, and reports the
movements it completed; the colony then applies all movements in a shuffled
order and hands each creature its consequences (food, pain, a feast shared,
a missed meal, the change in smell its movement brought) at the start of the
next round, together with the voices that reached it.  Because no creature
feels another's movement within a round, creatures can live in separate
processes (`workers`) with exactly the same result — a test checks that one
process and two worker processes produce identical lives.  Consequences
arrive 10 ms after a movement completes, well inside the ~0.3 s that its
synapses stay eligible.

### Days, fitness, and the genetic algorithm

At dawn each creature's day is scored: its **fitness** is the food it ate
(energy; meals 0.25, feast shares 0.4).  The day's best **stands out** if it
ate at least 25% (`margin`) and 0.5 energy (`min_gain`, two meals) more than
the median of the others (with `compare = "itself"`: the creature that
improved most on its own previous day, by as much).  Then:

* the champion lives on unchanged (elitism);
* every other place is taken by a variation of it, born within 2 units of
  it: its **genes** mutated
  (each gene with probability 0.5, multiplied by `exp(0.2 z)`, kept within
  bounds), and — with `inherit = "brain"`, the default — a copy of its brain,
  synapses and memories included, with 5% multiplicative noise on its
  learned synapses and fresh random streams.  This is exploit-and-explore as
  in population-based training (Jaderberg et al. 2017): Lamarckian, not
  biological.  `inherit = "genes"` builds fresh brains (new wiring) from the
  mutated genes instead — classic neuroevolution (Floreano, Dürr & Mattiussi
  2008) — which must learn everything again;
* `crossover` mixes the champion's genes with the runner-up's (uniform);
* the replaced creatures are archived (`retired/`, their final state saved)
  or, with `delete_old`, deleted.

If nobody stands out, nobody is replaced.  `selection = "random"` is a
control for experiments: the same turnover on the same days, but the parent
is a member picked at random instead of the best.  The genes are inborn settings,
not synapses:

| gene | default | range | what it sets |
|---|---|---|---|
| explore | 3.5 | 1.5–6 | noise of the movement channels |
| eagerness | 0.28 | 0.15–0.45 | the movement decoder's threshold (lower moves sooner) |
| move_lr | 0.05 | 0.01–0.25 | learning rate of the synapses that steer it |
| move_trace | 300 ms | 100–1500 | eligibility time constant of those synapses |
| appetite | 1.0 | 0.2–4 | reward for moving toward the smell of food |
| vigor | 3.0 | 0.5–8 | how much tonic dopamine energises movement |
| curiosity | 0.6 | 0.05–2.5 | reward for novelty when bored |
| sociability | 1.5 | 0.2–5 | reward per unit of loneliness relieved |
| loneliness | 90 s | 20–400 | how slowly loneliness builds |
| chatty | 3.0 | 1.5–5 | noise of the urge to speak |
| voice_lr | 0.06 | 0.01–0.25 | learning rate of what it says |

Genes live in a creature's saved state (brain spec, brain settings, body
settings), so a creature with new genes is built, saved and loaded like any
other.  `colony.json` keeps the lineage: every creature's parent,
generation, genes, mutations and fitness by day.

Whether selection helps is measured, not assumed (README, results 8): with
four creatures and ten-minute days a creature's food on one day did not
predict the next (r = −0.05), so the day's best is mostly the luckiest.
`anima.experiments.colony.repeatability` is the check to run before
trusting a selection scheme.

## 8. Running on a GPU (`anima/core/device.py`, `anima/core/engine.py`)

`Network(device=...)` — and through it `build_brain`, `Agent.create/load`,
`Colony.create/load` and every command's `--device` — chooses where the
spiking network runs.  `cpu` is the NumPy code of section 2, the reference.
Anything else hands the network, once built, to a `DeviceEngine`, written
once against a small NumPy-style array interface (`Backend`) with three
implementations: CuPy (NVIDIA CUDA; AMD with a ROCm build), PyTorch (CUDA,
ROCm, Apple `mps`, and `torch:cpu`), and NumPy itself (`emulate`: the device
engine on the CPU, which is how it is tested without a GPU).

**What lives where.**  On the device: membrane potentials, recovery
variables, the seven decaying per-neuron variables (three conductances,
STDP trace, readout, rate, slow adaptation), intrinsic-plasticity biases,
spike counts, the weight matrix (float32, as on the CPU), the learning
rules' state (masks as one byte per synapse slot, eligibility matrices,
budgets) and the noise buffer.  On the host: the network's structure (the
connection mask `M`), the per-neuron settings the brain writes every tick
(`I_ext`, `gain`, `adapt`, `bias_mod`, `noise_sigma`, `slow_inc` — uploaded
together when the brain marks them changed and at every window start), the
neuromodulator levels and flags, and everything outside the network (slow
systems, body, world, thought log).  Reading `net.v`, `net.W`, `net.rate` ...
returns a read-only host copy, cached until the next step; weights change
through `set_weights`, `scale_weights`, `jitter_weights`.

**One millisecond.**  The per-neuron update is the reference arithmetic in
the same order — as about 40 array operations, or as one fused CUDA kernel
with CuPy (checked against the array path on the actual device when the
engine starts, and dropped with a warning if they disagree).  Spikes are
then delivered: networks up to 2,500 neurons with one dense product
`[exc; inh] · fired @ W` (it reads all of W but needs no host round trip);
bigger ones gather only the rows of the neurons that fired, which needs one
wait for the device per millisecond to learn how many there were.

**Every 10 ms window.**  Each step records the spikes and the STDP traces
(after decay, before that step's spikes — what the reference rules read).
At the window's end, one wait for the device finds every neuron that fired
in it, and each plastic projection is updated with small matrix products
over the window, restricted to the columns (postsynaptic neurons) and rows
(presynaptic neurons) that fired:

* RewardSTDP: `E[:, post] += A+ · (X·s)ᵀ S_post`, `E[pre, :] −= A− · (S_pre·s)ᵀ Y`,
  with `s` each step's eligibility frame scale — exactly the per-spike rule,
  because eligibility updates do not depend on the eligibility or on the
  weight;
* Hebbian: the same sums, added to the weights;
* STDP: the same sums, with soft bounds taken from the weights at the start
  of the window (the per-spike rule compounds within a window: a
  second-order difference, ~1e-4 relative after 150 ms in the tests).

Weight changes of Hebbian and STDP therefore take effect at the end of the
window instead of the next millisecond.  Then come the reference engine's
slow updates (dopamine-driven weight changes, budget normalisation,
intrinsic plasticity, synaptic scaling), unchanged.

**Checks** (`tests/test_device.py`, `anima gpu-check`): with NumPy in
float64, the reference noise stream and plasticity off, the device engine's
spike trains, potentials and rates are identical to the reference engine's
for 2,000 ms (including intrinsic plasticity and synaptic scaling); with
plasticity on and identical spikes, eligibility traces and reward-learning
weights agree to rounding and STDP to ~1e-4; controlled single pairings
reproduce each rule (STDP soft and hard bounds, Hebbian, RewardSTDP with
normalisation and a weight floor) to 2e-5; files move between engines.

**Saving.**  `state_dict()` on a device returns the CPU format (host
copies, float64 neuron state), so a creature saved from a GPU loads on a
CPU and vice versa.  Device noise is re-seeded from the saved NumPy
generator state on load (deterministic, but not the stream an uninterrupted
run would have drawn).

## 9. Building your own

Everything above is data.  `anima spec --out brain.toml` writes the default
brain; edit populations, projections, rules and system settings, then
`anima live --spec brain.toml --new`.  In Python use `BrainBuilder`, and
subclass `anima.World` (five methods) to give it a different world —
see `examples/03_custom_brain.py`.

## References

* Izhikevich EM (2003) Simple model of spiking neurons. *IEEE TNN* 14:1569.
* Izhikevich EM (2007) Solving the distal reward problem through linkage of STDP and dopamine signaling. *Cereb Cortex* 17:2443.
* Bi GQ, Poo MM (1998) Synaptic modifications in cultured hippocampal neurons. *J Neurosci* 18:10464.
* Turrigiano GG et al. (1998) Activity-dependent scaling of quantal amplitude. *Nature* 391:892.
* Schultz W, Dayan P, Montague PR (1997) A neural substrate of prediction and reward. *Science* 275:1593.
* Doya K (2002) Metalearning and neuromodulation. *Neural Networks* 15:495.
* Aston-Jones G, Cohen JD (2005) An integrative theory of locus coeruleus–norepinephrine function. *Annu Rev Neurosci* 28:403.
* Hasselmo ME (2006) The role of acetylcholine in learning and memory. *Curr Opin Neurobiol* 16:710.
* Borbély AA (1982) A two process model of sleep regulation. *Hum Neurobiol* 1:195.
* Tononi G, Cirelli C (2014) Sleep and the price of plasticity. *Neuron* 81:12.
* Wilson MA, McNaughton BL (1994) Reactivation of hippocampal ensemble memories during sleep. *Science* 265:676.
* Gomperts SN, Kloosterman F, Wilson MA (2015) VTA neurons coordinate with the hippocampal reactivation of spatial experience. *eLife* 4:e05360.
* McClelland JL, McNaughton BL, O'Reilly RC (1995) Why there are complementary learning systems in the hippocampus and neocortex. *Psychol Rev* 102:419.
* Wolpert DM, Miall RC, Kawato M (1998) Internal models in the cerebellum. *Trends Cogn Sci* 2:338.
* Keramati M, Gutkin B (2014) Homeostatic reinforcement learning for integrating reward collection and physiological stability. *eLife* 3:e04811.
* Goldstein MH, King AP, West MJ (2003) Social interaction shapes babbling. *PNAS* 100:8030.
* Heyes C (2010) Where do mirror neurons come from? *Neurosci Biobehav Rev* 34:575.
* Compte A, Sanchez-Vives MV, McCormick DA, Wang XJ (2003) Cellular and network mechanisms of slow oscillatory activity (<1 Hz) and wave propagations in a cortical network model. *J Neurophysiol* 89:2707.
* Bazhenov M, Timofeev I, Steriade M, Sejnowski TJ (2002) Model of thalamocortical slow-wave sleep oscillations and transitions to activated states. *J Neurosci* 22:8691.
* Darwin CJ, Turvey MT, Crowder RG (1972) An auditory analogue of the Sperling partial report procedure: evidence for brief auditory storage. *Cogn Psychol* 3:255.
* von der Malsburg C (1973) Self-organization of orientation sensitive cells in the striate cortex. *Kybernetik* 14:85.
* Skinner BF (1951) How to teach animals. *Scientific American* 185(6):26.
* Pryor K (1984) *Don't Shoot the Dog: The New Art of Teaching and Training.* Simon & Schuster.
* Wood D, Bruner JS, Ross G (1976) The role of tutoring in problem solving. *J Child Psychol Psychiatry* 17:89.
* Bengio Y, Louradour J, Collobert R, Weston J (2009) Curriculum learning. *Proc 26th ICML*, 41–48.
* Thorndike EL (1898) Animal intelligence: an experimental study of the associative processes in animals. *Psychol Rev Monogr Suppl* 2(4).
* Skinner BF (1938) *The Behavior of Organisms.* Appleton-Century.
* Georgopoulos AP, Schwartz AB, Kettner RE (1986) Neuronal population coding of movement direction. *Science* 233:1416.
* Hochberg LR et al. (2006) Neuronal ensemble control of prosthetic devices by a human with tetraplegia. *Nature* 442:164.
* Baars BJ (1988) *A Cognitive Theory of Consciousness.* Cambridge University Press.
* Dehaene S, Changeux JP (2011) Experimental and theoretical approaches to conscious processing. *Neuron* 70:200.
* Raymond JE, Shapiro KL, Arnell KM (1992) Temporary suppression of visual processing in an RSVP task: an attentional blink? *J Exp Psychol Hum Percept Perform* 18:849.
* Skyrms B (2004) *The Stag Hunt and the Evolution of Social Structure.* Cambridge University Press.
* Holland JH (1975) *Adaptation in Natural and Artificial Systems.* University of Michigan Press.
* Jaderberg M et al. (2017) Population based training of neural networks. arXiv:1711.09846.
* Floreano D, Dürr P, Mattiussi C (2008) Neuroevolution: from architectures to learning. *Evol Intell* 1:47.
* Thompson RF, Spencer WA (1966) Habituation: a model phenomenon for the study of neuronal substrates of behavior. *Psychol Rev* 73:16.
* Toth LA, Gardiner TW (2000) Food and water restriction protocols: physiological and behavioral considerations. *Contemp Top Lab Anim Sci* 39(6):9.
