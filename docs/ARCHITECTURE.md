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
  is what later lets it echo you (Heyes 2010).

## 5. Teaching: a local LLM as the teacher (`anima/tutor`)

The creature learns from consequences; a teacher arranges the consequences.
`anima.tutor` lets a language model served locally by Ollama play the part
of a parent or animal trainer — without becoming part of the brain.

```
 creature thread ── 10 ms ticks, never waits ───────────────────────────────────►
     ▲  events: action, syllable, utterance, ate          ▲ tick hooks (every 10 ms)
     │                                                    │
     │      SkillMeter (passive measurement)   TeachingToolkit
     │        target share, echo rate,           reflex rules, e.g.
     │        meals, energy upkeep               reward_syllable: smile 0.5 s
     │      Exams on a clone (plasticity off)     after each "m…" syllable
     │        forage probe, hungry/sated calls   feed_on_call: food ahead when it
     │                │                           says "m…" while hungry
     │                ▼
 tutor thread ── every ~8 s: observe ─► teacher.decide ─► validate ─► execute ──►
                                       (Ollama /api/chat,   (lesson     say, praise,
                                        JSON schema;        rules,       give food,
                                        seconds)            welfare)     set rules
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
time of day, drives and energy, the syllables it said and the moves it made
since the last report, what it heard, the measured skills, the lesson (goal,
progress, tips, which rules and whether food are allowed), the active rules
and its own last actions and notes.  It answers `{"thought": …, "actions":
[…]}` with at most three of: `wait`, `say` (short syllables — the creature
hears phonemes, not words), `praise`, `scold`, `give_food` (ahead / left /
right / near), `set_rule` / `clear_rule`, `switch_lesson` (review a passed
lesson for three minutes) and `note`.  Requests use Ollama's structured
outputs (a JSON schema in `format`); replies are still parsed defensively
(prose, code fences and `<think>` blocks are skipped), and older servers that
reject schemas get plain JSON mode.

**Guard rails.**  Every decision goes through `validate()`: unknown tools and
rules are dropped, rules outside the current lesson are refused, food is
refused when the lesson forbids it (in "ask" a free meal would make the call
meaningless) unless the creature is starving, lessons cannot be skipped,
speech is reduced to letters and emoticons.  If the model is unreachable,
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
food* lesson before independence (`--with-ask`).  Copying comes before the
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

## 6. Building your own

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
* Toth LA, Gardiner TW (2000) Food and water restriction protocols: physiological and behavioral considerations. *Contemp Top Lab Anim Sci* 39(6):9.
