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
  thoughts: anima_brain.thoughts.jsonl  (anima thoughts --brain anima_brain.npz --follow)
anima is alive (1667 spiking neurons, 07:11 awake). It does not wait for you: it will
babble, wander, get hungry, bored, lonely and sleepy on its own.
Talk to it whenever you like; + or - after its answer gives it pleasure or pain (it
learns from them). /help for commands, /quit to leave.

anima › sumomo le   [07:16]
anima › midu   [07:19]
anima › lu   [07:22]
anima › kuhu   [07:27]
you › hello
anima › kima no   [07:31]
anima › baso   [07:37]
you › mama
anima › dewi   [07:41]
you › :)
anima › mi   [07:47]
anima › we   [07:52]
anima › ti   [07:56]
you › ba ba
anima › didima   [07:59]
you › -
      (anima feels pain 0.4 — it recalls "di di", 3.5 s ago)
you › ba ba
anima › bibipa   [08:09]
you › +
      (anima feels pleasure 0.5 — it recalls "bi pa", 3.1 s ago)
      (anima found food)
you › mo mo
      (anima found food)
anima › huwapo   [08:18]
you › -
      (anima feels pain 0.4 — it recalls "wa po", 3.1 s ago)
you › mo mo
anima › pulu pena   [08:28]
you › -
      (anima feels pain 0.4 — it recalls "pe na", 3.1 s ago)
you › /status
[08:34 awake] hunger 0.04 fatigue 0.23 boredom 0.38 loneliness 0.22 | DA 0.31 NE 0.37 ACh 0.82 5HT 0.54
you › /quit

saved anima_brain.npz — run again to continue its life.
```

That is a real session with a newborn creature.  The typing was done by a
small script, which answered each reply with `+` if it began with the
word's first sound and `-` if not; the only edits are wrapping the long
lines and shortening the paths.  The first four utterances happened before
anyone said anything.  The bracketed times are its own clock — one
simulated day lasts 20 real minutes by default, so it gets sleepy about 13
minutes in.  What it recalls is its answer: the syllables it said after you
had finished, as many as your word has ("bibipa" began before "ba ba" was
over, so its answer was "bi pa").

It can also use programs you allow, by learning which of its acts make
them do something good for it; it keeps a log of its thoughts, read out of
its own brain activity; other programs can connect to it while it lives
([below](#programs-thoughts-and-other-programs)); and several creatures can
share an arena, compete for food or share a feast, and — when one of them
does clearly better — be followed by it and variations of it, chosen by a
genetic algorithm ([a colony](#a-colony-living-together-competing-cooperating-evolving)).  Everything runs locally on
a CPU with **NumPy as the only dependency** — or, with CuPy or PyTorch
installed, its spiking network can run on a GPU, which is meant for brains
several times larger than the default ([running on a GPU](#running-on-a-gpu);
not yet tried on real GPU hardware).

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

In the chat, type anything at any time.  To teach it, give it **pleasure**
or **pain** for what it just said or did: `+` (or `/pleasure`), `-` (or
`/pain`) — `++`/`+++` and `--`/`---` are stronger ([below](#pain-and-pleasure-teaching-it-in-real-time)).
`:)` or `*smile*` is approval and `:(` disapproval (non-verbal, no sound,
and much weaker).  `/status`, `/world`, `/stats`, `/feed`, `/thoughts`,
`/mind` (print its thoughts as they happen), `/tools`, `/sleep`, `/wake`,
`/save`, `/quit`.

The live dashboard shows the world, its drives and neuromodulators, the
activity of each region, a live raster of cortical spikes and the
conversation (in v0.3 with its thoughts in between, `/mind` hides them):

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
are faster with `--think off`.  `--reinforcer smile` makes the reflexes
smile instead of giving pleasure, as in anima 0.5.  Once it has graduated,
nobody needs to be there: it forages, babbles and sleeps on its own.  In
`anima chat` you can still talk to it, teach it with `+` and `-`, smile at
it, and `/feed` it.

A real session with the built-in scripted teacher, in two sittings (paths
shortened, the line about the thought log left out; long stretches cut at
the "…"):

```
$ anima train --teacher scripted --minutes 9 --fast
teacher: scripted teacher (no LLM)
lesson 1 of 4: Find food — turn toward the smell of food and walk to it, without help
training for up to 9 creature-minutes as fast as the CPU allows; Ctrl-C stops (progress is saved)

   [07:11 forage] scripted: Let it practise; the guide keeps food within reach. → wait
      anima › susubo ti ko
   [07:21 forage] scripted: Let it practise; the guide keeps food within reach. → wait
      anima › we ka
      anima › to se
      anima › wuli
   …
   ✎ [13:39] exam 'forage': score 0.63 (chance 0.33; correct moves left 16/20, ahead 7/20, right 15/20)
   …
   ✎ [16:26] exam 'forage': score 0.78 (chance 0.33; correct moves left 15/20, ahead 16/20, right 16/20)
   …
★ [17:39] graduated 'Find food': exam 0.78 (need 0.60; chance 0.33), food 3.0/min (need 1.0)
   ── lesson 2: Copy me — repeat the word it just heard, starting with its first sound
   [17:39 echo] scripted: A new word to copy (wo wo). The reflex rewards it when its reply repeats the word. → say 'wo wo'
      anima › bano to mi bo
      anima › ko
   …
$ anima train --teacher scripted --minutes 12 --fast      # later: it carries on where it stopped
resumed anima_brain.npz (age 541s of simulated life)
  schooling: in lesson 2 of 4 (echo), mastered: forage  [anima_brain.tutor.json]
   …
   [18:29 echo] scripted: Say 'wo wo' again until it repeats it (0 times so far, said 3 times). → say 'wo wo'
      anima › woba lo
   [18:39 echo] scripted: Say 'wo wo' again until it repeats it (1 times so far, said 4 times). → say 'wo wo'
      anima › wawame
   [18:48 echo] scripted: Say 'wo wo' again until it repeats it (1 times so far, said 5 times). → say 'wo wo'
      anima › luwowa
   [18:58 echo] scripted: A new word to copy (pu pu); enough of 'wo wo'. The reflex rewards it when its reply repeats the word. → say 'pu pu'
      anima › wa
      anima › punu
   [19:08 echo] scripted: Say 'pu pu' again until it repeats it (1 times so far, said 1 times). → say 'pu pu'
   …
   [23:00] it is asleep — the teacher waits (replay consolidates the day)
   …
   ✎ [08:35] exam 'imitate': repeats the first sound 32% (chance 9%), the syllable 13% (chance 2%), the whole word 0% (n=31/33)
```

Its answer is what it says after the teacher has finished: "wawame" did
not count (its "wa wa" began while the teacher was still saying "wo wo"),
"luwowa" did ("wo wa"), so after two repeats the teacher moved on.  To pass
"Copy me" it needs a second exam like the last one.

With an LLM teacher the decision lines carry the model's name, its latency
and its own one-line thought instead of the script's.

**How it teaches.** Every ~8 seconds the model gets a short JSON report and
answers with up to three actions: say a short word, give **pleasure** or
**pain** for what the creature just said or did, smile, frown, give food
(ahead/left/right/near), switch a reflex rule on or off, go back to review
an earlier lesson, or leave itself a note.  The report is pre-digested so
that even a small model can teach well: what the teacher last said, what
the creature answered and how much of it matched (*nothing, the vowel, the
first sound, the first syllable, the whole word*) and what the reflexes
already gave for it; the word it is practising, how often it was said and
repeated; a suggestion for the next step; what the creature said last and
how long ago; drives, energy, its measured skills against chance, the
lesson's goal, progress and tips.  The model never touches a weight; all
learning is still the creature's own dopamine, spike timing and sleep.

Rewards only teach if they come within about a second of the behaviour (the
width of the eligibility traces), and a model takes seconds to think.  So
teaching runs at two speeds, as a human trainer's does with a clicker: the
LLM picks the strategy and the moments, and **reflex rules** running on the
creature's own 10 ms clock deliver the contingent part — pleasure the moment
its reply repeats the teacher's word (a little for the right first sound,
more for the right syllable, most for the whole word), pleasure half a
second after an *m* syllable, food in front of it when it calls while
hungry, food within reach while it learns to forage.  The model's own
pleasure and pain are *for* the creature's last words or movement: it
recalls them while it feels them, so they still teach although the model
answers seconds late.  The creature never waits for the model: a slow answer
just means fewer decisions.

| lesson | goal | reflexes | conditions | passed when (measured) |
|---|---|---|---|---|
| 1 forage | find food alone | `guide_food` | kept mildly hungry (energy ≤ 0.5) | two exams in a row: ≥ 60% correct moves with food held left / ahead / right (chance 33%); ≥ 1 meal/min |
| 2 echo | repeat the word it hears | `reward_imitation` | a little hungry (≤ 0.6) | two exams in a row: a clone hears 33 words and repeats their first sound ≥ 30% of the time, and ≥ 2.5× chance |
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
is starving), lessons can't be skipped, speech is cut to one short word (two
at most, four syllables, letters only), pleasure is capped at 1 and pain at
0.6, one of them per turn and not too often, and neither is given "for its
words" when it has not said anything for 8 seconds.  In "Copy me" the
teacher says the same word until the creature has repeated it twice (or
eight times in vain), then moves on to another consonant.  If
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
to teach your own, and add reflexes to `anima.tutor.RULES`.  Results are in [section 5](#5-what-a-teacher-adds)
and [section 9](#9-what-makes-it-learn-to-repeat-a-word).

## Pain and pleasure: teaching it in real time

Every animal is born able to feel pleasure and pain, and nothing teaches
faster.  In anima 0.6 they are part of the creature.  Pleasure (a treat,
being stroked) and pain (a pinch) reach its dopamine system the way food
does — a burst above baseline, or a dip below it — so whatever it just did
becomes more likely, or less likely, the next time it is in that situation.
Pain also makes it flinch: for a moment it freezes, and it is aroused.  The
feeling lingers for a second or two and shows up in its state, the
dashboard and its thought log (*"that feels good — I think of saying "ba
ba""*, *"ouch — that hurts"*).

```
you › ba ba
anima › didima   [07:59]
you › -
      (anima feels pain 0.4 — it recalls "di di", 3.5 s ago)
you › ba ba
anima › bibipa   [08:09]
you › +
      (anima feels pleasure 0.5 — it recalls "bi pa", 3.1 s ago)
```

**Late is fine.**  A reward only teaches if it comes within a second or two
of the act — the width of the synapses' eligibility traces — and a person
who reads the reply and types `+` is later than that.  So pleasure and pain
make the creature **recall what it just did**: it pauses and replays its
last words (or its last movement) — the syllables it said, and what its
ears and senses reported when it decided to say them — while the dopamine
is there.  Rats do this: at a reward, the hippocampus replays the path that
led there, in reverse (Foster & Wilson 2006).  The credit lands on its
reply, and the moment is stored as a memory and replayed again in its sleep.

| in the chat | dashboard (`anima live`) | what it gives |
|---|---|---|
| `+` `++` `+++` | the `+` key | pleasure 0.5 / 0.8 / 1.0 (the key: 0.6) |
| `-` `--` `---` | the `-` key | pain 0.4 / 0.7 / 1.0 (the key: 0.5) |
| `/pleasure [0-1] [speech \| move \| now]` | same | pleasure for its last words, its last movement, or nothing in particular (default, and for `+` / `-`: its answer to what you last said, if it answered in the last 15 s — even if it has babbled on since; otherwise its last words if it spoke in the last 8 s, else its last movement) |
| `/pain [0-1] [speech \| move \| now]` | same | pain, likewise |

The intensity sets how strong the dopamine signal is and how long it lasts:
pleasure 0.3 is about a smile, 0.6 about twice and 1.0 about four times
(and a smile also relieves loneliness, so a frown at a lonely creature may
even please it — pain has no such side effect).  The dashboard shows both
feelings as bars under the drives.  Other programs use `POST /pleasure` and
`POST /pain` on the bridge, and Python `agent.pleasure(0.6)` /
`agent.pain(0.4, credit="movement")`.  [`examples/07_pain_and_pleasure.py`](examples/07_pain_and_pleasure.py)
teaches it to repeat words with late pleasure and pain, the way a person in
the chat would.

**What it learns from them.**  Pleasure and pain teach through the same
dopamine-gated synapses as food: which sounds it makes and when, which way
it turns.  For "repeat what you heard", the synapses that matter are the
ones from hearing to speaking (the mirror map), and in anima 0.6 dopamine
gates those too: they learn while it speaks, and much faster while it
recalls a reply it is rewarded for — the sound it heard and the sound it
made are bound together — while replaying a moment that hurt binds nothing.
[Section 9](#9-what-makes-it-learn-to-repeat-a-word) measures what this does.
A creature raised with an earlier anima is brought up to date when it is
loaded (`anima chat` says so): its map keeps everything it learned and from
then on learns this way (`Agent.load(path, upgrade=False)` keeps the old rule).

## Programs, thoughts, and other programs

### It can use programs — the ones you allow

It cannot type and has no words, so each program you allow is bound to
something it can *do*: say a syllable, or reach with one of its hands (an
optional extra motor area).  The program's answer comes back as
consequences it can feel — a sound, food in front of it, a treat or a
buzzer — and it learns which programs are worth using the way it learns
everything else: from dopamine, within a second or two of acting.

```toml
# examples/tools.toml
[[tool]]
name = "vending"                             # a snack machine
hand = 0                                     # runs when it reaches with hand 0
command = ["{python}", "tools/vending.py"]   # never a shell; the call arrives as JSON on stdin

[[tool]]
name = "music"
say = "la"                                   # runs when it says "la"
command = ["{python}", "tools/music.py", "{boredom}"]
```

```bash
anima tools examples/tools.toml --try music                    # check the file, run one program once
anima chat --brain pet.npz --new --tools examples/tools.toml   # a new creature with 4 hands and these programs
```

A program answers with plain text (which the creature hears) or JSON:
`{"say": "...", "reward": -1..1, "food": "ahead" | "left" | "right" | "near"}`.
It can be a local command, a local web service (`url = ...`), a Python
function (`function = "pkg.mod:fn"`), or `remote = true`: answered by some
other program through `anima serve` (below).  Only listed programs run —
without a shell, with a timeout, an output cap, a cooldown and a limit on
calls per minute — and every call is an event and a line in its thought
log.  They run with your permissions, so list only programs you would let a
curious animal press the button of.  `/tools` in the chat shows how each
program has gone.  (Tool files in TOML need Python 3.11+ or `pip install
tomli`; JSON works everywhere.)

### It keeps a log of its thoughts

```
$ anima thoughts --brain pet.npz -n 0      # lines 52-84 of 213: a new creature with examples/tools.toml
09:16  intention  I'm inclined to turn right, toward the smell
09:20  program    I use clock (with hand 1) → 'it is 09:20'
09:21  speech     I say "pi ki di"
09:24  inner      I mouth "si" silently (I heard "it is 09:20")
09:27  speech     I say "bi"
09:30  program    I use clock (with hand 1) → 'it is 09:30'
09:31  perception a strong smell of food, ahead
09:34  inner      I mouth "pa" silently (I heard "it is 09:30")
09:37  perception a strong smell of food, to my right
09:38  action     I ate
09:40  speech     I say "lu mu"
09:48  speech     I say "no na ke"
09:49  program    I use vending (with hand 0) → 'clunk', food ahead
09:51  perception a strong smell of food, behind me
09:54  feeling    I'm very lonely (0.75)
09:57  action     ouch — I bumped into a wall
10:00  speech     I say "melu"
10:03  perception a strong smell of food, to my right
10:03  program    I use vending (with hand 0) → 'bzz'
10:06  inner      I mouth "to" silently (I heard "bzz")
10:09  perception a strong smell of food, all around
10:12  social     someone says "ba ba"
10:12  social     someone smiles at me
10:12  memory     I'll remember this: turning left, saying "ke" (+0.90)
10:12  program    I use vending (with hand 0) → 'bzz'
10:15  inner      I mouth "be" silently (I heard "bzz")
10:18  feeling    I'm getting tired (0.31)
10:19  program    I use music (by saying "la") → 'lo li la lu', reward +0.18
10:21  evaluation music: better than I expected (+0.31)
10:24  speech     I say "la hodipe"
10:27  action     I reach with hand 2 (nothing there)
10:30  speech     I say "time"
10:33  feeling    less lonely now (0.60)
```

A real log, unedited — a stretch from the first seven minutes of a new
creature's life with the example programs, spoken to now and then
([the whole log](results/programs/thoughts_sample.txt)).
It hears the clock's words only as sounds; the vending machine gives food at
most every 30 seconds and otherwise buzzes; the music rewards it more the more
bored it is.

These lines are not sentences it composes — it has no language.  They are
read out of the activity and state that drive its behaviour, the way a
brain–computer interface reads intended movements out of motor cortex:
drives crossing a level, which side a smell is on, which movement its
senses are pushing it toward, its mouth silently following a sound it
hears, dopamine bursts that were better or worse than expected and what
caused them, moments its hippocampus stores and replays in its dreams, the
programs it used, and every few minutes what its recent memories add up to.
As in global-workspace accounts of consciousness, the readouts compete:
every 100 ms the most salient candidate gets in, the workspace then closes
for a moment, and what was just thought habituates — so you get about 30
lines a minute awake and about ten asleep, not a firehose.

```bash
anima thoughts --follow                      # the log of anima_brain.npz, as it is written
anima thoughts --kinds intention,program     # some kinds only (--json for the numbers)
```

In `anima chat`, `/thoughts` shows the latest and `/mind` prints them as
they happen (or start with `--thoughts`); the dashboard shows them between
the lines of the conversation.  The log is JSON lines next to the brain
(`anima_brain.thoughts.jsonl`), each with the numbers it was read from.
When an LLM teaches it (`anima train`), the teacher's reports include its
recent thoughts.  How well the readouts match what it then does is measured
in [results 7](#7-is-the-thought-log-true).

### Other programs can connect to it

```bash
anima serve --tools examples/tools_remote.toml        # lives in real time at http://127.0.0.1:7777
curl 127.0.0.1:7777/state                             # drives, modulators, sleep, what it is doing
curl -N '127.0.0.1:7777/stream?kinds=utterance,thought'   # everything it does, as it happens
curl -d '{"text": "ba ba :)"}' 127.0.0.1:7777/hear     # talk to it (:) is a smile)
curl -d '{"amount": 0.5}' 127.0.0.1:7777/reward        # a treat (or a buzzer: -0.5)
curl -d '{"amount": 0.7}' 127.0.0.1:7777/pleasure      # pleasure for what it just said (or did)
curl -d '{"amount": 0.4, "for": "movement"}' 127.0.0.1:7777/pain   # pain for its last movement
```

| endpoint | |
|---|---|
| `GET /state`, `GET /thoughts?n=50&kinds=...` | its state; its latest thoughts |
| `GET /events?since=SEQ&kinds=...&wait=20` | events after sequence number SEQ (long polling); `GET /stream` is the same as server-sent events |
| `POST /hear {"text"}`, `/reward {"amount", "source"}`, `/feed {"where"}` | talk to it, reward it (clipped to ±1), put food near it |
| `POST /pleasure`, `POST /pain` `{"amount": 0-1, "for": "auto" \| "speech" \| "movement" \| "now"}` | pleasure or pain for what it just did — it recalls that while it feels it ([pain and pleasure](#pain-and-pleasure-teaching-it-in-real-time)); `pleasure` / `pain` events follow |
| `GET /pending`, `POST /tool_result {"call_id", "answer"}` | be one of its `remote` programs: answer its calls |
| `GET /tools` | its programs and how they have gone |

It listens on 127.0.0.1 only; to listen anywhere else it needs `--token`
(then every request carries `Authorization: Bearer <token>`).  From Python,
with the standard library only:

```python
from anima.bridge import BridgeClient

anima = BridgeClient("http://127.0.0.1:7777")
anima.hear("ba ba")
anima.pleasure(0.6)                       # (or anima.pain(0.4, "movement"))
for e in anima.follow(kinds=["utterance", "thought"]):
    print(e["kind"], e["text"])

anima.serve({"bell": lambda call: {"say": "ding", "reward": 0.3}})   # be its "bell" program
```

[`examples/05_external_program.py`](examples/05_external_program.py) is a
complete program that is the creature's bell (it starts its own creature if
none is being served).

## A colony: living together, competing, cooperating, evolving

Several creatures can share one arena.  Nobody tells them to compete or to
cooperate; the arena just makes both possible, and what each one does is
whatever its own brain learns:

* **Food is shared.**  Whoever gets to a piece first eats it.  A creature
  that was close by when another ate learns it *missed* it.  Food nobody
  eats withers after a minute or two and grows back somewhere else.
* **Feasts need two.**  Now and then a feast appears.  It smells twice as
  strong and is worth more to *each* eater than a meal, but one creature
  alone cannot eat it: two or more must be at it at the same time.  Going for
  it pays only if someone else comes too — the stag hunt of game theory.
* **They hear each other.**  What one says reaches the others nearby (more
  faintly further away) and relieves their loneliness, as your words do;
  they babble back and forth.  They can also sense where the others are.

At dawn every creature's day is scored by the food it ate.  If the best one
**stands out** — at least 25% and two meals more than the median of the
others — a **genetic algorithm** makes the next day's colony: the champion
and variations of it (its genes mutated; by default each starting from a
copy of its brain), while the creatures they replace are archived — or
**deleted, with `--delete-old`**.  If nobody stands out, everyone lives on.
Does it make better creatures?  Not yet at this scale: with four creatures
and ten-minute days a day's winner was mostly the luckiest, and the colonies
that kept everyone did as well ([results 8](#8-living-together-and-selecting-the-best)).

```bash
anima arena --days 3                          # 4 new creatures in ./colony live 3 days, then it is saved
anima arena --days 2                          # later: the same colony goes on
anima arena --status                          # who lives there now, and the family tree
anima arena --days 1 --delete-old             # creatures that lose their place are deleted, not archived
anima arena --new --from anima_brain.npz      # a colony of your creature and variations of it
anima arena --thoughts all --watch            # print their thoughts as they happen, draw the arena
```

A real session, with 10-minute days (`--day 600`).  On the first day nobody
stood out (the best, Dee, ate 18% more than the others' median); the second
day — the colony resumed, keeping its day length — went like this (the
lists of mutations are shortened here; [the whole
session](results/colony/arena_session.txt)):

```
$ anima arena --dir sample_colony --days 1 --workers 2
resumed the colony at sample_colony: day 2 (07:12), Ada (gen 0), Bo (gen 0), Cyd (gen 0), Dee (gen 0)
selection: a creature that eats 25% (and 0.5) more than the others' median makes the next day's population with 3 variations of itself (copies of its brain with mutated genes); those it replaces are archived in sample_colony/retired.
living 1 day(s) as fast as the CPU allows in 2 worker processes; Ctrl-C saves and stops

[day 2 09:00] Ada 0+1f e0.92 | Bo 1+0f e0.93 | Cyd 1+1f e0.98 | Dee 4+0f e0.94 | feasts 1, contested meals 2
[day 2 12:00] Ada 2+1f e0.91 | Bo 3+0f e0.88 | Cyd 3+2f e0.99 | Dee 6+1f e0.97 | feasts 2, contested meals 6
[day 2 15:00] Ada 2+1f e0.67 | Bo 3+1f e0.82 | Cyd 5+2f e0.97 | Dee 7+2f e0.98 | feasts 3, contested meals 8
[day 2 18:00] Ada 4+1f e0.92 | Bo 4+1f e0.82 | Cyd 5+3f e0.81 | Dee 9+3f e0.99 | feasts 4, contested meals 11
[day 2 21:00] Ada 5+1f e0.85 | Bo 4+1f e0.58 | Cyd 6+3f e0.82 | Dee 9+3f e0.75 | feasts 4, contested meals 12
[day 2 00:00] Ada 5+1f e0.63 (asleep) | Bo 4+2f e0.80 (asleep) | Cyd 7+4f e0.93 (asleep) | Dee 9+3f e0.57 (asleep) | feasts 5, contested meals 12
[day 2 03:00] Ada 5+1f e0.58 (asleep) | Bo 4+2f e0.75 (asleep) | Cyd 7+4f e0.88 (asleep) | Dee 9+3f e0.51 (asleep) | feasts 5, contested meals 12
[day 2 06:00] Ada 5+1f e0.52 (asleep) | Bo 4+2f e0.70 (asleep) | Cyd 7+4f e0.83 (asleep) | Dee 9+3f e0.46 (asleep) | feasts 5, contested meals 12
=== end of day 2 ===
  creature        gen  fitness  meals  feasts  beat  missed  near others  heard  mean energy
  Cyd (c3)          0     3.60      8       4     1       5         67%    287         0.89
  Dee (c4)          0     3.45      9       3     7       4         72%    275         0.76
  Bo (c2)           0     2.05      5       2     2       4         55%    271         0.79
  Ada (c1)          0     1.65      5       1     3       6         70%    292         0.73
  -> Cyd stands out: 3.60 vs the others' median 2.05 (+76%).
     Next day: Cyd and its variations: Eli (eagerness 0.28->0.196 (-30%), move_lr 0.0673->0.0602 (-11%), …), Fay (move_lr 0.0673->0.0742 (+10%), …), Gus (explore 3.5->3.1 (-12%), …).
     Retired: Ada, Bo, Dee (archived in retired/).

saved the colony (day 3) at sample_colony — continue: anima arena --dir sample_colony   family tree: anima arena --dir sample_colony --status
```

`7+4f` is 7 meals and 4 shared feasts so far; *beat* counts meals eaten with
a rival within reach, *missed* the reverse.  Each creature also keeps its
own thought log.  Bo's, on the first day, around a feast (Bo was retired
after day 2, so its files are now in `retired/`: `anima thoughts --brain
sample_colony/retired/c2.npz`):

```
16:43  social     Cyd says "bemo"
16:49  perception someone nearby, to my left
16:55  feeling    I'm getting hungry (0.32)
17:01  inner      I mouth "ma" silently (I heard "mi")
17:07  evaluation getting closer to food: worse than I expected (-0.29)
17:13  perception a strong smell of food, to my right
17:19  perception a feast — but it is too big to eat alone
17:25  evaluation getting closer to food: better than I expected (+0.50)
17:30  action     a feast, shared with Cyd
17:31  memory     I'll remember this: going ahead (+1.00)
17:37  social     Cyd says "dehu"
```

| | |
|---|---|
| what is scored | the food a creature ate that day, in energy (a meal 0.25, its share of a feast 0.4) |
| who stands out | the day's best, if it ate ≥ 25% (`--margin`) and ≥ 0.5 (`--min-gain`) more than the others' median — or, with `--gain-over itself`, the creature that improved most on its own previous day, by as much |
| the next day | the champion, unchanged, and `population − 1` variations of it, born next to it |
| a variation | the champion's 11 genes, each mutated with probability 0.5 by a factor `exp(0.2 z)` (`--mutation`, `--mutation-rate`); with `--inherit brain` (default) a copy of its brain — synapses, memories — with 5% synaptic noise; with `--inherit genes` a fresh brain that must learn everything again; `--crossover` first mixes its genes with the runner-up's |
| the genes | how much it explores, how readily it moves, how fast and over what time window its steering synapses learn, how rewarding food smell, novelty and company are, how fast it gets lonely, how chatty it is, how fast its voice learns ([the list](docs/ARCHITECTURE.md#7-a-colony-living-together-and-evolving-animaworldarenapy-animaevolve)) |
| the replaced | archived in `colony/retired/` with their thought logs, or deleted with `--delete-old` (`--new --delete-old` also deletes a whole old colony; `--from` never touches the file it copies) |

Colony creatures are built with 600 cortical neurons (`--cortex`) and a
sense of where the others are.  Each has its own brain file (about 0.7 MB)
and thought log in the colony directory, which also keeps the lineage of
every creature that ever lived there (`colony.json`); any of them can be
loaded on its own (`anima chat --brain colony/c7.npz`).  Archived creatures
add up over the days — `--delete-old` keeps only the living.  The creatures
run in parallel worker processes, one per CPU core (`--workers`), with
exactly the same results as in one process.  Four creatures live at about
0.5–0.7× real time on the 2-core development VM, so a 20-minute day takes
about half an hour there; `--day 600` makes shorter days.  From Python:
`anima.evolve.Colony` ([`examples/06_colony.py`](examples/06_colony.py)).

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
| voice | a "when to speak" gate + consonant × vowel competitions, breath, self-hearing, echoic memory, turn-taking reflex; a hearing→speaking mirror that learns while it speaks, faster with dopamine | spontaneous babbling in bouts that you can shape; answering, echoing, repeating your words |
| pain & pleasure | innate feelings you (or a teacher) give it: a dopamine burst or dip, a recall of what it just did, a flinch | the fastest way to teach it — even a few seconds late |
| neuromodulators | dopamine, noradrenaline, acetylcholine, serotonin | learning, arousal, sleep oscillations, impulsivity |
| ventral striatum | TD critic | dopamine = reward prediction error |
| cerebellum | forward model | surprise, curiosity, telling its own voice from yours |
| hippocampus | episodic store + prioritised replay | consolidation during slow-wave sleep |
| sleep | two-process model (sleep pressure x circadian clock), NREM/REM cycles | a day it lives through (default day = 20 min) |

Nothing is trained by back-propagation, and nothing is pre-trained.  Every
weight change is local: spike timing, the current weight, and a broadcast
neuromodulator.  See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the
full design and the neuroscience behind each piece.

## Running on a GPU

Every command that runs a creature takes `--device`, and so does the Python
API:

```bash
pip install cupy-cuda12x        # NVIDIA (cupy-cuda11x for CUDA 11) — or:
pip install torch               # NVIDIA, AMD (a ROCm build) or Apple silicon
anima gpu-check                 # finds the GPU and checks it against the CPU engine (a minute or two)
anima bench                     # how fast it lives here, by brain size, on the GPU and on the CPU
anima chat --device auto        # a GPU if there is one; or cuda, cuda:1, mps, torch:cuda, cupy:cuda ...
ANIMA_DEVICE=auto anima arena   # or set it once, for everything
```

From Python: `Agent.create(device="cuda")`, `Agent.load(path, device="mps")`,
`build_brain(spec, device="auto")`, `Colony.create(..., device="cuda")`.  A
creature file does not belong to a device: a brain raised on a GPU lives on
a CPU, and back.

What runs on the GPU is the spiking network — neurons, synapses, spike
delivery, the three learning rules, homeostasis
([`anima/core/engine.py`](anima/core/engine.py)).  Drives, neuromodulators,
sleep, critic, hippocampus, cerebellum, the body, the world and the thought
log stay on the CPU: they handle a few hundred numbers every 10 ms.  Three
things differ from the CPU engine, all so that the GPU computes instead of
waiting:

* numbers are float32 (the CPU engine keeps the neuron state in float64);
* the learning rules are applied once per 10 ms window, from that window's
  spikes, with small matrix products.  For reward learning's eligibility
  traces this is exact; a Hebbian or STDP weight change takes effect at the
  end of its window instead of the next millisecond;
* the background noise is drawn on the GPU.

**What to expect.**  The default creature is small (1,667 neurons) and
already lives 2–3x faster than real time on one CPU core.  Every simulated
millisecond still costs a GPU a handful of small kernel launches, so a GPU
will not make *it* much faster, and may make it slower.  The GPU is for
bigger brains.  The CPU engine slows with the square of the number of
neurons (its weight matrix is dense): on the development VM it ran 2.4x
real time with the default 1,667 neurons, 0.42x with 4,667 (`--cortex
4000`) and 0.046x with 10,667 (`--cortex 10000`), where one simulated
second took 22 seconds.  A GPU's time per millisecond should grow far more
slowly, until the weight matrix fills its memory (float32: 0.45 GB at
10,000 neurons, 1.7 GB at 20,000) — how much faster it actually is, `anima
bench` will tell.

**It has not met a GPU yet.**  The development VM has none, and package
downloads were blocked there, so neither CuPy nor PyTorch could even be
installed.  The device engine was built and tested on the CPU, with NumPy
standing in for the GPU library (`--device emulate`), against the CPU
engine.  Without plasticity, in float64 and with the same noise, its spike
trains are identical to the CPU engine's for 2,000 ms.  With plasticity,
the eligibility traces and reward-learning weights match to rounding and
STDP to ~1e-4 relative (150 ms, identical spikes); rules driven by
controlled spike pairings match to 2e-5.  And in the GPU configuration
(float32, noise drawn by the engine, plasticity per window) it learns to
forage like the CPU engine: correct choices 0.31/0.25 → **0.77/0.78**
after 10 minutes (seeds 1/2; the CPU engine: 0.38/0.33 → 0.75/0.77), frozen
controls 0.33/0.36 ([results](results/README.md#gpu--the-device-engine-checked-on-the-cpu-readme-running-on-a-gpu)).
Emulated, it is not fast: slower than the CPU engine for big brains — it
is there to check the code, not to run it.  The CuPy and PyTorch layers on top
([`anima/core/device.py`](anima/core/device.py)) are thin — the same array
operations — but they were never executed.  `anima gpu-check` runs the same
comparisons on your GPU (and `--device torch:cpu` runs the PyTorch path on
the CPU); if it reports a disagreement, please report it.

## Results

Every number here was produced by the shipped code: two seeds for each
learning experiment (three for the main comparison in 9), one simulated day
for the first.  The first four take roughly 1.5 hours on the 2-core
development VM, the tutoring comparison about 2.5 hours, the program and
thought-log checks (6–7) about 40 minutes with `--workers 2`, the colony (8)
about 3.5 hours, the teaching comparison (9) about 2 hours with `--workers 2`.
`anima experiment <name>` reruns any of them, and the raw numbers are in
[`results/`](results/README.md).  Nothing was tuned per seed.

**Which version.**  Sections 4, 5 and 9 were measured with anima 0.6;
1–3 and 6–8 with anima 0.5.  What 0.6 changed in the creature itself is its
voice: the hearing→speaking map learns only while it speaks (faster with
dopamine), and the echo of what it heard is masked when it starts to speak.
That leaves foraging (2), the programs used with hands (6) and the colony's
food (8) as they were, and may shift the babbling numbers of 1, 3 and the
voice parts of 6–7 a little.

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

Five minutes of babbling alone, hearing its own voice, then a partner says random syllable pairs ("dodo", "mimi", ...) every 4–8 s (`anima experiment imitation`; anima 0.6).

| | hearing→speaking map, consonants (diagonal / off-diagonal) | vowels | answered | echoes heard consonant (chance) | echoes heard vowel (chance) |
|---|---|---|---|---|---|
| mirror learning on, seed 1 | 0.98 → 3.94 | 0.98 → 14.48 | 48/48 | 0.25 (0.08) | 0.69 (0.20) |
| mirror learning on, seed 2 | 1.00 → 4.10 | 0.98 → 13.20 | 48/48 | 0.25 (0.09) | 0.38 (0.20) |
| mirror learning frozen, seed 1 | 0.98 → 0.98 | 0.98 → 0.98 | 48/48 | 0.15 (0.08) | 0.15 (0.20) |
| mirror learning frozen, seed 2 | 1.00 → 1.00 | 0.98 → 0.98 | 48/48 | 0.12 (0.08) | 0.19 (0.20) |

Babbling alone is enough to learn which sound each articulation makes (the map becomes strongly diagonal).  With that map it starts its answer with the consonant it just heard **24 times in 96**, against ~8 expected by chance (binomial p ≈ 6e-07), and with the vowel it heard **51 times in 96** against ~20 (p ≈ 1e-12); with mirror learning frozen, 13 against ~8 (p = 0.06) and vowels at chance.  It answered every utterance, thanks to the turn-taking reflex.  With anima 0.5, whose map also learned while it only listened, the same experiment gave 20 consonant echoes in 96 and vowel echoes no better than chance: learning only from its own speech made the vowel map four times sharper (diagonal 13–14 against 3.5–3.7) ([`results/v0.1/`](results/README.md) has the 0.5 numbers).

![imitation](docs/figures/imitation.png)

### 5. What a teacher adds

Two creatures per seed, with the same brain and world seeds.  One is raised
by the tutor through the four lessons — by the **scripted** teacher, whose
decisions go through exactly the same tools and validation as an LLM's (no
real model could be run here; see the honest limits).  The other lives
alone for the same length of time in the same world: natural food, nobody
to talk to.  It keeps its plasticity and learns whatever its life teaches
it.  Then both live on alone until 9 am the next morning and sit the same
exam, with plasticity off (`anima experiment tutoring --workers 2`, about
1.5 hours on the 2-core VM; anima 0.6):

| | lived | finds food (chance 0.33) | share of syllables starting with *m* | *m* share hungry / sated | copies the consonant it hears (chance) | repeats a word: first sound / syllable (chance 0.09 / 0.02) |
|---|---|---|---|---|---|---|
| tutored, seed 1 | 53 min | 0.86 | **0.26** | 0.33 / 0.22 | 9 of 30 (2.8 expected, p = 0.001) | 0.40 / 0.14 |
| tutored, seed 2 | 54 min | 0.77 | **0.56** | 0.68 / 0.59 | 9 of 30 (1.3 expected, p = 4e-06) | 0.55 / 0.20 |
| alone, seed 1 | 53 min | 0.89 | 0.02 | 0.00 / 0.03 | 11 of 30 (2.8 expected, p = 5e-05) | 0.32 / 0.20 |
| alone, seed 2 | 54 min | 0.90 | 0.18 | 0.11 / 0.16 | 19 of 30 (2.1 expected, p = 4e-15) | 0.52 / 0.23 |

Both tutored creatures passed all four lessons at the first attempt:
foraging after 8 minutes (exams 0.73 and 0.77), repeating words after 29
minutes (14 awake minutes of "Copy me", until a clone repeated the first
sound of at least 30% of the examiner's words in two exams in a row: 36% and
48% in the last), the target sound
after 42–43 minutes and living alone — ten awake minutes with no reflexes
and no help, energy 0.95 and 0.94 on average and never below 0.49 — after
53–54 minutes, two nights included.  The teacher put food down 14 and 10
times in all.

**Everybody copies now.**  With anima 0.5 the creatures that lived alone
copied the consonant they heard at chance (3 and 4 times in 30) and the
tutored ones barely (6 and 2).  Now that the hearing→speaking map learns
from the creature's own speech, babbling to itself is enough: the lone
creatures copy 11 and 19 times in 30 and repeat the first sound of a word
32–52% of the time.  The tutored ones copy too (9 and 9, 40–55%), though
less than their lone twins in the echo test, whose words never use the
target sound: after "make the sound", many of their answers start with
*m*.  What the teacher still adds is the target sound — 26% and 56% of the
tutored creatures' syllables against 2% and 18% (anima 0.5: 49–72% against
5–12%; "make the sound" now passes after 6–7 awake minutes of pleasure
instead of ~20 of smiles, and they keep less of it overnight).  And as
before, teaching did **not** make better foragers in the long run: a
creature living alone learns to forage from food by itself (0.89 and 0.90
against 0.86 and 0.77).  The trained *m* is still not a request: they say
it about as often when sated as when hungry (the optional *ask* lesson
tries to change that — see the honest limits).  The 0.5 run is in
[`results/tutoring/`](results/README.md).

![tutoring](docs/figures/tutoring.png)

### 6. It learns which program to use

A creature with four hands, and a program on each (`anima experiment
programs`).  For two minutes none of them does anything.  Then the program
on the hand it had used *least* starts answering with a reward of +0.6 — no
sound, no food, just the reward — and the other three keep doing nothing.
The control is the same creature with plasticity frozen at that moment (the
two runs are identical until then).  Share of its reaches that ran the
paying program (chance 0.25):

| | paying program (its reaches in the first 2 min) | 0–2 min | 2–4 min | 4–6 min | 6–8 min | 8–10 min | 10–20 min | reaches per minute, 10–20 min |
|---|---|---|---|---|---|---|---|---|
| plastic, seed 1 | hand 0 (0 of 10) | 0.00 | 0.38 | 0.88 | 1.00 | 1.00 | **1.00** | 10.8 |
| plastic, seed 2 | hand 0 (2 of 11) | 0.18 | 0.45 | 0.95 | 0.95 | 0.95 | **1.00** | 10.9 |
| frozen, seed 1 | hand 0 (0 of 10) | 0.00 | 0.23 | 0.50 | 0.50 | 0.44 | 0.28 | 5.0 |
| frozen, seed 2 | hand 0 (2 of 11) | 0.18 | 0.00 | 0.31 | 0.00 | 0.00 | 0.30 | 3.0 |

Within four minutes of the program starting to pay, nearly every reach
runs it, and it reaches about twice as often — close to the most its
5-second rest between reaches allows — while still foraging (35 and 54
meals in the 20 minutes).  Frozen, it keeps using all four hands at about
chance.  This is operant conditioning — a rat learning which lever gives
food — not an understanding of what the program does.

Programs can be bound to its voice too.  Here syllables starting with *b*
run a program that rewards it (+0.5, from minute 2) and syllables starting
with *d* one that does nothing (well fed, empty world).  Share of its
syllables:

| | 0–2 min | 2–4 min | 4–6 min | 6–8 min | 8–10 min |
|---|---|---|---|---|---|
| *b* (pays), seed 1 | 0.14 | 0.08 | 0.37 | 0.46 | 0.37 |
| *b* (pays), seed 2 | 0.07 | 0.24 | 0.31 | 0.37 | 0.43 |
| *d* (does not), seeds 1 / 2 | 0.07 / 0.06 | 0.07 / 0.07 | 0.08 / 0.09 | 0.06 / 0.03 | 0.05 / 0.07 |
| *b*, frozen, seed 1 | 0.14 | 0.17 | 0.13 | 0.08 | 0.05 |
| *b*, frozen, seed 2 | 0.07 | 0.11 | 0.11 | 0.05 | 0.09 |

The paying syllable rises from 0.10 to about **0.40** of everything it says
while the one that does nothing stays at ~0.05; frozen, nothing changes.
With its 2-second cooldown the program ran for about 105 of the ~150 *b*
syllables in each run and rewarded 94 and 98 of them (after minute 2).

![programs](docs/figures/programs.png)

### 7. Is the thought log true?

Most lines transcribe state the code can see directly — a drive's level,
the dopamine signal, a sleep stage, an episode the hippocampus stored — so
they are true by construction.  Three readouts make claims that can be
wrong, and `anima experiment programs` checks them:

* **"I'm inclined to turn left"** is read from the synaptic drive into the
  three movement channels, not from what it is doing.  After ten minutes of
  foraging practice, the movement it made most in the next 3 seconds was the
  one named in **0.55 and 0.46** of 29 and 26 inclinations (seeds 1 and 2),
  against 0.35 and 0.34 by chance and 0.41–0.42 for guessing "the same as
  the last 3 seconds".  Better than chance, far from certain: its movements
  are decided by fast fluctuations that no readout can see coming.
* **"I feel like using program0 (hand 0)"** is read from the synaptic drive
  into the four hands.  In both plastic runs the log said so before the
  habit showed — 38 and 63 s after the program started paying, versus 42
  and 103 s until it made up 3 of its last 6 reaches.  Over all reaches
  after minute 2 the readout favoured the hand that reached 94% and 96% of
  the time — but once the habit is there, "the same hand as last time" is
  right 91% and 93% of the time too.  The frozen creatures, which learned
  nothing, never had a clear urge, and never logged one.
* **"I mouth "ba" silently"** is read from its vocal channels while it
  listens (its voice is held back while someone speaks).  Spoken to every
  ~12 s, the first syllable it mouthed started with the consonant it heard
  in **14 of 26** and **11 of 21** cases (0.54 and 0.52), against 0.05 by
  chance (shuffling what was heard across trials).  That is far more than it
  echoes out loud (0.17–0.25 in section 4): the mirror map it learned while
  babbling is clearly at work, but mostly below the threshold of speech.

With the workspace competition the log ran at 30 lines a minute awake; about
320 further candidates per 8 minutes faded without getting in.

### 8. Living together, and selecting the best

Per seed, three colonies of four young creatures (600 cortical neurons,
genes varied around the defaults) share an arena for four 10-minute days
(`anima experiment colony`).  The three are one and the same colony until
the first evening on which someone stands out; from then on:

* **evolving** — the day's best and three variations of it (copies of its
  brain with mutated genes, born next to it) make up the next day;
* **random parent** — the same turnover on the same rule, but the parent is
  picked at random: the control for selection;
* **static** — nobody is replaced; they keep learning.

| food per creature (energy), seed 1 / seed 2 | day 1 | day 2 | day 3 | day 4 |
|---|---|---|---|---|
| evolving | 2.40 / 2.74 | 2.83 / 1.83 | **2.55** / 2.75 | 2.67 / 2.23 |
| random parent | 2.40 / 2.74 | 2.83 / 1.83 | 1.77 / 2.75 | 2.09 / 2.42 |
| static | 2.40 / 2.74 | 2.83 / 1.83 | 2.27 / 2.75 | **2.85** / **3.01** |

Nobody stood out on day 1.  In seed 1, Bo stood out on day 2 (3.85 against
the others' median 2.30), so from day 3 the evolving colony was Bo and its
variations, the random one Ada and hers, the static one everyone.  In seed
2 the first standout came on day 3, and the random draw happened to pick
the best creature too, so those two colonies differ only by the mutations
and by chance — and still ended day 4 0.19 apart.

**Selecting the best did not make a better colony.**  On the three days on
which the colonies differed, the one that kept everyone ate the most (2.71
per creature, against 2.48 for evolving and 2.09 for a random parent).  The
reason shows in the static colonies: **a creature's food on one day says
nothing about the next** (r = −0.05 over 24 pairs of days; the day's best
ate more than the others the next day on 3 of 6 days).  At four creatures
and ten minutes, a day's winner is mostly the luckiest, and copying luck
does not help.  The mechanism itself works as specified — it picks the
standout, copies and varies it, archives or deletes the rest — but for it
to select ability, judging has to cover more than one short day (longer
`--day`s, larger colonies, or several days per verdict; the last is not
built yet).

What the creatures did, in every colony:

* **They compete without deciding to.**  A rival was within reach for
  34–45% of the meals on day 1 and for 53–87% on day 4 — most of all in the
  evolving colonies, whose members were near-copies born side by side.
* **They share feasts.**  Feasts, which need two, were 11–42% of the food
  (1–6 a day per colony).
* **They stay together.**  On day 1 they spent 57–70% of their waking time
  within 3 units of another creature, where four creatures placed at random
  would spend 22%; on day 4, 79–84% in the static colonies and 85–95% in the
  others.  Probably for two reasons nobody wrote down as a rule: the same
  smell draws them to the same food, and hearing one another relieves
  loneliness, which is rewarding.

The first version of this experiment found a trap worth knowing about.
Food grew back only once eaten; the static colony herded in the middle of
the arena while all the food sat uneaten at the edges, and it starved (2.54
→ 0.64 food per creature in three days) with brains that still foraged
well alone.  Food nobody eats now withers and grows back elsewhere
([NOTES, v0.4](docs/NOTES.md), lessons 6–7; [`results/colony/first_run/`](results/colony/first_run/)).

![colonies by day, and one day vs the next](docs/figures/colony.png)

### 9. What makes it learn to repeat a word

A teacher says short words to a creature living in its arena — a syllable
said twice, "ba ba", "ki ki", every consonant but *h* — and keeps saying
each one until the creature has repeated at least its first sound twice
(or eight times in vain).  What follows the creature's replies differs.
Every five minutes a clone sits the imitation exam: the examiner says 44
words, plasticity off; does its reply repeat the first sound (chance ≈ 0.09),
the first syllable (≈ 0.02)?  Ten minutes of teaching after a first minute
alone, in the creature's morning, three seeds (`anima experiment teaching
--seeds 1,2,3 --workers 2`, about 2 hours; anima 0.6 unless the map is
"as in 0.5"):

| what follows its replies | its hearing→speaking map | repeats the first sound: before → 5 → 10 min (seeds at 10) | its first syllable at 10 min | pleasure / pain given |
|---|---|---|---|---|
| nothing: the teacher only talks | learns while it speaks | 0.09 → 0.17 → **0.42** (0.35, 0.48, 0.43) | 0.18 | — |
| a smile if the first sound is right (0.5's reflex) | learns while it speaks | 0.09 → 0.27 → **0.36** (0.39, 0.27, 0.41) | 0.16 | smiles |
| graded pleasure, with recall (0.6's reflex) | learns while it speaks | 0.09 → 0.19 → **0.44** (0.39, 0.50, 0.43) | 0.16 | 23 / 0 |
| graded pleasure, no recall | learns while it speaks | 0.09 → 0.12 → **0.37** (0.36, 0.37, 0.39) | 0.15 | 22 / 0 |
| a person, 2.5–4.5 s late: pleasure if it repeated something, pain if not, for its last words | learns while it speaks | 0.09 → 0.23 → **0.43** (0.36, 0.41, 0.51) | 0.19 | 20 / 80 |
| the same person, for its answer (what `+` / `-` do in the chat) | learns while it speaks | 0.09 → 0.26 → **0.38** (0.39, 0.30, 0.46) | 0.13 | 22 / 78 |
| the same person, no recall | learns while it speaks | 0.09 → 0.21 → **0.34** (0.33, 0.46, 0.25) | 0.15 | 22 / 78 |
| graded pleasure, with recall | as in 0.5 | 0.09 → 0.12 → **0.12** (0.16, 0.05, 0.14) | 0.03 | 14 / 0 |
| nothing: the teacher only talks | as in 0.5 | 0.09 → 0.12 → **0.15** (0.07, 0.26, 0.12) | 0.02 | — |

* **How its hearing→speaking map learns decides it.**  With the 0.6 rule
  (learning while it speaks, faster with dopamine) ten minutes of a teacher's
  words took copying of the first sound from 0.09 to 0.34–0.44 and of the
  first syllable from 0.01 to 0.13–0.19, whatever followed its replies.  With
  the 0.5 rule (learning all the time) they barely moved: 0.12–0.15 and
  0.02–0.03.  That was why the creatures were not learning to repeat what
  they heard.
* **What follows the replies matters much less, over ten minutes.**  A
  teacher who only talks did as well (0.42) as graded pleasure with recall
  (0.44): every word makes the creature answer (the turn-taking reflex), and
  every answer is practice for a map that learns from its own speech.  The
  differences between the ways of rewarding are within what three seeds can
  tell apart (seed-to-seed spread ±0.07); the recall gave a small edge both
  times it was tested (0.44 against 0.37 for pleasure, 0.43 against 0.34 for
  the person), not more than that.
* **Late is fine, with recall.**  The person's pleasure and pain came 2.5–4.5
  seconds after the reply had ended — too late for the synapses' traces —
  and taught about as well as the reflex's immediate pleasure (0.43 and
  0.38 against 0.44).
* **Where pleasure clearly helps: practising one word.**  "ba ba" said 40
  times, pleasure for each reply that repeated the *b* (`--parts practice`):
  the *b* row of the map (diagonal / off-diagonal) went 1.8 → 5.8 and 1.5 →
  3.3, against 1.8 → 3.1 and 1.5 → 2.1 with no reward, and it repeated the
  *b* in 5 and 4 of the last ten tries, against 1 and 2.  With the 0.5 rule
  the same practice *flattened* that row (1.8 → 1.3 and 1.4 → 1.2, with or
  without pleasure): each "ba ba" it heard became linked to whatever it said
  next.
* **The ceiling** (`--parts ceiling`): with its map made perfectly diagonal
  by hand (same synaptic budget), a creature repeats the first sound of 84–91%
  of words and the first syllable of 41–64% (from 14–23% and 0–4%); half-way
  diagonal, 50% and 14–18%.  There is room to grow.
* **Shaping babbling with late pleasure did not work in ten minutes.**  A
  person giving pleasure 2.5–4.5 s after every utterance that contained an
  *m* (20–36 times in ten minutes) left the share of *m* at 0.03–0.11, with
  recall or without.  A late "good" for an utterance credits every sound in
  it, and the creature's next words may already have begun; the smiles of
  section 3 came half a second after the *m* itself.

![what follows its replies](docs/figures/teaching.png)

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
  core/        spiking simulator: neurons, network, plasticity rules, monitors;
               the device engine (GPU) and its CuPy / PyTorch / NumPy backends
  brain/       drives, neuromodulators, sleep, hippocampus, cerebellum, critic,
               pain and pleasure (and the recall of what it just did),
               Brain orchestrator, spec builder, presets (incl. optional hands)
  body/        physiology, senses, motor decoding, voice and hearing
  world/       World interface, foraging arena, shared arena, empty world
  runtime/     the never-blocking life loop, save/load
  tools/       programs it may use: commands, web services, functions, remote
  mind/        the thought log: readouts, the workspace, JSONL log
  bridge/      `anima serve` (HTTP + server-sent events) and its client
  evolve/      colonies: rounds and worker processes, genes, the genetic algorithm
  tutor/       teaching: Ollama client, reflex rules (incl. the reward for repeating
               a word), skill meters, exams, curriculum, scripted and LLM teachers,
               the tutor loop
  ui/          terminal chat, curses dashboard (both with + / - for pleasure and pain)
  experiments/ the experiments below, runnable with `anima experiment ...`
  benchmark.py `anima gpu-check` (the GPU against the CPU engine) and `anima bench`
tests/         pytest suite (165 tests, about 5 minutes), incl. a mock Ollama server
examples/      01 alone · 02 talk from code · 03 build your own brain + world · 04 LLM tutor ·
               05 another program as its "bell" · 06 a colony · 07 teaching with late pleasure and
               pain · tools.toml + tools/ (example programs)
configs/       infant.toml — the default brain as an editable spec
docs/          ARCHITECTURE.md (how it works), NOTES.md (design notes, lessons learned), figures
results/       the raw numbers behind every table and figure, with a guide
```

## How far is this from an LLM agent?

Far — and not in a way that more neurons would close by themselves.  v0.3
was asked for "a human-like thinking agent on par with LLM agents"; this is
where it stands, measured rather than claimed:

| | an LLM agent | anima |
|---|---|---|
| language | reads and writes text fluently | hears 17 phonemes, babbles syllables, can be shaped and can echo; understands no words |
| using a program | picks among many tools by reading their descriptions, right away | learns by trial and reward which of a few bound acts pays off, over minutes ([results 6](#6-it-learns-which-program-to-use)) |
| arguments | composes them: JSON, code, queries | none — a program receives its state (drives, clock), never instructions |
| reasoning, planning | many steps, in text | none explicit; credit reaches back about two seconds, and sleep replays the day |
| knowledge | much of the written world, from pre-training | nothing it has not lived |
| learning while working | none: the weights are fixed, only the context changes | always: every reward changes synapses and sleep consolidates them |
| time | wakes up for each request | lives continuously; gets hungry, bored, lonely and sleepy; acts unprompted |
| working with others | negotiates, divides work and reports back, in language | shares an arena: competes for food, sometimes shares a feast that needs two, babbles back and forth — no plans, no agreements ([a colony](#a-colony-living-together-competing-cooperating-evolving)) |
| getting better over time | improved by its makers between versions | learns while it lives; a genetic algorithm can copy a colony's best and vary its inborn settings — so far, at four creatures and ten-minute days, it selects luck rather than ability ([results 8](#8-living-together-and-selecting-the-best)) |
| explaining itself | fluent explanations, which need not reflect how it computed | none; its thought log is read out of its activity by code, and the readouts are checked against what it does next ([results 7](#7-is-the-thought-log-true)) |
| hardware | GPUs, billions of parameters | one CPU core, ~1,700 neurons by default; a GPU engine for bigger brains (untested on real GPUs so far) |

What it has that LLM agents lack is a life: it runs continuously, learns
from consequences while it lives, has motives of its own, and what we read
out of it is tied to its actual dynamics.  What it lacks is most of what
makes LLM agents useful — language, knowledge, abstraction, composing
actions, planning.  Getting those from spiking dynamics is an open research
problem, not an engineering step.  For useful work today use an LLM; here an
LLM is put where it is strong, as the creature's teacher (`anima train`) or
as another program talking to it through `anima serve`.

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
* **Speed.** The CPU engine is pure NumPy on one core. On the development
  VM a whole creature lives 2.4x faster than real time with the default
  1,000-neuron cortex, 0.42x with a 4,000-neuron one and 0.046x with
  10,000; the dense weight matrix grows as N².  A GPU engine exists
  (`--device`), but it has never run on a GPU: whether it is faster on
  yours, and by how much, `anima bench` will tell (see [running on a
  GPU](#running-on-a-gpu)).
* **Reproducibility.** Runs are deterministic for a given seed and sequence
  of inputs; anything typed live makes a run unique, as it should.  So do
  real programs: a command or web service answers at wall-clock speed (the
  program experiments use in-line functions, which are deterministic).
* **The LLM teacher has not met a real model here.** The development VM
  could not download models, so the Ollama side (client, prompt, structured
  outputs, fallbacks, the never-waiting loop) is tested against an
  in-process mock of the Ollama API, and the teaching results in sections 5
  and 9 use the scripted teacher, which goes through exactly the same tools
  and validation. How good a real model's strategy is depends on the model:
  whether a small one follows the report's suggestion, keeps to one word at
  a time and uses pleasure and pain sparingly is untested. The reflexes, the
  validation (one short word, capped and spaced pleasure and pain) and the
  measured graduation are what keep a weak or erratic one from doing harm.
  `anima tutor-check` tells you in a few seconds whether yours produces
  usable decisions, including one about teaching a word.
* **Programs are buttons, not tools.** It can learn *which* of a few
  programs pays off, from their consequences, in minutes; it cannot read
  what a program says, pass it arguments, or know what a program will do
  before it has tried it.  Whatever a program does, it does with your
  user's permissions — the creature only decides when to press.
* **Its thoughts are our words.** The thought log is a readout written by
  this code, not introspection by the creature: the vocabulary, the
  thresholds and which readouts exist are design choices.  Most lines
  transcribe state the code can see directly (drives, dopamine, sleep,
  stored episodes) and are true by construction; the readouts that make a
  claim about what it will do or what it is echoing are measured in results
  7, and they are far from perfect.
* **Colonies: selection picks luck, and nobody decides anything.** With
  four creatures and ten-minute days, one day's food does not predict the
  next (r = −0.05), so the genetic algorithm mostly copies the luckiest;
  in two seeds the colonies that were never replaced did as well (results
  8).  Two seeds and four days are a pilot, not a verdict.  Children copy
  the champion's learned synapses (Lamarckian, as in population-based
  training), so most of what is passed on was learned, not inborn; fresh
  brains from mutated genes (`--inherit genes`) were not evaluated.
  Competing and cooperating are not choices: creatures compete because they
  want the same food, and share a feast when two happen to arrive together;
  there is no negotiation, and what they say to each other is babble.  Four
  creatures live at about 0.6× real time on two cores.
* **Pain and pleasure are given, not sensed.**  They come from you, a
  teacher or a program; nothing in its world hurts or pleases it except
  through them (food and walls keep their older rewards).  They are a
  functional model — a dopamine burst or dip, a flinch, arousal, a recall —
  not a model of what pain feels like.
* **The credit is simple.**  In the chat, `+` and `-` are for its answer to
  what you last said (the syllables after you had finished, as many as your
  word has) if it came in the last 15 s; otherwise for its last words (if
  it spoke in the last 8 s) or its last movement.  Nothing cleverer: it
  cannot tell which part of an utterance you meant, and a late "good" for
  babbling credits every sound in it, which is why late pleasure did not
  shape what it babbles (results 9).  `/pain move` / `/pleasure speech`
  say which kind of act you mean.
* **It repeats sounds, not words.**  It learns to start its reply with the
  sound it heard (34–44% of words after ten minutes of teaching), and more
  slowly the whole syllable (13–19%); "ba ba" for "ba ba" mostly comes
  from repeating a syllable it has just said.  It has no memory for
  sequences, so "ba di" is out of reach, and it still does not know what
  anything means.
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
