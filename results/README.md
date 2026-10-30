# Results

The raw numbers behind the tables and figures in the main README, as the
experiments wrote them.  Everything was run on a 2-core cloud VM with the
code in this repository; `anima experiment <name> --json out.json` reruns
any of them (runs are deterministic for a given seed).

## `v0.1/` — the creature on its own (README sections 1–4)

| file | experiment | README |
|---|---|---|
| `day_s1.json` | one simulated day alone, no input (`anima experiment spontaneous`); downsampled to 1 s — see its `note` | §1 |
| `foraging_{plastic,frozen}_s{1,2}.json` | foraging with plasticity on / frozen (`anima experiment foraging`) | §2 |
| `babble_{contingent,yoked,frozen}_s{1,2}.json` | smiles after "m" syllables vs random smiles vs frozen (`anima experiment babble`) | §3 |
| `imitation_{learned,frozen}_s{1,2}.json` | echoing after babbling alone (`anima experiment imitation`) | §4 |
| `summary.json` | the numbers quoted in the README, collected | §1–4 |

## `tutoring/` — what a teacher adds (README section 5)

`anima experiment tutoring`: per seed, one creature raised through the four
lessons by the scripted teacher (same tools and validation as an LLM
teacher), one living alone for the same time; both examined at 9 am the
next morning with plasticity off.

| file | contents |
|---|---|
| `tutored_seed{1,2}.json` | exam results (`forage`, `babble_share`, `ask_hungry`/`ask_sated`, `echo_rate`/`echo_chance`), the graduation record (`graduated`, `milestones`), exams taken during training, and a skill trace sampled every ~30 s |
| `alone_seed{1,2}.json` | the same exam for the creature that lived alone |
| `training_seed{1,2}.log` | everything the tutor printed while teaching: lessons, exams (✎), graduations (★), progress |

## `ask_lesson/` — why "ask for food" is optional (README honest limits)

Seed 1, branched from the start of the experimental *ask* lesson and trained
for ~21 more minutes: calls made when hungry are answered with food, calls
made when full with a frown (`frown_*`) or a double frown without the smile
(`firm_frown_*`).  The `.json` rows are the ask exams taken every ~3 minutes
(share of "m" held hungry vs sated); the `.log` files are the run output.
The share of "m" stayed about the same hungry and sated.

## `forage_measure/` — why foraging is tested by exam

An early version scored foraging passively ("did it move toward the nearest
food?").  These logs compare that score with the probe (food held left /
ahead / right, plasticity off) every 2.5 minutes.  In the experiment's
setting the probe rose to 0.70–0.77 while the passive score stayed at
0.33–0.35; in the tutor's setting the probe reached 0.63–0.66 and the
passive score 0.44–0.50.  With several pieces of food in the arena the
smell usually comes from several directions at once, so natural moves say
little about whether it can follow its nose — hence exams on a clone of the
creature.

## `echo_order/` — why "copy me" comes before "make the sound"

Runs of an earlier five-lesson curriculum that taught the target sound
first.  Once "m" dominated its babbling, the echo rate sank to 0.05–0.10:
seed 1 needed 28 creature-minutes (a night included) to scrape past the
0.15 bar, and seed 2 was still at 0.08 after 27 minutes when the run was
stopped.  With copying taught first, both passed at about 0.2 within 6–8
awake minutes (`tutoring/training_seed*.log`).

## `sessions/` — what `anima train` looks like

A real session with the scripted teacher (`anima train --teacher scripted
--minutes 8.5 --fast`) and its continuation two creature-minutes later,
including the forage exams and the first graduation.

## `programs/` — programs it learns to use, and whether its thoughts are true (README sections 6–7)

`anima experiment programs --workers 2` (about 40 minutes on the 2-core VM),
run in two parts: voice and mind first, then the hands part again with a
2-hour day (`--parts hands`) — the first hands run used the default 20-minute
day and slept through minutes 13–19 (same outcome; see `docs/NOTES.md`).

| file | contents |
|---|---|
| `programs.json` | everything below, as the experiment returned it |
| `hands_{plastic,frozen}_s{1,2}.json` | per run: which program paid, reaches per hand in 2-minute bins, program stats, the urge and inclination checks |
| `voice_{plastic,frozen}_s{1,2}.json` | per run: share of "b" (pays) and "d" (does not) syllables in 2-minute bins |
| `mind_s{1,2}.json` | the thought-log checks after 10 minutes of foraging practice: inclinations vs what it did in the next 3 s, silent echoes vs what it heard, thoughts per minute, and the first 80 lines of its log |
| `thoughts_sample.txt` | a stretch of a creature's thought log, with the example programs of `examples/tools.toml` |

## `colony/` — creatures together, and a genetic algorithm (README section 8)

`anima experiment colony --days 4 --seeds 1,2 --workers 2 --keep keep`
(about 3.5 hours on the 2-core VM; it was run in three parts,
`--policies evolving`, `random` and `static`, and put together with
`anima experiment colony --collect keep`): per seed, three colonies of four
young creatures with varied genes, identical until the first evening.  In
the *evolving* colony the day's best, when it stands out, is followed by
itself and three variations born next to it; in the *random* colony the
same happens on the same rule, but the parent is picked at random (the
control for selection); in the *static* colony nobody is ever replaced.
Days of 10 minutes (`--seconds 600`), 600 cortical neurons per creature.

| file | contents |
|---|---|
| `colony.json` | per colony and day: mean / best / worst food eaten, feasts, share of food from feasts, share of meals contested, food that withered, time spent near another creature, mean energy, the creatures' steering toward food and toward each other read from their synapses at the end of the day, whether a new generation followed, who its parent was and why (the decision's numbers), each creature's food; mean genes of the creatures alive each day; a summary per policy; `repeatability` — one creature's food on a day against the next day, in the static colonies |
| `keep/colony_{evolving,random,static}_s{1,2}.json` | each colony's own `colony.json` at the end: every creature's parent, generation, genes, mutations and day-by-day statistics, and every day's report |
| `arena_session.txt`, `arena_status.txt` | the `anima arena` session shown in the README, and `anima arena --status` after it |
| `first_run/` | the first run (seed 1, two colonies), before food withered and before children were born next to their parent: the static colony starved.  `colony_{static,evolving}_s1.json`, and `diagnosis.json` — the creatures' steering, how they foraged alone, and what happened to the starving herd when their sense of each other was silenced, the food moved, or they were scattered (docs/NOTES.md, v0.4, lessons 6–7) |
