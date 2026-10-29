"""Command line interface.

    anima chat                 talk to a creature in your terminal (any OS)
    anima live                 live dashboard: world, drives, spikes, chat (curses)
    anima train                a local LLM (Ollama) raises it through a curriculum
    anima tutor-check          check that Ollama and the model work as a teacher
    anima run --seconds 600    let it live headless, print what it does
    anima experiment foraging  reproduce the learning experiments
    anima spec --out my.toml   dump the default brain spec to edit
    anima info anima_brain.npz describe a saved creature

The creature is saved to ``anima_brain.npz`` (or ``--brain PATH``) when you
leave and resumed next time, so it keeps its memories and learned habits.
Use ``--new`` to start a fresh one.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__


def _make_agent(args: argparse.Namespace):
    from .brain.builder import load_spec
    from .runtime.agent import Agent

    path = Path(args.brain) if getattr(args, "brain", None) else None
    if path is not None and path.exists() and not args.new:
        agent = Agent.load(path)
        print(f"resumed {path} (age {agent.t_ms / 1000:.0f}s of simulated life)")
        _print_schooling(path)
        return agent, path
    kwargs: dict[str, Any] = {}
    spec = load_spec(args.spec) if getattr(args, "spec", None) else None
    if spec is None:
        kwargs["cortex_size"] = args.cortex
        kwargs["day_length_s"] = args.day
    agent = Agent.create(seed=args.seed, world=args.world, spec=spec, **kwargs)
    return agent, path


def _print_schooling(path: Path) -> None:
    """One line about its lessons, if a tutor has taught this creature."""
    from .tutor import tutor_path
    tp = tutor_path(path)
    if not tp.exists():
        return
    try:
        cur = json.loads(tp.read_text(encoding="utf-8")).get("curriculum", {})
    except (OSError, ValueError):
        return
    from .tutor import LESSON_NAMES
    names = cur.get("lessons") or list(LESSON_NAMES)
    passed = [g["lesson"] for g in cur.get("graduated", []) if g.get("mastered", True)]
    if cur.get("finished"):
        status = f"finished all {len(names)} lessons (mastered: {', '.join(passed) or 'none'}) — it lives on its own"
    else:
        i = int(cur.get("index", 0))
        name = names[i] if i < len(names) else "?"
        status = f"in lesson {i + 1} of {len(names)} ({name})" + (
            f", mastered: {', '.join(passed)}" if passed else "")
    print(f"  schooling: {status}  [{tp.name}]")


def _common(p: argparse.ArgumentParser, brain_default: str | None = "anima_brain.npz") -> None:
    p.add_argument("--brain", default=brain_default,
                   help="file the creature is loaded from / saved to (default: %(default)s)")
    p.add_argument("--new", action="store_true", help="start a new creature even if the file exists")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--world", choices=["foraging", "void"], default="foraging")
    p.add_argument("--spec", help="build the brain from a TOML/JSON spec instead of the default")
    p.add_argument("--cortex", type=int, default=1000, help="cortical neurons (default 1000)")
    p.add_argument("--day", type=float, default=1200.0,
                   help="seconds per simulated day-night cycle (default 1200)")
    p.add_argument("--speed", type=float, default=1.0, help="simulated seconds per real second")


def cmd_chat(args: argparse.Namespace) -> None:
    from .ui.chat import run_chat
    agent, path = _make_agent(args)
    run_chat(agent, path, speed=args.speed, show_events=not args.quiet)


def cmd_live(args: argparse.Namespace) -> None:
    try:
        from .ui.tui import run_dashboard
    except ImportError as exc:  # pragma: no cover - Windows without windows-curses
        sys.exit(f"the live dashboard needs curses ({exc}); on Windows: pip install windows-curses. "
                 "`anima chat` works everywhere.")
    agent, path = _make_agent(args)
    run_dashboard(agent, path, speed=args.speed)


def cmd_run(args: argparse.Namespace) -> None:
    agent, path = _make_agent(args)

    def show(e: dict[str, Any]) -> None:
        k = e["kind"]
        if k == "utterance":
            print(f"{e['t']:8.1f}s  says: {e['text']}")
        elif k == "stage":
            print(f"{e['t']:8.1f}s  -> {e['stage']}")
        elif k == "ate":
            print(f"{e['t']:8.1f}s  eats")
        elif k == "replay" and args.verbose:
            print(f"{e['t']:8.1f}s  replays a memory (reward {e['reward']:+.2f})")

    agent.on("*", show)
    t0 = time.time()
    end = agent.t_ms / 1000.0 + args.seconds
    report = max(10.0, args.seconds / 10)
    while agent.t_ms / 1000.0 < end:
        agent.run(min(report, end - agent.t_ms / 1000.0), realtime=args.realtime, speed=args.speed)
        print("   " + agent.describe())
    el = time.time() - t0
    print(f"lived {args.seconds:.0f}s in {el:.1f}s wall ({args.seconds / el:.2f}x realtime); "
          f"stats {agent.body.stats}")
    if path is not None:
        print(f"saved {agent.save(path)}")


def cmd_experiment(args: argparse.Namespace) -> None:
    seeds = tuple(int(s) for s in args.seeds.split(","))
    secs = args.seconds
    if args.name == "foraging":
        from .experiments.foraging import main
        res = main(seconds=secs or 600.0, seeds=seeds, plot=args.plot)
    elif args.name == "babble":
        from .experiments.babble import main
        res = main(seconds=secs or 600.0, seeds=seeds, plot=args.plot)
    elif args.name == "imitation":
        from .experiments.imitation import main
        res = main(seconds=secs or 300.0, seeds=seeds, plot=args.plot)
    elif args.name == "spontaneous":
        from .experiments.spontaneous import main
        res = main(seconds=secs, seed=seeds[0], plot=args.plot)  # default: one full day
    elif args.name == "tutoring":
        from .experiments.tutoring import main
        res = main(seeds=seeds, max_minutes=(secs or 7200.0) / 60.0, plot=args.plot)
    else:  # pragma: no cover
        sys.exit(f"unknown experiment {args.name}")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2, default=float))
        print(f"wrote {args.json}")


def _think_flag(value: str) -> bool | None:
    return {"auto": None, "on": True, "off": False}[value]


def _pull_hint(client) -> str:
    similar = client.similar_models()
    if similar:
        return f"  use one you have, e.g. --model {similar[0]}, or:  ollama pull {client.model}"
    return f"  ollama pull {client.model}   (or pick one from `ollama list` with --model)"


def _make_teacher(args: argparse.Namespace):
    """The teacher for ``anima train``: an Ollama model (checked first) or the script."""
    from .tutor import OllamaClient, OllamaTeacher, OllamaUnavailable, ScriptedTeacher

    if args.teacher == "scripted":
        return ScriptedTeacher(seed=args.seed), "scripted teacher (no LLM)"
    client = OllamaClient(model=args.model, host=args.host, think=_think_flag(args.think))
    try:
        ver = client.version()
        if not client.has_model():
            sys.exit(f"Ollama {ver} is running at {client.host}, but the model {args.model!r} is not "
                     f"pulled.\n{_pull_hint(client)}\nor train without an LLM:  --teacher scripted")
    except OllamaUnavailable as exc:
        sys.exit(f"{exc}\nOr train without an LLM:  anima train --teacher scripted")
    fallback = None if args.no_fallback else ScriptedTeacher(seed=args.seed)
    return OllamaTeacher(client, fallback=fallback), f"ollama:{args.model} at {client.host} (Ollama {ver})"


def cmd_train(args: argparse.Namespace) -> None:
    import threading

    from .tutor import Tutor, tutor_path

    from .brain.presets import CONSONANTS

    if args.target not in CONSONANTS:
        sys.exit(f"--target must be one of its consonants: {' '.join(CONSONANTS)}")
    teacher, desc = _make_teacher(args)
    agent, path = _make_agent(args)
    if not hasattr(agent.world, "place_food"):
        sys.exit("the lessons need the foraging world (--world foraging)")
    lock = threading.Lock()

    def out(line: str) -> None:
        with lock:
            print(line, flush=True)

    from .tutor import LESSONS, LESSONS_WITH_ASK, Curriculum
    curriculum = Curriculum(LESSONS_WITH_ASK if args.with_ask else LESSONS)
    tutor = Tutor(agent, teacher, target=args.target, think_every_s=args.think_every,
                  printer=out, verbose=not args.quiet, curriculum=curriculum)
    tpath = tutor_path(path) if path is not None else None
    if tpath is not None and tpath.exists() and not args.new:
        tutor.load(tpath)
        out(f"resumed teaching from {tpath}")
    cur = tutor.curriculum
    out(f"teacher: {desc}")
    if cur.finished:
        out("this creature has already passed the whole curriculum — it can live on its own:  "
            f"anima live --brain {path}")
        out(tutor.report())
        return
    lesson = cur.current
    out(f"lesson {cur.index + 1} of {len(cur.lessons)}: {lesson.title} — {lesson.goal}")
    if args.with_ask and "ask" not in [x.name for x in cur.lessons]:
        out("(--with-ask ignored: this creature's schooling started without it)")
    pace = "as fast as the CPU allows" if args.fast else f"at {args.speed:g}x real time"
    out(f"training for up to {args.minutes:g} creature-minutes {pace}; Ctrl-C stops (progress is saved)\n")
    if not args.quiet:
        name = "\033[36manima ›\033[0m" if sys.stdout.isatty() else "anima ›"
        agent.on("utterance", lambda e: out(f"      {name} {e['text']}"))

    def save() -> None:
        if path is not None:
            agent.save(path)
            tutor.save(tpath)

    report = None
    try:
        report = tutor.train(args.minutes, speed=args.speed, realtime=not args.fast,
                             report_every_min=args.report_every, checkpoint=save)
    except KeyboardInterrupt:
        out("\nstopping…")
    finally:
        tutor.stop()
        agent.stop()
        save()
    out("\n" + (report or tutor.report()))
    if path is not None:
        nxt = ("it can live on its own now:  anima live" if cur.finished else
               "run the same command again to continue the lessons, or meet it:  anima chat")
        out(f"saved {path} and {tpath} — {nxt} --brain {path}")


def cmd_tutor_check(args: argparse.Namespace) -> None:
    from .runtime.agent import Agent
    from .tutor import (DECISION_SCHEMA, OllamaClient, OllamaError, OllamaTeacher, OllamaUnavailable,
                        Tutor, validate)

    client = OllamaClient(model=args.model, host=args.host, think=_think_flag(args.think))
    try:
        ver = client.version()
    except OllamaUnavailable as exc:
        sys.exit(f"✗ {exc}")
    print(f"✓ Ollama {ver} at {client.host}")
    models = client.models()
    print(f"  installed models: {', '.join(models) if models else '(none — try: ollama pull llama3.2)'}")
    if not client.has_model():
        sys.exit(f"✗ model {args.model!r} is not pulled\n{_pull_hint(client)}")
    print(f"✓ model {args.model} is available")
    agent = Agent.create(seed=1)
    tutor = Tutor(agent, OllamaTeacher(client, fallback=None), printer=None, verbose=False)
    agent.run(5.0)
    obs = tutor.observe()
    msgs = tutor.teacher.messages(obs)
    print(f"  asking for {args.trials} teaching decision(s) about a newborn creature "
          f"(prompt ≈ {sum(len(m['content']) for m in msgs) // 4} tokens)…")
    times, good = [], 0
    for i in range(args.trials):
        try:
            raw, res = client.chat_json(msgs, schema=DECISION_SCHEMA)
        except (OllamaError, ValueError) as exc:
            print(f"✗ decision {i + 1}: {exc}")
            continue
        d = validate(raw, obs, client.model)
        times.append(res.seconds)
        ok = not d.rejected and d.thought != "(no thought given)"
        good += ok
        rej = f"  (rejected: {'; '.join(d.rejected)})" if d.rejected else ""
        print(f"{'✓' if ok else '~'} decision {i + 1} in {res.seconds:.1f}s: {d.thought!r} → {d.summary()}{rej}")
    if not times:
        sys.exit("✗ the model did not produce a usable decision; training would fall back to the "
                 "scripted teacher. Try another model (--model).")
    lat = min(times)
    print(f"\n{good}/{args.trials} clean decisions; fastest {lat:.1f}s. The creature never waits for the "
          f"teacher, so any speed works; with this latency use --think-every {max(8, round(2 * lat))} or more.")
    print(f"ready:  anima train --model {args.model}" + (f" --host {args.host}" if args.host else ""))


def cmd_spec(args: argparse.Namespace) -> None:
    from .brain.builder import save_spec
    from .brain.presets import PRESETS
    spec = PRESETS[args.preset](seed=args.seed)
    if args.out:
        save_spec(spec, args.out)
        print(f"wrote {args.out}: {len(spec['population'])} populations, "
              f"{len(spec['projection'])} projections")
    else:
        from .brain.builder import dumps_toml
        print(dumps_toml(spec))


def cmd_info(args: argparse.Namespace) -> None:
    from .runtime.agent import Agent
    agent = Agent.load(args.file)
    s = agent.latest
    print(f"{args.file}: {agent.brain.net.N} neurons, age {s['sim_time_s']:.0f}s, "
          f"{s['clock']} {s['stage']}")
    print("  " + agent.describe())
    print(f"  lifetime: {s['stats']}, memories {s['memories']}, replays {s['replays']}")
    hist = list(agent.body.voice.history)[-12:] if agent.body.voice else []
    if hist:
        print("  last syllables:", " ".join(x for _, x in hist))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="anima", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"anima {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("chat", help="talk to the creature in a plain terminal")
    _common(p)
    p.add_argument("--quiet", action="store_true", help="only show speech")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("live", help="live dashboard (curses)")
    _common(p)
    p.set_defaults(func=cmd_live)

    p = sub.add_parser("train", help="raise it with a teacher: a local LLM via Ollama, or a script",
                       description="A teacher raises the creature through four lessons (find food, copy "
                       "sounds, make one sound, live alone; --with-ask adds asking for food). The LLM decides what to teach and "
                       "says, rewards and feeds it; fast reflexes make rewards contingent; graduation is "
                       "measured. Progress is saved next to the brain file, so training builds up over "
                       "sessions.")
    _common(p)
    p.add_argument("--teacher", choices=["ollama", "scripted"], default="ollama")
    p.add_argument("--model", default="llama3.2", help="Ollama model (default: %(default)s)")
    p.add_argument("--host", default=None, help="Ollama server (default: $OLLAMA_HOST or 127.0.0.1:11434)")
    p.add_argument("--think", choices=["auto", "on", "off"], default="auto",
                   help="reasoning for thinking models (off is faster); auto = model default")
    p.add_argument("--minutes", type=float, default=30.0,
                   help="creature-minutes to train this session (default %(default)s)")
    p.add_argument("--fast", action="store_true", help="do not pace to the wall clock")
    p.add_argument("--think-every", type=float, default=8.0,
                   help="creature seconds between teacher decisions (default %(default)s)")
    p.add_argument("--target", default="m", help="the sound it learns to make and ask with (default m)")
    p.add_argument("--report-every", type=float, default=5.0, help="progress report / autosave interval (min)")
    p.add_argument("--no-fallback", action="store_true",
                   help="if the LLM fails, do nothing that turn instead of using the scripted teacher")
    p.add_argument("--with-ask", action="store_true",
                   help="add the experimental 'ask for food' lesson (see README: honest limits)")
    p.add_argument("--quiet", action="store_true", help="only lessons, milestones and progress")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("tutor-check", help="check Ollama and a model for `anima train`")
    p.add_argument("--model", default="llama3.2")
    p.add_argument("--host", default=None)
    p.add_argument("--think", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--trials", type=int, default=2)
    p.set_defaults(func=cmd_tutor_check)

    p = sub.add_parser("run", help="live headless for a while")
    _common(p, brain_default=None)
    p.add_argument("--seconds", type=float, default=120.0)
    p.add_argument("--realtime", action="store_true", help="pace to the wall clock")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("experiment", help="reproduce the learning experiments")
    p.add_argument("name", choices=["foraging", "babble", "imitation", "spontaneous", "tutoring"])
    p.add_argument("--seconds", type=float, default=None,
                   help="simulated seconds (default: 600; imitation 300 per phase; spontaneous one day; "
                        "tutoring: at most 7200 for the curriculum)")
    p.add_argument("--seeds", default="1,2")
    p.add_argument("--plot", help="save a figure (needs matplotlib)")
    p.add_argument("--json", help="save the numbers")
    p.set_defaults(func=cmd_experiment)

    p = sub.add_parser("spec", help="print or save a brain spec")
    p.add_argument("--preset", default="infant", choices=["infant", "minimal"])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out")
    p.set_defaults(func=cmd_spec)

    p = sub.add_parser("info", help="describe a saved creature")
    p.add_argument("file")
    p.set_defaults(func=cmd_info)

    args = ap.parse_args(argv)
    try:
        args.func(args)
    except BrokenPipeError:  # e.g. `anima run | head`: the reader went away
        import os
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        sys.exit(0)


if __name__ == "__main__":  # pragma: no cover
    main()
