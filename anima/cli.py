"""Command line interface.

    anima chat                 talk to a creature in your terminal (any OS)
    anima live                 live dashboard: world, drives, spikes, thoughts, chat (curses)
    anima serve                live in real time with a local HTTP connection for other programs
    anima thoughts --follow    read the log of its thoughts
    anima tools FILE           check the programs it may use (--try NAME runs one)
    anima arena --days 3       several creatures share an arena; the best of a day and its variations go on
    anima train                a local LLM (Ollama) raises it through a curriculum
    anima tutor-check          check that Ollama and the model work as a teacher
    anima run --seconds 600    let it live headless, print what it does
    anima experiment foraging  reproduce the learning experiments
    anima spec --out my.toml   dump the default brain spec to edit
    anima info anima_brain.npz describe a saved creature

The creature is saved to ``anima_brain.npz`` (or ``--brain PATH``) when you
leave and resumed next time, so it keeps its memories and learned habits;
its thoughts are logged next to it (``anima_brain.thoughts.jsonl``).  Use
``--new`` to start a fresh one, ``--tools FILE`` to give it programs to use.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__


def _load_tools(args: argparse.Namespace) -> list:
    """The programs named by --tools (read once, kept on ``args``)."""
    if not getattr(args, "tools", None):
        return []
    if getattr(args, "_tools", None) is None:
        from .tools import load_tools
        try:
            args._tools = load_tools(args.tools)
        except (OSError, ValueError, RuntimeError, ImportError, AttributeError, TypeError) as exc:
            sys.exit(f"--tools {args.tools}: {exc}")
    return args._tools


def _make_agent(args: argparse.Namespace):
    from .brain.builder import load_spec
    from .runtime.agent import Agent

    tools = _load_tools(args)
    need = max([t.hand + 1 for t in tools if t.hand is not None], default=0)
    hands = getattr(args, "hands", None)
    path = Path(args.brain) if getattr(args, "brain", None) else None
    if path is not None and path.exists() and not args.new:
        agent = Agent.load(path)
        print(f"resumed {path} (age {agent.t_ms / 1000:.0f}s of simulated life)")
        _print_schooling(path)
        have = len(agent.body.hands.labels) if agent.body.hands is not None else 0
        if hands is not None and hands != have:
            print(f"  (--hands only shapes a new creature; this one has {have})")
        return agent, path
    kwargs: dict[str, Any] = {}
    spec = load_spec(args.spec) if getattr(args, "spec", None) else None
    if spec is None:
        kwargs["cortex_size"] = args.cortex
        kwargs["day_length_s"] = args.day
        if hands is None:
            hands = max(4, need) if need else 0
        if hands:
            kwargs["hands"] = hands
    agent = Agent.create(seed=args.seed, world=args.world, spec=spec, **kwargs)
    return agent, path


def _attach(agent, path: Path | None, args: argparse.Namespace, quiet: bool = False):
    """Give the creature its programs (--tools) and start its thought log."""
    belt = mind = None
    tools = _load_tools(args)
    if tools:
        from .tools import ToolBelt
        try:
            belt = ToolBelt(agent, tools)
        except ValueError as exc:
            sys.exit(f"--tools {args.tools}: {exc}")
        if not quiet:
            print(f"  programs: {belt.describe()}")
    if not getattr(args, "no_thoughts", False):
        from .mind import ThoughtStream, thoughts_path
        tp = thoughts_path(path) if path is not None else None
        mind = ThoughtStream(agent, tp, tool_names=belt.hand_names() if belt else None)
        if tp is not None and not quiet:
            print(f"  thoughts: {tp}  (anima thoughts --brain {path} --follow)")
    return belt, mind


def _detach(belt, mind) -> None:
    if belt is not None:
        belt.close()
    if mind is not None:
        mind.close()


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
    p.add_argument("--tools", help="programs it may use: a TOML or JSON file (see examples/tools.toml)")
    p.add_argument("--hands", type=int, default=None,
                   help="hands for a new creature (default: 4 if --tools binds programs to hands, else none)")
    p.add_argument("--no-thoughts", action="store_true", help="do not keep a log of its thoughts")


def cmd_chat(args: argparse.Namespace) -> None:
    from .ui.chat import run_chat
    agent, path = _make_agent(args)
    belt, mind = _attach(agent, path, args)
    try:
        run_chat(agent, path, speed=args.speed, show_events=not args.quiet, belt=belt, mind=mind,
                 show_thoughts=args.thoughts)
    finally:
        _detach(belt, mind)


def cmd_live(args: argparse.Namespace) -> None:
    try:
        from .ui.tui import run_dashboard
    except ImportError as exc:  # pragma: no cover - Windows without windows-curses
        sys.exit(f"the live dashboard needs curses ({exc}); on Windows: pip install windows-curses. "
                 "`anima chat` works everywhere.")
    agent, path = _make_agent(args)
    belt, mind = _attach(agent, path, args, quiet=True)
    try:
        run_dashboard(agent, path, speed=args.speed, belt=belt, mind=mind)
    finally:
        _detach(belt, mind)


def cmd_run(args: argparse.Namespace) -> None:
    from .mind import format_thought
    agent, path = _make_agent(args)
    belt, mind = _attach(agent, path, args)
    color = sys.stdout.isatty()

    def show(e: dict[str, Any]) -> None:
        k = e["kind"]
        if args.thoughts:
            if k == "thought":
                print(f"{e['t']:8.1f}s  " + format_thought({**e, "kind": e["type"]}, color))
            return
        if k == "utterance":
            print(f"{e['t']:8.1f}s  says: {e['text']}")
        elif k == "stage":
            print(f"{e['t']:8.1f}s  -> {e['stage']}")
        elif k == "ate":
            print(f"{e['t']:8.1f}s  eats")
        elif k == "tool_result":
            print(f"{e['t']:8.1f}s  uses {e['tool']}: {e['summary']}")
        elif k == "replay" and args.verbose:
            print(f"{e['t']:8.1f}s  replays a memory (reward {e['reward']:+.2f})")

    if args.thoughts and mind is None:
        sys.exit("--thoughts needs the thought log (drop --no-thoughts)")
    agent.on("*", show)
    t0 = time.time()
    end = agent.t_ms / 1000.0 + args.seconds
    report = max(10.0, args.seconds / 10)
    try:
        while agent.t_ms / 1000.0 < end:
            agent.run(min(report, end - agent.t_ms / 1000.0), realtime=args.realtime, speed=args.speed)
            print("   " + agent.describe())
    except KeyboardInterrupt:
        print("stopping…")
    el = time.time() - t0
    lived = args.seconds - max(0.0, end - agent.t_ms / 1000.0)
    print(f"lived {lived:.0f}s in {el:.1f}s wall ({lived / max(el, 1e-9):.2f}x realtime); "
          f"stats {agent.body.stats}")
    if belt is not None:
        print("programs:\n" + belt.report())
    if mind is not None:
        st = mind.stats()
        print(f"thoughts: {st['total']} ({', '.join(f'{k} {v}' for k, v in st['counts'].items())})")
    _detach(belt, mind)
    if path is not None:
        print(f"saved {agent.save(path)}")


def cmd_serve(args: argparse.Namespace) -> None:
    from .bridge import Bridge
    agent, path = _make_agent(args)
    belt, mind = _attach(agent, path, args)
    try:
        bridge = Bridge(agent, host=args.host, port=args.port, token=args.token, thoughts=mind, belt=belt)
    except (ValueError, OSError) as exc:
        _detach(belt, mind)
        sys.exit(f"cannot serve on {args.host}:{args.port}: {exc}")
    bridge.start()
    agent.start(speed=args.speed)
    auth = f' -H "Authorization: Bearer {args.token}"' if args.token else ""
    print(f"anima is alive and listening on {bridge.url}  ({agent.brain.net.N} neurons, "
          f"{agent.latest['clock']} {agent.latest['stage']})")
    print(f"  curl{auth} {bridge.url}/state")
    print(f"  curl{auth} -N '{bridge.url}/stream?kinds=utterance,thought'")
    print(f"  curl{auth} -d '{{\"text\": \"ba ba :)\"}}' {bridge.url}/hear")
    print("Ctrl-C saves and stops." if path is not None else "Ctrl-C stops.")
    import signal

    def _stop(signum: int, frame: Any) -> None:  # e.g. a service manager's SIGTERM: save, then stop
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, _stop)
    except ValueError:  # pragma: no cover - not the main thread
        pass
    last_save = time.time()
    try:
        while agent.alive:
            time.sleep(0.5)
            if path is not None and args.autosave and time.time() - last_save >= args.autosave * 60:
                agent.save(path)
                last_save = time.time()
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        bridge.stop()
        agent.stop()
        _detach(belt, mind)
        if path is not None:
            print(f"saved {agent.save(path)} — run again to continue its life.")


def cmd_arena(args: argparse.Namespace) -> None:
    import shutil
    import signal
    import threading

    from .evolve import Colony, ColonyConfig
    from .mind import format_thought

    d = Path(args.dir)
    exists = (d / "colony.json").exists()
    if args.status:
        if not exists:
            sys.exit(f"no colony at {d} (start one: anima arena --dir {d})")
        col = Colony.load(d, printer=None)
        print(f"colony {d}: day {col.day} ({col.clock()}), {len(col.members)} creatures alive, "
              f"{len(col.records)} have lived here")
        for slot, cid in enumerate(col.members):
            info, r = col.info(slot), col.records[cid]
            print(f"  {info['name']} ({cid}, generation {r['generation']}, born day {r['born_day']}, {r['origin']}): "
                  f"energy {info['energy']:.2f}, {info['stage']}")
        if col.history:
            print("\n" + col.day_report(col.history[-1]))
        print("\nfamily tree (fitness = food eaten each day, in energy):\n" + col.lineage())
        col.close()
        return
    policy = {"margin": args.margin, "min_gain": args.min_gain, "compare": args.gain_over, "mutation": args.mutation,
              "mutation_rate": args.mutation_rate, "inherit": args.inherit,
              "crossover": True if args.crossover else None, "delete_old": True if args.delete_old else None,
              "thoughts": False if args.no_thoughts else None}
    if exists and args.new:
        if args.delete_old:
            shutil.rmtree(d)
            print(f"deleted the old colony at {d}")
        else:
            k = 1
            while d.with_name(f"{d.name}.old{k}").exists():
                k += 1
            d.rename(d.with_name(f"{d.name}.old{k}"))
            print(f"moved the old colony to {d.with_name(f'{d.name}.old{k}')} (--delete-old deletes it instead)")
        exists = False
    lock = threading.Lock()

    def out(line: str) -> None:
        with lock:
            print(line, flush=True)

    import os
    workers = args.workers if args.workers else min(args.population, os.cpu_count() or 1)
    if exists:
        if args.from_:
            sys.exit(f"{d} already holds a colony; --from starts a new one (add --new)")
        col = Colony.load(d, printer=out, workers=min(workers, 64), **policy)
        print(f"resumed the colony at {d}: day {col.day} ({col.clock()}), "
              + ", ".join(f"{n} (gen {col.records[c]['generation']})" for c, n in zip(col.members, col.names)))
    else:
        cfg = ColonyConfig(population=args.population, seed=args.seed, cortex=args.cortex,
                           day_length_s=args.day, kin=not args.no_kin,
                           **{k: v for k, v in policy.items() if v is not None})
        try:
            col = Colony.create(d, cfg, founder=args.from_, printer=out, workers=workers)
        except (ValueError, OSError, KeyError) as exc:
            sys.exit(f"cannot start a colony: {exc}")
        src = f" from a copy of {args.from_} (the file itself is left alone)" if args.from_ else ""
        print(f"a new colony at {d}{src}: " + ", ".join(col.names)
              + f" ({col.info(0)['neurons']} neurons each, day = {col.cfg.day_length_s:g} s)")
    c = col.cfg
    against = "the others' median" if c.compare == "others" else "it did the day before"
    print(f"selection: a creature that eats {c.margin:.0%} (and {c.min_gain:g}) more than {against} "
          f"makes the next day's population with {c.population - 1} variations of itself "
          f"({'copies of its brain' if c.inherit == 'brain' else 'fresh brains'} with mutated genes"
          f"{', crossed with the runner-up' if c.crossover else ''}); those it replaces are "
          f"{'DELETED' if c.delete_old else 'archived in ' + str(d / 'retired')}.")
    if args.thoughts:
        want = None if args.thoughts == "all" else {n.strip().lower() for n in args.thoughts.split(",")}

        def show(name: str, e: dict[str, Any]) -> None:
            if want is None or name.lower() in want:
                if e.get("type") not in ("speech", "dream", "state"):
                    out(f"  {name:>5}: " + format_thought({**e, "kind": e.get("type", "")}, color=sys.stdout.isatty()))
        col.listen("thought", show)
    if args.watch:
        col.printer = lambda line: out(line + ("\n" + "\n".join(col.arena.render()) if line.startswith("[") else ""))
    stop = threading.Event()

    def halt(signum: int, frame: Any) -> None:
        if stop.is_set():  # a second Ctrl-C: give up waiting
            raise KeyboardInterrupt
        stop.set()
        out("\nstopping at the end of this tick (Ctrl-C again to quit at once)…")
    old = [signal.signal(sg, halt) for sg in (signal.SIGINT, signal.SIGTERM)]
    pace = f"at {args.speed:g}x real time" if args.realtime else "as fast as the CPU allows"
    procs = f" in {col.workers} worker processes" if col.workers > 1 else ""
    print(f"living {args.days} day(s) {pace}{procs}; Ctrl-C saves and stops\n")
    try:
        col.run(days=args.days, realtime=args.realtime, speed=args.speed, stop=stop.is_set,
                report_every_h=args.report_every)
    finally:
        for sg, h in zip((signal.SIGINT, signal.SIGTERM), old):
            signal.signal(sg, h)
        col.save()
        col.close()
    print(f"\nsaved the colony (day {col.day}) at {d} — continue: anima arena --dir {d}   "
          f"family tree: anima arena --dir {d} --status")


def cmd_thoughts(args: argparse.Namespace) -> None:
    from .mind import KINDS, format_thought, read_thoughts, thoughts_path
    src = Path(args.file) if args.file else thoughts_path(args.brain)
    if src.suffix == ".npz":
        src = thoughts_path(src)
    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip()) if args.kinds else None
    bad = [k for k in kinds or () if k not in KINDS]
    if bad:
        sys.exit(f"unknown kind(s) {', '.join(bad)}; kinds: {', '.join(KINDS)}")
    if not src.exists() and not args.follow:
        sys.exit(f"no thought log at {src} (it is written while the creature lives: anima chat / live / serve)")
    color = sys.stdout.isatty() and not args.json

    def emit(rec: dict[str, Any]) -> None:
        print(json.dumps(rec) if args.json else format_thought(rec, color), flush=True)

    if src.exists():
        for rec in read_thoughts(src, args.n, kinds):
            emit(rec)
    if not args.follow:
        return
    pos = src.stat().st_size if src.exists() else 0
    try:
        while True:
            time.sleep(0.5)
            if not src.exists():
                continue
            size = src.stat().st_size
            if size < pos:  # rotated
                pos = 0
            if size == pos:
                continue
            with open(src, encoding="utf-8") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
            for line in chunk.splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if not kinds or rec.get("kind") in kinds:
                    emit(rec)
    except KeyboardInterrupt:
        pass


def cmd_tools(args: argparse.Namespace) -> None:
    from .tools import parse_answer  # noqa: F401  (import check)
    tools = _load_tools(args)
    if not tools:
        sys.exit(f"{args.tools}: no [[tool]] entries")
    print(f"{args.tools}: {len(tools)} program(s)")
    hands = 0
    for t in tools:
        print(f"  {t.describe()}")
        print(f"      cooldown {t.cooldown_s:g}s, timeout {t.timeout_s:g}s"
              + (", heard" if t.speak else ", not heard"))
        if t.hand is not None:
            hands = max(hands, t.hand + 1)
    if hands:
        print(f"hand-bound programs need a creature with at least {hands} hands "
              f"(a new one gets {max(4, hands)}: anima chat --brain pet.npz --new --tools {args.tools})")
    if args.try_:
        match = [t for t in tools if t.name == args.try_]
        if not match:
            sys.exit(f"no program named {args.try_!r}")
        t = match[0]
        call = {"call_id": 0, "tool": t.name, "how": "say" if t.say else "hand", "what": t.say or t.hand,
                "t": 0.0, "clock": "12:00", "stage": "awake", "energy": 0.5,
                "drives": {"hunger": 0.5, "fatigue": 0.2, "boredom": 0.5, "loneliness": 0.5}}
        if t.kind == "remote":
            print(f"{t.name} is answered from outside (POST /tool_result while `anima serve` runs)")
            return
        res = t.run(call)
        print(f"tried {t.name} with a sample call → {res.summary()}  ({res.seconds:.2f}s)")
        if res.raw:
            print("  raw answer: " + res.raw.strip()[:300])


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
    elif args.name == "programs":
        from .experiments.programs import main
        parts = tuple(x for x in (args.parts or "hands,voice,mind").split(",") if x)
        res = main(seconds=secs, seeds=seeds, parts=parts, workers=args.workers, plot=args.plot)
    elif args.name == "colony":
        from .experiments.colony import main
        if args.collect:
            from .experiments.colony import collect
            res = collect(args.collect, plot=args.plot)
        else:
            policies = tuple(x for x in (args.policies or "evolving,random,static").split(",") if x)
            res = main(days=args.days, seeds=seeds, day_length_s=secs or 600.0, workers=args.workers,
                       keep=args.keep, plot=args.plot, policies=policies)
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
    belt, mind = _attach(agent, path, args)
    lock = threading.Lock()

    def out(line: str) -> None:
        with lock:
            print(line, flush=True)

    from .tutor import LESSONS, LESSONS_WITH_ASK, Curriculum
    curriculum = Curriculum(LESSONS_WITH_ASK if args.with_ask else LESSONS)
    tutor = Tutor(agent, teacher, target=args.target, think_every_s=args.think_every,
                  printer=out, verbose=not args.quiet, curriculum=curriculum, mind=mind)
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
        _detach(belt, mind)
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
        _detach(belt, mind)
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
    p.add_argument("--thoughts", action="store_true", help="print its thoughts as they happen (/mind toggles)")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("live", help="live dashboard (curses)")
    _common(p)
    p.set_defaults(func=cmd_live)

    p = sub.add_parser("serve", help="live in real time and let other programs connect (HTTP)",
                       description="Runs the creature in real time with a small JSON web service next to it: "
                       "read its state, thoughts and events, talk to it, reward or feed it, and answer its "
                       "remote programs' calls. See `anima/bridge/server.py` for the endpoints.")
    _common(p)
    p.add_argument("--host", default="127.0.0.1", help="address to listen on (default: %(default)s)")
    p.add_argument("--port", type=int, default=7777)
    p.add_argument("--token", default=None, help="require 'Authorization: Bearer TOKEN' (needed off localhost)")
    p.add_argument("--autosave", type=float, default=5.0, help="save every N minutes (0: only on exit)")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("arena", help="several creatures in one arena: compete, cooperate, evolve",
                       description="Creatures share an arena: they compete for food, can share feasts that one "
                       "alone cannot eat, hear and sense each other. At the end of each day, if one ate clearly "
                       "more than the others, it and variations of it (a genetic algorithm) live the next day. "
                       "The colony is saved in --dir and continues where it stopped.")
    p.add_argument("--dir", default="colony", help="the colony's directory (default: %(default)s)")
    p.add_argument("--days", type=int, default=1, help="days to live this session (default 1)")
    p.add_argument("--new", action="store_true", help="start a new colony (the old one is moved aside, "
                   "or deleted with --delete-old)")
    p.add_argument("--from", dest="from_", metavar="BRAIN", help="start the colony from a copy of this "
                   "creature and variations of it (the file is not changed)")
    p.add_argument("--population", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cortex", type=int, default=600, help="cortical neurons per creature (default 600)")
    p.add_argument("--day", type=float, default=1200.0, help="seconds per day (default 1200)")
    p.add_argument("--no-kin", action="store_true", help="new creatures cannot sense where the others are")
    p.add_argument("--margin", type=float, default=None,
                   help="how much more (fraction) the best must eat than the others' median (default 0.25)")
    p.add_argument("--min-gain", type=float, default=None,
                   help="...and at least this much more food energy (default 0.5 = two meals)")
    p.add_argument("--gain-over", choices=["others", "itself"], default=None,
                   help="what a gain is measured against: the others that day (default), or the creature's "
                        "own previous day (the most improved one is chosen)")
    p.add_argument("--mutation", type=float, default=None, help="size of gene mutations (default 0.2)")
    p.add_argument("--mutation-rate", type=float, default=None, help="chance each gene mutates (default 0.5)")
    p.add_argument("--inherit", choices=["brain", "genes"], default=None,
                   help="children start from a copy of the champion's brain (default) or a fresh brain")
    p.add_argument("--crossover", action="store_true", help="mix the champion's genes with the runner-up's")
    p.add_argument("--delete-old", action="store_true",
                   help="delete creatures that lose their place (and an old colony with --new) instead of "
                        "archiving them")
    p.add_argument("--realtime", action="store_true", help="pace to the wall clock (x --speed)")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--report-every", type=float, default=3.0, help="progress line every N hours of its day")
    p.add_argument("--watch", action="store_true", help="draw the arena with every progress line")
    p.add_argument("--thoughts", metavar="NAMES", help="print these creatures' thoughts as they happen "
                   "(comma-separated names, or all)")
    p.add_argument("--no-thoughts", action="store_true", help="do not keep thought logs")
    p.add_argument("--status", action="store_true", help="show the colony and its family tree, then exit")
    p.add_argument("--workers", type=int, default=0,
                   help="worker processes for the creatures (default: one per creature, up to the CPU count; "
                        "1 = all in this process)")
    p.set_defaults(func=cmd_arena)

    p = sub.add_parser("thoughts", help="read the log of its thoughts")
    p.add_argument("file", nargs="?", help="a .thoughts.jsonl log, or a brain file (default: --brain)")
    p.add_argument("--brain", default="anima_brain.npz")
    p.add_argument("-n", type=int, default=40, help="how many of the latest to show (0: all)")
    p.add_argument("--kinds", help="only these kinds, e.g. intention,feeling,program")
    p.add_argument("--follow", "-f", action="store_true", help="keep printing new thoughts")
    p.add_argument("--json", action="store_true", help="print JSON lines")
    p.set_defaults(func=cmd_thoughts)

    p = sub.add_parser("tools", help="check a tools file (the programs it may use)")
    p.add_argument("tools", help="TOML or JSON file with [[tool]] entries")
    p.add_argument("--try", dest="try_", metavar="NAME", help="run one program once with a sample call")
    p.set_defaults(func=cmd_tools)

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
    p.add_argument("--thoughts", action="store_true", help="print its thoughts instead of its actions")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("experiment", help="reproduce the learning experiments")
    p.add_argument("name", choices=["foraging", "babble", "imitation", "spontaneous", "tutoring", "programs",
                                     "colony"])
    p.add_argument("--seconds", type=float, default=None,
                   help="simulated seconds (default: 600; imitation 300 per phase; spontaneous one day; "
                        "tutoring: at most 7200 for the curriculum; programs: 1200 hands, 600 voice, 480 mind; "
                        "colony: the length of a day, 600)")
    p.add_argument("--parts", help="programs only: any of hands,voice,mind (default all)")
    p.add_argument("--workers", type=int, default=1, help="programs, colony: runs in parallel processes")
    p.add_argument("--days", type=int, default=4, help="colony only: days per colony (default 4)")
    p.add_argument("--keep", help="colony only: copy each colony's colony.json (lineage, reports) here")
    p.add_argument("--policies", help="colony only: any of evolving,random,static (default all three)")
    p.add_argument("--collect", metavar="DIR", help="colony only: report on colonies kept earlier in DIR (with "
                   "--keep) instead of running new ones - to run a long experiment in parts")
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
