"""Another program talking to a living creature through ``anima serve``.

    anima serve --tools examples/tools_remote.toml           # terminal 1 (any creature)
    python examples/05_external_program.py                   # terminal 2

This script is the creature's bell.  Whenever the creature says a syllable
starting with "b", anima publishes the call and this program answers it: the
bell rings (the creature hears "ding") and, the lonelier the creature is,
the more it is rewarded.  Meanwhile it prints what the creature says and
thinks, and now and then says "ba ba" to it — which it may echo, ringing the
bell.  Only HTTP and JSON are used, so any language could do the same.

Without a running ``anima serve`` (or with ``--demo``) it starts its own
creature in this process, so you can watch it work in one terminal.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anima.bridge import BridgeClient, BridgeError  # noqa: E402


def ring(call: dict) -> dict:
    """The bell program: what happens when the creature rings."""
    lonely = float(call.get("drives", {}).get("loneliness", 0.0))
    return {"say": "ding", "reward": round(0.2 + 0.5 * lonely, 2)}


def demo_server():
    """A creature with a remote bell, served on a free local port."""
    from anima import Agent
    from anima.bridge import Bridge
    from anima.mind import ThoughtStream
    from anima.tools import RemoteTool, ToolBelt

    agent = Agent.create(seed=1)
    belt = ToolBelt(agent, [RemoteTool(name="bell", say="b", timeout_s=5.0, cooldown_s=4.0)])
    mind = ThoughtStream(agent)
    bridge = Bridge(agent, port=0, thoughts=mind, belt=belt).start()
    agent.start(speed=1.0)
    return bridge, lambda: (bridge.stop(), agent.stop(), belt.close(), mind.close())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:7777")
    ap.add_argument("--token", default=None)
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--demo", action="store_true", help="start a creature in this process")
    args = ap.parse_args()

    shutdown = None
    client = BridgeClient(args.url, args.token)
    if not args.demo:
        try:
            info = client.info()
        except BridgeError:
            print(f"no creature at {args.url}; starting one here (--demo)")
            args.demo = True
    if args.demo:
        bridge, shutdown = demo_server()
        client = BridgeClient(bridge.url)
        info = client.info()
    print(f"connected to {info['name']} ({info['neurons']} neurons), programs: {', '.join(info['tools']) or 'none'}")
    if not any(t.startswith("bell ") for t in info["tools"]):
        print("  (it has no 'bell' program: start it with --tools examples/tools_remote.toml)")

    end = time.time() + args.minutes * 60
    done = lambda: time.time() > end  # noqa: E731
    rung = []
    threading.Thread(target=client.serve, args=({"bell": ring},),
                     kwargs={"stop": done, "on_call": lambda c, r: rung.append(r)}, daemon=True).start()
    last_talk = 0.0
    try:
        for e in client.follow(kinds=["utterance", "thought", "tool_result"], stop=done):
            k = e["kind"]
            if k == "utterance":
                print(f"  anima says   {e['text']}")
            elif k == "thought" and e.get("type") not in ("speech", "social"):
                print(f"  anima thinks {e['text']}")
            elif k == "tool_result":
                print(f"  bell         {e['summary']}")
            if time.time() - last_talk > 20:
                client.hear("ba ba")
                print("  we say       ba ba")
                last_talk = time.time()
    except KeyboardInterrupt:
        pass
    print(f"\nthe bell rang {len(rung)} times; its stats: {client.tools()['stats']}")
    if shutdown is not None:
        shutdown()


if __name__ == "__main__":
    main()
