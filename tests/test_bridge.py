"""The local HTTP connection: other programs read, talk to and answer the creature."""

import json
import threading
import time
import urllib.request

import pytest

from anima import Agent
from anima.bridge import Bridge, BridgeClient, BridgeError
from anima.mind import ThoughtStream
from anima.tools import RemoteTool, ToolBelt


@pytest.fixture()
def served():
    a = Agent.create(seed=3, cortex_size=300)
    # bound to a syllable it cannot say, so only the tests ring it
    belt = ToolBelt(a, [RemoteTool(name="bell", say="qx", timeout_s=5.0, cooldown_s=0.0)])
    mind = ThoughtStream(a, None)
    bridge = Bridge(a, port=0, thoughts=mind, belt=belt).start()
    a.start(realtime=False)  # as fast as it goes
    yield a, bridge, belt, BridgeClient(bridge.url, timeout=5.0)
    bridge.stop()
    a.stop()
    belt.close()
    mind.close()


def _until(cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_info_state_and_thoughts(served):
    a, bridge, _, c = served
    info = c.info()
    assert info["alive"] and info["thoughts"] and info["tools"] == ["bell (remote, say 'qx')"]
    st = c.state()
    assert {"drives", "modulators", "stage", "clock"} <= set(st)
    assert _until(lambda: c.thoughts(5))
    assert all("text" in t for t in c.thoughts(5, kinds=["perception", "intention", "speech"]))


def test_hear_reward_and_feed_reach_the_creature(served):
    a, bridge, _, c = served
    start = c.events(0, ["-"])["last"]
    c.hear("ba ba :)")
    res = c.events(start, ["heard"], wait=5.0)
    assert res["events"][0]["text"] == "ba ba :)"
    assert c.reward(7.0, "test") == 1.0  # clipped
    assert _until(lambda: any(src == "bridge:test" for _, _, src in list(a.brain.reward_log)))
    n = len(a.world.food)
    c.feed("left")
    assert _until(lambda: len(a.world.food) == n + 1)
    with pytest.raises(BridgeError, match="400"):
        c.feed("upstairs")
    with pytest.raises(BridgeError, match="400"):
        c.request("POST", "/reward", {"amount": "lots"})
    with pytest.raises(BridgeError, match="404"):
        c.request("GET", "/nowhere")


def test_a_remote_program_answers_through_the_bridge(served):
    a, bridge, belt, c = served
    start = c.events(0, ["-"])["last"]
    a.call_soon(lambda ag: belt._on_syllable({"text": "qx"}))
    calls = c.events(start, ["tool_call"], wait=5.0)["events"]
    assert calls and calls[0]["tool"] == "bell" and calls[0]["program"] == "remote"
    assert c.pending()[0]["call_id"] == calls[0]["call_id"]
    assert c.tool_result(calls[0]["call_id"], {"say": "ding", "reward": 0.3})
    assert not c.tool_result(calls[0]["call_id"], "twice")  # already answered
    res = c.events(calls[0]["seq"], ["tool_result"], wait=5.0)["events"]
    assert res and res[0]["say"] == "ding" and res[0]["reward"] == 0.3
    assert c.tools()["stats"]["bell"]["ok"] == 1


def test_serve_answers_calls_until_stopped(served):
    a, bridge, belt, c = served
    answered = []
    stop = threading.Event()
    t = threading.Thread(target=c.serve, args=({"bell": lambda call: {"say": "ding"}},),
                         kwargs={"stop": stop.is_set, "on_call": lambda call, r: answered.append(call["call_id"])},
                         daemon=True)
    t.start()
    time.sleep(0.3)
    for k in (1, 2):
        a.call_soon(lambda ag: belt._on_syllable({"text": "qx"}))
        assert _until(lambda: len(answered) >= k)
    stop.set()
    t.join(15)
    assert len(answered) == 2 and belt.stats["bell"]["ok"] == 2


def test_event_stream(served):
    a, bridge, _, c = served
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(bridge.url + "/stream?kinds=heard", timeout=10) as resp:
        assert resp.headers["Content-Type"].startswith("text/event-stream")
        c.hear("hello")
        buf = b""
        while b"\n\n" not in buf:
            buf += resp.read1(4096)
    head, data = buf.split(b"data: ", 1)
    assert b"event: heard" in head
    assert json.loads(data.split(b"\n")[0])["text"] == "hello"


def test_token_and_host_rules():
    a = Agent.create(seed=3, cortex_size=300, world="void")
    with pytest.raises(ValueError, match="token"):
        Bridge(a, host="0.0.0.0", port=0)
    b = Bridge(a, port=0, token="s3cret").start()
    try:
        with pytest.raises(BridgeError, match="401"):
            BridgeClient(b.url).state()
        assert BridgeClient(b.url, token="s3cret").state()["stage"] == "awake"
        with pytest.raises(BridgeError, match="404"):  # no thought stream attached
            BridgeClient(b.url, token="s3cret").thoughts()
    finally:
        b.stop()


def test_unreachable_creature_gives_a_clear_error():
    with pytest.raises(BridgeError, match="anima serve"):
        BridgeClient("http://127.0.0.1:9", timeout=1.0).state()
