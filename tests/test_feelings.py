"""Pain and pleasure (anima.brain.hedonic, Brain.feel), the imitation reward,
and the ways to give them: chat, dashboard keys, bridge, Python API."""

import json
import urllib.request

import numpy as np
import pytest

from anima import Agent
from anima.brain.hedonic import Hedonic
from anima.tutor.skills import WordTrial, score_reply, target_syllables


def _speaking(seed=3, seconds=8.0, device=None):
    """A creature that has babbled for a while (so it has words to recall)."""
    a = Agent.create(seed=seed, device=device)
    a.run(seconds=seconds)
    assert any(x["kind"] == "speech" for x in a.brain.acts)
    return a


def _dopamine_after(a, fn, ticks=120):
    fn()
    trace = []
    for _ in range(ticks):
        a.tick()
        trace.append(a.brain.mods.dopamine)
    return np.array(trace)


def test_pleasure_teaches_more_than_a_smile_and_pain_is_its_opposite():
    a = _speaking()
    smile = _dopamine_after(a, lambda: a.hear(":)"))
    b = _speaking()
    strong = _dopamine_after(b, lambda: b.brain.feel("pleasure", 1.0, "test", "now"))
    c = _speaking()
    hurt = _dopamine_after(c, lambda: c.brain.feel("pain", 1.0, "test", "now"))
    assert strong.sum() > 3 * smile.sum() > 0  # dopamine integral: what a weight change scales with
    assert hurt.min() < -0.8 and hurt.sum() < -0.5 * strong.sum()
    assert b.brain.hedonic.counts["pleasure"] == 1 and c.brain.hedonic.counts["pain"] == 1
    assert c.brain.mods.noradrenaline > b.brain.mods.noradrenaline  # pain arouses


def test_it_recalls_its_last_words_and_pauses_meanwhile():
    a = _speaking()
    b = a.brain
    words = b._last_words()
    events, syllables = [], []
    a.on("pleasure", events.append)
    a.on("syllable", syllables.append)
    info = b.feel("pleasure", 0.6, "test", "speech")
    assert info["recalled"] == [b.describe_act(x) for x in words] and info["credit"] == "speech"
    assert b.frozen
    recalled = 0
    for _ in range(100):
        a.tick()
        recalled = max(recalled, b.recalls)
        if b._recalling:
            assert not syllables  # it does not say anything while it recalls
            ep = b._replay_episode
            assert ep is not None and "vocal.gate" not in ep.actions  # the voice's "go" is never replayed
    assert recalled == len(words) and not b.frozen
    assert len(events) == 1 and events[0]["recalled"] == info["recalled"]
    assert any(e.reward == pytest.approx(0.6) and e.t_s == words[0]["t"]  # and it remembers that moment
               for e in b.hippocampus.episodes)


def test_credit_by_act_ids_and_by_movement():
    a = _speaking()
    b = a.brain
    said = [x for x in b.acts if x["kind"] == "speech"][-2:]
    info = b.feel("pleasure", 0.5, "test", [x["id"] for x in said])
    assert info["recalled"] == [b.describe_act(x) for x in said]
    for _ in range(80):
        a.tick()
    moved = [x for x in b.acts if x["kind"] == "move"]
    info = b.feel("pain", 0.5, "test", "movement")
    assert info["recalled"] == [b.describe_act(moved[-1])] and info["recalled"][0] in ("forward", "left", "right")
    assert b.feel("pleasure", 0.5, "test", "now")["recalled"] == []
    with pytest.raises(ValueError):
        b.feel("pleasure", 0.5, "test", "yesterday")
    with pytest.raises(ValueError):
        b.feel("joy", 0.5)


def test_the_mirror_learns_while_speaking_faster_with_pleasure_and_not_from_pain():
    a = _speaking()
    b = a.brain
    seen = {"pain": [], "pleasure": []}
    for kind in ("pain", "pleasure"):
        a.run(2.0)  # (the dopamine dip after pain has passed)
        b.feel(kind, 0.6, "test", "speech")
        for _ in range(60):
            a.tick()
            if b._recalling:
                seen[kind].append(b.net.modulators["mirror"])
    assert seen["pain"] and set(seen["pain"]) == {0.0}  # reliving a moment that hurt binds nothing
    assert seen["pleasure"] and np.mean(seen["pleasure"]) > 3.0  # a rewarded reply is bound strongly
    gates, speaking = [], []
    for _ in range(300):
        a.tick()
        gates.append(b.net.modulators["mirror"])
        speaking.append(any(b.current_actions[g] >= 0 for g in ("vocal.C", "vocal.V")))
    gates, speaking = np.array(gates), np.array(speaking)
    assert speaking.any() and (~speaking).any()
    assert np.all(gates[~speaking] == 0.0) and np.all(gates[speaking] > 0.0) or gates[speaking].mean() > 0.5


def _drive_to(b, channel_slice, group="vocal.C"):
    """Mean learned weight into one channel of ``group`` from intero and cortex (dopamine-gated paths)."""
    tot = 0.0
    for name in (f"intero->{group}", f"cortex.E->{group}"):
        w = b.net.weights(name)
        tot += float(w[:, channel_slice].mean())
    return tot


@pytest.mark.parametrize("kind", ["pleasure", "pain"])
def test_recalled_syllable_gets_the_credit(kind):
    a = _speaking(seed=4, seconds=6.0)
    b = a.brain
    act = next(x for x in reversed(b.acts) if x["kind"] == "speech" and x["actions"].get("vocal.C", -1) >= 0)
    c = act["actions"]["vocal.C"]
    size = b.net.populations["vocal.C"].channel_size
    sl = slice(c * size, (c + 1) * size)
    rest = np.r_[0:c * size, (c + 1) * size:b.net.populations["vocal.C"].size]
    before = _drive_to(b, sl) - _drive_to(b, rest)
    for _ in range(3):  # say it was good (or bad) a few times, as a teacher might
        b.feel(kind, 1.0, "test", [act["id"]])
        for _ in range(150):
            a.tick()
    after = _drive_to(b, sl) - _drive_to(b, rest)
    assert (after > before) if kind == "pleasure" else (after < before)


def test_feelings_linger_and_are_saved(tmp_path):
    a = _speaking()
    a.brain.feel("pain", 0.8, "test", "speech")
    a.tick()
    assert a.brain.frozen and a.latest["feelings"]["pain"] > 0.7 or a.brain.hedonic.pain > 0.7
    a.save(tmp_path / "x.npz")
    b = Agent.load(tmp_path / "x.npz")
    hb, ha = b.brain.hedonic, a.brain.hedonic
    assert hb.pain == pytest.approx(ha.pain) and hb.counts == ha.counts and hb.pulses == ha.pulses
    assert [x["id"] for x in b.brain.acts] == [x["id"] for x in a.brain.acts]
    assert len(b.brain._recall_queue) == len(a.brain._recall_queue)
    assert all(e.reward < 0 for e in b.brain._recall_queue)  # still a painful moment after loading
    for _ in range(100):
        a.tick()
        b.tick()
    assert a.brain.recalls == b.brain.recalls
    np.testing.assert_array_equal(a.brain.net.spike_count, b.brain.net.spike_count)  # it continues exactly
    h = Hedonic()
    h.add("pleasure", 1.0, 0.5, 0.0)
    total = sum(h.step(0.01) for _ in range(100))
    assert total == pytest.approx(h.cfg.onset + h.cfg.rate * 0.5) and h.pleasure < 0.6


def test_a_brain_from_before_feelings_still_loads_and_is_brought_up_to_date(tmp_path):
    a = _speaking(seconds=2.0)
    s = a.state_dict()
    for k in ("hedonic", "acts", "act_id", "recall"):
        s["brain"].pop(k)
    for q in s["spec"]["projection"]:  # an anima 0.5 creature: its mirror had no gate
        if q.get("rule", {}).get("type") == "hebbian":
            q["rule"].pop("gate", None)
    b = Agent.from_state(s)
    assert len(b.upgraded) == 2 and all("learns while it speaks" in x for x in b.upgraded)
    assert b.brain.net.projections["audio->vocal.C"].rule.gate == "mirror"
    np.testing.assert_array_equal(b.brain.net.weights("audio->vocal.C"), a.brain.net.weights("audio->vocal.C"))
    b.run(seconds=1.0)
    b.brain.feel("pleasure", 0.5, "test")
    c = Agent.from_state(s, upgrade=False)
    assert not c.upgraded and c.brain.net.projections["audio->vocal.C"].rule.gate is None
    assert not b.clone().upgraded


def test_on_a_device_too():
    a = _speaking(device="emulate", seconds=4.0)
    got = []
    a.on("pleasure", got.append)
    a.pleasure(0.7, "speech")
    trace = _dopamine_after(a, lambda: None, ticks=60)
    assert got and got[0]["recalled"] and trace.max() > 0.5


def test_agent_api_checks_its_arguments():
    a = _speaking(seconds=2.0)
    with pytest.raises(ValueError):
        a.pleasure(0.5, credit="tomorrow")
    with pytest.raises(ValueError):
        a.pain(float("nan"))
    with pytest.raises(ValueError):
        a.feel("joy")
    got = []
    a.on("pain", got.append)
    a.pain(5.0, "now", source="me")  # clipped to 1
    a.run(0.05)
    assert got[0]["intensity"] == 1.0 and got[0]["source"] == "me" and got[0]["credit"] == "now"


def test_thought_log_reads_the_feeling(tmp_path):
    from anima.mind import ThoughtStream
    a = _speaking()
    mind = ThoughtStream(a, tmp_path / "t.jsonl")
    a.pleasure(0.8, "speech")
    a.run(0.5)
    feel = [t for t in mind.recent(30) if t["kind"] == "feeling" and t["data"].get("feeling") == "pleasure"]
    assert feel and feel[-1]["text"].startswith("that feels wonderful — I think of saying")
    mind.close()


# ------------------------------------------------------------------ copying words
def test_scoring_a_reply():
    t = target_syllables("ba ba :)")
    assert t == ["ba", "ba"]
    assert score_reply(t, ["ba", "ba", "mo"])["level"] == "word"
    assert score_reply(t, ["ba", "do"])["level"] == "syllable"
    assert score_reply(t, ["bi"])["level"] == "consonant"
    assert score_reply(t, ["da"])["level"] == "vowel"
    assert score_reply(t, ["ku"])["level"] == "none"
    assert not score_reply(t, [])["answered"]
    w = WordTrial("ki ki", 10.0)
    assert not w.syllable(10.2, "ki")  # the teacher is still talking: not a reply
    w.heard_end(10.5)
    assert not w.syllable(10.9, "ki", act=7)
    assert w.syllable(11.2, "ko", act=8) and w.result["level"] == "syllable" and w.acts == [7, 8]
    w = WordTrial("ki ki", 0.0)
    w.heard_end(1.0)
    assert not w.tick(3.0) and w.tick(4.1) and not w.result["answered"]  # no answer within 3 s


def test_reward_imitation_gives_graded_pleasure_for_the_reply():
    from anima.tutor import ScriptedTeacher, Tutor
    a = _speaking(seed=2)
    tutor = Tutor(a, ScriptedTeacher(), verbose=False, printer=None)
    kit = tutor.toolkit
    kit.clear_rules()
    kit.set_rule("reward_imitation")
    rule = kit._rules["reward_imitation"]
    said = [x for x in a.brain.acts if x["kind"] == "speech"][-2:]
    got = []
    a.on("pleasure", got.append)
    for heard, reply, level, amount in (("ba ba", ["ba", "ba"], "word", 1.0), ("ba ba", ["bo"], "consonant", 0.3),
                                        ("ba ba", ["ku"], "none", None)):
        w = WordTrial(heard, a.t_ms / 1000)
        w.heard_end(a.t_ms / 1000)
        for x, act in zip(reply, said):
            w.syllable(a.t_ms / 1000 + 0.1, x, act["id"])
        w.tick(a.t_ms / 1000 + 2.0)
        n = len(got)
        kit._reply_done(a.t_ms / 1000, w, [rule])
        a.run(0.05)
        assert kit.trials[-1]["match"] == level
        if amount is None:
            assert len(got) == n and kit.trials[-1]["rewarded"] == ""
        else:
            assert got[-1]["intensity"] == amount and "pleasure" in kit.trials[-1]["rewarded"]
            assert got[-1]["recalled"] == [a.brain.describe_act(x) for x in said[:len(reply)]]
        a.run(1.0)
    assert rule.counts == {"word": 1, "syllable": 0, "consonant": 1, "none": 1}


def test_imitation_exam_scores_a_clone():
    from anima.tutor.exams import imitate_exam, summarize
    a = _speaking(seed=1, seconds=5.0)
    t0, w0 = a.t_ms, a.brain.net.weights("audio->vocal.C").copy()
    r = imitate_exam(a, words=8, seed=3)
    assert r["trials"] == 8 and 0 <= r["consonant"] <= 1 and 0 <= r["consonant_chance"] <= 1
    assert r["word"] <= r["syllable"] <= r["consonant"]
    assert a.t_ms == t0 and np.array_equal(w0, a.brain.net.weights("audio->vocal.C"))  # the creature is untouched
    assert "repeats the first sound" in summarize("imitate", r)


# ------------------------------------------------------------------ the teacher
def test_teacher_tools_for_pleasure_and_pain():
    from anima.tutor.teachers import DECISION_SCHEMA, TOOLS, _system_prompt, validate
    obs = {"lesson": {"name": "echo"}, "its_last_words": {"text": "ba", "seconds_ago": 2.0}}
    d = validate({"thought": "t", "actions": [{"tool": "pleasure", "amount": 7, "for": "words"},
                                               {"tool": "pain", "amount": 0.3}]}, obs, "x")
    assert d.actions == [{"tool": "pleasure", "amount": 1.0, "for": "speech"}] and d.rejected
    d = validate({"thought": "t", "actions": [{"tool": "pain", "amount": 0.9, "for": "movement"}]}, obs, "x")
    assert d.actions == [{"tool": "pain", "amount": 0.6, "for": "movement"}]  # pain is capped
    stale = {"lesson": {}, "its_last_words": {"text": "ba", "seconds_ago": 30.0}}
    d = validate({"thought": "t", "actions": [{"tool": "pleasure", "for": "speech"}]}, stale, "x")
    assert d.actions == [{"tool": "wait"}] and "not said anything" in d.rejected[0]
    d = validate({"thought": "t", "actions": [{"tool": "say", "text": "Good job, say mama for me :)"}]}, obs, "x")
    assert d.actions == [{"tool": "say", "text": "good job"}]  # one short word, no smileys
    assert {"pleasure", "pain"} <= set(TOOLS)
    props = DECISION_SCHEMA["properties"]["actions"]["items"]["properties"]
    assert "amount" in props and props["for"]["enum"] == ["reply", "speech", "movement"]
    d = validate({"thought": "t", "actions": [{"tool": "pleasure", "amount": 0.5}]}, obs, "x")
    assert d.actions == [{"tool": "wait"}] and "no reply" in d.rejected[0]  # ("for" defaults to its reply)
    obs2 = {**obs, "last_reply": {"you_said": "ba ba", "it_replied": "bo", "seconds_ago": 6.0}}
    d = validate({"thought": "t", "actions": [{"tool": "pleasure", "amount": 0.5}]}, obs2, "x")
    assert d.actions == [{"tool": "pleasure", "amount": 0.5, "for": "reply"}]
    prompt = _system_prompt()
    assert '"tool": "pleasure"' in prompt and "reward_imitation" in prompt and "last_reply" in prompt


def test_tutor_reports_replies_and_limits_its_feelings():
    from anima.tutor import ScriptedTeacher, Tutor
    from anima.tutor.teachers import Decision
    a = Agent.create(seed=2)
    tutor = Tutor(a, ScriptedTeacher(), verbose=False, printer=None)
    tutor.toolkit.trials.append({"t": 1.0, "you_said": "ba ba", "it_replied": "ba", "match": "syllable",
                                 "rewarded": "pleasure 0.5 by reflex"})
    a.run(3.0)
    obs = tutor.observe()
    assert obs["last_reply"]["match"] == "the first syllable" and obs["last_reply"]["rewarded"].startswith("pleasure")
    assert "copies_syllable" in obs["skills"] and "feelings" in obs
    said = [x for x in a.brain.acts if x["kind"] == "speech"][-2:]
    tutor.toolkit.trials.append({"t": a.t_ms / 1000, "you_said": "ba ba", "it_replied": "bo", "match": "consonant",
                                 "rewarded": "", "acts": [x["id"] for x in said]})
    got = []
    a.on("pleasure", got.append)
    tutor.execute(Decision("t", [{"tool": "pleasure", "amount": 0.5, "for": "reply"}]))
    a.run(0.05)
    assert got and got[0]["recalled"] == [a.brain.describe_act(x) for x in said]  # exactly its reply
    d = Decision("t", [{"tool": "pain", "amount": 0.3, "for": "speech"}])
    tutor.execute(d)
    d2 = Decision("t", [{"tool": "pain", "amount": 0.3, "for": "speech"}])
    tutor.execute(d2)  # within 8 s of the last pain: refused
    assert d2.rejected and "too soon" in d2.rejected[0]
    a.run(0.1)
    assert a.brain.hedonic.counts["pain"] == 1


# ------------------------------------------------------------------ chat, bridge
def test_chat_commands_and_shortcuts():
    from anima.ui.chat import SHORTCUTS, ChatUI, describe_feeling, feel
    a = _speaking(seconds=3.0)
    out = []
    ui = ChatUI(a, None, out=out.append)
    got = []
    a.on("pleasure", got.append)
    a.on("pain", got.append)
    assert ui.command("/pleasure 0.8 speech") and ui.command("/pain") and ui.command("/good now")
    assert feel(a, "pain", ["2"]).startswith("the amount") and feel(a, "pain", ["later"]).startswith("/pain")
    a.run(0.05)
    assert [(e["kind"], e["intensity"]) for e in got] == [("pleasure", 0.8), ("pain", 0.5), ("pleasure", 0.6)]
    assert SHORTCUTS["+++"] == ("pleasure", 1.0) and SHORTCUTS["-"][0] == "pain"
    line = describe_feeling(got[0])
    assert line.startswith("(anima feels pleasure 0.8") and "it recalls" in line
    assert "for whatever it is doing right now" in describe_feeling(got[2])


def test_plus_and_minus_reach_its_answer_to_what_you_said():
    from anima.ui.chat import ChatUI
    a = _speaking(seed=2, seconds=3.0)
    ui = ChatUI(a, None, out=lambda line: None)
    got = []
    a.on("pleasure", got.append)
    for _ in range(6):  # say a word until it answers
        a.hear("ba ba")
        ui.replies.heard("ba ba")
        a.run(4.0)
        if ui.replies.last:
            break
    last = ui.replies.last
    assert last and last["said"] == "ba ba" and ui.replies.credit() == last["acts"]
    a.run(2.0)  # it may babble on meanwhile; "+" still means that answer
    assert ui.replies.credit() == last["acts"]
    a.feel("pleasure", 0.5, ui.replies.credit(), "you")
    a.run(0.05)
    assert got[-1]["recalled"] == last["reply"].split()
    a.run(16.0)
    assert ui.replies.credit() == "auto"  # too long ago: back to its last words or movement
    ui.replies.detach()


def test_bridge_pleasure_and_pain():
    from anima.bridge import Bridge, BridgeClient
    a = _speaking(seconds=2.0)
    a.start(speed=1.0)
    br = Bridge(a, port=0).start()
    try:
        c = BridgeClient(br.url)
        assert c.pleasure(0.7, "speech")["ok"]
        assert c.pain(0.4, "move")["for"] == "movement"
        for body, msg in (({"amount": 3}, "between 0 and 1"), ({"amount": 0.5, "for": "later"}, "for must be")):
            req = urllib.request.Request(br.url + "/pain", data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
            with pytest.raises(urllib.error.HTTPError) as err:
                urllib.request.urlopen(req, timeout=5)
            assert err.value.code == 400 and msg in json.loads(err.value.read())["error"]
        ev = c.events(kinds=["pleasure", "pain"], wait=3.0)["events"]
        assert {e["kind"] for e in ev} <= {"pleasure", "pain"} and ev
        assert "POST /pleasure" in c.info()["endpoints"]
    finally:
        br.stop()
        a.stop()
