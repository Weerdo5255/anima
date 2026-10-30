"""The LLM tutor: Ollama client, validation, reflexes, curriculum, end-to-end."""

import json
import math
import time

import numpy as np
import pytest

from anima import Agent
from anima.tutor import (DECISION_SCHEMA, OllamaClient, OllamaModelMissing, OllamaTeacher,
                         OllamaUnavailable, ScriptedTeacher, SkillReport, Tutor, extract_json, validate)
from anima.tutor.curriculum import LESSONS_WITH_ASK, Curriculum
from anima.tutor.ollama import resolve_host
from mock_ollama import MockOllama


# ------------------------------------------------------------------ client
def test_resolve_host_variants(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert resolve_host() == "http://127.0.0.1:11434"
    assert resolve_host("0.0.0.0:11434") == "http://127.0.0.1:11434"
    assert resolve_host("http://gpu-box:1234/") == "http://gpu-box:1234"
    monkeypatch.setenv("OLLAMA_HOST", "10.0.0.5")
    assert resolve_host() == "http://10.0.0.5:11434"


def test_extract_json_tolerates_chatty_models():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('<think>hmm {not json}</think>Sure! ```json\n{"a": {"b": "}"}}\n``` done') == {"a": {"b": "}"}}
    with pytest.raises(ValueError):
        extract_json("I would rather not.")


def test_asking_means_calling_more_when_hungry():
    from anima.tutor.curriculum import check_ask
    r = SkillReport()
    ok = {"ask": {"hungry": 0.55, "sated": 0.35, "n_hungry": 37, "n_sated": 36}}
    assert check_ask(r, ok)[0]
    assert not check_ask(r, {"ask": {**ok["ask"], "sated": 0.45}})[0]  # too small a difference
    assert not check_ask(r, {"ask": {**ok["ask"], "hungry": 0.2, "sated": 0.0}})[0]  # rarely says it
    assert not check_ask(r, {"ask": {**ok["ask"], "n_sated": 5}})[0]  # too few syllables to tell
    assert not check_ask(r, {})[0]


def test_client_talks_ollama_protocol():
    with MockOllama() as srv:
        c = OllamaClient(model="tiny", host=srv.url)
        assert c.version() == "0.12.6"
        assert c.has_model() and not c.has_model("other")
        obj, res = c.chat_json([{"role": "user", "content": "hi"}], schema=DECISION_SCHEMA)
        assert obj["actions"][0]["tool"] == "wait"
        req = srv.requests[-1]
        assert req["stream"] is False and req["model"] == "tiny"
        assert req["format"]["properties"]["actions"]["maxItems"] == 3
        assert "keep_alive" in req and "temperature" in req["options"]


def test_client_errors_are_explicit():
    with MockOllama() as srv:
        with pytest.raises(OllamaModelMissing, match="ollama pull nothere"):
            OllamaClient(model="nothere", host=srv.url).chat([{"role": "user", "content": "x"}])
    with pytest.raises(OllamaUnavailable, match="ollama serve"):
        OllamaClient(model="tiny", host="http://127.0.0.1:9", timeout=2).version()


def test_client_falls_back_to_json_mode_on_old_servers():
    with MockOllama(reject_schema=True) as srv:
        c = OllamaClient(model="tiny", host=srv.url)
        obj, _ = c.chat_json([{"role": "user", "content": "hi"}], schema=DECISION_SCHEMA)
        assert obj["thought"] == "watching"
        assert srv.requests[-1]["format"] == "json"


# ------------------------------------------------------------------ validation
def obs_for(lesson="forage", hunger=0.5, graduated=()):
    spec = next(x for x in LESSONS_WITH_ASK if x.name == lesson)
    return {"state": "awake", "target_sound": "m", "body": {"hunger": hunger},
            "lesson": {"name": lesson, "graduated": list(graduated),
                       "rules_you_may_use": list(spec.allowed_rules), "default_rules": list(spec.default_rules),
                       "you_may_give_food": spec.food_allowed}}


def test_validate_enforces_lesson_rules():
    raw = {"thought": "x", "actions": [
        {"tool": "give_food", "where": "left"}, {"tool": "fly"},
        {"tool": "set_rule", "rule": "guide_food"}, {"tool": "say", "text": "MA MA <script>"}]}
    d = validate(raw, obs_for("ask"), "test")
    tools = [a["tool"] for a in d.actions]
    assert "give_food" not in tools and "set_rule" not in tools  # not allowed in 'ask'
    assert {"say"} <= set(tools)
    assert d.actions[[a["tool"] for a in d.actions].index("say")]["text"] == "ma ma"  # one short word
    assert any("unknown tool" in r for r in d.rejected)
    starving = validate(raw, obs_for("ask", hunger=0.9), "test")
    assert starving.actions[0] == {"tool": "give_food", "where": "left"}  # welfare overrides
    assert len(validate({"thought": "", "actions": [{"tool": "praise"}] * 9}, obs_for(), "t").actions) == 1


def test_validate_blocks_skipping_ahead():
    d = validate({"thought": "", "actions": [{"tool": "switch_lesson", "lesson": "independence"}]},
                 obs_for("forage"), "t")
    assert d.actions == [{"tool": "wait"}] and d.rejected
    d = validate({"thought": "", "actions": [{"tool": "switch_lesson", "lesson": "forage"}]},
                 obs_for("babble", graduated=["forage"]), "t")
    assert d.actions == [{"tool": "switch_lesson", "lesson": "forage"}]
    assert validate("nonsense", obs_for(), "t").actions == [{"tool": "wait"}]
    ok = validate({"thought": "", "actions": [{"tool": "set_rule", "rule": "guide_food"},
                                              {"tool": "give_food", "where": "left"}]}, obs_for("forage"), "t")
    assert [a["tool"] for a in ok.actions] == ["set_rule", "give_food"] and not ok.rejected


def test_a_custom_curriculum():
    from anima.tutor import Lesson
    from anima.tutor.curriculum import check_babble
    only = (Lesson("chatter", "Chatter", "say lots of b", ("reward_syllable",), ("reward_syllable",), 0.0,
                   check_babble, "Smile at b."),)
    agent = Agent.create(seed=9)
    teacher = OllamaTeacher(OllamaClient(model="tiny", host="http://127.0.0.1:9"))
    tutor = Tutor(agent, teacher, target="b", curriculum=Curriculum(only), verbose=False, printer=None)
    assert "1. chatter: say lots of b" in teacher.system and "forage" not in teacher.system.split("Lessons")[1]
    agent.run(1)
    obs = tutor.observe()
    assert obs["lesson"]["name"] == "chatter" and obs["lesson"]["number"] == "1 of 1"
    assert tutor.toolkit.rule_names() == ["reward_syllable"] and "prefix=b" in tutor.toolkit.active_rules()[0]


# ------------------------------------------------------------------ reflexes
@pytest.fixture()
def tutored():
    agent = Agent.create(seed=2)
    tutor = Tutor(agent, ScriptedTeacher(), verbose=False, printer=None, hold_s=0.0)
    return agent, tutor


@pytest.fixture()
def smiling():
    """A tutor whose reflexes smile and frown, as in anima 0.5."""
    agent = Agent.create(seed=2)
    tutor = Tutor(agent, ScriptedTeacher(), verbose=False, printer=None, hold_s=0.0, reinforcer="smile")
    return agent, tutor


def test_reward_syllable_gives_pleasure_for_that_syllable(tutored):
    agent, tutor = tutored
    kit = tutor.toolkit
    kit.clear_rules()
    kit.set_rule("reward_syllable", prefix="m")
    got = []
    agent.on("pleasure", got.append)
    agent.run(3.0)
    said = [a for a in agent.brain.acts if a["kind"] == "speech"]
    assert said
    act = said[-1]
    t0 = agent.t_ms / 1000
    kit._on_syllable({"t": t0, "text": "ba", "act": act["id"]})  # not the target: nothing
    kit._on_syllable({"t": t0, "text": "mo", "act": act["id"]})
    agent.run(1.0)
    assert kit._rules["reward_syllable"].fired == 1 and len(got) == 1
    e = got[0]
    assert 0.4 <= e["t"] - t0 <= 0.62 and e["intensity"] == 0.5 and e["source"] == "tutor:reflex"
    assert e["recalled"] == [agent.brain.describe_act(act)]  # it recalls exactly that syllable
    assert not any(k == "heard" and d["text"] == ":)" for t, k, d in agent.log if t >= t0)  # no smile


def test_reward_syllable_smiles_within_half_a_second(smiling):
    agent, tutor = smiling
    kit = tutor.toolkit
    kit.clear_rules()
    kit.set_rule("reward_syllable", prefix="m")
    rule = kit._rules["reward_syllable"]
    agent.run(0.1)
    t0 = agent.t_ms / 1000
    kit._on_syllable({"t": t0, "text": "ba"})  # not the target: no smile
    kit._on_syllable({"t": t0, "text": "mo"})
    agent.run(1.0)
    smiles = [t for t, k, d in agent.log if k == "heard" and d["text"] == ":)" and t >= t0]
    assert rule.fired == 1 and len(smiles) == 1 and 0.4 <= smiles[0] - t0 <= 0.6
    kit._on_syllable({"t": agent.t_ms / 1000, "text": "mo"})  # within the refractory period
    assert rule.fired == 1


def test_feed_on_call_feeds_only_when_hungry(smiling):
    agent, tutor = smiling
    kit = tutor.toolkit
    kit.clear_rules()
    kit.set_conditions(None, True)  # no food restriction in this test
    kit.set_rule("feed_on_call", prefix="m")
    rule = kit._rules["feed_on_call"]
    w = agent.world
    placed = []
    original = w.place_food

    def spy(bearing_deg, distance, **kw):
        fx, fy = original(bearing_deg, distance, **kw)
        placed.append(math.degrees(math.atan2(fy - w.y, fx - w.x) - w.heading))
        return fx, fy

    w.place_food = spy
    agent.body.energy = 0.95
    agent.run(0.05)
    t0 = agent.t_ms / 1000
    rule.on_syllable(t0, "ma")
    agent.run(1.0)
    assert placed == []  # sated: no food...
    assert any(k == "heard" and d["text"] == ":(" for t, k, d in agent.log if t >= t0)  # ...but a frown
    assert rule.frowned == 1
    agent.body.energy = 0.2
    agent.run(0.05)
    rule.last = -1e9
    rule.on_syllable(agent.t_ms / 1000, "ma")
    agent.run(1.0)
    assert len(placed) == 1
    assert abs((placed[0] + 180) % 360 - 180) < 40  # right in front of it


def test_guide_food_keeps_food_within_reach(tutored):
    agent, tutor = tutored
    w = agent.world
    w.food = []
    w.regrow = [1e9]
    agent.body.energy = 0.3
    agent.run(2.5)
    assert w.food and w.nearest_food_distance() < 6.0


# ------------------------------------------------------------------ curriculum
def test_curriculum_requires_skill_and_practice():
    c = Curriculum()
    fed = SkillReport(food_per_min=2.0)
    passed = {"forage": {"score": 0.7}}
    assert not c.evaluate(fed, passed, passed)[0]  # no practice yet
    c.practice(6 * 60, awake=True)
    assert c.evaluate(fed, passed, passed)[0]
    assert not c.evaluate(fed, passed, None)[0]  # one passed exam is not enough
    assert not c.evaluate(fed, passed, {"forage": {"score": 0.5}})[0]
    assert not c.evaluate(fed, {"forage": {"score": 0.4}}, passed)[0]
    assert not c.evaluate(fed, {})[0]  # no exam taken yet
    assert not c.evaluate(SkillReport(food_per_min=0.2), passed, passed)[0]
    nxt = c.graduate(400.0, fed, passed)
    assert nxt.name == "echo" and c.graduated_names() == ["forage"] and c.awake_s == 0


def test_copy_me_is_graduated_by_an_imitation_exam():
    c = Curriculum()
    c.graduate(0.0, SkillReport(), {})  # -> echo
    assert c.current.name == "echo" and c.current.exam == "imitate"
    c.practice(6 * 60, awake=True)
    good = {"imitate": {"consonant": 0.4, "consonant_chance": 0.09, "syllable": 0.1, "syllable_chance": 0.02,
                        "n": 30, "trials": 33}}
    assert c.evaluate(SkillReport(), good, good)[0]
    assert not c.evaluate(SkillReport(), good, None)[0]  # twice in a row
    low = {"imitate": {**good["imitate"], "consonant": 0.2}}
    assert not c.evaluate(SkillReport(), low, good)[0]
    luck = {"imitate": {**good["imitate"], "consonant_chance": 0.2}}  # 0.4 is less than 2.5x chance
    assert not c.evaluate(SkillReport(), luck, luck)[0]
    silent = {"imitate": {**good["imitate"], "n": 10}}  # it must answer most words
    assert not c.evaluate(SkillReport(), silent, silent)[0]


def test_copy_me_takes_the_imitation_exam(tutored):
    agent, tutor = tutored
    tutor.curriculum.practice(6 * 60, True)
    tutor.meter.report = lambda: SkillReport(food_per_min=3.0)
    passed_exam(tutor)
    tutor._check_graduation()  # -> echo
    assert tutor.curriculum.current.name == "echo" and tutor.toolkit.rule_names() == ["reward_imitation"]
    agent.run(1.0)
    tutor.curriculum.practice(6 * 60, True)
    tutor._check_graduation()  # its practice is done: the exam is due
    e = tutor.exams["imitate"]
    assert e["trials"] == 33 and 0.0 <= e["consonant"] <= 1.0
    assert "imitate" in tutor.observe()["exams"] and "repeats the first sound" in tutor.progress_line()


def passed_exam(tutor, name="forage", times=2, **res):
    for _ in range(times):  # a lesson's exam has to be passed twice in a row
        e = {"score": 0.8, **res, "t": tutor.agent.t_ms / 1000.0,
             "lesson_index": tutor.curriculum.index, "lesson_awake_s": tutor.curriculum.awake_s}
        tutor.exams[name] = e
        tutor.exam_log.append({"exam": name, **e})


def test_exams_test_a_clone_and_leave_the_creature_alone():
    from anima.tutor.exams import ask_exam, forage_exam
    agent = Agent.create(seed=5)
    agent.run(2.0)
    t0, w0 = agent.t_ms, agent.brain.net.W.copy()
    pos = (agent.world.x, agent.world.y, agent.world.heading)
    res = forage_exam(agent, n_actions=6)
    assert 0.0 <= res["score"] <= 1.0 and sum(sum(v.values()) for v in res["choices"].values()) == 18
    ask = ask_exam(agent, block_s=3.0, blocks=1)
    assert ask["n_hungry"] + ask["n_sated"] > 0
    assert agent.t_ms == t0 and np.array_equal(agent.brain.net.W, w0)
    assert (agent.world.x, agent.world.y, agent.world.heading) == pos
    assert agent.brain.net.plasticity_enabled


def test_tutor_switches_reflexes_when_a_lesson_is_passed(tutored):
    agent, tutor = tutored
    assert tutor.toolkit.rule_names() == ["guide_food"]
    tutor.curriculum.practice(6 * 60, True)
    tutor.meter.report = lambda: SkillReport(food_per_min=3.0)
    passed_exam(tutor)
    tutor._check_graduation()
    assert tutor.curriculum.current.name == "echo"
    assert tutor.toolkit.rule_names() == ["reward_imitation"]
    assert tutor.milestones[0]["graduated"] == "forage"


# ------------------------------------------------------------------ end to end
def test_llm_tutor_never_makes_the_creature_wait():
    """A slow 'LLM' (1 s per decision): the creature keeps living meanwhile,
    and the model's decisions are executed."""
    said = {"n": 0}

    def responder(body):
        said["n"] += 1
        report = json.loads(body["messages"][1]["content"].split("\n")[1])
        assert report["lesson"]["name"] == "forage" and "skills" in report
        return json.dumps({"thought": "Greet it softly.", "actions": [{"tool": "say", "text": "ma ma"}]})

    with MockOllama(responder=responder, delay=1.0) as srv:
        agent = Agent.create(seed=3)
        teacher = OllamaTeacher(OllamaClient(model="tiny", host=srv.url))
        tutor = Tutor(agent, teacher, think_every_s=0.5, verbose=False, printer=None)
        agent.start(speed=1.0)
        tutor.start()
        t0, wall0 = agent.t_ms, time.time()
        time.sleep(4.0)
        tutor.stop()
        agent.stop()
        lived = (agent.t_ms - t0) / 1000.0
        assert lived >= 0.7 * (time.time() - wall0)  # real time kept despite 1 s LLM latency
        assert said["n"] >= 2
        assert any(e["source"].startswith("ollama") for e in tutor.history)
        assert any(k == "heard" and d["text"] == "ma ma" for _, k, d in agent.log)
        assert teacher.stats()["failures"] == 0


def test_broken_llm_falls_back_to_scripted_teacher():
    with MockOllama(responder=lambda body: "As an AI I cannot teach creatures.") as srv:
        agent = Agent.create(seed=4)
        teacher = OllamaTeacher(OllamaClient(model="tiny", host=srv.url))
        tutor = Tutor(agent, teacher, verbose=False, printer=None)
        agent.run(1)
        d = tutor.think_once()
        assert d.source == "fallback" and teacher.failures == 1
        for _ in range(4):  # three failures in a row: the model gets a rest
            tutor.think_once()
        assert teacher.failures == 3 and teacher.skipped == 2 and len(srv.requests) == 3
    quiet = OllamaTeacher(OllamaClient(model="tiny", host="http://127.0.0.1:9", timeout=1), fallback=None)
    d = quiet.decide({"lesson": {"name": "forage"}, "state": "awake"})
    assert d.source == "error" and d.actions == [{"tool": "wait"}]


def test_tutor_state_round_trip(tmp_path, tutored):
    agent, tutor = tutored
    agent.run(10)
    tutor.curriculum.practice(100, True)
    tutor.toolkit.set_rule("guide_food")
    tutor.toolkit.note("it likes the left")
    p = tutor.save(tmp_path / "t.tutor.json")
    other = Tutor(Agent.create(seed=2), ScriptedTeacher(), verbose=False, printer=None)
    other.load(p)
    assert other.curriculum.awake_s == pytest.approx(tutor.curriculum.awake_s)
    assert other.toolkit.rule_names() == tutor.toolkit.rule_names()
    assert len(other.meter.syllables) == len(tutor.meter.syllables)
    assert any("it likes the left" in x for _, x in other.toolkit.journal)


# ------------------------------------------------------------------ world & conditions
def test_gifts_do_not_regrow_and_wild_food_can_be_removed():
    from anima.world.foraging import ForagingWorld
    w = ForagingWorld(seed=0)
    w.offer_food("ahead")
    w.x, w.y = w.food[-1][0] - 0.5 * math.cos(w.heading), w.food[-1][1] - 0.5 * math.sin(w.heading)
    events = w.apply("forward")
    assert events and events[0][1]["gift"] is True and len(w.regrow) == 0  # eaten once, gone
    w.set_wild(False)
    assert w.food == [] and len(w.regrow) == 3  # natural food cleared away, will come back
    for _ in range(2000):
        w.step(0.01)
    assert w.food == []
    w.set_wild(True)
    for _ in range(1300):
        w.step(0.01)
    assert len(w.food) == 3 and not any(w.gift)
    w.offer_food("left")
    w.set_wild(False)  # a training condition: not saved with the world
    s = w.state_dict()
    w2 = ForagingWorld(seed=5)
    w2.load_state_dict(s)
    assert w2.food == w.food and w2.gift == w.gift == [True] and w2.wild


def test_lessons_set_their_training_conditions():
    agent = Agent.create(seed=2)
    tutor = Tutor(agent, ScriptedTeacher(), verbose=False, printer=None,
                  curriculum=Curriculum(LESSONS_WITH_ASK))
    agent.body.energy = 0.9
    agent.run(0.05)
    assert agent.body.energy <= 0.5  # forage: kept keen
    tutor.curriculum.index = 3
    tutor._apply_lesson("ask", announce=False)
    agent.run(0.05)
    assert agent.world.wild is False and agent.world.food == []
    agent.body.energy = 0.9
    agent.run(0.05)
    assert agent.body.energy > 0.85  # no cap in 'ask'
    tutor.curriculum.index = 4
    tutor._apply_lesson("independence", announce=False)
    agent.run(0.05)
    assert agent.world.wild is True and tutor.toolkit.rule_names() == []


def test_a_lesson_never_blocks_the_curriculum(tutored):
    agent, tutor = tutored
    tutor.curriculum.practice(46 * 60, True)  # 45 awake minutes and still not passed
    tutor.meter.report = lambda: SkillReport(food_per_min=0.0)
    tutor._check_graduation()
    g = tutor.curriculum.graduated[0]
    assert g["lesson"] == "forage" and g["mastered"] is False
    assert tutor.curriculum.current.name == "echo" and tutor.curriculum.mastered_names() == []
    assert "not mastered" in tutor.report() and tutor.milestones[0]["mastered"] is False


def test_a_saved_curriculum_comes_back_as_it_was(tmp_path):
    agent = Agent.create(seed=2)
    a = Tutor(agent, ScriptedTeacher(), verbose=False, printer=None, curriculum=Curriculum(LESSONS_WITH_ASK))
    a.curriculum.index = 3
    p = a.save(tmp_path / "x.tutor.json")
    b = Tutor(Agent.create(seed=2), ScriptedTeacher(), verbose=False, printer=None)  # default: no ask
    b.load(p)
    assert [x.name for x in b.curriculum.lessons] == [x.name for x in LESSONS_WITH_ASK]
    assert b.curriculum.current.name == "ask"


def test_a_test_must_keep_passing_to_graduate():
    agent = Agent.create(seed=2)
    tutor = Tutor(agent, ScriptedTeacher(), verbose=False, printer=None, hold_s=2.0)
    tutor.curriculum.practice(6 * 60, True)
    tutor.meter.report = lambda: SkillReport(food_per_min=3.0)
    passed_exam(tutor)
    tutor._check_graduation()
    assert tutor.curriculum.index == 0  # passing once is not enough
    agent.run(1.0)
    tutor.meter.report = lambda: SkillReport(food_per_min=0.1)  # a dip resets the clock
    tutor._check_graduation()
    tutor.meter.report = lambda: SkillReport(food_per_min=3.0)
    agent.run(1.5)
    tutor._check_graduation()
    assert tutor.curriculum.index == 0
    agent.run(2.1)
    tutor._check_graduation()
    assert tutor.curriculum.index == 1


def test_review_a_passed_lesson_then_return(tutored):
    agent, tutor = tutored
    tutor.curriculum.practice(6 * 60, True)
    tutor.meter.report = lambda: SkillReport(food_per_min=3.0)
    passed_exam(tutor)
    tutor._check_graduation()  # -> echo
    tutor.review_minutes = 0.05
    d = validate({"thought": "refresh", "actions": [{"tool": "switch_lesson", "lesson": "forage"}]},
                 tutor.observe(), "t")
    tutor.execute(d)
    assert tutor.reviewing == "forage" and tutor.toolkit.rule_names() == ["guide_food"]
    assert tutor.observe()["lesson"]["reviewing"] == "forage"
    awake0 = tutor.curriculum.awake_s
    agent.run(4.0)
    assert tutor.curriculum.awake_s == awake0  # the current lesson's clock pauses
    tutor.think_once()
    assert tutor.reviewing is None and tutor.toolkit.rule_names() == ["reward_imitation"]


def test_feed_command(tutored):
    from anima.ui.chat import ChatUI
    agent, _ = tutored
    said = []
    ui = ChatUI(agent, None, out=said.append)
    n = len(agent.world.food)
    ui.command("/feed left")
    agent.run(0.02)
    assert len(agent.world.food) == n + 1 and "left" in said[-1]
    ui.command("/feed sideways")
    assert "ahead" in said[-1]


# ------------------------------------------------------------------ command line
def test_cli_train_and_resume_with_mock_ollama(tmp_path, capsys):
    from anima.cli import main
    brain = tmp_path / "pupil.npz"
    with MockOllama(models=("llama3.2:latest",)) as srv:
        main(["train", "--host", srv.url, "--brain", str(brain), "--minutes", "0.25", "--fast",
              "--think-every", "3", "--quiet"])
        out = capsys.readouterr().out
        assert "teacher: ollama:llama3.2" in out and "lesson 1 of 4" in out
        assert brain.exists() and (tmp_path / "pupil.tutor.json").exists()
        state = json.loads((tmp_path / "pupil.tutor.json").read_text())
        assert state["curriculum"]["awake_s"] > 10 and len(state["history"]) >= 3
        main(["train", "--host", srv.url, "--brain", str(brain), "--minutes", "0.1", "--fast", "--quiet"])
        assert "resumed teaching" in capsys.readouterr().out


def test_cli_explains_missing_ollama_and_model(tmp_path, capsys):
    from anima.cli import main
    with pytest.raises(SystemExit, match="--teacher scripted"):
        main(["train", "--host", "127.0.0.1:9", "--brain", str(tmp_path / "b.npz"), "--minutes", "0.1"])
    with MockOllama(models=("qwen3:4b",)) as srv:
        with pytest.raises(SystemExit, match="--model qwen3:4b"):
            main(["train", "--host", srv.url, "--model", "qwen3", "--brain", str(tmp_path / "b.npz")])
        main(["tutor-check", "--host", srv.url, "--model", "qwen3:4b", "--trials", "1"])
        out = capsys.readouterr().out
        assert "decision 1" in out and "'Copy me', after it answered 'bo bo' with 'bu'" in out
