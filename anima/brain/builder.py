"""Build brains from declarative specs.

A *spec* is plain data (a dict, or a TOML / JSON file) with three parts::

    [[population]]          # groups of spiking neurons
    name = "cortex.E"
    size = 800
    kind = "RS"             # Izhikevich cell type
    role = "cortex"         # how neuromodulators act on it
    noise = 4.0             # spontaneous background drive
    ...

    [[projection]]          # synapses between populations
    pre = "cortex.E"
    post = "motor"
    p = 0.1
    weight = 0.004
    [projection.rule]       # optional local learning rule
    type = "reward_stdp"
    tau_e = 1000.0

    [brain.drives]          # settings of the non-spiking systems
    day_length_s = 1200.0

Populations become input ports with an ``[population.input]`` table and
output groups with ``[population.output]``.  Roles understood by the brain:
``sensory``, ``audio``, ``intero``, ``cortex``, ``motor``, ``vocal``; any
other role is simulated but not neuromodulated.

Use :class:`BrainBuilder` to assemble a spec in Python, or
:func:`load_spec` / :func:`save_spec` to read and write files.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from ..core.network import Network, Population, Projection
from ..core.plasticity import make_rule
from .brain import Brain

_POP_FIELDS = {"name", "size", "kind", "inhibitory", "noise", "bias", "jitter",
               "nmda_ratio", "target_rate", "channels"}
_PROJ_FIELDS = {"pre", "post", "p", "weight", "weight_jitter", "w_max", "topology",
                "scaling", "autapses", "pre_channel_gain", "post_channel_gain", "fixed_indegree",
                "name"}


def build_network(spec: dict[str, Any], seed: int | None = None) -> Network:
    seed = spec.get("seed") if seed is None else seed
    net = Network(seed=seed)
    for p in spec.get("population", []):
        unknown = set(p) - _POP_FIELDS - {"role", "input", "output", "tags"}
        if unknown:
            raise ValueError(f"population {p.get('name')!r}: unknown fields {sorted(unknown)}")
        kwargs = {k: v for k, v in p.items() if k in _POP_FIELDS}
        tags = dict(p.get("tags", {}))
        tags["role"] = p.get("role", "other")
        if "input" in p:
            tags["input"] = dict(p["input"])
        if "output" in p:
            tags["output"] = dict(p["output"])
        net.add_population(Population(tags=tags, **kwargs))
    for q in spec.get("projection", []):
        unknown = set(q) - _PROJ_FIELDS - {"rule"}
        if unknown:
            raise ValueError(f"projection {q.get('pre')}->{q.get('post')}: unknown fields {sorted(unknown)}")
        kwargs = {k: v for k, v in q.items() if k in _PROJ_FIELDS}
        net.connect(Projection(rule=make_rule(copy.deepcopy(q.get("rule"))), **kwargs))
    return net.build()


def build_brain(spec: dict[str, Any], seed: int | None = None) -> Brain:
    """Compile a spec into a running :class:`Brain`."""
    spec = copy.deepcopy(spec)
    if seed is not None:
        spec["seed"] = seed
    net = build_network(spec)
    return Brain(net, spec)


class BrainBuilder:
    """Fluent helper for writing specs in Python.

    >>> b = BrainBuilder("tiny", seed=1)
    >>> b.population("E", 80, kind="RS", noise=4.0, role="cortex")       # doctest: +SKIP
    >>> b.connect("E", "E", p=0.1, weight=0.015, rule={"type": "stdp"})  # doctest: +SKIP
    >>> brain = b.build()                                                # doctest: +SKIP
    """

    def __init__(self, name: str = "brain", seed: int | None = None) -> None:
        self.spec: dict[str, Any] = {"name": name, "population": [], "projection": [], "brain": {}}
        if seed is not None:
            self.spec["seed"] = seed

    def population(self, name: str, size: int, **kwargs: Any) -> "BrainBuilder":
        self.spec["population"].append({"name": name, "size": size, **kwargs})
        return self

    def input(self, name: str, size: int, channels: int = 1, labels: list[str] | None = None,
              max_current: float = 10.0, role: str = "sensory", gated: bool | None = None,
              **kwargs: Any) -> "BrainBuilder":
        inp: dict[str, Any] = {"max_current": max_current}
        if labels is not None:
            inp["labels"] = list(labels)
        if gated is not None:
            inp["gated"] = gated
        return self.population(name, size, channels=channels, role=role, input=inp, **kwargs)

    def output(self, name: str, size: int, channels: int, labels: list[str] | None = None,
               role: str = "motor", **kwargs: Any) -> "BrainBuilder":
        out: dict[str, Any] = {}
        if labels is not None:
            out["labels"] = list(labels)
        return self.population(name, size, channels=channels, role=role, output=out, **kwargs)

    def connect(self, pre: str, post: str, **kwargs: Any) -> "BrainBuilder":
        self.spec["projection"].append({"pre": pre, "post": post, **kwargs})
        return self

    def configure(self, system: str, **kwargs: Any) -> "BrainBuilder":
        """Settings for a non-spiking system: drives, sleep, modulators,
        hippocampus, cerebellum or effects."""
        self.spec["brain"].setdefault(system, {}).update(kwargs)
        return self

    def build(self, seed: int | None = None) -> Brain:
        return build_brain(self.spec, seed=seed)


# ------------------------------------------------------------------ files
def load_spec(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover
        try:
            import tomli as tomllib  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("reading TOML needs Python 3.11+ or `pip install tomli`; "
                               "or use a .json spec") from exc
    return tomllib.loads(text)


def save_spec(spec: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    else:
        path.write_text(dumps_toml(spec), encoding="utf-8")


_BARE = re.compile(r"^[A-Za-z0-9_-]+$")


def _key(k: str) -> str:
    return k if _BARE.match(k) else json.dumps(k)


def _val(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        r = repr(v)
        return r if any(c in r for c in ".einf") else r + ".0"
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_val(x) for x in v) + "]"
    raise TypeError(f"cannot write {type(v).__name__} to TOML")


def _is_table_list(v: Any) -> bool:
    return isinstance(v, list) and len(v) > 0 and all(isinstance(x, dict) for x in v)


def dumps_toml(d: dict[str, Any]) -> str:
    """Write a spec as TOML (subset sufficient for anima specs)."""
    out: list[str] = []

    def scalars(table: dict[str, Any]) -> None:
        for k, v in table.items():
            if v is None or isinstance(v, dict) or _is_table_list(v):
                continue
            out.append(f"{_key(k)} = {_val(v)}")

    def tables(path: list[str], table: dict[str, Any]) -> None:
        for k, v in table.items():
            if isinstance(v, dict):
                out.append("")
                out.append(f"[{'.'.join(_key(p) for p in path + [k])}]")
                scalars(v)
                tables(path + [k], v)
            elif _is_table_list(v):
                for item in v:
                    out.append("")
                    out.append(f"[[{'.'.join(_key(p) for p in path + [k])}]]")
                    scalars(item)
                    tables(path + [k], item)

    scalars(d)
    tables([], d)
    return "\n".join(out).lstrip("\n") + "\n"
