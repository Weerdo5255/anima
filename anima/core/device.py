"""Where a network's arrays live: the CPU, or a GPU.

anima's reference simulator (:class:`~anima.core.network.Network` with
``device="cpu"``) is plain NumPy.  Any other device runs the network with
:class:`~anima.core.engine.DeviceEngine`, written once against the small
array interface below and backed by

* **CuPy** — NVIDIA GPUs (CUDA), and AMD GPUs with a ROCm build of CuPy;
* **PyTorch** — NVIDIA (CUDA), AMD (ROCm, which PyTorch also calls
  ``cuda``), Apple silicon (``mps``), and ``torch:cpu`` for trying the
  PyTorch path on any computer;
* **NumPy** itself (``device="emulate"``): the device engine on the CPU,
  which is how it is tested without a GPU.

Device strings::

    cpu                     the reference NumPy engine (the default)
    auto                    a GPU if one can be used, else cpu
    cuda, cuda:1            NVIDIA/AMD GPU: CuPy if installed, else PyTorch
    mps                     Apple silicon GPU (PyTorch)
    cupy:cuda:0, torch:cuda, torch:mps, torch:cpu   a library chosen explicitly
    emulate                 the device engine on the CPU, with NumPy

The ``ANIMA_DEVICE`` environment variable sets the default for everything
that builds or loads a creature.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

INSTALL_HINT = (
    "To run on a GPU, install one of\n"
    "  NVIDIA:        pip install cupy-cuda12x   (or cupy-cuda11x)   — or PyTorch\n"
    "  AMD (ROCm):    a ROCm build of PyTorch or CuPy\n"
    "  Apple silicon: pip install torch           (then --device mps)\n"
    "Run `anima gpu-check` to see what anima finds.")


class DeviceError(RuntimeError):
    pass


@dataclass(frozen=True)
class DeviceSpec:
    """A resolved device: which array library, which device of it."""

    library: str  # "numpy" (reference), "emulate", "cupy" or "torch"
    device: str = "cpu"  # "cpu", "cuda:0", "mps", ...

    @property
    def is_reference(self) -> bool:
        return self.library == "numpy"

    def __str__(self) -> str:
        if self.library == "numpy":
            return "cpu"
        if self.library == "emulate":
            return "emulate"
        return f"{self.library}:{self.device}"


CPU = DeviceSpec("numpy", "cpu")


def default_device() -> str:
    return os.environ.get("ANIMA_DEVICE", "").strip() or "cpu"


def resolve_device(device: Any = None) -> DeviceSpec:
    """Turn a device string (or None: ``ANIMA_DEVICE``, else cpu) into a
    :class:`DeviceSpec`, checking that the library and the device exist."""
    if isinstance(device, DeviceSpec):
        return device
    text = (str(device).strip() if device is not None else default_device()).lower()
    if text in ("", "cpu", "numpy"):
        return CPU
    if text == "emulate":
        return DeviceSpec("emulate", "cpu")
    if text == "auto":
        found = available_gpus()
        return found[0] if found else CPU
    lib, _, rest = text.partition(":")
    if lib in ("cupy", "torch"):
        spec = DeviceSpec(lib, _normalize(rest or ("cuda" if lib == "cupy" else "cuda")))
    elif lib in ("cuda", "gpu", "rocm"):
        dev = _normalize("cuda" + (f":{rest}" if rest else ""))
        if not _has("cupy") and not _has("torch"):
            raise DeviceError(f"device {device!r} needs CuPy or PyTorch; neither is installed.\n{INSTALL_HINT}")
        spec = DeviceSpec("cupy", dev) if _has("cupy") else DeviceSpec("torch", dev)
    elif lib == "mps":
        spec = DeviceSpec("torch", "mps")
    else:
        raise DeviceError(f"unknown device {device!r} (try cpu, auto, cuda, mps, torch:cpu or emulate)")
    check_device(spec)
    return spec


def _normalize(dev: str) -> str:
    if dev == "cuda":
        return "cuda:0"
    return dev


def _has(module: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(module) is not None


def check_device(spec: DeviceSpec) -> None:
    """Raise :class:`DeviceError` (with what to install) if ``spec`` cannot be used."""
    if spec.library in ("numpy", "emulate"):
        return
    if not _has(spec.library):
        raise DeviceError(f"device {spec} needs {'CuPy' if spec.library == 'cupy' else 'PyTorch'}, "
                          f"which is not installed.\n{INSTALL_HINT}")
    if spec.library == "cupy":
        import cupy
        try:
            n = cupy.cuda.runtime.getDeviceCount()
        except Exception as exc:  # no driver, no GPU
            raise DeviceError(f"CuPy is installed but cannot reach a GPU: {exc}\n{INSTALL_HINT}") from exc
        k = _index(spec.device)
        if not spec.device.startswith("cuda") or k >= n:
            raise DeviceError(f"device {spec}: CuPy sees {n} GPU(s)")
        return
    import torch
    if spec.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise DeviceError(f"PyTorch {torch.__version__} cannot use a CUDA/ROCm GPU here "
                              f"(torch.cuda.is_available() is False).\n{INSTALL_HINT}")
        if _index(spec.device) >= torch.cuda.device_count():
            raise DeviceError(f"device {spec}: PyTorch sees {torch.cuda.device_count()} GPU(s)")
    elif spec.device == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise DeviceError(f"PyTorch {torch.__version__} cannot use the Apple GPU (mps) here.")
    elif spec.device != "cpu":
        raise DeviceError(f"unknown PyTorch device {spec.device!r}")


def _index(dev: str) -> int:
    _, _, k = dev.partition(":")
    return int(k) if k else 0


def available_gpus() -> list[DeviceSpec]:
    """The GPUs anima can use here, best first (CuPy before PyTorch)."""
    out: list[DeviceSpec] = []
    for spec in (DeviceSpec("cupy", "cuda:0"), DeviceSpec("torch", "cuda:0"), DeviceSpec("torch", "mps")):
        try:
            check_device(spec)
        except Exception:
            continue
        out.append(spec)
    return out


def describe_device(spec: DeviceSpec) -> str:
    """One line: the library, its version, the device's name and memory."""
    if spec.library == "numpy":
        return f"cpu: the reference NumPy engine (NumPy {np.__version__})"
    if spec.library == "emulate":
        return f"emulate: the device engine on the CPU (NumPy {np.__version__})"
    if spec.library == "cupy":
        import cupy
        k = _index(spec.device)
        props = cupy.cuda.runtime.getDeviceProperties(k)
        name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        mem = props.get("totalGlobalMem", 0) / 2 ** 30
        return f"{spec}: {name}, {mem:.1f} GB (CuPy {cupy.__version__})"
    import torch
    if spec.device.startswith("cuda"):
        p = torch.cuda.get_device_properties(_index(spec.device))
        return f"{spec}: {p.name}, {p.total_memory / 2 ** 30:.1f} GB (PyTorch {torch.__version__})"
    return f"{spec}: PyTorch {torch.__version__}"


# ======================================================================
# The array interface the device engine is written against
# ======================================================================
class Backend:
    """NumPy-style array operations on one device.  Everything the device
    engine does goes through these methods (or through ``+ - * / @``,
    slicing and integer/boolean indexing, which all three libraries share),
    so a new library needs only this class."""

    name = "numpy"

    def __init__(self, spec: DeviceSpec, dtype: Any = np.float32) -> None:
        self.spec = spec
        self.xp = np
        self.float = np.dtype(dtype)
        self.int = np.int64
        self.bool = np.bool_

    # -------------------------------------------------- moving data
    def asarray(self, a: Any, dtype: Any = None) -> Any:
        return np.array(a, dtype=dtype, copy=True)

    def to_host(self, x: Any) -> np.ndarray:
        return np.array(x, copy=True)

    def sync(self) -> None:
        pass

    def activate(self) -> None:
        """Make this device current for the calling thread (a creature may
        live in another thread than the one that built it)."""

    # -------------------------------------------------- creating
    def zeros(self, shape: Any, dtype: Any = None) -> Any:
        return self.xp.zeros(shape, dtype=dtype or self.float)

    def full(self, shape: Any, value: float, dtype: Any = None) -> Any:
        return self.xp.full(shape, value, dtype=dtype or self.float)

    def copy(self, x: Any) -> Any:
        return x.copy()

    def astype(self, x: Any, dtype: Any) -> Any:
        return x.astype(dtype)

    # -------------------------------------------------- maths
    def exp(self, x: Any) -> Any:
        return self.xp.exp(x)

    def expm1(self, x: Any) -> Any:
        return self.xp.expm1(x)

    def where(self, cond: Any, a: Any, b: Any) -> Any:
        return self.xp.where(cond, a, b)

    def clip_(self, x: Any, lo: float | None = None, hi: float | None = None) -> Any:
        """In place, like ``np.clip(x, lo, hi, out=x)``."""
        if lo is not None and hi is not None:
            self.xp.clip(x, lo, hi, out=x)
        elif hi is not None:
            self.xp.minimum(x, hi, out=x)
        elif lo is not None:
            self.xp.maximum(x, lo, out=x)
        return x

    def maximum(self, x: Any, y: Any) -> Any:
        return self.xp.maximum(x, y)

    def minimum(self, x: Any, y: Any) -> Any:
        return self.xp.minimum(x, y)

    def stack(self, arrays: list[Any]) -> Any:
        return self.xp.stack(arrays)

    def sum(self, x: Any, axis: int | None = None) -> Any:
        return x.sum(axis=axis)

    def mean(self, x: Any, axis: int | None = None) -> Any:
        return x.mean(axis=axis)

    def any(self, x: Any) -> bool:
        return bool(x.any())

    def isnan(self, x: Any) -> Any:
        return self.xp.isnan(x)

    def nonzero(self, mask: Any) -> Any:
        """1-D indices of the True entries (this waits for the device)."""
        return self.xp.flatnonzero(mask)

    def any_axis0(self, x: Any) -> Any:
        """Per column: was any entry non-zero?  (``x`` is 2-D.)"""
        return (x != 0).any(axis=0)

    def take_rows_sum(self, W: Any, idx: Any) -> Any:
        """``W[idx].sum(axis=0)``: what the neurons ``idx`` send."""
        return W[idx].sum(axis=0)

    # -------------------------------------------------- randomness
    def rng(self, seed: int) -> Any:
        return np.random.default_rng(seed)

    def standard_normal(self, gen: Any, shape: tuple[int, ...]) -> Any:
        return gen.standard_normal(shape, dtype=self.float)

    # -------------------------------------------------- the fused neuron update
    def neuron_kernel(self) -> Any:
        """A fused kernel for the per-neuron update, or None (use the
        array path).  See :meth:`anima.core.engine.DeviceEngine._neurons`."""
        return None


class CupyBackend(Backend):
    name = "cupy"

    def __init__(self, spec: DeviceSpec, dtype: Any = np.float32) -> None:
        super().__init__(spec, dtype)
        import cupy
        self.cupy = cupy
        self.xp = cupy
        self._dev = cupy.cuda.Device(_index(spec.device))
        self._dev.use()

    def asarray(self, a: Any, dtype: Any = None) -> Any:
        with self._dev:
            return self.cupy.array(np.asarray(a), dtype=dtype)

    def to_host(self, x: Any) -> np.ndarray:
        return self.cupy.asnumpy(x)

    def sync(self) -> None:
        self.cupy.cuda.Stream.null.synchronize()

    def activate(self) -> None:
        self._dev.use()

    def zeros(self, shape: Any, dtype: Any = None) -> Any:
        with self._dev:
            return self.cupy.zeros(shape, dtype=dtype or self.float)

    def full(self, shape: Any, value: float, dtype: Any = None) -> Any:
        with self._dev:
            return self.cupy.full(shape, value, dtype=dtype or self.float)

    def rng(self, seed: int) -> Any:
        with self._dev:
            return self.cupy.random.RandomState(int(seed) % (2 ** 63))

    def standard_normal(self, gen: Any, shape: tuple[int, ...]) -> Any:
        with self._dev:
            return gen.standard_normal(shape, dtype=self.float)

    def neuron_kernel(self) -> Any:
        from .engine import cupy_neuron_kernel
        return cupy_neuron_kernel(self.cupy, self.float)


class TorchBackend(Backend):
    name = "torch"

    def __init__(self, spec: DeviceSpec, dtype: Any = np.float32) -> None:
        super().__init__(spec, dtype)
        import torch
        self.torch = torch
        self.dev = torch.device(spec.device)
        if spec.device == "mps" and np.dtype(dtype) == np.float64:
            raise DeviceError("Apple's mps device has no float64; use float32")
        self._t = {np.dtype(np.float32): torch.float32, np.dtype(np.float64): torch.float64,
                   np.dtype(np.int64): torch.int64, np.dtype(np.bool_): torch.bool}
        self.float = torch.float32 if np.dtype(dtype) == np.float32 else torch.float64
        self.int = torch.int64
        self.bool = torch.bool
        self.xp = torch

    def _dtype(self, dtype: Any) -> Any:
        if dtype is None:
            return None
        if isinstance(dtype, self.torch.dtype):
            return dtype
        return self._t[np.dtype(dtype)]

    def asarray(self, a: Any, dtype: Any = None) -> Any:
        arr = np.ascontiguousarray(np.asarray(a))
        if arr.dtype == np.float64 and dtype is None and self.float == self.torch.float32:
            dtype = self.float
        t = self.torch.from_numpy(arr.copy())
        return t.to(device=self.dev, dtype=self._dtype(dtype) or t.dtype)

    def to_host(self, x: Any) -> np.ndarray:
        return x.detach().to("cpu").numpy().copy()

    def sync(self) -> None:
        if self.dev.type == "cuda":
            self.torch.cuda.synchronize(self.dev)
        elif self.dev.type == "mps":
            self.torch.mps.synchronize()

    def zeros(self, shape: Any, dtype: Any = None) -> Any:
        return self.torch.zeros(shape, dtype=self._dtype(dtype) or self.float, device=self.dev)

    def full(self, shape: Any, value: float, dtype: Any = None) -> Any:
        return self.torch.full(shape if isinstance(shape, tuple) else (shape,), value,
                               dtype=self._dtype(dtype) or self.float, device=self.dev)

    def copy(self, x: Any) -> Any:
        return x.clone()

    def astype(self, x: Any, dtype: Any) -> Any:
        return x.to(self._dtype(dtype))

    def exp(self, x: Any) -> Any:
        return self.torch.exp(x)

    def expm1(self, x: Any) -> Any:
        return self.torch.expm1(x)

    def where(self, cond: Any, a: Any, b: Any) -> Any:
        if not isinstance(a, self.torch.Tensor):
            a = self.torch.tensor(a, dtype=b.dtype, device=self.dev)
        if not isinstance(b, self.torch.Tensor):
            b = self.torch.tensor(b, dtype=a.dtype, device=self.dev)
        return self.torch.where(cond, a, b)

    def clip_(self, x: Any, lo: float | None = None, hi: float | None = None) -> Any:
        return x.clamp_(min=lo, max=hi)

    def maximum(self, x: Any, y: Any) -> Any:
        if not isinstance(y, self.torch.Tensor):
            return x.clamp(min=y)
        return self.torch.maximum(x, y)

    def minimum(self, x: Any, y: Any) -> Any:
        if not isinstance(y, self.torch.Tensor):
            return x.clamp(max=y)
        return self.torch.minimum(x, y)

    def stack(self, arrays: list[Any]) -> Any:
        return self.torch.stack(arrays)

    def sum(self, x: Any, axis: int | None = None) -> Any:
        return x.sum() if axis is None else x.sum(dim=axis)

    def mean(self, x: Any, axis: int | None = None) -> Any:
        return x.mean() if axis is None else x.mean(dim=axis)

    def any(self, x: Any) -> bool:
        return bool(x.any().item())

    def isnan(self, x: Any) -> Any:
        return self.torch.isnan(x)

    def nonzero(self, mask: Any) -> Any:
        return self.torch.nonzero(mask).flatten()

    def any_axis0(self, x: Any) -> Any:
        return (x != 0).any(dim=0)

    def take_rows_sum(self, W: Any, idx: Any) -> Any:
        return W.index_select(0, idx).sum(dim=0)

    def rng(self, seed: int) -> Any:
        try:
            g = self.torch.Generator(device=self.dev)
        except (RuntimeError, TypeError):  # a device without its own generator
            g = self.torch.Generator()
        g.manual_seed(int(seed) % (2 ** 63))
        return g

    def standard_normal(self, gen: Any, shape: tuple[int, ...]) -> Any:
        if gen.device.type == self.dev.type:
            return self.torch.randn(shape, generator=gen, device=self.dev, dtype=self.float)
        return self.torch.randn(shape, generator=gen, dtype=self.float).to(self.dev)


def make_backend(spec: DeviceSpec, dtype: Any = np.float32) -> Backend:
    if spec.library == "emulate":
        return Backend(spec, dtype)
    if spec.library == "cupy":
        return CupyBackend(spec, dtype)
    if spec.library == "torch":
        return TorchBackend(spec, dtype)
    raise DeviceError(f"{spec} has no device backend (the reference engine runs it)")


__all__ = ["DeviceSpec", "DeviceError", "CPU", "resolve_device", "default_device", "available_gpus",
           "describe_device", "check_device", "Backend", "CupyBackend", "TorchBackend", "make_backend",
           "INSTALL_HINT"]
