"""Generic spiking-network simulator used by anima's brains."""

from .device import DeviceError, DeviceSpec, available_gpus, describe_device, resolve_device
from .monitors import Probe, RateMonitor, SpikeRecorder, isi_cv, power_spectrum
from .network import Network, Population, Projection
from .neurons import PRESETS, IzhikevichParams
from .plasticity import STDP, Hebbian, RewardSTDP, Rule, make_rule

__all__ = [
    "Network", "Population", "Projection", "PRESETS", "IzhikevichParams",
    "Rule", "STDP", "RewardSTDP", "Hebbian", "make_rule",
    "SpikeRecorder", "RateMonitor", "Probe", "isi_cv", "power_spectrum",
    "resolve_device", "available_gpus", "describe_device", "DeviceSpec", "DeviceError",
]
