"""Save / Load — round-trip a model's structure + parameters to disk.

File format: numpy `.npz` archive containing
  - `_structure_`: 0-d unicode array holding a JSON-encoded Config dict
  - one entry per parameter, keyed by the hierarchical StateDict name

This avoids pickle (no arbitrary-code execution on load) and stays inspectable
with `numpy.load` + `json.loads`.
"""

import json
import numpy as np

from .core.module import Module


# ---------------------------------------------------------------------------
# Type registry — maps the `"type"` string in Config back to a class
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[Module]] = {}


def Register(cls: type[Module]) -> type[Module]:
    """Decorator / function that registers a Module subclass for Load()."""
    _REGISTRY[cls.__name__] = cls
    return cls


def _BuiltinRegistry() -> None:
    """Populate the registry with the built-in modules. Idempotent."""
    from .layers import Linear, Sequential, Conv2D, Flatten, MaxPool2D
    from .activations import ReLU, Sigmoid, Tanh, Softmax
    for cls in (Linear, Sequential, Conv2D, Flatten, MaxPool2D, ReLU, Sigmoid, Tanh, Softmax):
        _REGISTRY.setdefault(cls.__name__, cls)


# ---------------------------------------------------------------------------
# Model construction from a Config dict
# ---------------------------------------------------------------------------

def BuildFromConfig(config: dict) -> Module:
    """Reconstruct a Module hierarchy from a Config dict."""
    _BuiltinRegistry()
    type_name = config["type"]
    if type_name not in _REGISTRY:
        raise KeyError(
            f"Unknown module type {type_name!r}. "
            f"Use serialize.Register on custom modules before Load()."
        )
    cls = _REGISTRY[type_name]

    if type_name == "Sequential":
        from .layers import Sequential  # local import — concrete type for the kwarg
        children = [BuildFromConfig(c) for c in config["modules"]]
        raw_shape = config.get("inputShape")
        inputShape = tuple(raw_shape) if raw_shape is not None else None
        return Sequential(*children, inputShape=inputShape)

    return cls(**config.get("args", {}))


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

_STRUCTURE_KEY = "_structure_"


def Save(model: Module, path: str) -> None:
    """Save structure + parameters to a `.npz` archive at `path`."""
    structure = json.dumps(model.Config())
    arrays = {k: np.asarray(v) for k, v in model.StateDict().items()}
    arrays[_STRUCTURE_KEY] = np.array(structure)
    np.savez(path, **arrays)


def Load(path: str) -> Module:
    """Reconstruct a full model (structure + weights) from `path`."""
    with np.load(path, allow_pickle=False) as data:
        if _STRUCTURE_KEY not in data.files:
            raise ValueError(
                f"{path}: missing structure entry — was this file written by Save()?"
            )
        structure = json.loads(str(data[_STRUCTURE_KEY]))
        model = BuildFromConfig(structure)
        sd = {k: data[k] for k in data.files if k != _STRUCTURE_KEY}
    model.LoadStateDict(sd)
    return model


def SaveStateDict(model: Module, path: str) -> None:
    """Save only the parameters (no structure) — for transfer-learning patterns."""
    np.savez(path, **{k: np.asarray(v) for k, v in model.StateDict().items()})


def LoadStateDict(model: Module, path: str, strict: bool = True) -> None:
    """Load parameters into an already-constructed model."""
    with np.load(path, allow_pickle=False) as data:
        sd = {k: data[k] for k in data.files}
    model.LoadStateDict(sd, strict=strict)
