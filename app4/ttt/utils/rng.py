from __future__ import annotations

import random
from typing import Any

import torch


def get_rng_state() -> dict[str, Any]:
    """
    Return a torch-saveable RNG snapshot for deterministic resume.

    Note: This is NOT JSON-serializable; store it via `torch.save`.
    """
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }

    # numpy is a runtime dependency (needed by some torch distributed object collectives).
    try:  # pragma: no cover
        import numpy as np  # type: ignore

        state["numpy"] = np.random.get_state()
    except Exception:
        state["numpy"] = None

    if torch.cuda.is_available():  # pragma: no cover
        state["cuda"] = torch.cuda.get_rng_state_all()
    else:
        state["cuda"] = None

    return state


def set_rng_state(state: dict[str, Any]):
    if not state:
        return

    py = state.get("python", None)
    if py is not None:
        random.setstate(py)

    t = state.get("torch", None)
    if t is not None:
        torch.set_rng_state(t)

    np_state = state.get("numpy", None)
    if np_state is not None:
        try:  # pragma: no cover
            import numpy as np  # type: ignore

            np.random.set_state(np_state)
        except Exception:
            pass

    cuda = state.get("cuda", None)
    if cuda is not None and torch.cuda.is_available():  # pragma: no cover
        torch.cuda.set_rng_state_all(cuda)

