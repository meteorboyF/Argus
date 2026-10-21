"""Keep optional conversion frameworks out of production inference processes.

CTranslate2 eagerly attempts an optional torch import through its converter
and model-spec modules. Those sites handle ImportError; Whisper's compiled
inference and Silero ONNX do not require torch. Diagnostic/model-conversion
commands run separately and do not install this process-wide policy.
"""
from __future__ import annotations

import importlib.abc
import sys


class _NoTorchInference(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ModuleNotFoundError(
                'PyTorch is excluded from the ARGUS production inference memory budget; '
                'run model conversion or torch diagnostics in a separate process.',
                name=fullname)
        return None


_POLICY = _NoTorchInference()


def enforce_inference_import_budget():
    """Install once, before inference models load; never unload an active torch."""
    if sys.modules.get('torch') is not None:
        raise RuntimeError('PyTorch is already loaded; start ARGUS in a fresh process '
                           'to preserve the production inference memory budget.')
    if _POLICY not in sys.meta_path:
        sys.meta_path.insert(0, _POLICY)
