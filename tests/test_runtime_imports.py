"""Process isolation prevents this production policy affecting test tooling."""
import subprocess
import sys


def run(code):
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_optional_torch_import_fails_cleanly_without_blocking_numpy():
    run('''
from argus.runtime_imports import enforce_inference_import_budget
import sys
enforce_inference_import_budget()
try:
    import torch
except ImportError as exc:
    assert exc.name == 'torch'
else:
    raise AssertionError('torch was loaded')
import numpy
assert 'torch' not in sys.modules
''')


def test_existing_framework_is_not_unloaded_or_disguised():
    run('''
from argus.runtime_imports import enforce_inference_import_budget
import sys, types
existing=types.ModuleType('torch')
sys.modules['torch']=existing
try:
    enforce_inference_import_budget()
except RuntimeError as exc:
    assert 'fresh process' in str(exc)
else:
    raise AssertionError('preloaded torch accepted')
assert sys.modules['torch'] is existing
''')


def test_policy_is_idempotent():
    run('''
from argus.runtime_imports import enforce_inference_import_budget, _POLICY
import sys
enforce_inference_import_budget()
enforce_inference_import_budget()
assert sys.meta_path.count(_POLICY) == 1
''')


def test_unknown_production_object_never_loads_live_encoder():
    from argus.grounding import Grounder, UnsupportedObjectError
    from argus.config import GroundingConfig
    import pytest
    g=Grounder.__new__(Grounder)
    g.cfg=GroundingConfig()
    g.vocab=None
    g._embed_live=lambda _: pytest.fail('production imported live encoder')
    with pytest.raises(UnsupportedObjectError, match='No installed embedding'):
        g._embed('uninstalled object')


def test_installed_vector_is_still_available_without_live_encoder():
    from argus.grounding import Grounder
    from argus.config import GroundingConfig
    from types import SimpleNamespace
    import numpy as np
    vector=np.ones((1,1,512),np.float32)
    g=Grounder.__new__(Grounder)
    g.cfg=GroundingConfig()
    g.vocab=SimpleNamespace(get=lambda _:vector)
    assert g._embed('chair') is vector


def test_explicit_diagnostic_encoding_remains_available():
    from argus.grounding import Grounder
    from argus.config import GroundingConfig
    g=Grounder.__new__(Grounder)
    g.cfg=GroundingConfig(allow_torch_fallback=True)
    g.vocab=None
    g._embed_live=lambda _: 'diagnostic vector'
    assert g._embed('uninstalled object') == 'diagnostic vector'


def test_unsupported_object_has_honest_spoken_response():
    from argus.orchestrator import Orchestrator
    from argus.grounding import UnsupportedObjectError
    from types import SimpleNamespace
    import threading
    o=Orchestrator.__new__(Orchestrator)
    o._query_lock=threading.Lock()
    def unsupported(_):
        raise UnsupportedObjectError('no installed embedding')
    o.dialogue=SimpleNamespace(handle=unsupported)
    said=[]
    o.speaker=SimpleNamespace(speak=said.append)
    o.handle_query('find an uninstalled object')
    assert said == ["I can't identify that object yet. Try another object."]
