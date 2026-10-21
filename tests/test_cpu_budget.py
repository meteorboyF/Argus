"""Resource policy is independent of model availability."""
import sys
from types import SimpleNamespace

from argus.cpu_budget import ort_options


def test_cpu_sessions_do_not_create_default_pools_or_spin(monkeypatch):
    class Options:
        def __init__(self):
            self.entries = {}
        def add_session_config_entry(self, key, value):
            self.entries[key] = value
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(SessionOptions=Options))
    options = ort_options()
    assert options.intra_op_num_threads == options.inter_op_num_threads == 1
    assert options.entries == {'session.intra_op.allow_spinning': '0',
                               'session.inter_op.allow_spinning': '0'}
