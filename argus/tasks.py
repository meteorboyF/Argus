"""One cancellable slow task; the microphone never waits for model inference."""
from __future__ import annotations
import threading


class TaskRunner:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread = None
        self._cancel = threading.Event()
        self.label = ''
        self._closed = False
        self.local = threading.local()

    @property
    def busy(self):
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def submit(self, label, action, on_error=None):
        with self._lock:
            if self._closed or (self._thread is not None and self._thread.is_alive()):
                return False
            self.label = label[:80]
            cancel = self._cancel = threading.Event()
            def run():
                from .telemetry import bus
                self.local.cancel = cancel
                bus.set(task_busy=True)
                try:
                    action()
                except Exception as exc:
                    if on_error is not None and not cancel.is_set():
                        on_error(exc)
                finally:
                    bus.set(task_busy=False)
                    self.local.cancel = None
                    with self._lock:
                        self.label = ''
            self._thread = threading.Thread(target=run, name='argus-assistant', daemon=True)
            self._thread.start()
            return True

    def cancelled(self):
        token = getattr(self.local, 'cancel', None)
        return token is not None and token.is_set()

    def cancel(self):
        self._cancel.set()

    def close(self):
        with self._lock:
            self._closed = True
            thread = self._thread
            self._cancel.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)
