import threading

from argus.depth_preview import LatestWorker


def test_slow_preview_drops_backlog_without_blocking_safety():
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    seen = []
    def consume(item):
        seen.append(item)
        if item == 1:
            entered.set()
            assert release.wait(2)
        if item == 100:
            finished.set()
    worker = LatestWorker(consume)
    try:
        worker.submit(1)
        assert entered.wait(2)
        # Renderer is blocked; the producer must still advance to latest.
        for value in range(2, 101):
            worker.submit(value)
        release.set()
        assert finished.wait(2)
        assert seen == [1, 100]
    finally:
        release.set()
        worker.stop()


def test_preview_error_does_not_kill_worker():
    finished = threading.Event()
    def consume(item):
        if item == 'bad':
            raise ValueError('display failed')
        finished.set()
    worker = LatestWorker(consume)
    try:
        worker.submit('bad')
        # Wait until consumed before submitting recovery, using condition.
        import time
        limit = time.monotonic()+2
        while worker._pending is not None and time.monotonic() < limit:
            time.sleep(.001)
        worker.submit('recovery')
        assert finished.wait(2)
    finally:
        worker.stop()
