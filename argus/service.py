"""Own the headless runtime and model server without privileged operations."""
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.request import urlopen


def healthy(url):
    try:
        with urlopen(url.rstrip('/') + '/health', timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def stop_child(child):
    """Only signal a child we created; never search for arbitrary processes."""
    if child is None or child.poll() is not None:
        return
    child.send_signal(signal.SIGINT)
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def supervise(repo, config=None, log_dir=Path('/opt/argus/logs')):
    from .config import load_config
    from .guide import Guide
    cfg = load_config(config)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    server = runtime = None
    failure = None
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        with (log_dir/'llama-server.log').open('ab', buffering=0) as server_log:
            if not healthy(cfg.agent.server_url):
                if cfg.agent.server_url.rstrip('/') != 'http://127.0.0.1:8080':
                    raise RuntimeError('Configured model server is unavailable; automatic launcher supports localhost port 8080 only')
                server = subprocess.Popen(['bash', str(repo/'scripts/run_llama_server.sh')],
                                          cwd=repo, stdout=server_log, stderr=subprocess.STDOUT)
                deadline = time.monotonic()+240
                while not healthy(cfg.agent.server_url):
                    if stop.wait(1):
                        return 0
                    if server.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError('Model server did not become ready. Check llama-server.log')
            command = [sys.executable, '-m', 'argus']
            if config:
                command += ['--config', str(config)]
            command += ['run']
            runtime = subprocess.Popen(command, cwd=repo)
            failed_health = 0
            while not stop.wait(2):
                code = runtime.poll()
                if code is not None:
                    if code:
                        raise RuntimeError('ARGUS stopped with an error. Check runtime.log')
                    return 0  # intentional voice shutdown must stay shut down
                if server is not None and server.poll() is not None:
                    raise RuntimeError('Model server stopped unexpectedly')
                failed_health = 0 if healthy(cfg.agent.server_url) else failed_health+1
                if failed_health >= 3:
                    raise RuntimeError('Model server is unavailable')
    except Exception as exc:
        failure = str(exc)
        print(f'[service] {failure}', flush=True)
    finally:
        stop_child(runtime)
        stop_child(server)
    if failure:
        try:
            with Guide(cfg.speech, show=False) as guide:
                guide.failed(failure)
        except Exception as exc:
            print(f'[service] Failure announcement unavailable: {exc}', flush=True)
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--log-dir', type=Path, default=Path('/opt/argus/logs'))
    args = parser.parse_args()
    return supervise(args.repo, args.config, args.log_dir)


if __name__ == '__main__':
    raise SystemExit(main())
