#!/usr/bin/env python3
"""Install the headless user unit; --enable starts it on future user sessions.

Installation never starts or stops the running prototype. Boot before login
requires user lingering, which must be enabled separately by Fardeen if needed.
"""
import argparse
from pathlib import Path
import subprocess
import sys


def render(repo, python):
    def quoted(path):
        return '"'+str(path).replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%')+'"'
    template = (repo/'deploy/argus.service').read_text()
    return template.replace('@REPO@', str(repo).replace('%', '%%')).replace('@PYTHON@', quoted(python))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--enable', action='store_true')
    parser.add_argument('--out', type=Path, default=Path.home()/'.config/systemd/user/argus.service')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    Path('/opt/argus/logs').mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(repo, Path(sys.executable)))
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    if args.enable:
        subprocess.run(['systemctl', '--user', 'enable', 'argus.service'], check=True)
    print(f'Installed {args.out}. Start later with: systemctl --user start argus')


if __name__ == '__main__':
    main()
