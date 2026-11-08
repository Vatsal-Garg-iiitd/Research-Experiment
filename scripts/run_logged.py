"""Run one project command with an isolated, timestamped combined output log.

The process lock prevents concurrent calls through this wrapper. It protects
scripts that append rows to shared result CSV files from interleaved writers.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "results" / "run_logs"
LOCK = LOG_DIR / ".experiment-run.lock"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Unique readable run label")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after --")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("supply a command after --")
    if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run_id):
        parser.error("run-id may contain only letters, digits, underscore, and hyphen")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        print(f"Another logged experiment holds {LOCK}; refusing to overlap runs.", file=sys.stderr)
        return 2

    started = datetime.now(timezone.utc)
    log_path = LOG_DIR / f"{started.strftime('%Y%m%dT%H%M%SZ')}_{args.run_id}.log"
    try:
        os.write(lock_fd, f"pid={os.getpid()} run_id={args.run_id} started={started.isoformat()}\n".encode())
        os.close(lock_fd)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        with log_path.open("x", encoding="utf-8", buffering=1) as log:
            log.write(f"run_id: {args.run_id}\nstarted_utc: {started.isoformat()}\n")
            log.write(f"cwd: {ROOT}\ncommand: {subprocess.list2cmdline(command)}\n")
            log.write(f"python: {sys.executable}\nsource_revision: ")
            revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
            log.write(revision.stdout.strip() if revision.returncode == 0 else "unavailable")
            log.write("\n--- combined stdout/stderr ---\n")
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)

            def stop_child(signum, _frame):
                if proc.poll() is None:
                    proc.send_signal(signum)

            signal.signal(signal.SIGINT, stop_child)
            signal.signal(signal.SIGTERM, stop_child)
            assert proc.stdout is not None
            for line in proc.stdout:
                stamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
                rendered = f"[{stamp}] {line}"
                log.write(rendered)
                sys.stdout.write(rendered)
                sys.stdout.flush()
            code = proc.wait()
            ended = datetime.now(timezone.utc)
            log.write(f"\nended_utc: {ended.isoformat()}\nexit_code: {code}\n")
        return code
    finally:
        try:
            LOCK.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
