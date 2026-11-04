#!/usr/bin/env python3
"""
Run jev and keep it up to date. Every --every seconds this fetches; when the branch's upstream has new commits that
fast-forward onto a clean checkout, it pulls them. Unless they only touch files the bot never loads (CI, docs, tests, dev
scripts), it stops the bot and replaces itself with the new run.py, which starts the bot on the new code: so a change to
run.py takes effect too, with no service restart. If the bot exits on its own, it waits and does the same.

    python3 run.py                          # runs: mise exec -- uv run --with-requirements requirements.txt jev_bot.py
    python3 run.py -- python jev_bot.py     # or any other command

Restarting and Ctrl-C send the bot one SIGTERM, which it takes as a Ctrl-C, and wait for it to exit (if the bot finishes
its replies before stopping, a restart doesn't cut one off). Ctrl-C again sends another; a third time kills it.
"""

import argparse
import fnmatch
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Through mise when it's installed: it installs a newly bumped pin (.python-version, mise.toml) and execs uv in place.
DEFAULT_CMD = (["mise", "exec", "--"] if shutil.which("mise") else []) + [
    "uv", "run", "--with-requirements", "requirements.txt", "jev_bot.py"]
# Files the running bot never loads: pulling changes to only these doesn't restart it. Anything else does.
NO_RESTART = [".github/*", "scripts/*", "*.md", "test_*.py", "fakes.py", "jev_eval.py", "rocky_replay.py",
              "renovate.json", ".gitignore"]
FINISH_TIMEOUT = 300    # seconds to wait for the bot to exit after asking it to stop, before killing it


def say(msg):
    print(f"[run] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def git(*args):
    return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True,
                          env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})


def needs_restart(changed):
    return any(not any(fnmatch.fnmatch(path, pattern) for pattern in NO_RESTART) for path in changed)


# Pull if the upstream moved and fast-forwards cleanly; returns the files that changed, empty if nothing was pulled
def update():
    if (fetch := git("fetch", "--quiet")).returncode:
        say(f"fetch failed: {fetch.stderr.strip()}")
        return []
    head, upstream = git("rev-parse", "HEAD").stdout.strip(), git("rev-parse", "@{u}").stdout.strip()
    if not upstream or head == upstream:
        return []
    if git("merge-base", "--is-ancestor", "HEAD", "@{u}").returncode:
        say("not updating: this checkout has commits the upstream doesn't")
        return []
    if git("diff", "--quiet", "HEAD").returncode:
        say("not updating: this checkout has uncommitted changes")
        return []
    if (merge := git("merge", "--ff-only", "--quiet", "@{u}")).returncode:
        say(f"not updating: {merge.stderr.strip()}")
        return []
    log = git("log", "--format=  %h %s", f"{head}..HEAD").stdout.rstrip()
    say(f"updated {head[:7]} -> {upstream[:7]}\n{log}")
    return git("diff", "--name-only", head, "HEAD").stdout.splitlines()


# Run the checkout's run.py, as pulled, in this process's place: same pid, arguments and terminal. Every restart of the
# bot goes through here, so run.py itself is always the pulled one.
def reexec():
    os.execv(sys.executable, [sys.executable, str(HERE / "run.py"), *sys.argv[1:]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--every", type=int, default=60, help="seconds between update checks (default 60)")
    ap.add_argument("cmd", nargs="*", help="command that runs the bot")
    args = ap.parse_args()
    cmd = args.cmd or DEFAULT_CMD

    # The bot runs in its own session, so the terminal's Ctrl-C reaches only this script, which passes it on.
    # Signal the bot's own process, not its group: `uv run` forwards a signal to Python, so the group would get it twice.
    # SIGTERM, not SIGINT: with a terminal attached, uv assumes the terminal sent a SIGINT to Python itself and doesn't
    # forward it — but the bot is in its own session, so it never got it, and kept running until killed.
    bot, presses = None, 0
    def on_signal(sig, frame):
        nonlocal presses
        presses += 1
        if bot and bot.poll() is None:
            if presses < 3:
                os.kill(bot.pid, signal.SIGTERM)
            else:
                os.killpg(bot.pid, signal.SIGKILL)
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    if update():  # start on the latest code, this script included
        reexec()
    say(f"starting: {' '.join(cmd)}")
    started = time.monotonic()
    bot = subprocess.Popen(cmd, cwd=HERE, start_new_session=True)
    next_check, restarting = started + args.every, False
    while bot.poll() is None and not presses and not restarting:
        time.sleep(1)
        if time.monotonic() >= next_check:
            next_check = time.monotonic() + args.every
            if restarting := needs_restart(changed := update()):
                say("restarting on the new code")
                os.kill(bot.pid, signal.SIGTERM)
            elif changed:
                say("not restarting: only files the bot doesn't load changed")
    try:
        bot.wait(timeout=FINISH_TIMEOUT)
    except subprocess.TimeoutExpired:
        say(f"bot still running after {FINISH_TIMEOUT}s — killing it")
        os.killpg(bot.pid, signal.SIGKILL)
        bot.wait()
    if not restarting and not presses:
        # Exited on its own: start again, waiting longer each time it dies within a minute (a broken update, say)
        quick_exits = int(os.environ.get("RUN_QUICK_EXITS", 0)) + 1 if time.monotonic() - started < 60 else 0
        os.environ["RUN_QUICK_EXITS"] = str(quick_exits)  # kept across reexec()
        delay = min(5 * 2 ** quick_exits, 300)
        say(f"bot exited with {bot.returncode}; restarting in {delay}s")
        for _ in range(delay):
            if not presses:
                time.sleep(1)
    if presses:
        say("stopped")
    else:
        reexec()


if __name__ == "__main__":
    main()
