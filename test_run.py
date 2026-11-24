import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

RUN_PY = (Path(__file__).resolve().parent / "run.py").read_text()
ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
       "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
# Stands in for the bot: notes the commit it started on, then runs until it's stopped or run.py goes away.
BOT = """
import os, subprocess, sys, time
parent = os.getppid()
head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
with open(sys.argv[1], "a") as f:
    f.write(head + "\\n")
while os.getppid() == parent:
    time.sleep(0.1)
"""


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, env=ENV, check=True, capture_output=True, text=True).stdout.strip()


def commit(repo, files):
    for path, text in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)
        git(repo, "add", path)
    git(repo, "commit", "-qm", ", ".join(files))
    return git(repo, "rev-parse", "HEAD")


class Deploy(unittest.TestCase):
    """run.py running from a clone of a throwaway upstream, checking every second, with the stub bot."""

    def test_deploy(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        upstream, clone, self.starts, self.log = root / "upstream", root / "clone", root / "starts", root / "run.log"
        git(root, "init", "-q", "-b", "master", str(upstream))
        commit(upstream, {"run.py": RUN_PY, "rocky_bot.py": "1", "README.md": "docs"})
        git(root, "clone", "-q", str(upstream), str(clone))

        # It pulls before it starts the bot
        first = commit(upstream, {"rocky_bot.py": "2"})
        with open(self.log, "w") as log:
            run = subprocess.Popen([sys.executable, str(clone / "run.py"), "--every", "1", "--",
                                    sys.executable, "-c", BOT, str(self.starts)],
                                   env=ENV, stdout=log, stderr=subprocess.STDOUT)
        self.addCleanup(self.stop, run)
        self.wait_for(lambda: self.started() == [first])

        # A code commit upstream restarts the bot on it, run.py included: it execs the new run.py
        new_run_py = RUN_PY.replace("def main():\n", "def main():\n    say('the new run.py')\n")
        code = commit(upstream, {"run.py": new_run_py, "requirements.txt": "x"})
        self.wait_for(lambda: self.started() == [first, code])
        self.assertIn("the new run.py", self.log.read_text())

        # One that only touches files the bot never loads is pulled without a restart
        docs = commit(upstream, {"README.md": "more docs", ".github/workflows/tests.yml": "ci",
                                 "test_rocky.py": "test", "renovate.json": "{}", "scripts/setup.sh": "setup"})
        self.wait_for(lambda: "not restarting" in self.log.read_text())
        self.assertEqual(git(clone, "rev-parse", "HEAD"), docs)
        self.assertEqual(self.started(), [first, code])

        # Something else fast-forwards the clone before run.py's own check can: it restarts the bot all the same.
        # The commit reaches master upstream only after the clone has it, so run.py never pulls it itself.
        git(upstream, "checkout", "-q", "-b", "side")
        outside = commit(upstream, {"viewer/app.js": "js"})
        git(upstream, "checkout", "-q", "master")
        git(clone, "fetch", "-q", str(upstream), "side")  # by path, so it races run.py's fetch for no ref
        git(clone, "merge", "-q", "--ff-only", outside)
        git(upstream, "merge", "-q", "--ff-only", "side")
        self.wait_for(lambda: self.started() == [first, code, outside])

        # A local commit is what the checkout runs: the bot restarts on it, but only once there's nothing uncommitted
        (clone / "rocky_bot.py").write_text("<<<<<<< a conflict")
        local = commit(clone, {"mise.toml": "local"})
        self.wait_for(lambda: self.log.read_text().count("uncommitted changes") >= 2)
        self.assertEqual(self.started(), [first, code, outside])
        git(clone, "checkout", "rocky_bot.py")
        self.wait_for(lambda: self.started() == [first, code, outside, local])
        # and run.py keeps it, rather than pulling what the upstream has since
        commit(upstream, {"rocky_bot.py": "3"})
        refused = self.log.read_text().count("commits the upstream doesn't")
        self.wait_for(lambda: self.log.read_text().count("commits the upstream doesn't") >= refused + 2)
        self.assertEqual(git(clone, "rev-parse", "HEAD"), local)
        self.assertEqual(self.started(), [first, code, outside, local])

    def started(self):
        return self.starts.read_text().split() if self.starts.exists() else []

    def wait_for(self, condition, timeout=30):
        deadline = time.monotonic() + timeout
        while not condition():
            if time.monotonic() > deadline:
                self.fail(f"timed out; bot started on {self.started()}; run.py said:\n{self.log.read_text()}")
            time.sleep(0.1)

    def stop(self, run):
        run.send_signal(signal.SIGTERM)
        try:
            run.wait(timeout=10)
        except subprocess.TimeoutExpired:
            run.kill()
            run.wait()


if __name__ == "__main__":
    unittest.main()
