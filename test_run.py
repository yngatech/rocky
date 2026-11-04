import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run


def sh(cwd, *args):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True)


class NeedsRestart(unittest.TestCase):
    def test_only_files_the_bot_never_loads(self):
        self.assertFalse(run.needs_restart([".github/workflows/tests.yml", "README.md", "test_run.py", "renovate.json"]))

    def test_any_file_the_bot_loads(self):
        for path in ["jev_bot.py", "requirements.txt", ".python-version", "mise.toml", "viewer/app.js", "run.py"]:
            self.assertTrue(run.needs_restart(["README.md", path]), path)


class Update(unittest.TestCase):
    """update() on a real clone of a throwaway upstream: returns the files a pull changed."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.upstream, self.clone = root / "upstream", root / "clone"
        sh(root, "git", "init", "-q", "-b", "master", str(self.upstream))
        self.commit("jev_bot.py", "first")
        sh(root, "git", "clone", "-q", str(self.upstream), str(self.clone))
        patcher = mock.patch.object(run, "HERE", self.clone)
        patcher.start()
        self.addCleanup(patcher.stop)

    def commit(self, path, text):
        (self.upstream / path).parent.mkdir(parents=True, exist_ok=True)
        (self.upstream / path).write_text(text)
        sh(self.upstream, "git", "add", path)
        sh(self.upstream, "git", "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", path)

    def test_nothing_new(self):
        self.assertEqual(run.update(), [])

    def test_returns_the_pulled_files(self):
        self.commit(".github/workflows/tests.yml", "ci")
        self.commit("README.md", "docs")
        with mock.patch.object(run, "say"):
            changed = run.update()
        self.assertEqual(sorted(changed), [".github/workflows/tests.yml", "README.md"])
        self.assertFalse(run.needs_restart(changed))
        self.assertEqual((self.clone / "README.md").read_text(), "docs")

    def test_a_code_change_restarts(self):
        self.commit("jev_bot.py", "second")
        with mock.patch.object(run, "say"):
            self.assertTrue(run.needs_restart(run.update()))


if __name__ == "__main__":
    unittest.main()
