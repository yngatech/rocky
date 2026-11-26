"""Offline behavioral checks for making, publishing and keeping statuses.

    uv run --with-requirements requirements.txt python -m unittest test_status
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fakes import answer, api_error, claude, patch_all, traced
import rocky_bot as j


class StatusUpdateTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        traced(self)
        self.old = j.discord.CustomActivity("I feel rested today")
        # what save_status would write: the new status already in the history
        self.saved = Mock(side_effect=lambda *args: setattr(self, "saved_recent", j.recent_statuses))
        self.shown, self.posted, self.written = AsyncMock(), AsyncMock(), Mock()
        self.recent = [f"I feel day {n}." for n in range(j.STATUS_RECENT)]
        patch_all(self, [(j, "has_credit", True), (j, "status", self.old), (j, "status_turn", 0), (j, "heard", True),
                         (j, "tried_at", None), (j, "STATUS_EVERY", 180), (j, "gen_lock", asyncio.Lock()),
                         (j, "recent_statuses", self.recent), (j, "rocky_prompt", lambda *a, **k: "voice"),
                         (j, "recent_chat", Mock(return_value="Speaker: dinner tonight")),
                         (j.bot._connection, "user", SimpleNamespace(display_name="Testbot")),
                         (j, "save_status", self.saved), (j, "show_credit", self.shown),
                         (j, "post_status", self.posted), (j, "write_trace", self.written)])

    async def test_a_status_is_written_shown_posted_and_kept(self):
        with claude(answer("Pasta, question?", uncached=2_000, cached=0, output=9)) as api:
            await j.update_status.coro()
        mood = "I feel pasta, question?"
        self.assertEqual(j.status.name, mood)
        self.assertEqual(api.call_count, 1)  # one request: nothing judges it
        self.saved.assert_called_once()
        self.assertEqual(self.saved.call_args.args[:2], (mood, 0))
        self.shown.assert_awaited_once()
        self.posted.assert_awaited_once_with(mood)
        t = self.written.call_args.args[0]
        self.assertEqual((t["status"], t["llm"], t["requests"]), (mood, "rocky", 1))
        self.assertAlmostEqual(t["cost"], (2_000 * .10 + 9 * .50) / 1e6)
        self.assertEqual([c["purpose"] for c in t["api_calls"]], ["status"])
        self.assertIn("Speaker: dinner tonight", j.context_text(t["context"]))
        self.assertNotIn("status_attempts", t)
        self.assertEqual(self.saved_recent, self.recent[1:] + [mood])  # the oldest makes room
        self.assertFalse(j.heard)
        self.assertEqual(j.status_turn, 1)

    async def test_no_answer_keeps_the_previous_status(self):
        with claude(api_error(529, "overloaded_error", "Overloaded")):
            await j.update_status.coro()
        self.assertIs(j.status, self.old)
        self.assertEqual(j.recent_statuses, self.recent)
        self.saved.assert_not_called()
        self.shown.assert_not_awaited()
        self.posted.assert_not_awaited()
        self.assertTrue(j.heard)
        self.assertEqual(j.status_turn, 1)
        self.assertIsNone(self.written.call_args.args[0]["status"])

    async def test_quiet_or_out_of_credit_cycle_does_not_generate(self):
        for heard, credit in ((False, True), (True, False)):
            with self.subTest(heard=heard, credit=credit), patch.object(j, "heard", heard), \
                 patch.object(j, "has_credit", credit), patch.object(j, "llm_status", AsyncMock()) as generate:
                await j.update_status.coro()
                generate.assert_not_awaited()
        self.saved.assert_not_called()
        self.posted.assert_not_awaited()
        self.written.assert_not_called()

    async def test_an_error_keeps_the_previous_status_and_is_logged(self):
        with patch.object(j, "llm_status", AsyncMock(side_effect=RuntimeError("synthetic failure"))), \
             self.assertLogs(j.log, level="ERROR"):
            await j.update_status.coro()
        self.assertIs(j.status, self.old)
        self.saved.assert_not_called()
        self.posted.assert_not_awaited()
        self.assertTrue(j.heard)
        self.assertIn("synthetic failure", self.written.call_args.args[0]["error"])

    async def test_every_status_gets_the_chat_and_the_starts_take_turns(self):
        for turn in range(4):
            with self.subTest(turn=turn), patch.object(j, "status_turn", turn), patch.object(j, "heard", True), \
                 patch.object(j, "tried_at", None), \
                 patch.object(j, "llm_status", AsyncMock(return_value="I feel good")) as generate:
                await j.update_status.coro()
                generate.assert_awaited_once_with("Testbot", j.STATUS_STARTS[turn % 3], "Speaker: dinner tonight")


class StatusFileTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "status.json"
        patch_all(self, [(j, "STATUS_PATH", self.path), (j, "status", None), (j, "status_turn", 0),
                         (j, "status_at", None), (j, "tried_at", None), (j, "recent_statuses", [])])

    def test_a_restart_keeps_the_earlier_statuses(self):
        j.recent_statuses = ["I feel rain is loud.", "I wonder why toast falls, question?"]
        j.save_status(j.recent_statuses[-1], 4, "2026-11-20T18:00:00+00:00")
        j.recent_statuses = []
        j.load_status()
        self.assertEqual(j.recent_statuses, ["I feel rain is loud.", "I wonder why toast falls, question?"])
        self.assertEqual((j.status.name, j.status_turn), ("I wonder why toast falls, question?", 5))

    def test_a_file_from_before_has_just_its_status(self):
        self.path.write_text(json.dumps({"status": "I feel tired.", "turn": 2, "at": "2026-11-20T18:00:00+00:00"}))
        j.load_status()
        self.assertEqual(j.recent_statuses, ["I feel tired."])


if __name__ == "__main__":
    unittest.main()
