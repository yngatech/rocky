"""Offline checks for diagnostic actions, using synthetic logs and Discord sends.

    uv run --with-requirements requirements.txt python -m unittest test_explanations
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fakes import patch_all
import discord
import rocky_bot as j


class ExplanationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.user = SimpleNamespace(id=100, display_name="Testbot")
        self.guild = SimpleNamespace(me=self.user, emojis=[])
        self.channel = SimpleNamespace(id=300)
        self.logged = {"at": "2026-10-01T12:00:00+00:00", "channel_id": 300, "message_id": 10, "reply_id": 20,
                       "author": "Speaker", "bot_name": "Testbot", "message": "tea?", "history": [],
                       "reply": "Tea", "transcript": "Speaker: tea?\n\nTestbot: ", "asked": .9,
                       "steps": [{"word": "tea", "top": [["tea", .8, .8], ["coffee", .2, .2]]}]}
        self.sent = []
        self.uploads = {}
        patch_all(self, [(j, "LOG_DIR", Path(self.dir.name)), (j.bot._connection, "user", self.user)])

    def target(self, t=None, *, original=False):
        t = self.logged if t is None else t
        return Mock(spec=discord.Message, id=t["message_id"] if original else t.get("status_id", t.get("reply_id")),
                    author=SimpleNamespace(id=500) if original else self.user,
                    content=t["message"] if original else t.get("status", t.get("reply")))

    # What Discord was sent: (content, the first file, its bytes, the send's options); files go in self.uploads
    async def capture(self, content=None, **kwargs):
        file = data = None
        for uploaded in kwargs.get("files") or []:
            uploaded_data = uploaded.fp.read()
            if data is None:
                file, data = uploaded, uploaded_data
            self.uploads[uploaded.filename] = uploaded_data
            uploaded.close()
            uploaded.fp.close()
        self.sent.append((content, file, data, kwargs))

    def embeds(self, sent=None):
        options = (sent or self.sent[-1])[3]
        return options.get("embeds") or [options["embed"]]

    def fields(self, sent=None):
        return {field.name: field.value for field in self.embeds(sent)[0].fields}

    def interaction(self):
        return SimpleNamespace(channel_id=self.channel.id, guild=self.guild,
                               response=SimpleNamespace(defer=AsyncMock()),
                               followup=SimpleNamespace(send=AsyncMock(side_effect=self.capture)))

    async def test_context_actions_send_inline_and_freshly_attached_transcripts(self):
        for text in ("Speaker: tea? @everyone ```\n\nTestbot: ", "Speaker: " + "tea " * 600):
            with self.subTest(attached=len(text) > j.CONTEXT_LIMIT):
                self.logged["transcript"] = text
                j.write_trace(self.logged)
                self.sent.clear()
                await j.context_action.callback(self.interaction(), self.target())
                await j.context_action.callback(i := self.interaction(), self.target())
                i.response.defer.assert_awaited_once_with(ephemeral=True, thinking=True)
                first, second = self.embeds(self.sent[0])[0], self.embeds(self.sent[1])[0]
                self.assertEqual(first.to_dict(), second.to_dict())
                if len(text) > j.CONTEXT_LIMIT:
                    # Uploaded each time: an attachment opens in Discord's viewer, a link to an old one doesn't
                    self.assertEqual([(s[1].filename, s[2]) for s in self.sent], [("context.txt", text.encode())] * 2)
                    self.assertIsNone(first.description)
                else:
                    self.assertIn(j.unfence(text), first.description)
                    self.assertEqual(self.uploads, {})
                self.assertTrue(self.sent[1][3]["ephemeral"])
                self.assertEqual(self.sent[1][3]["allowed_mentions"].to_dict()["parse"], [])

    async def test_context_uses_exact_window_and_cache_counts_and_preserves_the_transcript(self):
        for long in (False, True):
            with self.subTest(long=long):
                t = dict({k: v for k, v in self.logged.items() if k not in ("history", "steps", "asked")},
                         llm="rocky", llm_window=1_000_000, llm_prompt_tokens=18_673,
                         llm_cached_tokens=14_533, llm_messages=291, llm_completion_tokens=24,
                         transcript="Speaker: " + "tea " * (600 if long else 1))
                j.write_trace(t)
                self.sent.clear()
                self.uploads.clear()
                await j.context_action.callback(self.interaction(), self.target(t))
                embed, = self.embeds(self.sent[0])
                self.assertEqual({f.name: f.value for f in embed.fields},
                                 {"Model": "Rocky", "Chat messages": "291", "Output": "24 tokens",
                                  "Context window": f"`▰{'▱' * 19}` 1.9% · 18,673 / 1,000,000 tokens",
                                  "Input cache": f"`{'▰' * 16}{'▱' * 4}` 77.8% · 14,533 cached · 4,140 uncached"})
                self.assertFalse(embed.image.url)
                if long:
                    self.assertEqual(self.uploads, {"context.txt": t["transcript"].encode()})
                else:
                    self.assertEqual(self.uploads, {})
                    self.assertIn(t["transcript"], embed.description)
                self.assertNotIn("estimated", json.dumps(embed.to_dict()))

    def test_text_bars_show_any_share_and_fill_only_when_full(self):
        self.assertEqual([j.text_bar(part, 1000, cells=4) for part in (0, 1, 500, 999, 1000)],
                         ["`▱▱▱▱`", "`▰▱▱▱`", "`▰▰▱▱`", "`▰▰▰▱`", "`▰▰▰▰`"])

    async def test_context_for_jev_or_an_older_log_has_a_transcript_without_made_up_usage(self):
        for t in (self.logged, dict(self.logged, llm="haiku")):
            with self.subTest(llm=t.get("llm")):
                j.write_trace(t)
                await j.context_action.callback(self.interaction(), self.target(t))
                embed, = self.embeds()
                self.assertEqual(embed.title, "What Testbot saw")
                self.assertIn(t["transcript"], embed.description)
                self.assertEqual(embed.fields[0].value, "Haiku" if t.get("llm") else "Jev")
                self.assertNotIn("Context window", [f.name for f in embed.fields])
                self.assertFalse(embed.image.url)

    async def test_stats_privately_show_only_model_total_cost_and_context_without_a_model_call(self):
        t = dict(self.logged, llm="haiku", llm_model="synthetic/previous-haiku", cost=.0016, byok_cost=.0015,
                 requests=2, seconds=1.2, llm_window=1_000_000, llm_prompt_tokens=9_800,
                 llm_cached_tokens=8_000, llm_messages=420, llm_completion_tokens=18,
                 transcript="Synthetic chat that Stats should not quote")
        j.write_trace(t)
        interaction = self.interaction()
        with patch.object(j, "llm_message", new_callable=AsyncMock) as llm:
            await j.stats_action.callback(interaction, self.target(t))
            await j.stats_action.callback(self.interaction(), self.target(t, original=True))
        self.assertEqual(self.fields(), {"Model": "synthetic/previous-haiku", "Total cost": "$0.001600 USD",
                                        "Context": "9,800 / 1,000,000 tokens · 1.0% used\n420 chat messages"})
        self.assertEqual(self.embeds(self.sent[0])[0].to_dict(), self.embeds(self.sent[1])[0].to_dict())
        self.assertNotIn(t["transcript"], str(self.fields()))
        self.assertEqual(len(self.uploads), 0)
        llm.assert_not_awaited()
        interaction.response.defer.assert_awaited_once_with(ephemeral=True, thinking=True)
        self.assertTrue(self.sent[0][3]["ephemeral"])
        self.assertEqual(self.sent[0][3]["allowed_mentions"].to_dict()["parse"], [])

    async def test_stats_find_jev_replies_reactions_and_posted_statuses(self):
        with patch.object(j, "STATUS_CHANNEL", self.channel.id):
            for name, t, original in [
                    ("reply", dict(self.logged, cost=.02, requests=8), False),
                    ("reaction", dict(self.logged, reaction="👍", reply=None, cost=.00001, requests=2,
                                      asked_transcript="Speaker: @Testbot tea?\nTestbot: "), True),
                    ("status", {"at": self.logged["at"], "status": "I feel tea is good", "status_id": 40,
                                "bot_name": "Testbot", "transcript": "Testbot's diary, today: I feel",
                                "cost": .04008, "requests": 10}, False)]:
                with self.subTest(name=name):
                    j.write_trace(t)
                    await j.stats_action.callback(self.interaction(), self.target(t, original=original))
                    self.assertEqual(self.fields()["Model"], "Jev")
                    self.assertEqual(self.fields()["Total cost"], f"${t['cost']:.6f} USD")
                    self.assertTrue(self.fields()["Context"].startswith("Not recorded"))
                    await j.context_action.callback(self.interaction(), self.target(t, original=original))
                    self.assertIn(j.logged_context(t), self.embeds()[0].description)

    async def test_rocky_reacting_as_well_as_replying_is_still_its_turn(self):
        # Rocky picks its own emoji: a reaction on its turn isn't Jev's reply-or-react check
        t = dict({k: v for k, v in self.logged.items() if k != "steps"},  # Jev's words: none in a turn of rocky's
                 llm="rocky", llm_model="synthetic/haiku", reply="Teal. Good colour.", reaction="🎨",
                 cost=.00115, requests=2, llm_window=1_000_000, llm_prompt_tokens=49_027, llm_cached_tokens=48_813,
                 llm_messages=474, transcript="testperson: remember my colour\nTestbot: Teal. Good colour.")
        j.write_trace(t)
        await j.stats_action.callback(self.interaction(), self.target(t))
        self.assertEqual(self.fields()["Model"], "synthetic/haiku")
        self.assertTrue(self.fields()["Context"].startswith("49,027 / 1,000,000 tokens"))
        self.assertIn(t["transcript"], j.logged_context(t))
        self.assertFalse(j.jev_reacted(t))
        self.assertTrue(j.jev_reacted(dict(self.logged, reaction="👍")))
        only_reacted = {k: v for k, v in t.items() if k != "reply"}
        next(j.LOG_DIR.glob("*.jsonl")).unlink()
        j.write_trace(only_reacted)  # the only entry for the message: a turn that just reacted
        await j.context_action.callback(self.interaction(), self.target(only_reacted, original=True))
        self.assertIn(t["transcript"], self.embeds()[0].description)  # its turn's log, not a Jev question check

    async def test_stats_for_an_llm_status_show_accepted_usage_and_total_retry_cost(self):
        t = {"at": self.logged["at"], "status": "I feel tea is good", "status_id": 40, "bot_name": "Testbot",
             "llm": "haiku", "llm_model": "synthetic/haiku", "cost": .00108, "byok_cost": .001,
             "requests": 4, "llm_window": 1_000_000, "llm_prompt_tokens": 3_000,
             "status_attempts": [{"accepted": False, "llm_prompt_tokens": 2_000},
                                 {"accepted": True, "llm_prompt_tokens": 3_000}]}
        with patch.object(j, "STATUS_CHANNEL", self.channel.id):
            j.write_trace(t)
            await j.stats_action.callback(self.interaction(), self.target(t))
        self.assertEqual(self.fields()["Total cost"], "$0.001080 USD")
        self.assertEqual(self.fields()["Context"], "3,000 / 1,000,000 tokens · 0.3% used")

    async def test_stats_handle_missing_usage_zero_cost_and_out_of_credit(self):
        for fields, model, cost in [
                ({}, "Jev", "Not recorded"),
                ({"llm": "haiku"}, "Haiku", "Not recorded"),
                ({"llm": "kimi", "cost": 0, "llm_completion_tokens": 0}, "Kimi", "$0.000000 USD"),
                ({"reply": j.NO_MONEY[0], "cost": 0, "requests": 0}, "None (out of credit)", "$0.000000 USD"),
                ({"llm": "haiku", "reply": j.NO_MONEY[0], "cost": 0, "requests": 0},
                 "None (out of credit)", "$0.000000 USD")]:
            with self.subTest(model=model):
                t = dict(self.logged, **fields)
                j.write_trace(t)
                await j.stats_action.callback(self.interaction(), self.target(t))
                self.assertEqual(self.fields()["Model"], model)
                self.assertEqual(self.fields()["Total cost"], cost)
                self.assertTrue(self.fields()["Context"].startswith("No generation" if model.startswith("None")
                                                                    else "Not recorded"))

    async def test_stats_handle_zero_input_and_missing_window_in_partial_logs(self):
        for tokens, window, expected in [(100, 262_144, "100 / 262,144 tokens · <0.1% used"),
                                        (0, 262_144, "0 / 262,144 tokens · 0.0% used"),
                                        (0, None, "0 tokens")]:
            with self.subTest(tokens=tokens, window=window):
                t = dict(self.logged, llm="kimi", llm_window=window, llm_prompt_tokens=tokens)
                j.write_trace(t)
                await j.stats_action.callback(self.interaction(), self.target(t))
                self.assertTrue(self.fields()["Context"].startswith(expected))

    async def test_diagnostics_explain_request_purposes_routing_and_billing_privately(self):
        t = dict(self.logged, llm="haiku", cost=.000658, byok_cost=.000577, requests=2, seconds=2.5,
                 llm_cached_tokens=14_533, llm_completion_tokens=24, api_calls=[
                     {"model": j.JEV_MODEL, "purpose": "decision", "status": 200, "attempt": 0,
                      "router_cost": .000081, "is_byok": False, "key_cost": 0},
                     {"model": j.LLM_MODEL, "purpose": "reply", "status": 200, "attempt": 0,
                      "router_cost": 0, "is_byok": True, "key_cost": .000577}])
        j.write_trace(t)
        with patch.object(j, "llm_message", new_callable=AsyncMock) as llm:
            await j.diagnostics_action.callback(self.interaction(), self.target(t))
            await j.diagnostics_action.callback(self.interaction(), self.target(t, original=True))
        fields = self.fields()
        self.assertEqual(fields["Requests"], "2")
        self.assertEqual(fields["Total time"], "2.5s")
        self.assertIn("Jev · 1 request · paid through OpenRouter", fields["Check whether to reply or react"])
        self.assertIn("$0.000081 USD", fields["Check whether to reply or react"])
        self.assertIn("Rocky · 1 request · paid through Anthropic key", fields["Write the reply"])
        self.assertIn("$0.000577 USD", fields["Write the reply"])
        self.assertEqual(fields["Final generation"], "14,533 cached input tokens · 24 output tokens")  # no input count
        self.assertIn("Routed through OpenRouter", self.embeds()[0].footer.text)
        self.assertEqual(self.embeds(self.sent[0])[0].to_dict(), self.embeds(self.sent[1])[0].to_dict())
        self.assertTrue(self.sent[-1][3]["ephemeral"])
        llm.assert_not_awaited()

    async def test_diagnostics_show_the_final_generations_input_with_its_cached_part(self):
        for cached, expected in [(0, "71,141 input tokens (0 cached) · 70 output tokens"),
                                 (70_998, "71,141 input tokens (70,998 cached) · 70 output tokens")]:
            with self.subTest(cached=cached):
                t = dict(self.logged, llm="rocky", cost=.007, llm_prompt_tokens=71_141, llm_cached_tokens=cached,
                         llm_completion_tokens=70)
                j.write_trace(t)
                await j.diagnostics_action.callback(self.interaction(), self.target(t))
                self.assertEqual(self.fields()["Final generation"], expected)

    async def test_diagnostics_handle_old_logs_without_inventing_request_purposes(self):
        t = dict(self.logged, llm="haiku", cost=.001, byok_cost=.0009, requests=2)
        j.write_trace(t)
        await j.diagnostics_action.callback(self.interaction(), self.target(t))
        self.assertEqual(self.fields()["Requests"], "2")
        self.assertEqual(self.fields()["Model ID"], "Not recorded")
        self.assertIn("OpenRouter: $0.000100 USD", self.fields()["Billing"])
        self.assertNotIn("Write the reply", self.fields())
        self.assertIn("not recorded", self.embeds()[0].footer.text)

    async def test_diagnostics_count_failed_attempts_and_do_not_price_unknown_failures_as_free(self):
        t = dict(self.logged, llm="haiku", cost=.001, requests=1, api_calls=[
            {"model": j.LLM_MODEL, "purpose": "reply", "status": 429, "attempt": 0,
             "router_cost": None, "is_byok": False, "key_cost": 0},
            {"model": j.LLM_MODEL, "purpose": "reply", "status": 200, "attempt": 1,
             "router_cost": .001, "is_byok": False, "key_cost": 0}])
        j.write_trace(t)
        await j.diagnostics_action.callback(self.interaction(), self.target(t))
        self.assertEqual(self.fields()["Requests"], "2")
        self.assertIn("1 failed attempt", self.fields()["Write the reply"])
        self.assertIn("$0.001000 USD reported", self.fields()["Write the reply"])
        self.assertIn("1 HTTP retries", self.fields()["Retries"])
        self.assertNotIn("Stopped early", self.fields()["Write the reply"])  # older logs have no endings

    async def test_diagnostics_show_refusals_and_other_early_stops(self):
        call = {"model": j.LLM_MODEL, "purpose": "reply", "status": 200, "router_cost": 0,
                "is_byok": False, "key_cost": 0, "id": "gen-synthetic", "provider": "Anthropic"}
        t = dict(self.logged, llm="haiku", cost=0, requests=4, api_calls=[
            call | {"attempt": 0, "finish_reason": "content_filter", "native_finish_reason": "refusal"},
            call | {"attempt": 0, "finish_reason": "content_filter", "native_finish_reason": "refusal"},
            call | {"attempt": 0, "finish_reason": "length", "native_finish_reason": "max_tokens"},
            call | {"attempt": 0, "finish_reason": "stop", "native_finish_reason": "end_turn"}])
        j.write_trace(t)
        await j.diagnostics_action.callback(self.interaction(), self.target(t))
        self.assertIn("Stopped early: refusal ×2, max_tokens", self.fields()["Write the reply"])

    async def test_stats_require_a_matching_target_in_the_same_channel_and_existing_logs(self):
        j.write_trace(dict(self.logged, cost=.02))
        await j.stats_action.callback(self.interaction(), self.target())
        edited = self.target()
        edited.content = "Changed reply"
        await j.stats_action.callback(self.interaction(), edited)
        wrong_id = self.target()
        wrong_id.id = 99
        await j.stats_action.callback(self.interaction(), wrong_id)
        elsewhere = self.interaction()
        elsewhere.channel_id = 301
        await j.stats_action.callback(elsewhere, self.target())
        next(j.LOG_DIR.glob("*.jsonl")).unlink()
        await j.stats_action.callback(self.interaction(), self.target())
        self.assertTrue(all(s[0] == "Nothing logged for that" and s[3]["ephemeral"] for s in self.sent[1:]))

    async def test_each_target_finds_its_own_log_entry(self):
        j.write_trace(self.logged)
        newer = dict(self.logged, message_id=11, reply_id=21, reply="Coffee", transcript="Speaker: coffee?\n\nTestbot: ")
        j.write_trace(newer)
        await j.context_action.callback(self.interaction(), self.target())
        await j.context_action.callback(self.interaction(), self.target(newer))
        self.assertIn("tea?", self.embeds(self.sent[0])[0].description)
        self.assertIn("coffee?", self.embeds(self.sent[1])[0].description)

    async def test_missing_and_deleted_logs_have_nothing_to_show(self):
        await j.context_action.callback(self.interaction(), self.target())
        j.write_trace(self.logged)
        await j.context_action.callback(self.interaction(), self.target())
        next(j.LOG_DIR.glob("*.jsonl")).unlink()
        await j.context_action.callback(self.interaction(), self.target())
        self.assertEqual([s[0] for s in self.sent[::2]], ["Nothing logged for that"] * 2)
        self.assertEqual(self.embeds(self.sent[1])[0].title, "What Testbot saw")

    async def test_actions_find_a_converted_user_mention(self):
        logged = dict(self.logged, reply="@Moss knows", sent="<@100000000000000002> knows")
        j.write_trace(logged)
        target = self.target(logged)
        target.content = logged["sent"]
        target.clean_content = logged["reply"]
        await j.context_action.callback(self.interaction(), target)
        self.assertIn("What Testbot saw", self.embeds()[0].title)
        self.assertEqual(j.find_trace(self.channel.id, target, lambda t: True), logged)
        target.content = "<@100000000000000002> edited"
        self.assertIsNone(j.find_trace(self.channel.id, target, lambda t: True))  # the new sent field still detects edits

    async def test_legacy_converted_mentions_match_readable_text_but_require_the_same_id(self):
        logged = dict(self.logged, reply="@Moss knows")
        j.write_trace(logged)
        target = self.target(logged)
        target.content = "<@100000000000000002> knows"
        target.clean_content = "@Moss knows"
        self.assertEqual(j.find_trace(self.channel.id, target, lambda t: True), logged)
        target.id += 1
        self.assertIsNone(j.find_trace(self.channel.id, target, lambda t: True))


if __name__ == "__main__":
    unittest.main()
