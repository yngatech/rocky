"""Messages to Rocky answered end to end, and the Context, Stats and Diagnostics actions on what was posted, with
Discord and OpenRouter faked and the bot's logs in a temporary directory.

    uv run --with-requirements requirements.txt python -m unittest test_actions
"""

import asyncio
import json
from collections import defaultdict
import tempfile
import unittest
from pathlib import Path

from fakes import Discord, answer, completion, isolate_rocky, openrouter, patch_all
import rocky_bot as j
import rocky_harness as rh

LONG = ("Granite is hard and old and it holds up mountains. " * 50).strip()  # past Discord's 2000 characters
OUT_OF_CREDIT = (402, {"error": {"code": 402, "message": "Insufficient credits"}})


class AnsweringTests(unittest.IsolatedAsyncioTestCase):
    maxDiff = None

    def setUp(self):
        here = tempfile.TemporaryDirectory()
        self.addCleanup(here.cleanup)
        self.discord = Discord(self, "bot-spam", "diary")
        isolate_rocky(self, here.name)
        patch_all(self, [(j, "LOG_DIR", Path(here.name) / "logs"), (j, "STATUS_PATH", Path(here.name) / "status.json"),
                         (j, "STATUS_CHANNEL", self.discord.channels["diary"].id), (j, "STATUS_EVERY", 180),
                         (j, "has_credit", True),
                         (j, "channel_history", defaultdict(list)), (j, "history_loaded", {}), (j, "taken", {}),
                         (j, "heard", False), (j, "tried_at", None), (j, "status", None), (j, "status_turn", 0),
                         (j, "recent_statuses", []), (j, "gen_lock", asyncio.Lock())])

    async def message(self, text, *replies):
        # pip tagging Rocky in #bot-spam, and OpenRouter answering Rocky's requests with `replies` in turn
        m = self.discord.say("bot-spam", "pip", text)
        with openrouter(*replies) as s:
            s.set_credit.side_effect = lambda ok: setattr(j, "has_credit", ok)  # what the bot then knows of its credit
            await j.on_message(m)
        return m

    async def conversation(self):
        # pip asks and Rocky answers: plainly, after a tool round trip, and at length; then a status, and out of credit.
        # Each message pip sent, with what the bot posted in answer, and the status.
        said = [await self.message("which rock is best", (200, completion(answer(
                    "Granite. Strong, strong, strong.", [("react", {"emoji": "🪨"})])))),
                await self.message("my cat is called biscuit", (200, completion(answer(
                    calls=[("remember", {"about": "pip", "fact": "has a cat called Biscuit"})]))),
                    (200, completion(answer("Biscuit. Good name.")))),
                await self.message("tell me everything about granite", (200, completion(answer(LONG))))]
        j.heard = True
        with openrouter((200, completion(answer("I feel granite is best rock."), cost=.0004))):
            await j.update_status.coro()
        said.append(await self.message("and basalt?", OUT_OF_CREDIT))
        said.append(await self.message("hello?", OUT_OF_CREDIT))  # still out: no request
        status, = self.discord.channels["diary"].messages()
        return [(m, self.discord.answer(m)) for m in said], status

    async def test_a_message_to_rocky_is_answered_and_logged(self):
        answered, status = await self.conversation()
        (plain, plain_reply), (remembered, remembered_reply), (long, parts), (ran_out, gif), (broke, gif_again) = answered

        # Each answer is a reply to the message, the long one in two parts, with what Rocky did under it
        self.assertEqual([m.content for m in plain_reply], ["Granite. Strong, strong, strong."])
        self.assertEqual(self.discord.reactions, [(plain.id, "🪨")])
        self.assertEqual([m.content for m in remembered_reply],
                         ["Biscuit. Good name.\n-# 🧠 remembered: pip has a cat called Biscuit"])
        self.assertEqual([m.reference and m.reference.message_id for m in parts], [long.id, None])
        self.assertEqual(" ".join(m.content for m in parts), LONG)
        self.assertTrue(all(len(m) == 1 and m[0].content in j.NO_MONEY for m in (gif, gif_again)))
        self.assertEqual(status.content, "I feel granite is best rock.")

        # logs/ has a line for each, with what it cost, every request, and the ids of what was posted
        lines = [json.loads(line) for path in sorted(j.LOG_DIR.glob("*.jsonl")) for line in path.read_text().splitlines()]
        self.assertEqual([t.get("message_id") for t in lines],
                         [plain.id, remembered.id, long.id, None, ran_out.id, broke.id])
        turns = {t.get("message_id"): t for t in lines}
        for m, posted, requests in ((plain, plain_reply, 1), (remembered, remembered_reply, 2), (long, parts, 1)):
            t = turns[m.id]
            self.assertEqual((t["llm_model"], t["requests"], t["reply_id"]), (j.LLM_MODEL, requests, posted[0].id))
            self.assertAlmostEqual(t["cost"], .0016 * requests)
            self.assertEqual([(c["purpose"], c["status"]) for c in t["api_calls"]], [("reply", 200)] * requests)
        self.assertEqual(turns[long.id]["reply_ids"], [m.id for m in parts])
        self.assertEqual(turns[plain.id]["reaction"], "🪨")
        self.assertEqual((lines[3]["status_id"], lines[3]["api_calls"][0]["purpose"]), (status.id, "status"))
        self.assertEqual([c["status"] for c in turns[ran_out.id]["api_calls"]], [402])
        self.assertNotIn("api_calls", turns[broke.id])
        self.assertEqual((turns[ran_out.id]["reply_id"], turns[broke.id]["reply_id"]), (gif[0].id, gif_again[0].id))

        # Rocky's log of the channel has the conversation, each of its messages by the id it was posted with
        rh.flush()
        saved = json.loads((rh.LOG_DIR / f"{plain.channel.id}.json").read_text())
        self.assertEqual([(e["type"], e["text"]) for e in saved["events"] if e["type"] in ("message", "rocky") and e["text"]],
                         [("message", "@Rocky which rock is best"), ("rocky", "Granite. Strong, strong, strong."),
                          ("message", "@Rocky my cat is called biscuit"), ("rocky", "Biscuit. Good name."),
                          ("message", "@Rocky tell me everything about granite"), ("rocky", LONG),
                          ("message", "@Rocky and basalt?"), ("message", "@Rocky hello?")])
        self.assertEqual({int(k) for k, v in saved["ids"].items() if v == "you"},
                         {m.id for m in plain_reply + remembered_reply + parts})
        self.assertEqual(rh.logs[long.channel.id].rocky_text(parts[1].id), LONG)  # a reaction to either part is to it
        # and so does the channel's history, without the lines under a reply
        history = {e["id"]: (e.get("reply"), e.get("reply_id")) for e in j.channel_history[long.channel.id]}
        self.assertEqual((history[remembered.id], history[long.id]),
                         (("Biscuit. Good name.", remembered_reply[0].id), (LONG, parts[0].id)))
        self.assertEqual(rh.memory("400")["facts"][0]["fact"], "has a cat called Biscuit")

    async def test_the_actions_explain_what_was_posted(self):
        answered, status = await self.conversation()
        (plain, [reply]), (_, [with_actions]), (_, [first, second]), (_, [gif]), (_, [gif_again]) = answered
        targets = {"plain reply": reply, "message that woke it": plain, "reply with action lines": with_actions,
                   "long reply's first part": first, "its second part": second, "status": status,
                   "out-of-credit gif": gif, "gif while still out of credit": gif_again}
        answers = {}  # (Context, Stats, Diagnostics) for each: (text, embed, attached files)
        for name, target in targets.items():
            answers[name] = []
            for action in (j.context_action, j.stats_action, j.diagnostics_action):
                await action.callback(self.discord.interaction(target, answers[name]), target)

        def fields(name, action):
            return {f["name"]: f["value"] for f in answers[name][action][1].get("fields", [])}

        # Only Context on a gif sent without asking Rocky has nothing to show: it had no transcript
        self.assertEqual({name: [text for text, _, _ in sent if text] for name, sent in answers.items()},
                         {name: ["Nothing logged for that"] if name == "gif while still out of credit" else []
                          for name in targets})

        self.assertIn("which rock is best", answers["plain reply"][0][1]["description"])
        self.assertEqual(fields("plain reply", 1)["Model"], j.LLM_MODEL)
        self.assertEqual(answers["message that woke it"], answers["plain reply"])
        self.assertIn("my cat is called biscuit", answers["reply with action lines"][0][1]["description"])
        self.assertEqual(fields("reply with action lines", 1)["Total cost"], "$0.003200 USD")
        self.assertIn("Rocky · 2 requests", fields("reply with action lines", 2)["Write the reply"])
        self.assertIn("tell me everything about granite", answers["long reply's first part"][0][2]["context.txt"].decode())
        self.assertEqual(answers["its second part"], answers["long reply's first part"])
        self.assertIn("Rocky · 1 request", fields("status", 2)["Write the status"])

        # Out of credit, no model wrote the gif, and Diagnostics names none
        for name in ("out-of-credit gif", "gif while still out of credit"):
            self.assertEqual(fields(name, 1)["Model"], "None (out of credit)")
            self.assertNotIn("Model ID", fields(name, 2))
        self.assertIn("1 failed attempt", fields("out-of-credit gif", 2)["Write the reply"])
        self.assertNotIn("footer", answers["gif while still out of credit"][2][1])  # nothing routed


if __name__ == "__main__":
    unittest.main()
