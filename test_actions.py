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
        # pip asks and Rocky answers: plainly, after a tool round trip, after a secret note and a look at the bar, and
        # at length; then a status, and out of credit. Each message pip sent, with what the bot posted in answer, and
        # the status.
        said = [await self.message("which rock is best", (200, completion(answer(  # the first, so nothing to read
                    "Granite. Strong, strong, strong.", [("react", {"emoji": "🪨"})]), cached=0, written=8990))),
                await self.message("my cat is called biscuit", (200, completion(answer(
                    calls=[("remember", {"about": "pip", "fact": "has a cat called Biscuit"})]))),
                    (200, completion(answer("Biscuit. Good name."))))]
        said.append(await self.message("pick a number, and is the bar open?", (200, completion(answer(
            "Seven! Bar open!", [("note", {"text": "picked 7"}), ("bar", {})],  # written before it knew
            reasoning=[{"type": "reasoning.text", "text": "seven, they said", "signature": "synthetic"}]))),
            (200, completion(answer("Rocky pick number. Secret. Bar closed until four, sad sad sad.")))))
        said.append(await self.message(f"tell me everything about granite. {LONG}", (200, completion(answer(LONG)))))
        j.heard = True
        with openrouter((200, completion(answer("I feel granite is best rock."), cost=.0004))):
            await j.update_status.coro()
        said.append(await self.message("and basalt?", OUT_OF_CREDIT))
        said.append(await self.message("hello?", OUT_OF_CREDIT))  # still out: no request
        status, = self.discord.channels["diary"].messages()
        return [(m, self.discord.answer(m)) for m in said], status

    async def test_a_message_to_rocky_is_answered_and_logged(self):
        answered, status = await self.conversation()
        ((plain, plain_reply), (remembered, remembered_reply), (secret, secret_reply), (long, parts), (ran_out, gif),
         (broke, gif_again)) = answered

        # Each answer is a reply to the message, the long one in two parts, with what Rocky did under it
        self.assertEqual([m.content for m in plain_reply], ["Granite. Strong, strong, strong."])
        self.assertEqual(self.discord.reactions, [(plain.id, "🪨")])
        self.assertEqual([m.content for m in remembered_reply],
                         ["Biscuit. Good name.\n-# 🧠 remembered: pip has a cat called Biscuit"])
        # under the message it wrote once it had seen what the bar said, with its note shown as made, but not what's in it
        self.assertEqual([m.content for m in secret_reply],
                         ["Rocky pick number. Secret. Bar closed until four, sad sad sad.\n-# 📝 made a note\n"
                          "-# 🍺 checked the bar"])
        self.assertFalse(any("picked 7" in m.content or "Seven" in m.content for m in self.discord.messages.values()))
        self.assertEqual([m.reference and m.reference.message_id for m in parts], [long.id, None])
        self.assertEqual(" ".join(m.content for m in parts), LONG)
        self.assertTrue(all(len(m) == 1 and m[0].content in j.NO_MONEY for m in (gif, gif_again)))
        self.assertEqual(status.content, "I feel granite is best rock.")

        # logs/ has a line for each, with what it cost, every request, and the ids of what was posted
        lines = [json.loads(line) for path in sorted(j.LOG_DIR.glob("*.jsonl")) for line in path.read_text().splitlines()]
        self.assertEqual([t.get("message_id") for t in lines],
                         [plain.id, remembered.id, secret.id, long.id, None, ran_out.id, broke.id])
        turns = {t.get("message_id"): t for t in lines}
        for m, posted, requests in ((plain, plain_reply, 1), (remembered, remembered_reply, 2), (secret, secret_reply, 2),
                                    (long, parts, 1)):
            t = turns[m.id]
            self.assertEqual((t["llm_model"], t["requests"], t["reply_id"]), (j.LLM_MODEL, requests, posted[0].id))
            self.assertAlmostEqual(t["cost"], .0016 * requests)
            self.assertEqual([(c["purpose"], c["status"]) for c in t["api_calls"]], [("reply", 200)] * requests)
        self.assertEqual(turns[long.id]["reply_ids"], [m.id for m in parts])
        self.assertEqual(turns[plain.id]["reaction"], "🪨")
        self.assertEqual((lines[4]["status_id"], lines[4]["api_calls"][0]["purpose"]), (status.id, "status"))
        self.assertEqual([c["status"] for c in turns[ran_out.id]["api_calls"]], [402])
        self.assertNotIn("api_calls", turns[broke.id])
        self.assertEqual((turns[ran_out.id]["reply_id"], turns[broke.id]["reply_id"]), (gif[0].id, gif_again[0].id))

        # Rocky's log of the channel has the conversation, each of its messages by the id it was posted with
        rh.flush()
        saved = json.loads((rh.LOG_DIR / f"{plain.channel.id}.json").read_text())
        self.assertEqual([(e["type"], e["text"]) for e in saved["events"] if e["type"] in ("message", "rocky") and e["text"]],
                         [("message", "@Rocky which rock is best"), ("rocky", "Granite. Strong, strong, strong."),
                          ("message", "@Rocky my cat is called biscuit"), ("rocky", "Biscuit. Good name."),
                          ("message", "@Rocky pick a number, and is the bar open?"),
                          ("rocky", "Rocky pick number. Secret. Bar closed until four, sad sad sad."),
                          ("message", f"@Rocky tell me everything about granite. {LONG}"), ("rocky", LONG),
                          ("message", "@Rocky and basalt?"), ("message", "@Rocky hello?")])
        self.assertEqual({int(k) for k, v in saved["ids"].items() if v == "you"},
                         {m.id for m in plain_reply + remembered_reply + secret_reply + parts})
        self.assertEqual(rh.logs[long.channel.id].rocky_text(parts[1].id), LONG)  # a reaction to either part is to it
        # and so does the channel's history, without the lines under a reply
        history = {e["id"]: (e.get("reply"), e.get("reply_id")) for e in j.channel_history[long.channel.id]}
        self.assertEqual((history[remembered.id], history[secret.id], history[long.id]),
                         (("Biscuit. Good name.", remembered_reply[0].id),
                          ("Rocky pick number. Secret. Bar closed until four, sad sad sad.", secret_reply[0].id),
                          (LONG, parts[0].id)))
        self.assertEqual(rh.memory("400")["facts"][0]["fact"], "has a cat called Biscuit")

        # A turn that only reacts is saved too, though nothing in the channel changes after it, so a restart then keeps it
        async def thinking():  # longer than SAVE_AFTER: the message's own save is written meanwhile
            rh.flush()
            return 200, completion(answer(calls=[("react", {"emoji": "🗿"})]))
        j.has_credit = True
        nod = await self.message("granite forever", thinking())
        self.assertEqual((self.discord.answer(nod), self.discord.reactions[-1]), ([], (nod.id, "🗿")))
        rh.flush()  # what a restart writes
        saved = json.loads((rh.LOG_DIR / f"{nod.channel.id}.json").read_text())
        self.assertEqual([(e["type"], e.get("text")) for e in saved["events"][-3:]],
                         [("message", "@Rocky granite forever"), ("rocky", ""), ("result", None)])
        self.assertEqual(saved["asked"], rh.logs[nod.channel.id].asked.isoformat())

        # A log started afresh reads the channel back from Discord: Rocky's replies come back as it wrote them, and
        # the lines under them don't, so it never reads them as its own words
        rh.logs.clear()
        (rh.LOG_DIR / f"{plain.channel.id}.json").unlink()
        j.channel_history.clear()
        j.history_loaded.clear()
        j.has_credit = True
        with openrouter((200, completion(answer("Rocky not say.")))) as s:
            await j.on_message(self.discord.say("bot-spam", "pip", "what did you pick?"))
        sent = json.dumps(s.post.call_args.kwargs["json"]["messages"], ensure_ascii=False)
        self.assertIn("Biscuit. Good name.", sent)
        self.assertIn("Rocky pick number. Secret. Bar closed until four, sad sad sad.", sent)
        self.assertNotIn("-#", sent)

    async def test_the_actions_explain_what_was_posted(self):
        answered, status = await self.conversation()
        (plain, [reply]), (_, [with_actions]), (_, [secret]), (_, [first, second]), (_, [gif]), (_, [gif_again]) = answered
        # and a turn logged before Context kept the request, with the log as text
        old = self.discord.post(plain.channel, "Old answer.", self.discord.me)
        with open(j.LOG_DIR / f"{old.created_at:%Y-%m-%d}.jsonl", "a") as f:
            f.write(json.dumps({"at": old.created_at.isoformat(), "channel_id": old.channel.id, "message_id": 1,
                                "reply": "Old answer.", "reply_id": old.id, "llm_model": j.LLM_MODEL,
                                "transcript": "<memories>\n[1] pip: likes soup\n</memories>\n<message id=\"1\" from=\"pip\">hi</message>\nrocky: Old answer."}) + "\n")
        targets = {"plain reply": reply, "message that woke it": plain, "reply with action lines": with_actions,
                   "reply after a note and the bar": secret, "old answer": old,
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

        # Context: the request as it was built, a block per message. What came before is only said, with the system
        # prompt and memories by their size; this turn's messages are in full
        def context(name):
            text, embed, files = answers[name][0]
            labels = [line for line in embed["description"].splitlines() if line.startswith("**")]
            return embed, labels, files.get("context.txt", b"").decode()
        embed, labels, file = context("plain reply")
        self.assertRegex(labels[0], r"^\*\*system\*\* · [\d,]+ characters, not shown · cache breakpoint$")
        self.assertEqual(labels[1:], ["**user** · <memories>: 0 facts, not shown · cache breakpoint",
                                      "**user** · cache breakpoint", "**assistant**", "**↳ calls react**",
                                      "**tool** · react's result"])
        self.assertIn('woke_you="yes">@Rocky which rock is best</message>\n```', embed["description"])
        self.assertIn('```json\n{"emoji": "\\ud83e\\udea8"}\n```', embed["description"])  # as the API had it
        self.assertEqual(file, "")  # all of it is there
        # The first request read nothing from the cache, and wrote it all for the next
        self.assertEqual(fields("plain reply", 0)["Input cache"].splitlines()[1], "0 read · 8,990 written · 10 uncached")
        self.assertEqual(fields("reply with action lines", 0)["Input cache"].splitlines()[1], "8,000 read · 1,000 uncached")

        # A round trip: the call, its result, then what it said. A note shows it was made, never what it says, and
        # thinking only that there was some
        embed, labels, file = context("reply after a note and the bar")
        self.assertEqual(labels[2:], ["**7 earlier messages** · in context.txt", "**user** · cache breakpoint",
                                      "**assistant** · 1 thinking block, hidden", "**↳ calls note** · what it notes is secret",
                                      "**↳ calls bar**", "**tool** · note's result", "**tool** · bar's result", "**assistant**"])
        self.assertTrue(embed["description"].endswith(
            "**assistant**\n```\nRocky pick number. Secret. Bar closed until four, sad sad sad.\n```"))
        self.assertIn("my cat is called biscuit", file)  # the earlier messages, as labelled blocks
        self.assertIn("── tool · remember's result ──\nRemembered as [1].", file)
        for shown in (embed["description"], file):
            self.assertNotIn("picked 7", shown)
            self.assertNotIn("seven, they said", shown)
        # A turn too long to show whole has the middle of its longest blocks cut, and all of it attached
        embed, labels, file = context("long reply's first part")
        self.assertIn("characters cut, all in context.txt", embed["description"])
        self.assertLessEqual(len(embed["description"]), 4096)
        self.assertIn(LONG, file)
        # A status: what it was asked, and what it wrote
        embed, labels, file = context("status")
        self.assertEqual([label.split(" · ")[0] for label in labels], ["**system**", "**user**", "**assistant**"])
        self.assertIn("I feel granite is best rock.", embed["description"])
        # An old trace's log as text, as Context showed it then, but for what Rocky remembers
        self.assertEqual(context("old answer")[0]["description"],
                         '```\n<message id="1" from="pip">hi</message>\nrocky: Old answer.\n```')

        self.assertEqual(fields("plain reply", 1)["Model"], j.LLM_MODEL)
        self.assertEqual(answers["message that woke it"], answers["plain reply"])
        self.assertEqual(fields("reply with action lines", 1)["Total cost"], "$0.003200 USD")
        self.assertIn("Rocky · 2 requests", fields("reply with action lines", 2)["Write the reply"])
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
