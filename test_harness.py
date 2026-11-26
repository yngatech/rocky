"""Offline checks for rocky_harness: the channel log, the requests it makes, Rocky's turns, and keeping all of it.
The model's answers are synthetic.

    uv run --with-requirements requirements.txt python -m unittest test_harness
"""

import contextlib
import json
import os
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fakes import answer, isolate_rocky, patch_all, session, traced
import rocky_bot as j
import rocky_harness as rh

T0 = datetime(2026, 10, 12, 18, 0, tzinfo=timezone.utc)
git_head = rh.head  # the real one, for KeepingTests.repo


class _Case(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.trace = traced(self)
        isolate_rocky(self, self.dir.name)
        patch_all(self, [(j, "TIMEZONE", j.ZoneInfo("UTC")), (j, "ROCKY_LINES", ["Amaze, amaze, amaze!"])])
        self.log = rh.Log("bot-spam", "400", 300)

    def said(self, author, text, minutes, to_bot=False, **kwargs):
        return self.log.message(author, text, T0 + timedelta(minutes=minutes), to_bot=to_bot, **kwargs)

    def thinking(self, request):  # where each of its thinking blocks is in a request: (message, block)
        return [(i, k) for i, m in enumerate(request["messages"]) if m["role"] == "assistant"
                for k, b in enumerate(m["content"]) if b["type"] in ("thinking", "redacted_thinking")]

    def remember_soup(self):  # in the server's memories
        rh.use(self.log, {"function": {"name": "remember", "arguments": '{"about": "pip", "fact": "likes soup"}'}}, {"n": 1})


class RenderTests(_Case):
    def test_people_are_user_turns_and_rocky_the_assistant(self):
        self.said("kettle", "anyone up?", 0)
        q = self.said("pip", "rocky <3 soup?", 1, to_bot=True, discord_id=900)
        q["woke"] = True
        self.log.add("rocky", text="Soup good.", calls=[], id=901)
        self.log.ids["901"] = "you"
        self.said("pip", "why", 2, to_bot=True, reply_to="you", you_said="Soup good.")
        self.log.add("reaction", on="you", you_said="Soup good.", emoji="😂", author="moss")
        self.log.add("edit", on=1, text="anyone up??")
        request = rh.render(self.log, "rocky")
        system, (memories, first, rocky, second) = request["system"], request["messages"]
        self.assertIn("woke_you", system[0]["text"])
        self.assertEqual(system[0]["cache_control"], rh.CACHE)
        self.assertEqual(memories["content"][0]["text"], "<memories>\nNothing yet.\n</memories>")
        self.assertEqual(first["content"], '<day>Mon 12 Oct</day>\n<message id="1" from="kettle" at="18:00">anyone up?</message>\n'
                                           '<message id="2" from="pip" at="18:01" to="you" woke_you="yes">rocky &lt;3 soup?</message>')
        self.assertEqual(rocky, {"role": "assistant", "content": [{"type": "text", "text": "Soup good."}]})
        text = second["content"][0]["text"]
        self.assertIn('to="you" reply_to="your message: Soup good.">why</message>', text)
        self.assertIn('<reaction from="moss" emoji="😂" on="your message: Soup good."/>', text)
        self.assertIn('<edit id="1">anyone up??</edit>', text)
        self.assertEqual(second["content"][0]["cache_control"], rh.CACHE)  # the end of the request

    def test_a_new_turn_leaves_everything_before_it_as_it_was(self):
        q = self.said("pip", "soup?", 0, to_bot=True)
        q["woke"] = True
        before = rh.render(self.log, "rocky")
        self.log.add("rocky", text="", calls=[{"id": "c1", "name": "remember", "arguments": '{"about": "pip", "fact": "likes soup"}'}],
                     reasoning=[{"type": "reasoning.encrypted", "data": "x"}])
        rh.use(self.log, {"function": {"name": "remember", "arguments": '{"about": "pip", "fact": "likes soup"}'}}, q)
        self.log.add("result", id="c1", content="Remembered as [1].")
        self.said("kettle", "rocky hi", 1, to_bot=True)["woke"] = True
        after = rh.render(self.log, "rocky")
        self.assertEqual(after["system"], before["system"])
        texts = lambda request: [j.text_of(m["content"]) for m in request["messages"]]
        self.assertEqual(texts(after)[:len(before["messages"])], texts(before))  # memories too: a snapshot
        self.assertEqual(after["messages"][2], {"role": "assistant", "content": [
            {"type": "tool_use", "id": "c1", "name": "remember", "input": {"about": "pip", "fact": "likes soup"}}]})
        self.assertEqual(after["messages"][3], {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "Remembered as [1]."}]})
        self.log.reorganise()  # at a jump
        self.assertIn("[1] pip: likes soup", j.text_of(rh.render(self.log, "rocky")["messages"][0]["content"]))

    def test_only_the_turn_in_progress_sends_its_thinking_as_it_was_kept(self):
        # A log as it was kept through OpenRouter: a turn's thinking as its reasoning_details, with their summaries
        kept = lambda n: [{"type": "reasoning.text", "text": f"summary {n}", "signature": f"signature-{n}",
                           "format": "anthropic-claude-v1", "index": 0}]
        self.said("pip", "rocky soup?", 0, to_bot=True)["woke"] = True
        self.log.add("rocky", text="Soup good.", calls=[], reasoning=kept(1))
        self.said("pip", "rocky bar?", 1, to_bot=True)["woke"] = True
        self.log.add("rocky", text="", calls=[{"id": "c1", "name": "bar", "arguments": "{}"}], reasoning=kept(2))
        self.log.add("result", id="c1", content="Closed for the night.")
        follow_up = rh.render(self.log, "rocky")
        # the request after its call: that turn's thinking, first in its message and unaltered; none from before
        self.assertEqual(self.thinking(follow_up), [(4, 0)])
        self.assertEqual(follow_up["messages"][4]["content"][0],
                         {"type": "thinking", "thinking": "summary 2", "signature": "signature-2"})
        self.log.add("rocky", text="Bar shut.", calls=[])
        self.said("pip", "rocky ok", 2, to_bot=True)["woke"] = True
        self.assertEqual(self.thinking(rh.render(self.log, "rocky")), [])  # the next turn: none
        self.assertEqual(self.log.events[1]["reasoning"], kept(1))  # and the log keeps it all
        # A warm-up refreshes the cache up to the people's turn before Rocky's last: not that turn, nor its thinking
        self.log.bot_name = "rocky"
        warm = rh.warm_request(self.log)["messages"]
        self.assertEqual((self.thinking({"messages": warm}), j.text_of(warm[-1]["content"]).endswith("rocky bar?</message>")),
                         ([], True))

    def test_what_the_api_wont_take_from_an_old_log_is_left_out(self):
        self.said("pip", "rocky lol", 0, to_bot=True)
        # a call cut off mid-way through OpenRouter, kept half written; a reply read back from Discord with no words
        self.log.add("rocky", text="", calls=[{"id": "c1", "name": "react", "arguments": '{"emoji": "😂'}])
        self.log.add("result", id="c1", content="That didn't work")
        self.log.add("rocky", text="", calls=[])
        self.said("pip", "rocky hi", 1, to_bot=True)
        messages = rh.render(self.log, "rocky")["messages"]
        self.assertEqual([m["role"] for m in messages], ["user", "user", "assistant", "user", "user"])
        self.assertEqual(messages[2]["content"], [{"type": "tool_use", "id": "c1", "name": "react", "input": {}}])

    def test_after_a_tool_call_the_cache_mark_stays_where_the_last_request_ended(self):
        q = self.said("pip", "rocky remember soup", 0, to_bot=True)
        q["woke"] = True
        first = rh.render(self.log, "rocky")["messages"]
        self.log.add("rocky", text="", calls=[{"id": "c1", "name": "remember", "arguments": '{"about": "pip", "fact": "soup"}'}])
        self.log.add("result", id="c1", content="Remembered as [1].")
        follow_up = rh.render(self.log, "rocky")["messages"]
        self.assertEqual(follow_up[1], first[1])  # people's turn, marked as the first request had it
        self.assertEqual(follow_up[1]["content"][0]["cache_control"], rh.CACHE)
        self.assertEqual(follow_up[-1]["content"][0]["type"], "tool_result")

    def test_memories_are_the_servers_not_the_channels(self):
        self.remember_soup()
        other = rh.Log("general", "400", 301)
        other.reorganise()
        self.assertIn("pip: likes soup", rh.memories_text(other))
        self.assertNotIn("soup", rh.memories_text(rh.Log("dm", "dm-5", 5)))

    def test_a_log_without_memories_has_none_and_cant_keep_any(self):
        self.remember_soup()
        log = rh.Log("favourite rock?", None, 900)
        log.message("pip", "favourite rock?", T0, to_bot=True)["woke"] = True
        log.reorganise()
        request = rh.render(log, "rocky")
        (first,) = request["messages"]  # no <memories> before it
        prompt = request["system"][0]["text"]
        self.assertIn(rh.NO_MEMORIES, prompt)
        self.assertNotIn("<memories>", prompt)
        self.assertNotIn("- remember:", prompt)
        self.assertIn("favourite rock?", j.text_of(first["content"]))
        self.assertEqual([t["name"] for t in rh.tools_for(log)], ["react", "note", "bar", "order", "look"])
        self.assertEqual(rh.use(log, {"function": {"name": "remember", "arguments": '{"about": "pip", "fact": "likes rocks"}'}},
                                {"n": 1}), (None, "No memories here."))
        self.assertEqual([f["fact"] for f in rh.memory("400")["facts"]], ["likes soup"])
        # and a server's log is as it was
        self.assertIn("- remember:", j.text_of(rh.render(self.log, "rocky")["system"]))
        self.assertEqual([t["name"] for t in rh.tools_for(self.log)],  # think_back only with the book
                         ["react", "remember", "forget", "note", "bar", "order", "look"])


class ToolTests(_Case):
    def call(self, name, **args):
        return rh.use(self.log, {"function": {"name": name, "arguments": json.dumps(args)}}, {"n": 7})

    def test_react_defaults_to_the_message_that_woke_it(self):
        self.assertEqual(self.call("react", emoji="😂"), (("react", 7, "😂"), "Reacted."))
        self.assertEqual(self.call("react", emoji="🎉", message=3)[0], ("react", 3, "🎉"))

    def test_remember_keeps_short_settled_new_facts_and_forget_drops_them(self):
        self.assertEqual(self.call("remember", about="@pip", fact="is vegetarian.")[1], "Remembered as [1].")
        self.assertEqual(self.call("remember", about="Pip", fact="Is vegetarian")[1], "Already remembered as [1].")
        self.assertIn("Not kept", self.call("remember", about="rocky", fact="likes Rocky — no, Project Hail Mary? Avoid")[1])
        self.assertIn("Not kept", self.call("remember", about="pip", fact="x" * 101)[1])
        self.assertEqual(rh.memory("400")["facts"][0]["fact"], "is vegetarian")
        self.assertEqual(self.call("forget", number=1), (("forget", 1, "pip: is vegetarian"), "Forgot it."))
        self.assertEqual(self.call("forget", number=1)[1], "No such fact.")
        self.assertIn("didn't work", rh.use(self.log, {"function": {"name": "remember", "arguments": '{"about"'}}, {"n": 1})[1])

    def test_a_note_is_kept_as_its_call_and_left_out_of_context(self):
        self.assertEqual(self.call("note", text="picked 4"), (("note", None, None), "Noted."))  # shown as made, no more
        self.assertEqual(self.call("note", text=" ")[1], "Nothing to note.")
        self.log.add("rocky", text="", calls=[{"id": "c1", "name": "note", "arguments": '{"text": "picked 4"}'},
                                              {"id": "c2", "name": "react", "arguments": '{"emoji": "🤫"}'}])
        request = rh.render(self.log, "rocky")
        self.assertEqual(request["messages"][-1]["content"][0]["input"], {"text": "picked 4"})  # Rocky sees its note
        record = json.dumps(j.request_items(**request), ensure_ascii=False)
        self.assertNotIn("picked 4", record)  # Context, which anyone can open, doesn't
        self.assertIn('[{"name": "note"}, {"name": "react", "arguments": "{\\"emoji\\": \\"🤫\\"}"}]', record)

    def bar_at(self, at):
        patch_all(self, [(rh, "clock", lambda: at)])

    def day(self, shut=False, out=False):  # the first day from T0 that's like this
        return next(d for d in (T0.date() + timedelta(days=i) for i in range(365))
                    if bool(rh.bar_today(d)[0]) == shut and bool(rh.bar_today(d)[1]) == out)

    def test_the_bar_keeps_hours_and_some_days_doesnt_open(self):
        day = self.day()
        opens, closes = rh.HOURS[day.weekday()]
        at = lambda h, m=0: datetime(day.year, day.month, day.day, h, m, tzinfo=timezone.utc)
        self.bar_at(at(opens, 30))
        self.assertEqual(self.call("bar")[0], ("bar", None, None))
        self.assertIn("is open", self.call("bar")[1])
        self.assertIn("Blip-A Lager £4.90", self.call("bar")[1])
        self.assertEqual(rh.bar_state(at(opens - 1)), (False, f"Closed. Opens at {opens}:00."))
        self.assertIn("Last orders!", rh.bar_state(at(closes - 1, 50))[1])
        shut = self.day(shut=True)
        is_open, says = rh.bar_state(datetime(shut.year, shut.month, shut.day, 18, tzinfo=timezone.utc))
        self.assertFalse(is_open)
        self.assertRegex(says, r"^Closed today: .+\. Next open \w+ from \d+:00\.$")
        # and the same whenever it's asked that day
        self.assertEqual(rh.bar_today(shut), rh.bar_today(shut))

    def test_an_order_is_for_whats_on_while_its_open(self):
        day = self.day(out=True)
        out = rh.bar_today(day)[1]
        other = next(d for d in rh.MENU if d != out)
        self.bar_at(datetime(day.year, day.month, day.day, 20, tzinfo=timezone.utc))
        self.assertEqual(self.call("order", drink=other.lower()), (("order", other, None), f"Ordered: {other}, £{rh.MENU[other]:.2f}."))
        self.assertEqual(self.call("order", drink=other, **{"for": "@pip"})[1], f"Ordered: {other} for pip, £{rh.MENU[other]:.2f}.")
        self.assertIn(f"{out}'s run out", self.call("order", drink=out)[1])
        self.assertIn(f"{out} £{rh.MENU[out]:.2f} (run out)", self.call("bar")[1])
        self.assertTrue(self.call("order", drink="Guinness")[1].startswith("Not on the menu."))
        self.assertEqual(self.call("order", drink=out)[0], ("unordered", out, None))
        self.bar_at(datetime(day.year, day.month, day.day, 3, tzinfo=timezone.utc))
        self.assertEqual(self.call("order", drink=other), (("unordered", other, None), rh.bar_state()[1]))  # closed: what the bar says

    def test_a_wordless_answer_is_nothing_and_a_leading_tag_of_the_asker_goes(self):
        for text in ("(silent)", ".", "*nothing*", "[no reply]"):
            self.assertEqual(rh.said(text, "rocky"), "")
        self.assertEqual(rh.said("rocky: Soup good.", "rocky"), "Soup good.")
        self.assertEqual(rh.unaddressed("@Pip, soup good", "pip"), "soup good")
        self.assertEqual(rh.unaddressed("@Pipsqueak soup", "pip"), "@Pipsqueak soup")
        self.assertEqual(rh.unaddressed("soup @pip", "pip"), "soup @pip")


class TurnTests(_Case):
    async def turn(self, *answers):
        q = self.said("pip", "rocky my cat is called biscuit", 0, to_bot=True)
        llm = AsyncMock(side_effect=list(answers))
        with patch.object(j, "llm_message", llm):
            turn = await rh.take_turn(self.log, q)
        return turn, llm

    async def test_it_speaks_and_acts_in_one_response(self):
        turn, llm = await self.turn(answer("Biscuit. Good name.", [("react", {"emoji": "🐱"}),
                                                                    ("remember", {"about": "pip", "fact": "has a cat called Biscuit"})],
                                           thinking=[("a cat fact", "signature-cat")]))
        self.assertEqual((turn.text, turn.reactions, turn.kept, turn.requests),
                         ("Biscuit. Good name.", [(1, "🐱")], [("remember", 1, "pip: has a cat called Biscuit")], 1))
        (max_tokens,), request = llm.call_args
        self.assertEqual(max_tokens, rh.MAX_TOKENS)
        self.assertEqual({k: request[k] for k in ("tools", "thinking", "output_config")},
                         {"tools": rh.tools_for(self.log), **rh.THINKING})
        messages = request["messages"]
        self.assertIn('woke_you="yes"', j.text_of(messages[-1]["content"]))
        # marked for the cache: without the marks, every turn read the whole log at full price
        self.assertEqual([b["cache_control"] for b in (request["system"][0], messages[0]["content"][0],
                                                       messages[-1]["content"][0])], [rh.CACHE] * 3)
        rocky = next(e for e in self.log.events if e["type"] == "rocky")
        self.assertIs(turn.spoke, rocky)
        # its thinking kept as it always was, in OpenRouter's shape, so the log reads the same either way
        self.assertEqual(rocky["reasoning"], [{"type": "reasoning.text", "text": "a cat fact", "signature": "signature-cat",
                                               "format": "anthropic-claude-v1", "index": 0}])
        self.assertEqual([e["content"] for e in self.log.events if e["type"] == "result"], ["Reacted.", "Remembered as [1]."])

    async def test_after_only_remembering_or_a_note_it_gets_a_turn_to_speak(self):
        for call, result in [(("remember", {"about": "pip", "fact": "has a cat called Biscuit"}), "Remembered as [1]."),
                             (("note", {"text": "picked 4"}), "Noted.")]:
            with self.subTest(call=call[0]):
                typed = []

                @contextlib.asynccontextmanager
                async def typing():
                    typed.append(llm.await_count)
                    yield

                q = self.said("pip", "rocky my cat is called biscuit", 0, to_bot=True)
                llm = AsyncMock(side_effect=[answer(calls=[call]), answer("Biscuit. Good name.")])
                with patch.object(j, "llm_message", llm):
                    turn = await rh.take_turn(self.log, q, typing=typing)
                self.assertEqual(typed, [1])  # typing only for the request after the first: it's going to speak
                self.assertEqual((turn.text, turn.requests), ("Biscuit. Good name.", 2))
                # Both its responses and the result between them, for the day's log (and rocky_viewer)
                self.assertEqual([e["type"] for e in turn.events], ["rocky", "result", "rocky"])
                self.assertEqual(turn.events, self.log.events[-3:])
                self.assertEqual(llm.call_args.kwargs["messages"][-1],
                                 {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_0", "content": result}]})

    async def test_at_the_bar_it_sees_what_happened_before_it_speaks(self):
        with patch.object(rh, "bar_state", lambda now=None: (True, "The Hail Mary is open until 23:00. Menu: Blip-A Lager £4.90.")), \
                patch.object(rh, "bar_today", lambda d: (None, None)):  # nothing's run out, whatever the date
            turn, llm = await self.turn(answer("Rocky get beer!", [("bar", {}), ("react", {"emoji": "🍺"})]),
                                        answer(calls=[("order", {"drink": "Blip-A Lager"})]),
                                        answer("Rocky have beer. Good, good, good."))
        self.assertEqual((turn.text, turn.reactions, turn.requests), ("Rocky have beer. Good, good, good.", [(1, "🍺")], 3))
        # What it wrote before it knew isn't posted, nor shown to it as said
        self.assertEqual([e["text"] for e in self.log.events if e["type"] == "rocky"], ["", "", "Rocky have beer. Good, good, good."])
        self.assertEqual(llm.call_args.kwargs["messages"][-1]["content"][-1]["content"], "Ordered: Blip-A Lager, £4.90.")

    async def test_what_it_did_is_shown_in_call_order(self):
        rh.memory("400")["facts"].append({"n": 9, "about": "pip", "fact": "likes lager", "at": ""})
        with patch.object(rh, "bar_state", lambda now=None: (True, "The Hail Mary is open until 23:00. Menu: lots.")), \
                patch.object(rh, "bar_today", lambda d: (None, None)):
            turn, llm = await self.turn(
                answer(calls=[("order", {"drink": "eridani stout", "for": "@pip"}), ("note", {"text": "picked 4"}),
                              ("forget", {"number": 9}), ("forget", {"number": 99}),
                              ("remember", {"about": "pip", "fact": "has a cat called Biscuit"}),
                              ("remember", {"about": "pip", "fact": "has a cat called biscuit"}),
                              ("remember", {"about": "pip", "fact": "likes it? maybe? avoid"}),
                              ("react", {"emoji": "🍺"}), ("bar", {})]),
                answer("ignored, it went to the bar", [("bar", {}), ("order", {"drink": "Moon\nshine https://example.org/moon"})]),
                answer("Stout for Pip."))
        self.assertEqual(turn.text, "Stout for Pip.")
        self.assertEqual([k[0] for k in turn.kept], ["forget", "forget", "remember"])  # orders aren't memories
        # Not the note's text, a forget of no fact, a fact already kept or one not kept, nor the bar twice; and whatever
        # it wrote in a call stays on its own line, links unpreviewed
        self.assertEqual(j.action_lines(turn.shown), ["-# 🍺 ordered Eridani Stout for pip",
                                                      "-# 📝 made a note",
                                                      "-# 🗑️ forgot: pip likes lager",
                                                      "-# 🧠 remembered: pip has a cat called Biscuit",
                                                      "-# 🍺 checked the bar",
                                                      "-# 🍺 couldn't order Moon shine <https://example.org/moon>"])

    async def test_only_reacting_or_nothing_is_one_request(self):
        turn, llm = await self.turn(answer(calls=[("react", {"emoji": "😂"})]))
        self.assertEqual((turn.text, turn.reactions, llm.await_count), ("", [(1, "😂")], 1))
        events = len(self.log.events)
        turn, llm = await self.turn(answer("(silent)"))
        self.assertEqual((turn.text, turn.reactions, llm.await_count), ("", [], 1))
        self.assertEqual(len(self.log.events), events + 1)  # just the message — an empty turn isn't kept
        turn, llm = await self.turn(answer(calls=[("note", {"text": "picked 4"}), ("react", {"emoji": "🤫"})]))
        self.assertEqual((turn.reactions, llm.await_count), ([(3, "🤫")], 1))

    async def test_a_message_that_comes_while_it_waits_goes_after_its_turn(self):
        first = self.said("pip", "rocky soup?", 0, to_bot=True)
        meanwhile = []

        async def llm(max_tokens, messages, **kwargs):
            if not meanwhile:  # a second message to it arrives during the first request, as live
                meanwhile.append(self.said("moss", "rocky also tea?", 0, to_bot=True))
                return answer("Soup good.", [("remember", {"about": "pip", "fact": "likes soup"})])
            self.assertEqual(messages[-1]["role"], "user")  # Haiku 5.5 refuses a request ending with Rocky's turn
            return answer("Tea good too.")

        with patch.object(j, "llm_message", AsyncMock(side_effect=llm)) as called:
            turn = await rh.take_turn(self.log, first)
            self.assertEqual(turn.text, "Soup good.")
            self.assertEqual([e["type"] for e in self.log.events], ["message", "rocky", "result", "message"])
            turn = await rh.take_turn(self.log, meanwhile[0])
        self.assertEqual((turn.text, called.await_count), ("Tea good too.", 2))
        self.assertIn("also tea?", j.text_of(called.call_args.kwargs["messages"][-1]["content"]))

    async def test_a_message_that_comes_during_its_tool_calls_still_has_its_own_turn(self):
        first = self.said("pip", "rocky soup?", 0, to_bot=True)
        meanwhile = []

        @contextlib.asynccontextmanager
        async def typing():  # Discord's typing is a request too: a message can arrive while it's sent
            meanwhile.append(self.said("kettle", "rocky biscuit?", 0, to_bot=True))
            yield

        async def llm(max_tokens, messages, **kwargs):
            text = "".join(j.text_of(m["content"]) for m in messages if m["role"] == "user")
            if "biscuit" in text:  # only a later turn sees it
                self.assertEqual(messages[-1]["role"], "user")
                return answer("Tea good, biscuit good.")
            if not isinstance(messages[-1]["content"], list) or messages[-1]["content"][-1]["type"] != "tool_result":
                meanwhile.append(self.said("moss", "rocky also tea?", 0, to_bot=True))  # during the first request
                return answer("", [("look", {})])
            self.assertNotIn("also tea?", text)  # its follow-up is the turn it began, nothing since
            return answer("Soup good.")

        with patch.object(j, "llm_message", AsyncMock(side_effect=llm)) as called, \
                patch.object(rh, "pictures", AsyncMock(return_value=[])):
            turn = await rh.take_turn(self.log, first, typing=typing)
            self.assertEqual(turn.text, "Soup good.")
            self.assertEqual([e["type"] for e in self.log.events],
                             ["message", "rocky", "result", "rocky", "message", "message"])
            tea = await rh.take_turn(self.log, meanwhile[0])
            biscuit = await rh.take_turn(self.log, meanwhile[1])
        self.assertTrue(tea.requests == 1 and biscuit.requests == 0)  # tea's request saw biscuit: one answer for both
        self.assertEqual(called.await_count, 3)

    async def test_with_nothing_new_since_its_turn_it_makes_no_request(self):
        first = self.said("pip", "rocky soup?", 0, to_bot=True)
        second = self.said("moss", "rocky also tea?", 0, to_bot=True)  # in the first turn's request
        llm = AsyncMock(side_effect=[answer("Soup good, tea good.")])
        with patch.object(j, "llm_message", llm):
            await rh.take_turn(self.log, first)
            turn = await rh.take_turn(self.log, second)
        self.assertEqual((turn.text, turn.requests, llm.await_count), ("", 0, 1))
        # Both were the first turn's to answer, and its request said so
        self.assertTrue(first["woke"] and second["woke"])
        self.assertEqual(j.text_of(llm.await_args.kwargs["messages"][-1]["content"]).count('woke_you="yes"'), 2)

    async def test_a_message_answered_elsewhere_isnt_this_turns_nor_does_a_status_answer_one(self):
        earlier = self.said("kettle", "rocky biscuit?", 0, to_bot=True)  # answered by Rocky outside a turn
        self.log.add("rocky", text="Biscuit good.", calls=[], id=901, elsewhere=True)
        chatter = self.said("moss", "tea anyone", 1)
        q = self.said("pip", "rocky soup?", 2, to_bot=True)
        self.log.add("rocky", text="Mood: soup.", calls=[], id=902, elsewhere=True)  # a status, as spoke_elsewhere adds it
        llm = AsyncMock(side_effect=[answer("Soup good.")])
        with patch.object(j, "llm_message", llm):
            turn = await rh.take_turn(self.log, q)
        self.assertEqual(("woke" in earlier, "woke" in chatter, q["woke"]), (False, False, True))
        self.assertEqual((turn.text, llm.await_count), ("Soup good.", 1))
        self.assertEqual(llm.await_args.kwargs["messages"][-1]["role"], "user")  # never ending with Rocky's

    async def test_after_a_status_its_turn_keeps_what_it_was_answering_in_its_follow_ups(self):
        q = self.said("pip", "rocky bar?", 0, to_bot=True)
        self.log.add("rocky", text="Mood: soup.", calls=[], id=902, elsewhere=True)  # a status, after what woke it
        llm = AsyncMock(side_effect=[answer(calls=[("bar", {})], thinking=[("bar first", "signature-bar")]),
                                     answer("Bar shut.")])
        with patch.object(j, "llm_message", llm):
            await rh.take_turn(self.log, q)
        first, follow_up = (c.kwargs["messages"] for c in llm.await_args_list)
        self.assertEqual(first[-1], rh.your_turn())
        # the same, then its turn: its thinking first in its own message, not after the status's words
        self.assertEqual(follow_up[:len(first)], first)
        self.assertEqual(follow_up[len(first)]["content"][0], {"type": "thinking", "thinking": "bar first",
                                                               "signature": "signature-bar"})


class LookTests(_Case):
    PICTURE = "data:image/jpeg;base64,AAAA"

    def png(self, size, mode="RGBA", fmt="PNG"):
        out = __import__("io").BytesIO()
        rh.Image.new(mode, size, (255, 128, 0, 0) if mode == "RGBA" else 7).save(out, fmt)
        return out.getvalue()

    async def look(self, *answers, found=None, error=None):
        q = self.said("pip", "rocky look at my cat [photo]", 0, to_bot=True, discord_id=900)
        llm = AsyncMock(side_effect=list(answers))
        pictures = AsyncMock(return_value=found, side_effect=error)
        with patch.object(j, "llm_message", llm), patch.object(rh, "pictures", pictures):
            turn = await rh.take_turn(self.log, q)
        return turn, llm, pictures

    async def test_it_sees_the_pictures_before_it_speaks(self):
        turn, llm, pictures = await self.look(answer("Cute cat!", [("look", {})]), answer("Cat orange. Fat, fat, fat."),
                                              found=[self.PICTURE])
        self.assertEqual(pictures.await_args.args[1], 1)  # the message that woke it
        self.assertEqual((turn.text, turn.requests), ("Cat orange. Fat, fat, fat.", 2))
        self.assertEqual(j.action_lines(turn.shown), ["-# 👀 looked at the picture"])
        # what it guessed before it looked isn't posted, nor shown to it as said
        self.assertEqual([e["text"] for e in self.log.events if e["type"] == "rocky"], ["", "Cat orange. Fat, fat, fat."])
        self.assertEqual(llm.call_args.kwargs["messages"][-1], {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_0", "content": [
                {"type": "text", "text": "Message 1's picture:"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"}}]}]})
        # Context counts it, without it
        self.assertEqual(j.request_items(**rh.render(self.log, "rocky"))[-2],
                         {"role": "tool", "name": "look", "content": "Message 1's picture:", "pictures": 1})
        # and it's still there in later requests, after a restart too
        rh.save_log(self.log)
        rh.logs.clear()
        again = rh.log_for(SimpleNamespace(id=300, name="bot-spam", guild=SimpleNamespace(id=400)))
        self.assertEqual(rh.render(again, "rocky"), rh.render(self.log, "rocky"))

    async def test_what_it_couldnt_see_it_is_told(self):
        for found, error, told in [([], None, "Message 1 has no pictures."), (None, None, "No message 1 to look at."),
                                   (None, j.discord.NotFound(SimpleNamespace(status=404, reason=""), "gone"),
                                    "Message 1 was deleted."),
                                   (None, j.aiohttp.ClientError("down"), "Couldn't see message 1's pictures.")]:
            with self.subTest(told=told):
                self.log = rh.Log("bot-spam", "400", 300)
                turn, llm, _ = await self.look(answer(calls=[("look", {"message": 1})]), answer("Rocky not see."),
                                               found=found, error=error)
                self.assertEqual(llm.call_args.kwargs["messages"][-1]["content"][-1]["content"], told)
                self.assertEqual(j.action_lines(turn.shown), ["-# 👀 couldn't see a picture"])
                self.assertNotIn('"image"', json.dumps(llm.call_args.kwargs["messages"]))

    async def test_a_messages_pictures_are_fetched_again_and_shrunk(self):
        big, gif, text = self.png((3000, 1500)), self.png((40, 40), "P", "GIF"), b"not a picture"
        files = {"https://cdn/a.png": big, "https://cdn/b.txt": text, "https://tenor/thumb.png": gif,
                 "https://cdn/sticker.png": self.png((160, 160))}
        m = SimpleNamespace(
            attachments=[SimpleNamespace(url="https://cdn/a.png", content_type="image/png"),
                         SimpleNamespace(url="https://cdn/b.txt", content_type="text/plain")],
            embeds=[SimpleNamespace(image=SimpleNamespace(url=None), thumbnail=SimpleNamespace(url="https://tenor/thumb.png"))],
            stickers=[SimpleNamespace(url="https://cdn/sticker.png", format=SimpleNamespace(name="png")),
                      SimpleNamespace(url="https://cdn/lottie.json", format=SimpleNamespace(name="lottie"))])
        channel = SimpleNamespace(fetch_message=AsyncMock(return_value=m))

        def get(url, **k):
            r = session()  # a response is its own `async with` too
            # in pieces, as it arrives, then b"" at its end
            chunks = [files[url][i:i + 1000] for i in range(0, len(files[url]), 1000)]
            r.status, r.content.read = 200, AsyncMock(side_effect=chunks + [b""])
            return r

        s = session()
        s.get.side_effect = get

        self.said("pip", "look [photo]", 0, discord_id=900)
        bot = SimpleNamespace(get_channel=lambda i: channel if i == 300 else None)
        with patch.object(j, "bot", bot), patch.object(j.aiohttp, "ClientSession", return_value=s):
            found = await rh.pictures(self.log, 1)
            self.assertIsNone(await rh.pictures(self.log, 2))  # not in the log
        channel.fetch_message.assert_awaited_once_with(900)
        self.assertEqual([c.args[0] for c in s.get.call_args_list],
                         ["https://cdn/a.png", "https://tenor/thumb.png", "https://cdn/sticker.png"])
        sizes = [rh.Image.open(__import__("io").BytesIO(__import__("base64").b64decode(u.split(",", 1)[1]))).size
                 for u in found]
        self.assertEqual(sizes, [(1024, 512), (40, 40), (160, 160)])
        self.assertTrue(all(u.startswith("data:image/jpeg;base64,") for u in found))
        self.assertIsNone(rh.shrunk(text))

    async def test_a_deleted_messages_pictures_go_when_the_cache_is_rebuilt(self):
        rh.logs[300] = self.log
        await self.look(answer(calls=[("look", {})]), answer("Cat."), found=[self.PICTURE])
        size = rh.chars(self.log)
        self.assertGreater(size, rh.PICTURE_CHARS)  # counted at its tokens, not its base64
        rh.deleted(300, 900)
        self.assertIn('"data": "AAAA"', json.dumps(rh.render(self.log, "rocky")))  # until the cache goes
        rh.tidy(self.log)
        self.assertNotIn('"data": "AAAA"', json.dumps(rh.render(self.log, "rocky")))
        self.assertIn("Message 1 was deleted.", json.dumps(rh.render(self.log, "rocky")))


class LongMessageTests(_Case):
    def test_a_message_too_long_for_discord_is_cut_at_sentences(self):
        self.assertEqual(j.message_parts("Short."), ["Short."])
        text = ("Rocky like beer. " * 150).strip()  # 2549 characters
        parts = j.message_parts(text)
        self.assertEqual(len(parts), 2)
        self.assertTrue(all(len(p) <= j.MESSAGE_LIMIT for p in parts))
        self.assertTrue(parts[0].endswith("beer."))
        self.assertEqual(" ".join(parts), text)
        self.assertEqual(j.message_parts("word " * 500, limit=100)[0], ("word " * 20).strip())  # no sentences: spaces
        self.assertEqual(j.message_parts("x" * 250, limit=100), ["x" * 100, "x" * 100, "x" * 50])


class KeepingTests(_Case):
    def channel(self):
        return SimpleNamespace(id=300, name="bot-spam", guild=SimpleNamespace(id=400))

    def test_a_reply_to_a_message_the_log_doesnt_have_shows_what_it_replied_to(self):
        user = SimpleNamespace(id=1, display_name="rocky")
        channel = self.channel()
        mine = SimpleNamespace(id=50, author=user, clean_content="Is good. Soup my favourite too.")
        theirs = SimpleNamespace(id=51, author=SimpleNamespace(id=7, display_name="kettle"), clean_content="soup is soup")
        sent = lambda i, reply: SimpleNamespace(id=i, channel=channel, author=SimpleNamespace(id=8, display_name="pip"),
                                                clean_content="this one ^^", created_at=T0, reference=SimpleNamespace(
                                                    message_id=reply.id, resolved=reply))
        with patch.object(j.bot._connection, "user", user), \
                patch.object(j, "should_respond", lambda m: True), patch.object(j, "message_text", lambda m: m.clean_content):
            rh.observe(sent(900, mine))
            rh.observe(sent(901, theirs))
            gone = sent(902, mine)
            gone.reference.resolved = j.discord.DeletedReferencedMessage(gone.reference)
            with patch.object(j.bot._connection, "_get_message", lambda i: None):
                rh.observe(gone)
        text = rh.people_turn(rh.log_for(channel).events, {})
        self.assertIn('reply_to="your older message: Is good. Soup my favourite too.">this one ^^', text)
        self.assertIn('reply_to="an older message from kettle: soup is soup">this one ^^', text)
        self.assertIn('<message id="3" from="pip" at="18:00" to="you">this one ^^', text)  # deleted: nothing to show

    def test_a_log_comes_back_after_a_restart(self):
        log = rh.log_for(self.channel())
        log.message("pip", "soup?", T0, to_bot=True, discord_id=900)
        log.add("rocky", text="Soup good.", calls=[], id=901, reasoning=[{"type": "reasoning.encrypted", "data": "x"}])
        log.ids["901"] = "you"
        log.asked = T0
        rh.save_log(log)
        rh.logs.clear()
        back = rh.log_for(self.channel())
        self.assertIsNot(back, log)
        self.assertEqual((back.space, back.ids, back.next_n, back.asked), ("400", {"900": 1, "901": "you"}, 2, T0))
        self.assertEqual(rh.render(back, "rocky"), rh.render(log, "rocky"))

    async def test_shutting_down_writes_what_was_waiting(self):
        log = rh.log_for(self.channel())
        log.message("pip", "soup?", T0, to_bot=True, discord_id=900)
        rh.changed(log)
        rh.use(log, {"function": {"name": "remember", "arguments": '{"about": "pip", "fact": "likes soup"}'}}, {"n": 1})
        self.assertFalse((rh.LOG_DIR / "300.json").exists())  # waiting, so a burst writes once
        rh.flush()
        self.assertEqual(json.loads((rh.LOG_DIR / "300.json").read_text())["ids"], {"900": 1})
        self.assertEqual(json.loads(rh.MEMORY_PATH.read_text())["400"]["facts"][0]["fact"], "likes soup")
        self.assertEqual((rh.log_saves, rh.memories_save), ({}, None))

    def repo(self):
        here = Path(self.dir.name) / "repo"
        here.mkdir()
        patch_all(self, [(rh, "HERE", here), (rh, "head", git_head)])
        git = lambda *a, **env: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *a],
                                               cwd=here, check=True, capture_output=True, text=True,
                                               env={**os.environ, **env}).stdout.strip()
        git("init", "-q")

        def commit(path, subject, **env):
            (here / path).write_text(subject)
            git("add", path)
            git("commit", "-qm", subject, **env)
            return git("rev-parse", "HEAD")
        return commit

    def test_after_an_update_to_rocky_its_logs_are_told(self):
        commit = self.repo()
        first = commit("rocky_harness.py", "generate: give Rocky a note tool (#100)")
        log = rh.log_for(self.channel())
        self.assertEqual((log.code, log.events), (first, []))  # a new log starts as Rocky is now
        log.message("pip", "soup?", T0, to_bot=True)
        rh.save_log(log)
        rh.logs.clear()
        commit("rocky_bot.py", "status: something about Jev (#101)")
        bar = commit("rocky_harness.py", "generate: give Rocky a bar to order drinks from (#106)")
        quiet = commit("rocky_harness.py", f"generate: send Rocky's requests another way (#107)\n\nWhy.\n\n{rh.NO_UPDATE}")
        before = rh.render(log, "rocky")
        back = rh.log_for(self.channel())
        self.assertEqual([(e["commit"], e["subject"]) for e in back.events if e["type"] == "update"],
                         [(bar, "give Rocky a bar to order drinks from")])  # not the one marked as no update to him
        self.assertEqual(back.code, quiet)
        after = rh.render(back, "rocky")
        self.assertEqual(after["system"], before["system"])  # appended: the request before it is as it was
        self.assertEqual(after["messages"][:-1], before["messages"][:-1])
        self.assertIn("<update>give Rocky a bar to order drinks from</update>", j.text_of(after["messages"][-1]["content"]))
        rh.save_log(back)
        rh.logs.clear()
        self.assertEqual(len(rh.log_for(self.channel()).events), 2)  # told once

    def test_a_log_from_before_updates_gets_them_where_they_happened(self):
        commit = self.repo()
        date = lambda minutes: {"GIT_COMMITTER_DATE": (T0 + timedelta(minutes=minutes)).isoformat()}
        commit("rocky_harness.py", "generate: before the log began (#1)", **date(-60))
        commit("rocky_harness.py", "generate: give Rocky a note tool (#100)", **date(2))
        commit("rocky_bot.py", "status: something about Jev (#101)", **date(3))
        bar = commit("rocky_harness.py", "generate: give Rocky a bar to order drinks from (#106)", **date(30))
        log = rh.log_for(self.channel())
        log.code = None
        log.message("pip", "pick a number", T0, to_bot=True)
        log.add("rocky", text="", calls=[{"id": "c1", "name": "note", "arguments": "{}"}], at=T0 + timedelta(minutes=1))
        log.add("result", id="c1", content="Noted.", at=T0 + timedelta(minutes=3))  # after the note tool's commit
        log.message("pip", "is it 4", T0 + timedelta(minutes=10), to_bot=True)
        rh.save_log(log)
        rh.logs.clear()
        back = rh.log_for(self.channel())
        self.assertEqual([e.get("subject") or e["type"] for e in back.events],
                         ["message", "rocky", "result", "give Rocky a note tool", "message",
                          "give Rocky a bar to order drinks from"])
        self.assertEqual(back.code, bar)
        rendered = rh.render(back, "rocky")["messages"]  # a turn's calls still come straight before their results
        self.assertEqual([m["role"] for m in rendered[1:4]], ["user", "assistant", "user"])
        self.assertEqual(rendered[3]["content"], [{"type": "tool_result", "tool_use_id": "c1", "content": "Noted."}])

    def test_memories_come_back_after_a_restart(self):
        self.remember_soup()
        rh.save_memories()
        rh.memories.clear()
        rh.memories_loaded = False
        self.assertEqual(rh.memory("400")["facts"][0]["fact"], "likes soup")

    def test_the_first_sync_fills_in_history_before_and_later_ones_append(self):
        log = rh.log_for(self.channel())
        live = log.message("pip", "rocky hi", T0 + timedelta(minutes=5), to_bot=True, discord_id=905)
        history = [{"name": "kettle", "content": "soup?", "at": T0, "id": 900, "to_bot": True,
                    "reply": "Soup good.", "reply_id": 901, "reply_at": T0},
                   {"name": "moss", "content": "lol", "at": T0 + timedelta(minutes=1), "id": 902,
                    "to_bot": True, "reaction": "😂"},
                   {"name": "pip", "content": "rocky hi", "at": live["at"], "id": 905}]
        rh.sync(log, history)
        self.assertEqual([e["type"] for e in log.events], ["message", "rocky", "message", "rocky", "result", "message"])
        self.assertIs(log.events[-1], live)
        self.assertEqual(log.ids["901"], "you")
        # its reaction as it would have called it, the emoji itself: not "\ud83d\ude02", which Haiku didn't read as one
        self.assertEqual(log.events[3]["calls"][0]["arguments"], '{"emoji": "😂", "message": 3}')
        before = rh.render(log, "rocky")["messages"]
        rh.sync(log, history + [{"name": "kettle", "content": "back", "at": T0 + timedelta(minutes=9), "id": 909},
                                {"name": "kettle", "content": "old", "at": T0 - timedelta(days=1), "id": 899}])
        after = rh.render(log, "rocky")["messages"]
        self.assertEqual([j.text_of(m.get("content")) for m in after[:len(before) - 1]],
                         [j.text_of(m.get("content")) for m in before[:-1]])
        self.assertIn(">back</message>", j.text_of(after[-1]["content"]))  # added at the end, not where it was
        self.assertNotIn(">old</message>", j.text_of(after[-1]["content"]))

    def test_a_deleted_message_is_marked_now_and_blanked_when_the_cache_is_rebuilt(self):
        rh.logs[300] = self.log
        q = self.said("pip", "my password is hunter2", 0, discord_id=900)
        self.log.add("edit", on=q["n"], text="my password is hunter3")
        self.said("kettle", "oops", 1, discord_id=901)
        self.log.add("rocky", text="Hunter. Is name.", calls=[], id=902)
        self.log.ids["902"] = "you"
        self.remember_soup()
        before = [j.text_of(m.get("content")) for m in rh.render(self.log, "rocky")["messages"]]
        rh.deleted(300, 900)
        rh.deleted(300, 902)
        rh.deleted(300, 900)  # once is enough
        rh.deleted(300, 12345)  # never in the log
        after = [j.text_of(m.get("content")) for m in rh.render(self.log, "rocky")["messages"]]
        self.assertEqual(after[:len(before) - 1], before[:-1])  # nothing earlier changes yet
        self.assertIn("hunter2", after[1])  # Rocky still saw it said...
        self.assertTrue(after[-1].endswith('<deleted id="1"/>\n<deleted id="your message: Hunter. Is name."/>'))  # ...and gone
        rh.tidy(self.log)
        tidied = "\n".join(j.text_of(m.get("content")) for m in rh.render(self.log, "rocky")["messages"])
        self.assertNotIn("hunter", tidied.lower().replace("hunter. is name", ""))
        self.assertIn('<message id="1" from="pip" at="18:00" deleted="yes"/>', tidied)
        self.assertNotIn('<deleted id="1"/>', tidied)
        self.assertIn("[1] pip: likes soup", tidied)  # <memories> brought up to date at the same time

    def test_past_its_size_the_oldest_half_goes_at_a_message(self):
        for i in range(40):
            self.said("pip", "x" * 100, i, discord_id=1000 + i)
            self.log.add("rocky", text="", calls=[{"id": f"c{i}", "name": "react", "arguments": "{}"}])
            self.log.add("result", id=f"c{i}", content="Reacted.")
        self.remember_soup()
        with patch.object(rh, "LOG_TOKENS", (rh.chars(self.log) - 1) / rh.CHARS_PER_TOKEN):
            self.assertTrue(rh.jump(self.log))
        self.assertEqual(self.log.events[0]["type"], "message")
        self.assertEqual(len(self.log.events), 60)
        self.assertNotIn("1000", self.log.ids)
        self.assertIn("1039", self.log.ids)
        self.assertIn("likes soup", rh.memories_text(self.log))
        self.assertFalse(rh.jump(self.log))


if __name__ == "__main__":
    unittest.main()
