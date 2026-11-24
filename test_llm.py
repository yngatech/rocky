"""Offline behavioral checks for the LLM: its requests and statuses, with the LLM's answers synthetic.

    uv run --with-requirements requirements.txt python -m unittest test_llm
"""

import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fakes import openrouter, patch_all, session, traced
import rocky_bot as j


# What the LLM said to messages
async def said(messages):
    return ((await j.llm_message(messages)) or {}).get("content") or ""


class _Case(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.trace = traced(self)
        patch_all(self, [(j, "has_credit", True), (j, "channel_history", defaultdict(list)),
                         (j, "TIMEZONE", j.ZoneInfo("UTC")), (j, "recent_statuses", [])])  # not whatever .env sets


class RequestTests(_Case):
    async def test_a_request_is_recorded_with_its_billing_and_ending(self):
        with openrouter((200, {"id": "gen-synthetic-2", "provider": "Anthropic",
                               "choices": [{"message": {"content": "Coffee first. Then walk."}, "finish_reason": "stop",
                                            "native_finish_reason": "stop"}],
                               "usage": {"cost": 0, "is_byok": True,
                                         "cost_details": {"upstream_inference_cost": .000577}}})), \
                patch.object(j.log, "warning") as warning:
            reply = await said([{"role": "user", "content": "coffee?"}])
        self.assertEqual(reply, "Coffee first. Then walk.")
        call, = self.trace["api_calls"]
        self.assertEqual((call["purpose"], call["model"], call["is_byok"]), ("reply", j.LLM_MODEL, True))
        # paid on our own Anthropic key
        self.assertEqual(round(call["router_cost"] + call["key_cost"], 6), .000577)
        self.assertAlmostEqual(self.trace["byok_cost"], .000577)
        self.assertEqual((call["finish_reason"], call["provider"], call["id"]), ("stop", "Anthropic", "gen-synthetic-2"))
        self.assertIsNone(j.stopped_early(call))  # an ordinary ending, without a warning
        warning.assert_not_called()
        self.assertEqual(self.trace["requests"], 1)
        self.assertAlmostEqual(self.trace["cost"], .000577)

    async def test_a_request_logs_the_prompt_size_for_context(self):
        with openrouter((200, {"choices": [{"message": {"content": "Is Biscuit."}}],
                               "usage": {"prompt_tokens": 3_100, "completion_tokens": 18, "cost": 0.001,
                                         "prompt_tokens_details": {"cached_tokens": 2_900}}})) as s:
            await said([{"role": "system", "content": "voice"}, {"role": "user", "content": "cat?"}])
        messages = s.post.call_args.kwargs["json"]["messages"]
        self.assertEqual(self.trace["llm_window"], 1_000_000)
        self.assertEqual(self.trace["llm_prompt_tokens"], 3_100)
        self.assertEqual(self.trace["llm_completion_tokens"], 18)
        self.assertEqual(self.trace["llm_cached_tokens"], 2_900)
        self.assertEqual(self.trace["llm_prompt_chars"], sum(len(m["content"]) for m in messages))

    async def test_a_rate_limited_request_is_tried_again(self):
        limited = (429, {"error": {"code": 429}})
        with openrouter(limited, limited, (200, {"choices": [{"message": {"content": "Is Biscuit."}}],
                                                 "usage": {"prompt_tokens": 10}})):
            text = await said([{"role": "user", "content": "cat?"}])
        self.assertEqual(text, "Is Biscuit.")
        self.assertEqual([r["status"] for r in self.trace["api_calls"]], [429, 429, 200])
        self.assertEqual([r["attempt"] for r in self.trace["api_calls"]], [0, 1, 2])
        self.assertTrue(all(r["router_cost"] is None for r in self.trace["api_calls"]))

    async def test_a_refusal_is_recorded_and_warned_about_without_what_was_said(self):
        with openrouter((502, {"error": {"code": 502}}),
                        (200, {"id": "gen-synthetic-1", "provider": "Anthropic",
                               "choices": [{"message": {"content": "Secret refusal words."},
                                            "finish_reason": "content_filter", "native_finish_reason": "refusal"}],
                               "usage": {"cost": .0001}})), self.assertLogs(j.log, "WARNING") as logs:
            await said([{"role": "user", "content": "cat?"}])
        failed, refused = self.trace["api_calls"]  # nothing on why the failure stopped: it had no answer
        self.assertEqual({k: failed[k] for k in ["finish_reason", "native_finish_reason", "id", "provider"]},
                         dict.fromkeys(["finish_reason", "native_finish_reason", "id", "provider"]))
        self.assertEqual((refused["finish_reason"], refused["native_finish_reason"], refused["id"],
                          refused["provider"]), ("content_filter", "refusal", "gen-synthetic-1", "Anthropic"))
        self.assertEqual(j.stopped_early(refused), "refusal")
        warning, = [line for line in logs.output if "refused" in line]
        self.assertIn("content_filter, refusal", warning)
        self.assertIn("gen-synthetic-1", warning)
        self.assertNotIn("Secret", "".join(logs.output))

    async def test_an_odd_response_is_still_retried(self):
        with openrouter((502, {"choices": [None]}), (200, ["not", "a", "completion"]),
                        (200, {"choices": [{"message": {"content": "Is Biscuit."}, "finish_reason": "stop"}], "usage": {}})):
            reply = await said([{"role": "user", "content": "cat?"}])
        self.assertEqual(reply, "Is Biscuit.")
        self.assertEqual([r["finish_reason"] for r in self.trace["api_calls"]], [None, None, "stop"])

class CreditTests(_Case):
    """A request OpenRouter won't pay for: out of credit, or over the key's limit or its workspace's budget."""

    BUDGET = {"error": {"code": 403, "message": "Workspace budget exceeded",
                        "metadata": {"limit_source": "openrouter_workspace_budget", "limit_usd": 1, "spend_usd": 1}}}

    async def test_a_spent_budget_or_credit_stops_without_retrying(self):
        for status, body in [(403, self.BUDGET), (402, {"error": {"code": 402, "message": "Insufficient credits"}}),
                             (402, "not json")]:
            with self.subTest(status=status, body=body):
                with openrouter((status, body)) as s:
                    self.assertEqual(await said([{"role": "user", "content": "cat?"}]), "")
                self.assertEqual(s.post.call_count, 1)
                s.set_credit.assert_awaited_once_with(False)

    async def test_any_other_forbidden_is_retried(self):
        body = {"error": {"code": 403, "message": "Flagged", "metadata": {"reasons": ["synthetic"]}}}
        with openrouter((403, body)) as s:
            await said([{"role": "user", "content": "cat?"}])
        self.assertEqual(s.post.call_count, 3)
        s.set_credit.assert_not_awaited()

    async def test_credit_left_is_the_lowest_of_credit_key_limit_and_budget(self):
        for key, want in [({"limit_remaining": None}, 7.5), ({"limit_remaining": 4.0}, 4.0),
                          ({"limit_remaining": 4.0, "effective_budget": {"remaining_usd": 0.25}}, 0.25),
                          ({"limit_remaining": None, "effective_budget": None}, 7.5)]:
            with self.subTest(key=key):
                replies = iter([{"data": {"total_credits": 10, "total_usage": 2.5}}, {"data": key}])
                s = session()
                s.get.return_value.__aenter__.side_effect = lambda: SimpleNamespace(json=AsyncMock(return_value=next(replies)))
                with patch.object(j.aiohttp, "ClientSession", return_value=s):
                    self.assertEqual(await j.credit_left(), want)


class StatusTests(_Case):
    async def test_status_keeps_its_opener_on_one_line(self):
        for answer, want in [("I wonder if sleep is same\nwhen no one watches.", "I wonder if sleep is same when no one watches."),
                             ("Why cat knock tea, question?", "I wonder why cat knock tea, question?"),
                             ("I wonder " + "very " * 40, None)]:
            with self.subTest(answer=answer), patch.object(j, "llm_message", AsyncMock(return_value={"content": answer})):
                mood = await j.llm_status("rocky", "I wonder", "kettle: my cat knocked my tea over")
            self.assertLessEqual(len(mood), 128)
            if want:
                self.assertEqual(mood, want)
        self.assertIn("kettle: my cat knocked my tea over", self.trace["transcript"])
        self.assertEqual(self.trace["llm"], "rocky")

    async def test_a_status_sees_what_it_posted_on_earlier_days(self):
        for recent in ([], ["I feel rain is loud.", "I wonder why toast falls butter down, question?"]):
            with self.subTest(recent=recent), patch.object(j, "recent_statuses", recent), \
                    patch.object(j, "llm_message", AsyncMock(return_value={"content": "I'm thinking about soup."})):
                await j.llm_status("rocky", "I'm thinking about", "kettle: soup tonight")
            ask = self.trace["transcript"]
            self.assertEqual("earlier days" in ask, bool(recent))
            if recent:  # oldest first, before the chat the new one is about
                self.assertLess(ask.index(recent[0]), ask.index(recent[1]))
                self.assertLess(ask.index(recent[1]), ask.index("kettle: soup tonight"))

    async def test_a_status_gets_rockys_lines_in_a_fresh_order(self):
        with patch.object(j, "rocky_prompt", return_value="voice") as prompt, \
                patch.object(j, "llm_message", AsyncMock(return_value={"content": "I feel good."})):
            await j.llm_status("rocky", "I feel")
        prompt.assert_called_with("rocky", shuffle=True)

    def test_clean_takes_the_first_paragraph_without_a_name(self):
        self.assertEqual(j.clean_llm('Rocky: "No. Just no."\n\nkettle: ok', "rocky"), "No. Just no.")
        self.assertEqual(j.clean_llm(None, "rocky"), "")


if __name__ == "__main__":
    unittest.main()
