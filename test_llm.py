"""Offline behavioral checks for the LLM: its requests and statuses, with the LLM's answers synthetic.

    uv run --with-requirements requirements.txt python -m unittest test_llm
"""

import unittest
from collections import defaultdict
from unittest.mock import AsyncMock, patch

from fakes import answer, api_error, claude, patch_all, traced
import rocky_bot as j


# What the LLM said to a message
async def said(text):
    response = await j.llm_message(messages=[{"role": "user", "content": text}])
    return j.text_in(response) if response else ""


class _Case(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.trace = traced(self)
        patch_all(self, [(j, "has_credit", True), (j, "channel_history", defaultdict(list)),
                         (j, "TIMEZONE", j.ZoneInfo("UTC")), (j, "recent_statuses", [])])  # not whatever .env sets


class RequestTests(_Case):
    async def test_a_request_is_recorded_with_its_tokens_cost_and_ending(self):
        # The sizes of a cold request on the live log, and what OpenRouter billed for them on our key: $0.01432927
        with claude(answer("Coffee first. Then walk.", uncached=4, cached=9537, written=71095, output=29)) as api, \
                patch.object(j.log, "warning") as warning:
            reply = await said("coffee?")
        self.assertEqual(reply, "Coffee first. Then walk.")
        request = api.call_args.kwargs
        self.assertEqual((request["model"], request["max_tokens"]), ("claude-haiku-5-5", j.LLM_MAX_TOKENS))
        self.assertFalse({"temperature", "top_p", "top_k", "frequency_penalty"} & set(request))
        call, = self.trace["api_calls"]
        self.assertEqual((call["purpose"], call["model"], call["status"], call["stop_reason"], call["id"]),
                         ("reply", "claude-haiku-5-5", 200, "end_turn", "msg_synthetic"))
        self.assertAlmostEqual(call["cost"], .01432927)
        self.assertAlmostEqual(self.trace["cost"], .01432927)
        # the prompt as a whole, of it what was read from the cache and written to it
        self.assertEqual({k: self.trace[k] for k in ("llm_prompt_tokens", "llm_cached_tokens", "llm_cache_write_tokens",
                                                     "llm_completion_tokens", "llm_window")},
                         {"llm_prompt_tokens": 80636, "llm_cached_tokens": 9537, "llm_cache_write_tokens": 71095,
                          "llm_completion_tokens": 29, "llm_window": 1_000_000})
        self.assertIsNone(j.stopped_early(call))  # an ordinary ending, without a warning
        warning.assert_not_called()
        self.assertEqual(self.trace["requests"], 1)

    async def test_a_prompt_past_100k_tokens_costs_five_times_as_much_for_all_of_it(self):
        for cached, want in [(99_000, 99_000 * .01 / 1e6), (120_000, 120_000 * .05 / 1e6)]:
            with self.subTest(cached=cached), claude(answer("Is Biscuit.", uncached=0, cached=cached, output=0)):
                t = traced(self)
                await said("cat?")
            self.assertAlmostEqual(t["cost"], want)

    async def test_a_failed_request_is_recorded_and_gives_nothing(self):
        with claude(api_error(400, "invalid_request_error", "messages.3.content.0: Invalid `signature` in `thinking` block")) as api, \
                self.assertLogs(j.log, "WARNING"):
            self.assertEqual(await said("cat?"), "")
        call, = self.trace["api_calls"]
        self.assertEqual((call["status"], call["cost"], call["stop_reason"]), (400, None, None))
        api.set_credit.assert_not_awaited()  # not for want of money
        self.assertEqual(self.trace["requests"], 0)

    async def test_running_out_of_tokens_gives_nothing_but_says_so(self):
        with claude(answer("Coffee first. Then", stop="max_tokens")), self.assertLogs(j.log, "WARNING"):
            self.assertEqual(await said("coffee?"), "")
        self.assertTrue(self.trace["llm_cut_off"])
        self.assertEqual(j.stopped_early(self.trace["api_calls"][0]), "max_tokens")


class CreditTests(_Case):
    """A request Anthropic won't run for want of money: credit run out, a spend limit or the tier's monthly cap."""

    async def test_no_money_stops_without_answering(self):
        for error in [api_error(402, "billing_error", "There's an issue with your billing."),
                      api_error(429, "rate_limit_error", "You have reached your API usage limits: ...",
                                "enforced_spend_limit_reached"),
                      api_error(400, "invalid_request_error", "You have reached your specified API usage limits. "
                                                              "You will regain access on 2026-12-01 at 00:00 UTC."),
                      api_error(400, "invalid_request_error", "Your credit balance is too low to access the Anthropic API.")]:
            with self.subTest(error=type(error).__name__, message=error.api_message):
                t = traced(self)
                with claude(error) as api:
                    self.assertEqual(await said("cat?"), "")
                api.set_credit.assert_awaited_once_with(False)
                self.assertEqual(t["error"], "out of credit")

    async def test_money_is_checked_with_a_request_that_costs_nothing(self):
        for reply, want in [(answer(stop="max_tokens", uncached=8, cached=0, output=0), True),
                            (api_error(402, "billing_error", "There's an issue with your billing."), False),
                            (api_error(529, "overloaded_error", "Overloaded"), None)]:
            with self.subTest(want=want), claude(reply) as api:
                self.assertEqual(await j.has_money(), want)
            self.assertEqual(api.call_args.kwargs["max_tokens"], 0)


class StatusTests(_Case):
    async def test_status_keeps_its_opener_on_one_line(self):
        for text, want in [("I wonder if sleep is same\nwhen no one watches.", "I wonder if sleep is same when no one watches."),
                           ("Why cat knock tea, question?", "I wonder why cat knock tea, question?"),
                           ("I wonder " + "very " * 40, None)]:
            with self.subTest(text=text), patch.object(j, "llm_message", AsyncMock(return_value=answer(text))):
                mood = await j.llm_status("rocky", "I wonder", "kettle: my cat knocked my tea over")
            self.assertLessEqual(len(mood), 128)
            if want:
                self.assertEqual(mood, want)
        self.assertIn("kettle: my cat knocked my tea over", j.context_text(self.trace["context"]))
        self.assertEqual(self.trace["llm"], "rocky")

    async def test_a_status_sees_what_it_posted_on_earlier_days(self):
        for recent in ([], ["I feel rain is loud.", "I wonder why toast falls butter down, question?"]):
            with self.subTest(recent=recent), patch.object(j, "recent_statuses", recent), \
                    patch.object(j, "llm_message", AsyncMock(return_value=answer("I'm thinking about soup."))):
                await j.llm_status("rocky", "I'm thinking about", "kettle: soup tonight")
            ask = j.context_text(self.trace["context"])
            self.assertEqual("earlier days" in ask, bool(recent))
            if recent:  # oldest first, before the chat the new one is about
                self.assertLess(ask.index(recent[0]), ask.index(recent[1]))
                self.assertLess(ask.index(recent[1]), ask.index("kettle: soup tonight"))

    async def test_a_status_gets_rockys_lines_in_a_fresh_order_without_thinking(self):
        with patch.object(j, "rocky_prompt", return_value="voice") as prompt, \
                claude(answer("I feel good.")) as api:
            await j.llm_status("rocky", "I feel")
        prompt.assert_called_with("rocky", shuffle=True)
        request = api.call_args.kwargs
        self.assertEqual((request["system"], request["thinking"]), ("voice", {"type": "disabled"}))
        self.assertNotIn("output_config", request)  # the model's default effort

    def test_clean_takes_the_first_paragraph_without_a_name(self):
        self.assertEqual(j.clean_llm('Rocky: "No. Just no."\n\nkettle: ok', "rocky"), "No. Just no.")
        self.assertEqual(j.clean_llm(None, "rocky"), "")


if __name__ == "__main__":
    unittest.main()
