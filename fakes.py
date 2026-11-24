"""What the offline tests share: rocky_bot imported without real keys, and fakes for Discord, OpenRouter, Rocky's
answers and Rocky's state. Not a test module itself: unittest only collects test_*.py.
"""

import contextlib
import itertools
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("DISCORD_TOKEN_JEV", "test")
os.environ.setdefault("OPENROUTER_API_KEY", "test")

import discord  # noqa: E402
import rocky_bot as j  # noqa: E402
import rocky_harness as rh  # noqa: E402


def patch_all(case, patches):
    # [(obj, attr, value), ...], each undone when the test ends
    for obj, attr, value in patches:
        p = patch.object(obj, attr, value)
        p.start()
        case.addCleanup(p.stop)


def traced(case):
    # A fresh trace for the test, as each reply and status gets
    t = {"cost": 0.0, "requests": 0}
    case.addCleanup(j.trace.reset, j.trace.set(t))
    return t


def isolate_rocky(case, here):
    # Rocky's logs and memories, empty and kept in `here`; and none of the checkout's own commits
    patch_all(case, [(rh, "LOG_DIR", Path(here) / "rocky_logs"), (rh, "MEMORY_PATH", Path(here) / "rocky_memories.json"),
                     (rh, "logs", {}), (rh, "log_saves", {}), (rh, "memories", {}), (rh, "memories_loaded", True),
                     (rh, "memories_save", None), (rh, "head", lambda: None),
                     (rh, "saved_memoryless", {})])


def answer(text="", calls=(), reasoning=None):
    # Rocky's response to a request: what it says, and the tools it calls as [(name, arguments)]
    return {"role": "assistant", "content": text or None, "reasoning_details": reasoning,
            "tool_calls": [{"id": f"call_{i}", "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}
                           for i, (name, args) in enumerate(calls)] or None}


def completion(message, cost=.0016):
    # OpenRouter's body for a chat completion: Rocky's message (answer()), with its usage and cost
    return {"choices": [{"message": message, "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
            "usage": {"prompt_tokens": 9000, "completion_tokens": 20, "cost": cost,
                      "prompt_tokens_details": {"cached_tokens": 8000}}}


def session():
    # An aiohttp session, its own `async with`
    s = MagicMock()
    s.__aenter__.return_value = s
    return s


@contextlib.contextmanager
def openrouter(*replies):
    # OpenRouter answering each request with the next (status, body), the last one again once they run out. Yields the
    # session, to see what was asked, with set_credit on it; retries don't wait.
    replies = list(replies)

    def reply():
        status, body = replies.pop(0) if len(replies) > 1 else replies[0]
        return SimpleNamespace(status=status, json=AsyncMock(return_value=body),
                               text=AsyncMock(return_value=json.dumps(body)))

    s = session()
    s.post.return_value.__aenter__.side_effect = reply
    with patch.object(j.aiohttp, "ClientSession", return_value=s), \
            patch.object(j, "set_credit", AsyncMock()) as s.set_credit, patch.object(j.asyncio, "sleep", AsyncMock()):
        yield s


class Message(SimpleNamespace):
    # discord.py's own formatting of mentions, on a message made here
    clean_content = property(discord.Message.clean_content.function)
    raw_mentions = property(discord.Message.raw_mentions.function)


class Discord:
    """A server as discord.py hands it to the bot: its channels by name, people's messages in them (say), what the bot
    posts and reacts with there, and the actions' private answers (interaction). Undone when the test ends."""

    def __init__(self, case, *channels):
        self.me = SimpleNamespace(id=100000000000000001, bot=True, display_name="Rocky")
        self.guild = SimpleNamespace(id=400, me=self.me, self_role=None, emojis=[], get_member=lambda _: None,
                                     get_role=lambda _: None, _resolve_channel=lambda _: None)
        self.ids = itertools.count(1000)
        self.messages = {}        # id: message, everything said
        self.reactions = []       # (message id, emoji), the bot's
        self.channels = {name: self.channel(300 + i, name) for i, name in enumerate(channels)}
        by_id = {c.id: c for c in self.channels.values()}
        patch_all(case, [(j.bot._connection, "user", self.me), (j.bot, "get_channel", by_id.get)])

    def channel(self, id, name):
        c = SimpleNamespace(id=id, name=name, guild=self.guild, typing=contextlib.nullcontext)

        async def send(content, **kwargs):
            return self.post(c, content, self.me)

        async def history(limit=None, before=None, **kwargs):  # newest first
            for m in sorted((m for m in self.messages.values() if m.channel is c and (before is None or m.id < before.id)),
                            key=lambda m: m.id, reverse=True)[:limit]:
                yield m

        async def react(target, emoji):
            self.reactions.append((target, emoji))

        c.send, c.history = send, history
        c.messages = lambda: [m for m in self.messages.values() if m.channel is c]
        c.get_partial_message = lambda i: SimpleNamespace(add_reaction=lambda emoji: react(i, emoji))
        return c

    def post(self, channel, content, author, to=None):
        m = Message(id=next(self.ids), content=content, author=author, guild=self.guild, channel=channel,
                    mentions=[self.me] if f"<@{self.me.id}>" in content else [], role_mentions=[], embeds=[],
                    attachments=[], stickers=[], reactions=[], created_at=datetime.now(timezone.utc),
                    reference=to and SimpleNamespace(message_id=to.id, resolved=to))

        async def reply(content, **kwargs):
            return self.post(channel, content, self.me, to=m)

        m.reply = reply
        self.messages[m.id] = m
        return m

    def say(self, channel, name, text):
        # someone tagging the bot in a channel
        author = SimpleNamespace(id=500 + sum(map(ord, name)), bot=False, display_name=name)
        return self.post(self.channels[channel], f"<@{self.me.id}> {text}", author)

    def answer(self, m):
        # what the bot posted in m's channel after m, before anyone spoke there again
        later = [x for x in self.messages.values() if x.channel is m.channel and x.id > m.id]
        return list(itertools.takewhile(lambda x: x.author is self.me, later))

    def interaction(self, target, answers):
        # Someone using an action on target; what only they see goes in answers, as (text, embed as a dict, attached
        # files by name). Private from the start, with no pings.
        async def defer(ephemeral=False, **kwargs):
            assert ephemeral

        async def send(content=None, embed=None, files=None, ephemeral=False, allowed_mentions=None, **kwargs):
            assert ephemeral and allowed_mentions.to_dict()["parse"] == []
            answers.append((content or None, embed.to_dict() if embed else None,
                            {f.filename: f.fp.read() for f in files or ()}))

        return SimpleNamespace(channel_id=target.channel.id, guild=self.guild,
                               response=SimpleNamespace(defer=defer), followup=SimpleNamespace(send=send))
