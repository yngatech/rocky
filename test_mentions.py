"""Offline behavioral checks for mention routing, history, startup catch-up, and keeping Rocky's cache warm.

    uv run --with-requirements requirements.txt python -m unittest test_mentions
"""

import asyncio
import contextlib
import copy
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fakes import answer, isolate_rocky, openrouter, patch_all, traced
import discord
import rocky_bot as j
import rocky_harness as rh
import rocky_viewer


class _Message(SimpleNamespace):
    # Exercise discord.py's real formatter rather than mocking its output.
    clean_content = property(discord.Message.clean_content.function)
    raw_mentions = property(discord.Message.raw_mentions.function)


class _Discord(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.reactors_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.reactors_dir.cleanup)
        self.user = SimpleNamespace(id=100000000000000001, bot=True, display_name="Testbot")
        self.role = SimpleNamespace(id=200000000000000001, mention="<@&200000000000000001>", name="Testbot")
        self.guild = SimpleNamespace(id=400, self_role=self.role, me=self.user, threads=[], get_thread=lambda _: None,
                                     get_member=lambda _: None, get_role=lambda _: None,
                                     _resolve_channel=lambda _: None)
        self.channel = Mock(id=300)
        self.channel.permissions_for.return_value = SimpleNamespace(read_messages=True, read_message_history=True)
        self.guild.text_channels = [self.channel]
        self.target = Mock(spec=discord.Message, id=700, author=self.user, reactions=[])
        self.target.reference = None
        self.target.created_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        patch_all(self, [
            (j.bot._connection, "user", self.user),
            (j.bot._connection, "_guilds", {400: self.guild}),
            (j, "DM_USERS", {500}),
            (j, "channel_history", defaultdict(list)),
            (j, "heard", False),
            (j, "taken", {}),
            (j, "reactor_names", {}),
            (j, "reactor_cache", {}),
            (j, "reactors_save", None),
            (j, "REACTORS_PATH", Path(self.reactors_dir.name) / "reactors.json"),
        ])
        isolate_rocky(self, self.reactors_dir.name)  # no checkout's logs, and none left behind


class ReactionContextTests(_Discord):
    def setUp(self):
        super().setUp()
        self.entry = {"name": "Speaker", "content": "hello", "id": 600,
                      "reply": "Hello", "reply_id": 700}
        j.channel_history[self.channel.id] = [self.entry]

    def payload(self, uid=501, name="Moss", emoji="😂"):
        return SimpleNamespace(channel_id=self.channel.id, message_id=700, guild_id=400,
                               user_id=uid, member=Mock(spec=discord.Member, id=uid, display_name=name, guild=self.guild), emoji=emoji)

    # Who reacted to the bot's reply, as people Rocky can ping (mention_targets)
    def names(self, entry=None):
        entry = {"name": "Speaker", **(entry or self.entry)}
        return {name for name, uid in j.mention_targets([entry], "Testbot", reactors=True).items()
                if uid and name != entry["name"]}  # not whoever asked

    def reaction(self, emoji="😂", fail=False):
        async def users():
            if fail:
                raise discord.HTTPException(SimpleNamespace(status=403, reason="Forbidden"), "denied")
            for user in [self.user, SimpleNamespace(id=501, display_name="Moss"),
                         SimpleNamespace(id=502, display_name="Pip")]:
                yield user
        return SimpleNamespace(emoji=emoji, count=3, me=True, users=users)

    async def test_live_names_and_removal(self):
        await j.on_raw_reaction_add(self.payload())
        await j.on_raw_reaction_add(self.payload(502, "Pip"))
        await j.on_raw_reaction_add(self.payload(self.user.id, "Testbot"))
        self.assertEqual(self.names(), {"Moss", "Pip"})
        self.assertEqual(self.entry["reply_reactions"], {"😂": 2})  # not its own
        await j.on_raw_reaction_remove(self.payload())
        self.assertEqual(self.names(), {"Pip"})
        await j.on_raw_reaction_remove(self.payload(502, "Pip"))
        self.assertEqual(self.entry["reply_reactions"], {})

    async def test_clear_one_and_all_reactions(self):
        await j.on_raw_reaction_add(self.payload())
        await j.on_raw_reaction_add(self.payload(502, "Pip", "💀"))
        await j.on_raw_reaction_clear_emoji(self.payload())
        self.assertEqual(self.names(), {"Pip"})  # only on 💀 now
        await j.on_raw_reaction_clear(self.payload())
        self.assertNotIn("reply_reactors", self.entry)
        self.assertNotIn("reply_reactions", self.entry)

    async def test_names_restored_from_channel_history(self):
        question = MentionTests.message(self)
        question.id = 600
        reply = _Message(id=700, author=self.user, reference=SimpleNamespace(message_id=600), guild=None,
                         content="Hello", reactions=[self.reaction()], created_at=question.created_at,
                         mentions=[], role_mentions=[], embeds=[], attachments=[], stickers=[])
        async def history(**kwargs):
            for message in [reply, question]:
                yield message
        self.channel.history = history
        j.channel_history[self.channel.id] = []
        await j.load_history(SimpleNamespace(channel=self.channel))
        self.entry = j.channel_history[self.channel.id][0]
        self.assertEqual(self.names(), {"Moss", "Pip"})

    async def test_standalone_reply_names(self):
        question = MentionTests.message(self)
        reply = _Message(id=700, author=self.user, reference=SimpleNamespace(message_id=question.id), guild=None,
                         content="Hello", reactions=[self.reaction()], created_at=question.created_at,
                         mentions=[], role_mentions=[], embeds=[], attachments=[], stickers=[])
        standalone = {"role": "assistant", "content": "Hello"}
        await j.attach_reactions(standalone, reply, prefix="")
        self.assertEqual(self.names(standalone), {"Moss", "Pip"})

    async def test_http_failure_keeps_counts_and_old_logs_render(self):
        reply = SimpleNamespace(id=700, reactions=[self.reaction(fail=True)])
        await j.attach_reactions(self.entry, reply)
        self.assertEqual(self.entry["reply_reactions"], {"😂": 2})
        self.assertNotIn("700", j.reactor_cache)  # asked again next time
        p = self.payload()
        p.member, p.guild_id = None, None
        with patch.object(j.bot, "get_user", return_value=None), patch.object(
            j.bot, "fetch_user", new_callable=AsyncMock, return_value=SimpleNamespace(display_name="Moss")
        ):
            await j.on_raw_reaction_add(p)
        self.assertEqual(self.names(), {"Moss"})

    async def test_a_rebuild_reuses_saved_names_without_asking_discord(self):
        j.reactor_cache["700"] = {"counts": {"😂": 2}, "reactors": {"😂": {"501": "Moss", "502": "Pip"}}}
        unasked = SimpleNamespace(emoji="😂", count=3, me=True, users=Mock(side_effect=AssertionError("asked Discord")))
        await j.attach_reactions(self.entry, SimpleNamespace(id=700, reactions=[unasked]))
        self.assertEqual(self.names(), {"Moss", "Pip"})

    async def test_only_an_emoji_whose_count_changed_is_asked_about(self):
        j.reactor_cache["700"] = {"counts": {"😂": 2, "💀": 1}, "reactors": {"😂": {"501": "Moss", "502": "Pip"},
                                                                           "💀": {"503": "Fern"}}}
        unasked = SimpleNamespace(emoji="😂", count=3, me=True, users=Mock(side_effect=AssertionError("asked Discord")))
        await j.attach_reactions(self.entry, SimpleNamespace(id=700, reactions=[unasked, self.reaction("💀")]))
        self.assertEqual(self.entry["reply_reactors"], {"😂": {"501": "Moss", "502": "Pip"},
                                                        "💀": {"501": "Moss", "502": "Pip"}})
        self.assertEqual(j.reactor_cache["700"]["counts"], {"😂": 2, "💀": 2})

    async def test_live_reactions_keep_the_saved_names_current(self):
        await j.on_raw_reaction_add(self.payload())
        await j.on_raw_reaction_add(self.payload(502, "Pip", "💀"))
        await j.on_raw_reaction_remove(self.payload())
        self.assertEqual(j.reactor_cache["700"], {"counts": {"💀": 1}, "reactors": {"💀": {"502": "Pip"}}})
        self.assertIsNotNone(j.reactors_save)  # written a moment later, once
        j.save_reactors()
        self.assertEqual(j.load_reactors(), j.reactor_cache)
        await j.on_raw_reaction_clear(self.payload())
        self.assertNotIn("700", j.reactor_cache)

    async def test_history_names_are_server_nicknames(self):
        # Reaction users come back as plain users with global names; the member lookup gives the nickname,
        # once per person however many replies they reacted to.
        fetch = AsyncMock(side_effect=lambda uid: SimpleNamespace(display_name={501: "Mossy", 502: "Pipsqueak"}[uid]))
        guild = SimpleNamespace(id=400, get_member=lambda _: None, fetch_member=fetch)
        replies = [SimpleNamespace(id=700 + i, guild=guild, reactions=[self.reaction()]) for i in range(2)]
        entries = [{"reply": "Hello"} for _ in replies]
        await asyncio.gather(*(j.attach_reactions(e, r) for e, r in zip(entries, replies)))
        self.assertEqual(entries[1]["reply_reactors"]["😂"], {"501": "Mossy", "502": "Pipsqueak"})
        self.assertEqual(fetch.await_count, 2)

class MentionTests(_Discord):
    def message(self, *, content="hello", ping=False, role=False, reply=True, dm=False, author_bot=False):
        return _Message(
            id=600, content=content, guild=None if dm else self.guild, channel=self.channel,
            author=SimpleNamespace(id=500, bot=author_bot, display_name="Speaker"),
            mentions=[self.user] if ping else [], role_mentions=[self.role] if role else [],
            reference=SimpleNamespace(message_id=self.target.id, resolved=self.target) if reply else None,
            created_at=datetime.now(timezone.utc), reactions=[], attachments=[], stickers=[], embeds=[],
        )

    async def test_mentions_stay_in_place(self):
        for mention, options in (("<@100000000000000001>", {"ping": True}), ("<@!100000000000000001>", {"ping": True}),
                                 ("<@&200000000000000001>", {"role": True})):
            with self.subTest(mention=mention):
                m = self.message(content=f"think {mention} is drunk?", **options)
                text = j.message_text(m)
                self.assertEqual(text, "think @Testbot is drunk?")
                self.assertEqual(j.history_entry(m)["content"], "think @Testbot is drunk?")

    async def test_a_message_from_history_reads_as_it_did_live(self):
        m = self.message(content="<@100000000000000001> tea or coffee?\n\nasking for a friend", ping=True, reply=False)
        filled, live = rh.Log("general", "400", 300), rh.Log("general", "400", 300)
        rh.sync(filled, [j.history_entry(m)])
        rh.observe(m, live)
        text = lambda log: next(e["text"] for e in log.events if e["type"] == "message")
        self.assertEqual(text(filled), "@Testbot tea or coffee?\n\nasking for a friend")
        self.assertEqual(text(filled), text(live))
        self.assertEqual(rh.render(filled, "rocky")[2], rh.render(live, "rocky")[2])

    async def test_other_mentions_and_possessives_are_preserved(self):
        m = self.message(content="<@100000000000000001>'s friend <@!100000000000000002> likes <@&200000000000000002>", ping=True)
        m.mentions.append(SimpleNamespace(id=100000000000000002, display_name="Friend"))
        m.role_mentions.append(SimpleNamespace(id=200000000000000002, name="Readers"))
        self.assertEqual(j.message_text(m), "@Testbot's friend @Friend likes @Readers")

    async def test_discord_formatter_handles_channels_and_deleted_mentions(self):
        self.guild._resolve_channel = lambda id: SimpleNamespace(name="general") if id == 300000000000000001 else None
        m = self.message(content="<@100000000000000001> see <#300000000000000001>", ping=True)
        self.assertEqual(j.message_text(m), "@Testbot see #general")
        m.content = "<@100000000000000099> <@&200000000000000099> <#300000000000000099>"
        self.assertEqual(j.message_text(m), "@deleted-user @deleted-role #deleted-channel")
        m.content = "@everyone @here"
        self.assertEqual(j.message_text(m), "@\u200beveryone @\u200bhere")

    async def test_dm_mentions_keep_readable_names(self):
        m = self.message(content="<@100000000000000001>'s friend", ping=True, dm=True)
        self.assertEqual(j.message_text(m), "@Testbot's friend")

    async def test_unpinged_reply_is_chatter_without_a_response(self):
        m = self.message()
        with patch.object(j, "respond", new_callable=AsyncMock) as respond:
            await j.on_message(m)
        respond.assert_not_awaited()
        entry, = j.channel_history[self.channel.id]
        self.assertFalse(entry["to_bot"])
        self.assertEqual(entry["content"], "hello")

    async def test_mentions_trigger_responses_with_or_without_reply(self):
        for options in ({"ping": True}, {"ping": True, "reply": False},
                        {"role": True}, {"role": True, "reply": False}):
            with self.subTest(options=options):
                m = self.message(**options)
                with patch.object(j, "respond", new_callable=AsyncMock) as respond:
                    await j.on_message(m)
                respond.assert_awaited_once_with(m)

    async def test_allowed_dm_needs_no_mention(self):
        m = self.message(dm=True, reply=False)
        with patch.object(j, "respond", new_callable=AsyncMock) as respond:
            await j.on_message(m)
        respond.assert_awaited_once_with(m)

    async def test_plain_names_wake_bot_and_mark_history(self):
        self.guild.me = SimpleNamespace(display_name="Garden Bot")
        for text in ("hello testbot", "TESTBOT!", "Testbot's turn", "what does Testbot think?",
                     "Garden Bot, hello", "hi garden bot!", "TESTBOTTTTTTT!", "hello garden bottttt",
                     "thanks rock", "ROCKKK!"):
            for reply in (False, True):
                with self.subTest(text=text, reply=reply):
                    m = self.message(content=text, reply=reply)
                    with patch.object(j, "respond", new_callable=AsyncMock) as respond:
                        await j.on_message(m)
                    respond.assert_awaited_once_with(m)
                    entry = j.history_entry(m)
                    self.assertTrue(entry["to_bot"])
                    self.assertEqual(entry["content"], text)

    async def test_name_substrings_stay_chatter(self):
        for text in ("testbotany", "mytestbot", "testbot_2", "étestbot", "testboté",
                     "mytestbottt", "testbotttany", "testbottt_2", "testbottté", "testbo",
                     "rocks", "rocked", "rocket", "Rockstar", "rocking"):
            with self.subTest(text=text):
                m = self.message(content=text, reply=False)
                with patch.object(j, "respond", new_callable=AsyncMock) as respond:
                    await j.on_message(m)
                respond.assert_not_awaited()
                self.assertFalse(j.history_entry(m)["to_bot"])

    async def test_name_matching_treats_punctuation_literally(self):
        self.guild.me = SimpleNamespace(display_name="Garden.Bot")
        self.assertTrue(j.should_respond(self.message(content="hello Garden.Bot!")))
        self.assertFalse(j.should_respond(self.message(content="hello GardenXBot!")))

    async def test_plain_names_do_not_bypass_bot_or_dm_restrictions(self):
        for options in ({"author_bot": True}, {"dm": True}):
            with self.subTest(options=options), patch.object(j, "DM_USERS", set()):
                m = self.message(content="hello Testbot", **options)
                with patch.object(j, "respond", new_callable=AsyncMock) as respond:
                    await j.on_message(m)
                respond.assert_not_awaited()

    async def test_catch_up_answers_plain_name(self):
        m = self.message(content="hello Testbot", reply=False)

        async def history(**kwargs):
            yield m
            yield self.target

        self.channel.history.side_effect = history
        with patch.object(j, "DM_USERS", set()), patch.object(j, "respond", new_callable=AsyncMock) as respond:
            await j.catch_up()
        respond.assert_awaited_once_with(m, caught_up=True)

    async def test_other_bots_cannot_trigger_responses(self):
        m = self.message(ping=True, author_bot=True)
        with patch.object(j, "respond", new_callable=AsyncMock) as respond:
            await j.on_message(m)
        respond.assert_not_awaited()
        entry, = j.channel_history[self.channel.id]
        self.assertEqual(entry["content"], "hello")
        self.assertFalse(entry["to_bot"])
        self.assertFalse(j.heard)

    async def test_own_posts_stay_out_of_history(self):
        m = self.message(author_bot=True, reply=False)
        m.author = self.user
        with patch.object(j, "respond", new_callable=AsyncMock) as respond:
            await j.on_message(m)
        respond.assert_not_awaited()
        self.assertFalse(j.channel_history[self.channel.id])
        self.assertFalse(j.heard)

    async def test_catch_up_does_not_answer_other_apps(self):
        m = self.message(ping=True, author_bot=True, reply=False)

        async def history(**kwargs):
            yield m

        self.channel.history.side_effect = history
        with patch.object(j, "DM_USERS", set()), patch.object(j, "respond", new_callable=AsyncMock) as respond:
            await j.catch_up()
        respond.assert_not_awaited()

    async def test_catch_up_skips_unpinged_reply_and_answers_pinged_reply(self):
        for ping in (False, True):
            with self.subTest(ping=ping):
                m = self.message(ping=ping)

                async def history(**kwargs):
                    yield m
                    yield self.target

                self.channel.history.side_effect = history
                with patch.object(j, "DM_USERS", set()), \
                        patch.object(j, "respond", new_callable=AsyncMock) as respond:
                    await j.catch_up()
                if ping:
                    respond.assert_awaited_once_with(m, caught_up=True)
                else:
                    respond.assert_not_awaited()


    async def test_catch_up_skips_message_already_being_answered(self):
        # Just after a restart: on_message is answering m when catch_up() finds it, its reply not sent yet
        m = self.message(ping=True, reply=False)
        started, finish = asyncio.Event(), asyncio.Event()

        async def answering(m, **extra):
            started.set()
            await finish.wait()

        async def history(**kwargs):
            yield m

        self.channel.history.side_effect = history
        with patch.object(j, "DM_USERS", set()), patch.object(j, "taken", {}), \
                patch.object(j, "respond_traced", side_effect=answering) as traced:
            live = asyncio.create_task(j.on_message(m))
            await started.wait()
            await asyncio.wait_for(j.catch_up(), 1)  # a second answer would wait on finish too
            finish.set()
            await live
            await j.respond(m, caught_up=True)  # and once it's answered, too
        traced.assert_awaited_once_with(m)


class EmbedTests(_Discord):
    message = MentionTests.message

    def test_standalone_embed_includes_title_and_description(self):
        m = self.message(content="", author_bot=True, reply=False)
        m.embeds = [discord.Embed(url="https://example.com/milestone", title="Contribution milestone", description=(
            "[synthetic-user](https://example.com/synthetic-user) has passed **1,000 contributions** in 2026."))]
        self.assertEqual(j.message_text(m),
                         "[embed: Contribution milestone synthetic-user has passed 1,000 contributions in 2026.]")

    def test_multiline_description_and_fields_share_one_word_limit(self):
        m = self.message(content="", reply=False)
        embed = discord.Embed(description="Build finished\nAll checks passed")
        embed.set_author(name="Build App")
        embed.add_field(name="Result", value=" ".join(f"word{i}" for i in range(30)))
        m.embeds = [embed]
        expected = "Build App Build finished All checks passed Result: " + " ".join(f"word{i}" for i in range(12))
        self.assertEqual(j.message_text(m), f"[embed: {expected} ...]")

    def test_link_preview_is_not_repeated_as_a_standalone_embed(self):
        for url in ("https://example.com/post", "https://example.com/redirect"):
            with self.subTest(url=url):
                m = self.message(content=url, reply=False)
                m.embeds = [discord.Embed(url="https://example.com/post", title="Release notes", description="Preview text")]
                self.assertEqual(j.message_text(m), "[link: Release notes]")

    def test_multiple_links_and_extra_embed(self):
        m = self.message(content="https://example.com/first https://example.com/second", reply=False)
        m.embeds = [discord.Embed(url="https://example.com/first", title="First article"),
                    discord.Embed(url="https://example.com/second", title="Second article"),
                    discord.Embed(title="Build", description="All checks passed")]
        self.assertEqual(j.message_text(m), "[link: First article] [link: Second article] [embed: Build All checks passed]")

    def test_gif_tweet_attachments_and_stickers_keep_their_tags(self):
        m = self.message(content="https://tenor.com/view/happy-dance-gif https://example.com/post", reply=False)
        gif = discord.Embed(url="https://tenor.com/view/happy-dance-gif", type="gifv", title="Happy dance")
        tweet = discord.Embed(url="https://example.com/post", description="**Good news**\nOther text")
        tweet.set_author(name="Synthetic User (@synthetic)")
        m.embeds = [gif, tweet, discord.Embed().set_image(url="https://example.com/image.png")]
        m.attachments = [SimpleNamespace(content_type="image/png")]
        m.stickers = [SimpleNamespace(name="Wave")]
        self.assertEqual(j.message_text(m),
                         "[gif: Happy dance] [link: Synthetic User: Good news] [photo] [sticker: Wave]")


class ActionTests(_Discord):
    """Diagnostic actions, from right-clicking a message: answered only to whoever asked."""

    def test_actions_are_message_context_menus(self):
        menus = {c.name: c.type for c in j.bot.tree.get_commands(type=discord.AppCommandType.message)}
        self.assertEqual(menus, {"Context": discord.AppCommandType.message, "Stats": discord.AppCommandType.message,
                                 "Diagnostics": discord.AppCommandType.message})

    async def test_failed_registration_doesnt_stop_startup(self):
        error = discord.HTTPException(Mock(status=503, reason="Service Unavailable"), "unavailable")
        with patch.object(j.bot.tree, "sync", AsyncMock(side_effect=error)), self.assertLogs(j.log, "WARNING") as logs:
            await j.bot.setup_hook()
        self.assertIn("Registering app commands failed", logs.output[0])


class WakingTests(_Discord):
    """Discord shows a bot online as soon as it connects; it stays idle until it has caught up."""

    async def test_idle_until_caught_up_then_online_with_its_status(self):
        presence = AsyncMock()
        seen_while_catching_up = []

        async def catch_up():
            await j.set_credit(True)  # credit coming back mid-catch-up mustn't put it online early
            seen_while_catching_up.extend(presence.call_args_list)

        patch_all(self, [(j.bot, "change_presence", presence), (j.bot, "is_ready", lambda: True)])
        with patch.object(j, "has_credit", None), patch.object(j, "credit_left", AsyncMock(return_value=5.0)), \
                patch.object(j, "catch_up", catch_up), patch.object(j, "STATUS_EVERY", 0), \
                patch.object(j, "status", discord.CustomActivity("I feel tea")), patch.object(j, "waking", True):
            self.assertEqual((j.bot.status, j.bot.activity), (discord.Status.idle, j.WAKING))  # from the first connect
            await j.on_ready()
            self.assertEqual([c.kwargs["status"] for c in seen_while_catching_up], [discord.Status.idle])
            self.assertEqual(presence.call_args.kwargs, {"status": discord.Status.online, "activity": j.status})
            self.assertFalse(j.waking)


class _Handling(_Discord):
    """handle() end to end, with what Rocky would say patched in."""

    def setUp(self):
        super().setUp()
        self.load_history = j.load_history
        self.trace = traced(self)
        self.channel.typing = lambda: contextlib.nullcontext()
        self.channel.fetch_message = AsyncMock(side_effect=lambda i: self.by_id[i])
        self.channel.name, self.channel.guild = "bot-spam", self.guild
        self.by_id = {}
        self.next_id = 800
        patch_all(self, [(j, "has_credit", True), (j, "history_loaded", {}), (j, "emoji_vocabulary", lambda g: {}),
                         (j, "load_history", AsyncMock())])
        # Chatter before the conversation, which a !nocontext thread shouldn't see
        j.channel_history[self.channel.id] = [
            {"name": "kettle", "content": "the toaster is on fire again", "id": 1,
             "at": datetime.now(timezone.utc) - timedelta(minutes=10), "to_bot": False}]

    def said(self, content, *, to=None, bot=False, name="pip", ping=True, channel=None):
        m = Mock(spec=discord.Message, id=self.next_id, content=(f"<@{self.user.id}> " if ping and not bot else "") + content,
                 guild=self.guild, channel=channel or self.channel, mentions=[self.user] if ping and not bot else [],
                 role_mentions=[], created_at=datetime.now(timezone.utc), reactions=[], attachments=[], stickers=[],
                 embeds=[], author=self.user if bot else SimpleNamespace(id=500, bot=False, display_name=name))
        m.clean_content = discord.Message.clean_content.function(m)
        m.raw_mentions = discord.Message.raw_mentions.function(m)
        # Discord only resolves the message m replies to, not the ones above it
        m.reference = to and SimpleNamespace(message_id=to.id, resolved=None)
        self.by_id[m.id] = m
        self.next_id += 1
        m.reply = AsyncMock(side_effect=lambda content, **kwargs: self.discord_reply(m, content))
        return m

    # What Rocky's log of where it answered m had before m, as (who, what), "you" for Rocky — answering with `says`
    async def answer(self, m, says="Geology good"):
        if m.reference:
            m.reference.resolved = self.by_id[m.reference.message_id]
        m.reply = AsyncMock(side_effect=lambda content, **kwargs: self.discord_reply(m, content))
        with patch.object(j, "llm_message", AsyncMock(return_value=answer(says))):
            await j.handle(m)
        log = rh.logs[getattr(self, "threads", {}).get(m.id, m.channel).id]
        n = log.ids[str(m.id)]
        events = log.events[:next(i for i, e in enumerate(log.events) if e.get("n") == n)]
        return [("you" if e["type"] == "rocky" else e["author"], e["text"]) for e in events
                if e["type"] in ("message", "rocky")]

    def discord_reply(self, m, content):
        users = {u.id: u for message in self.by_id.values() for u in [message.author, *message.mentions]}
        self.sent = self.said(content, to=m, bot=True, channel=m.channel)
        self.sent.mentions = [users[uid] for uid in self.sent.raw_mentions if uid in users]
        self.sent.clean_content = discord.Message.clean_content.function(self.sent)
        return self.sent


class RockyHarnessTests(_Handling):
    """Rocky's turn from the channel's log, posted and acted on."""

    def setUp(self):
        super().setUp()
        rh.log_for(self.channel)  # a channel Rocky has talked in, so it follows what's said there
        self.partials = {}
        self.channel.get_partial_message = lambda i: self.partials.setdefault(i, Mock(add_reaction=AsyncMock()))

    async def test_rocky_speaks_reacts_and_remembers_from_the_channels_log(self):
        await j.on_message(self.said("anyone seen my keys", name="kettle", ping=False))
        await j.on_message(self.said("!why", ping=False))  # no longer a command: just something said
        m = self.said("my cat is called biscuit")
        llm = AsyncMock(return_value=answer("@pip Biscuit. Good name.", [
            ("react", {"emoji": "🐱"}), ("remember", {"about": "pip", "fact": "has a cat called Biscuit"})]))
        with patch.object(j, "llm_message", llm):
            await j.handle(m)  # as catch-up does: not seen live, so the turn adds it
        m.reply.assert_awaited_once()
        # it's a reply to pip already, and says what Rocky remembered under it
        self.assertEqual(m.reply.call_args.args[0], "Biscuit. Good name.\n-# 🧠 remembered: pip has a cat called Biscuit")
        self.partials[m.id].add_reaction.assert_awaited_once_with("🐱")
        log = rh.logs[self.channel.id]
        people = [e for e in log.events if e["type"] == "message"]
        self.assertEqual([(e["author"], e["text"]) for e in people][-3:],
                         [("kettle", "anyone seen my keys"), ("pip", "!why"), ("pip", "@Testbot my cat is called biscuit")])
        self.assertTrue(people[-1]["woke"])
        self.assertEqual(log.ids[str(self.sent.id)], "you")
        self.assertEqual(rh.memory("400")["facts"][0]["fact"], "has a cat called Biscuit")
        self.assertEqual((self.trace["llm"], self.trace["reply"], self.trace["reaction"]), ("rocky", "@pip Biscuit. Good name.", "🐱"))
        self.assertIn("my cat is called biscuit", j.context_text(self.trace["context"]))
        self.assertEqual(j.channel_history[self.channel.id][-1]["reply"], "Biscuit. Good name.")

        # what happens to Rocky's message next is in the log too
        await j.on_raw_reaction_add(SimpleNamespace(channel_id=self.channel.id, message_id=self.sent.id, guild_id=400,
                                                    user_id=501, emoji="😂", member=Mock(spec=discord.Member, id=501,
                                                    display_name="Moss", guild=self.guild)))
        reply = self.said("why biscuit", to=self.sent, ping=False)
        reply.reference.resolved = self.sent  # as Discord gives a reply to a message it has
        rh.observe(reply)
        reaction, asked = [e for e in log.events if e["type"] in ("reaction", "message")][-2:]
        # its turn is kept as it wrote it — only the posted message lost the tag
        self.assertEqual((reaction["author"], reaction["on"], reaction["you_said"]), ("Moss", "you", "@pip Biscuit. Good name."))
        # a reply with its ping off isn't to Rocky, but it's still shown as answering its message
        self.assertEqual((asked["reply_to"], asked["you_said"], asked["to_bot"]), ("you", "@pip Biscuit. Good name.", False))

    async def test_after_a_restart_what_happens_before_anything_is_said_still_reaches_its_log(self):
        said = self.said("my password is hunter2", ping=False)
        await j.on_message(said)
        with patch.object(j, "llm_message", AsyncMock(return_value=answer("Hunter. Is name."))):
            await j.handle(self.said("rocky what's a good password"))
        rh.flush()
        rh.logs.clear()  # a restart, and nothing said there since
        with patch.object(j.bot, "get_channel", lambda i: None):  # not cached yet either
            await j.on_raw_message_delete(SimpleNamespace(channel_id=self.channel.id, message_id=said.id))
            await j.on_raw_reaction_add(SimpleNamespace(channel_id=self.channel.id, message_id=self.sent.id, guild_id=400,
                                                        user_id=501, emoji="😂", member=Mock(spec=discord.Member, id=501,
                                                        display_name="Moss", guild=self.guild)))
        llm = AsyncMock(return_value=answer("Is good."))
        with patch.object(j, "llm_message", llm):
            await j.handle(self.said("thanks rocky"))
        latest = j.text_of(llm.await_args.args[0][-1]["content"])
        self.assertIn('<deleted id="1"/>', latest)
        self.assertIn('<reaction from="Moss" emoji="😂" on="your message: Hunter. Is name."/>', latest)

    async def test_a_status_posted_in_the_channel_is_something_it_said(self):
        await j.on_message(self.said("anyone seen my keys", name="kettle", ping=False))
        rh.flush()
        rh.logs.clear()  # a restart, and nothing said there since
        self.channel.send = AsyncMock(side_effect=lambda content, **kwargs: self.said(content, bot=True))
        with patch.object(j, "STATUS_CHANNEL", self.channel.id), patch.object(j.bot, "get_channel", lambda i: self.channel):
            await j.post_status("I feel bad. AI copy self, no need human.")
        log = rh.logs[self.channel.id]
        self.assertEqual([e["type"] for e in log.events], ["message", "rocky"])  # after what was saved, not instead
        self.assertEqual(log.events[-1]["text"], "I feel bad. AI copy self, no need human.")

        m = self.said("why are you scared of this", to=self.by_id[self.next_id - 1])
        m.reference.resolved = self.by_id[m.reference.message_id]
        rh.observe(m)
        llm = AsyncMock(return_value=answer("Rocky not scared. Rocky worried."))
        with patch.object(j, "llm_message", llm):
            await j.handle(m)
        request = llm.await_args.args[0]
        self.assertEqual([j.text_of(x["content"]) for x in request if x["role"] == "assistant"],
                         ["I feel bad. AI copy self, no need human."])
        self.assertIn('reply_to="your message: I feel bad. AI copy self, no need human."', j.text_of(request[-1]["content"]))

    async def test_only_reacting_posts_nothing_and_never_shows_typing(self):
        m = self.said("lol")
        self.channel.typing = Mock(side_effect=lambda: contextlib.nullcontext())
        with patch.object(j, "llm_message", AsyncMock(return_value=answer(calls=[("react", {"emoji": "😂"})]))):
            await j.handle(m)
        m.reply.assert_not_awaited()
        self.channel.typing.assert_not_called()  # it would have lingered ~10s after the reaction
        self.partials[m.id].add_reaction.assert_awaited_once_with("😂")
        self.assertEqual(j.channel_history[self.channel.id][-1]["reaction"], "😂")

    async def test_a_channel_has_no_log_until_rocky_is_woken_there(self):
        rh.logs.clear()
        await j.on_message(self.said("anyone seen my keys", name="kettle", ping=False))
        self.assertEqual((rh.logs, list(Path(rh.LOG_DIR).glob("*"))), ({}, []))
        with patch.object(j, "llm_message", AsyncMock(return_value=answer("Hello."))):
            await j.handle(self.said("hello?"))
        log = rh.logs[self.channel.id]
        # the first log has what came before
        self.assertEqual([(e["type"], e.get("text")) for e in log.events][-3:],
                         [("message", "anyone seen my keys"), ("message", "@Testbot hello?"), ("rocky", "Hello.")])
        await j.on_message(self.said("thanks", name="kettle", ping=False))  # and from then on, what's said there
        self.assertEqual(log.events[-1]["text"], "thanks")

    async def test_a_new_log_is_filled_from_further_back_in_the_channel(self):
        rh.logs.clear()
        start = datetime.now(timezone.utc) - timedelta(days=2)
        scanned = []  # oldest first, ids in order like Discord's
        for i in range(40):
            scanned.append(self.said(f"old {i:02}", name="kettle", ping=False))
            if i == 20:
                scanned.append(self.said("!nocontext a secret", ping=False))
            if i == 25:
                scanned.append(asked := self.said("old question", name="kettle"))
                scanned.append(self.said("Old answer.", to=asked, bot=True))
        for k, m in enumerate(scanned):
            m.created_at = start + timedelta(minutes=k)
        befores = []

        def history(limit=None, before=None, **kwargs):  # newest first
            befores.append(before and before.id)
            async def page():
                for m in [m for m in reversed(scanned) if before is None or m.id < before.id][:limit]:
                    yield m
            return page()

        self.channel.history = history
        j.channel_history[self.channel.id] = []
        target = 20 * (len("old 00") + rh.EVENT_CHARS)  # about 20 old messages' worth
        with patch.object(j, "HISTORY_SCAN", 5), patch.object(j, "load_history", self.load_history), \
                patch.object(rh, "LOG_TOKENS", target / rh.CHARS_PER_TOKEN / rh.FILL), \
                patch.object(j, "llm_message", AsyncMock(return_value=answer("Hi."))):
            await j.handle(m := self.said("what did I miss"))
        texts = [e.get("text") for e in rh.logs[self.channel.id].events]
        # read back from where the channel's history ends
        self.assertEqual(befores, [m.id, min(e["id"] for e in j.channel_history[self.channel.id])])
        first = int(texts[0].split()[1])
        self.assertTrue(10 < first < 25, texts[0])  # not all of it: about the target, less what history had
        olds = [t for t in texts if t.startswith("old ")]
        self.assertEqual(olds, [f"old {i:02}" for i in range(first, 40)])
        self.assertNotIn("!nocontext a secret", texts)
        self.assertEqual(texts[texts.index("@Testbot old question") + 1], "Old answer.")  # its reply under it
        self.assertEqual(texts[-2:], ["@Testbot what did I miss", "Hi."])

    async def test_an_edit_that_names_it_wakes_it_once_while_the_message_is_recent(self):
        def edit(m, content):  # as discord.py hands it over, the message as it is now
            m.content = content
            m.clean_content = discord.Message.clean_content.function(m)
            return SimpleNamespace(message=m, cached_message=None)

        # asked without its name, so nothing happens — then its name edited in
        asked = self.said("can you look in the back?", ping=False)
        await j.on_message(asked)
        llm = AsyncMock(return_value=answer("Rocky look. Back is empty, question."))
        traces = []
        with patch.object(j, "llm_message", llm), patch.object(j, "write_trace", traces.append):
            await j.on_raw_message_edit(edit(asked, "can you look in the back testbot?"))
        asked.reply.assert_awaited_once()  # a reply to the message itself
        self.assertEqual(asked.reply.call_args.args[0], "Rocky look. Back is empty, question.")
        # its log has the message as it was first sent, and the edit as what woke it
        n = rh.logs[self.channel.id].ids[str(asked.id)]
        sent = j.text_of(llm.await_args.args[0][-1]["content"])
        self.assertRegex(sent, rf'<message id="{n}" from="pip" at="[0-9:]+">can you look in the back\?</message>\n'
                               rf'<edit id="{n}" to="you" woke_you="yes">can you look in the back testbot\?</edit>$')
        self.assertTrue(j.channel_history[self.channel.id][-1]["to_bot"])
        # and Context on the message shows the edit at the end of the turn's user block
        [t] = traces
        self.assertEqual(t["message_id"], asked.id)
        self.assertIn('<edit id="1" to="you" woke_you="yes">can you look in the back testbot?</edit>\n```\n'
                      "**assistant**", j.context_response(t, "Testbot")[0].description)

        # one that woke it by its name, and one it reacted to, woken by another
        named = self.said("testbot is the kettle on", ping=False)
        with patch.object(j, "llm_message", AsyncMock(return_value=answer("Kettle on."))):
            await j.on_message(named)
        await j.on_message(news := self.said("the kettle broke", name="kettle", ping=False))
        react = answer(calls=[("react", {"emoji": "😢", "message": rh.logs[self.channel.id].ids[str(news.id)]})])
        with patch.object(j, "llm_message", AsyncMock(return_value=react)):
            await j.on_message(self.said("testbot did you hear", ping=False))
        self.partials[news.id].add_reaction.assert_awaited_once_with("😢")
        await j.on_message(milk := self.said("where's the milk", ping=False))
        # then Discord's embed for it, typos fixed, its name taken out and put back — after a restart, from Rocky's
        # log alone: each already had its turn
        rh.flush()
        rh.logs.clear()
        j.taken.clear()
        j.channel_history.clear()
        with patch.object(j, "llm_message", llm):
            for m, content in ((named, named.content), (named, "testbot is the kettel on"), (named, "is the kettle on"),
                               (named, "testbot is the kettle on"), (asked, "can you look in the back, testbot?"),
                               (news, "testbot the kettle broke")):
                await j.on_raw_message_edit(edit(m, content))
        self.assertEqual(llm.await_count, 1)
        self.assertEqual([e["text"] for e in rh.logs[self.channel.id].events if e["type"] == "edit"][-2:],
                         ["can you look in the back, testbot?", "testbot the kettle broke"])
        # but one that was never to it does, from its log
        with patch.object(j, "llm_message", llm):
            await j.on_raw_message_edit(edit(milk, "testbot where's the milk"))
        milk.reply.assert_awaited_once()
        self.assertEqual(llm.await_count, 2)

        # nor does an edit to a message older than EDIT_WAKES minutes, or to one nothing has a record of
        old = self.said("anyone want tea", ping=False)
        await j.on_message(old)
        missed = self.said("what's for tea", ping=False)  # sent while the bot was down
        with patch.object(j, "llm_message", llm):
            await j.on_raw_message_edit(edit(missed, "testbot what's for tea"))
            with patch.object(rh, "clock", lambda: old.created_at + timedelta(minutes=j.EDIT_WAKES, seconds=1)):
                await j.on_raw_message_edit(edit(old, "testbot want tea?"))
        self.assertEqual(llm.await_count, 2)
        last = rh.logs[self.channel.id].events[-1]
        self.assertEqual((last["type"], last["text"], last.get("to_bot")), ("edit", "testbot want tea?", None))

    async def test_an_edit_and_a_nocontext_message(self):
        m = self.said("rocky soup or stew", ping=False)
        await j.on_message(m)
        m.content = "rocky soup or salad"
        m.clean_content = discord.Message.clean_content.function(m)
        await j.on_raw_message_edit(SimpleNamespace(message=m, cached_message=m))
        self.assertEqual(rh.logs[self.channel.id].events[-1], {**rh.logs[self.channel.id].events[-1], "type": "edit",
                                                                 "on": 1, "text": "rocky soup or salad"})
        await j.on_raw_message_delete(SimpleNamespace(channel_id=self.channel.id, message_id=m.id))
        self.assertEqual(rh.logs[self.channel.id].events[-1]["type"], "deleted")
        side = self.said("!nocontext secret soup")
        side.create_thread = AsyncMock(side_effect=discord.Forbidden(Mock(status=403), "no threads"))
        with patch.object(j, "llm_message", AsyncMock(return_value=answer("Soup good."))):
            await j.on_message(side)
        side.reply.assert_awaited_once()  # answered alone, with no thread to answer in
        self.assertNotIn(str(side.id), rh.logs[self.channel.id].ids)
        rh.flush()
        self.assertEqual([p.name for p in Path(rh.LOG_DIR).glob("*")], [f"{self.channel.id}.json"])  # its log isn't kept


class KeepWarmTests(_Handling):
    """Rocky's prompt cache kept warm while a channel is quiet: the bot's warm loop ticking each minute on a fake clock,
    with OpenRouter and Discord faked."""

    T = datetime(2026, 11, 2, 18, 0, tzinfo=timezone.utc)

    def setUp(self):
        super().setUp()
        self.now = self.T
        self.requests = []  # (when, the request's body)
        self.dir = Path(self.reactors_dir.name)
        patch_all(self, [(rh, "clock", lambda: self.now), (j, "LOG_DIR", self.dir / "logs")])
        j.channel_history[self.channel.id][0]["at"] = self.T - timedelta(minutes=10)
        self.channel.send = AsyncMock()
        self.partials = {}
        self.channel.get_partial_message = lambda i: self.partials.setdefault(i, Mock(add_reaction=AsyncMock()))

    # OpenRouter: Rocky answers a turn; a warm-up (one token at most) stops at its first, having read its prompt — or,
    # with self.held set, not until the test sets it
    held = None
    WARMED = {"choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "length",
                           "native_finish_reason": "max_tokens"}],
              "usage": {"prompt_tokens": 9000, "completion_tokens": 1, "cost": 0.0008,
                        "prompt_tokens_details": {"cached_tokens": 8996, "cache_write_tokens": 0}}}

    def api(self, body):
        self.requests.append((self.now, copy.deepcopy(body)))
        if body["max_tokens"] == 1 and self.held:
            return self.held_warm_up(self.held)
        if body["max_tokens"] == 1:
            return 200, self.WARMED
        return 200, {"choices": [{"message": answer("Granite. Strong, strong, strong."), "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 9000, "completion_tokens": 20, "cost": 0.0016}}

    async def held_warm_up(self, release):
        self.holding.set()
        await release.wait()
        return 200, self.WARMED

    def said(self, *args, **kwargs):
        m = super().said(*args, **kwargs)
        m.created_at = self.now
        return m

    async def minutes(self, n):  # the warm loop's ticks, a minute apart
        for _ in range(n):
            self.now += timedelta(minutes=1)
            await j.keep_warm.coro()

    def warm_ups(self):
        return [(when, body) for when, body in self.requests if body["max_tokens"] == 1]

    def prompt(self, body):
        return "\n".join(j.text_of(m.get("content")) for m in body["messages"])

    def same_request(self, warm, turn):  # all a cache entry depends on: OpenRouter's body but for max_tokens
        return {k: v for k, v in warm.items() if k != "max_tokens"} == {k: v for k, v in turn.items() if k != "max_tokens"}

    async def test_a_quiet_channel_is_kept_warm_for_over_a_day_after_its_last_turn(self):
        with openrouter(self.api):
            m = self.said("which rock is best")
            await j.respond(m)
            (_, turn), = self.requests
            m.reply.assert_awaited_once()
            self.now += timedelta(minutes=1)
            brb = self.said("brb, dinner", name="kettle", ping=False)
            await j.on_message(brb)  # said since: not part of a warm-up
            log = rh.logs[self.channel.id]
            events, facts = copy.deepcopy(log.events), copy.deepcopy(rh.memories)

            # Before the hour since its turn is up, the request the turn made goes again, for one token
            await self.minutes(48)
            self.assertEqual(self.warm_ups(), [])
            await self.minutes(1)
            (when, warm), = self.warm_ups()
            self.assertEqual(when, self.T + timedelta(minutes=50))
            self.assertTrue(self.same_request(warm, turn))
            self.assertNotIn("dinner", self.prompt(warm))
            # and whatever it said is thrown away
            m.reply.assert_awaited_once()
            self.channel.send.assert_not_awaited()
            self.assertEqual(self.partials, {})
            self.assertEqual((log.events, rh.memories), (events, facts))

            # A turn in progress holds the next back, and so does being out of credit, while there's time
            async with j.gen_lock:
                await self.minutes(50)
            self.assertEqual(len(self.warm_ups()), 1)
            await self.minutes(1)
            with patch.object(j, "has_credit", False):
                await self.minutes(50)
            self.assertEqual(len(self.warm_ups()), 2)
            await self.minutes(1)
            self.assertEqual([when - self.T for when, _ in self.warm_ups()],
                             [timedelta(minutes=50), timedelta(minutes=101), timedelta(minutes=152)])

            # A message deleted is blanked at the next warm-up, not kept in view as long as the cache is
            await j.on_raw_message_delete(SimpleNamespace(channel_id=self.channel.id, message_id=1))
            await self.minutes(50)
            blanked = self.warm_ups()[-1][1]
            self.assertNotIn("toaster", self.prompt(blanked))
            self.assertIn('<message id="1" from="kettle"', self.prompt(blanked))
            self.assertIn('deleted="yes"/>', self.prompt(blanked))
            await self.minutes(50)
            self.assertTrue(self.same_request(self.warm_ups()[-1][1], blanked))

            # It goes on while its last turn is within KEEP_WARM_HOURS, never an hour without a request, then stops
            await self.minutes((self.T + timedelta(hours=rh.KEEP_WARM_HOURS + 2) - self.now) // timedelta(minutes=1))
            times = [when for when, _ in self.requests]
            self.assertLess(max(b - a for a, b in zip(times, times[1:])), timedelta(hours=1))
            self.assertLess(times[-1], self.T + timedelta(hours=rh.KEEP_WARM_HOURS))
            self.assertGreater(times[-1], self.T + timedelta(hours=rh.KEEP_WARM_HOURS - 1))
            warmed = len(self.warm_ups())
            self.assertEqual((m.reply.await_count, self.channel.send.await_count, self.partials), (1, 0, {}))

            # Its next turn, its cache gone: deleted since is blanked first, and its Stats say what kept it warm
            await j.on_raw_message_delete(SimpleNamespace(channel_id=self.channel.id, message_id=brb.id))
            again = self.said("and the worst rock?")
            await j.respond(again)
            self.assertFalse(self.requests[-1][1]["max_tokens"] == 1)
            self.assertNotIn("dinner", self.prompt(self.requests[-1][1]))

        traces = [json.loads(line) for path in sorted((self.dir / "logs").glob("*.jsonl"))
                  for line in path.read_text().splitlines()]
        warm = [t for t in traces if t.get("warm")]
        self.assertEqual(len(warm), warmed)
        self.assertEqual({(t["channel_id"], t["cost"], t["api_calls"][0]["purpose"]) for t in warm},
                         {(self.channel.id, 0.0008, "warm")})
        # a warm-up isn't one of Rocky's turns on its page, nor found by an action on a message
        self.assertEqual(len(rocky_viewer.Files(self.dir).turns()), 2)
        latest = j.find_trace(self.channel.id, self.sent, lambda t: t.get("reply"))
        self.assertEqual(latest["warm_ups"], warmed)
        kept = next(f for f in j.stats_response(latest).fields if f.name == "Kept warm")
        self.assertEqual(kept.value, f"{warmed} warm-ups since the turn before · {j.money(0.0008 * warmed)}")

    async def test_after_a_restart_it_carries_on_from_what_was_saved_a_channel_at_a_time(self):
        channels = {}
        for name, cid in [("bot-spam", 300), ("garden", 301), ("kitchen", 302)]:
            channel = self.channel if cid == 300 else Mock(id=cid)
            channel.name, channel.guild, channel.typing = name, self.guild, lambda: contextlib.nullcontext()
            channels[name] = channel
        with openrouter(self.api):
            self.now = self.T - timedelta(hours=2)
            await j.respond(self.said("is basalt good", channel=channels["kitchen"]))
            self.now = self.T
            for name in ("bot-spam", "garden"):  # two turns at once
                await j.respond(self.said("which rock is best", channel=channels[name]))
            turns = {name: body for (_, body), name in zip(self.requests, ["kitchen", "bot-spam", "garden"])}
            self.now = self.T + timedelta(minutes=30)
            rh.flush()
            rh.logs.clear()  # a restart, and nothing said since
            await self.minutes(19)
            # A slow warm-up keeps no reply waiting
            release = asyncio.Event()
            self.held, self.holding = release, asyncio.Event()
            tick = asyncio.create_task(self.minutes(1))
            await asyncio.wait_for(self.holding.wait(), 1)
            self.held = None
            m = self.said("is basalt still good", channel=channels["kitchen"])
            await asyncio.wait_for(j.respond(m), 1)
            m.reply.assert_awaited_once()
            release.set()
            await tick
            await self.minutes(10)
            warm_ups = self.warm_ups()
        # the two due together a minute apart; kitchen, an hour past its last request when the bot came back, not at all
        self.assertEqual([when - self.T for when, _ in warm_ups], [timedelta(minutes=50), timedelta(minutes=51)])
        self.assertEqual(sorted(name for _, body in warm_ups for name in turns if self.same_request(body, turns[name])),
                         ["bot-spam", "garden"])


class OutgoingMentionTests(_Handling):
    def test_name_matching_and_ping_allowlist(self):
        targets = j.mention_targets([{"name": name, "users": {str(uid): name}} for name, uid in
                                    [("Moss", 501), ("Pip Meadow", 502), ("Shared", 503), ("shared", 504)]], "Testbot")
        for text, expected, ids in [
                ("Moss @moss, @Pip Meadow!", "Moss <@501>, <@502>!", [501, 502]),
                ("@Moss Fernando", None, []), ("@Moss knows", None, []), ("@Shared!", None, []),
                ("@Moss\n@Pip Meadow", "<@501>\n<@502>", [501, 502]),
                ("https://example.com/@Moss `@Moss` @Moss!", "https://example.com/@Moss `@Moss` <@501>!", [501]),
                ("@Unknown <@501> <@&600> @everyone @here @Testbot", None, [])]:
            with self.subTest(text=text):
                content, allowed = j.resolve_mentions(text, targets)
                self.assertEqual(content, expected or text)
                self.assertEqual(allowed.to_dict(), {"parse": [], "users": ids})

    async def test_rocky_can_ping(self):
        uid = 100000000000000002
        m = self.said(f"ask <@{uid}>")
        m.mentions.append(SimpleNamespace(id=uid, display_name="Moss"))
        m.clean_content = discord.Message.clean_content.function(m)
        with patch.object(j, "llm_message", AsyncMock(return_value=answer("@Moss!"))):
            await j.handle(m)
        self.assertEqual(m.reply.call_args.args[0], f"<@{uid}>!")
        allowed = m.reply.call_args.kwargs["allowed_mentions"].to_dict()
        self.assertEqual((allowed["users"], allowed["parse"]), ([uid], []))
        self.assertFalse(m.reply.call_args.kwargs["mention_author"])

    async def test_mentions_survive_a_restart(self):
        uid = 100000000000000002

        async def turn(m, says):
            if m.reference:
                m.reference.resolved = self.by_id[m.reference.message_id]
            with patch.object(j, "llm_message", AsyncMock(return_value=answer(says))):
                await j.handle(m)

        j.channel_history[self.channel.id] = []
        root = self.said(f"ask <@{uid}>")
        root.mentions.append(SimpleNamespace(id=uid, display_name="Moss"))
        root.clean_content = discord.Message.clean_content.function(root)
        await turn(root, "@moss!")
        live = j.channel_history[self.channel.id][0].copy()
        reply = self.sent
        j.channel_history[self.channel.id] = []
        rh.logs.clear()

        async def history(**kwargs):
            for message in (reply, root):
                yield message
        self.channel.history = history
        await self.load_history(SimpleNamespace(channel=self.channel))
        question = self.said("again", to=reply)
        await turn(question, "@Moss!")
        rebuilt = j.channel_history[self.channel.id][0]
        self.assertEqual(live["reply"], rebuilt["reply"])
        self.assertEqual(live["reply_users"], rebuilt["reply_users"])
        self.assertEqual(question.reply.call_args.args[0], f"<@{uid}>!")


class AppHistoryTests(_Handling):
    def app_post(self):
        m = self.said("", name="Milestone App", ping=False)
        m.author.bot = True
        m.embeds = [discord.Embed(title="Contribution milestone", description="synthetic-user passed **1,000 contributions**.")]
        return m

    async def test_live_app_embed_is_seen_when_answering_a_human(self):
        app = self.app_post()
        with patch.object(j, "respond", new_callable=AsyncMock) as respond:
            await j.on_message(app)
        respond.assert_not_awaited()
        question = self.said("say congrats")
        llm = AsyncMock(return_value=answer("Congrats, synthetic-user!"))
        with patch.object(j, "llm_message", llm):
            await j.handle(question)
        chat = j.text_of(llm.await_args.args[0][-1]["content"])
        self.assertIn('from="Milestone App"', chat)
        self.assertIn(">[embed: Contribution milestone synthetic-user passed 1,000 contributions.]</message>", chat)
        self.assertIn("say congrats", chat)
        self.assertIn("[embed: Contribution milestone synthetic-user passed 1,000 contributions.]", j.context_text(self.trace["context"]))
        self.assertEqual(question.reply.call_args.args[0], "Congrats, synthetic-user!")

    async def test_app_posts_rebuilt_after_restart_and_own_status_excluded(self):
        app = self.app_post()
        own = self.said("I feel happy", bot=True)
        j.channel_history[self.channel.id] = []

        async def history(**kwargs):
            for m in (own, app):
                yield m

        self.channel.history = history
        with patch.object(j, "load_history", self.load_history):
            seen = await self.answer(self.said("say congrats"))
        self.assertIn(("Milestone App", "[embed: Contribution milestone synthetic-user passed 1,000 contributions.]"), seen)
        self.assertNotIn("I feel happy", [text for _, text in seen])

    async def test_app_embed_edits_update_context(self):
        app = self.app_post()
        await j.on_message(app)
        after = self.app_post()
        after.id = app.id
        after.author.display_name = "Renamed App"
        after.embeds = [discord.Embed(title="Contribution milestone", description="synthetic-user passed **2,000 contributions**.")]
        await j.on_raw_message_edit(SimpleNamespace(message=after, cached_message=app))
        seen = await self.answer(self.said("say congrats"))
        self.assertIn(("Milestone App", "[embed: Contribution milestone synthetic-user passed 2,000 contributions.]"), seen)


class NoContextThreadTests(_Handling):
    """"@rocky !nocontext ..." opens a thread on the message, a channel of its own: answered there with nothing from
    before, every message in it to the bot, and without Rocky's memories."""

    def setUp(self):
        super().setUp()
        self.threads = {}
        self.partials = {}
        self.channel.get_partial_message = lambda i: self.partials.setdefault(i, Mock(add_reaction=AsyncMock()))

    # A !nocontext message to the bot, which a thread can be opened on — with the message's id, as Discord's are
    def asked(self, content):
        m = self.said(content)

        async def create_thread(name, **kwargs):
            t = Mock(spec=discord.Thread, id=m.id, guild=self.guild)
            t.name = name
            # not the channel's Mock itself, which assigning would make the thread's child
            t.parent = SimpleNamespace(id=self.channel.id, fetch_message=lambda i: self.channel.fetch_message(i))
            t.typing = lambda: contextlib.nullcontext()
            t.send = AsyncMock(side_effect=lambda content, **kwargs: self.said(content, bot=True, channel=t))
            t.get_partial_message = lambda i: self.partials.setdefault(("thread", i), Mock(add_reaction=AsyncMock()))

            def history(limit=None, before=None, **kwargs):  # newest first: the thread's, not the message it's on
                async def page():
                    for x in sorted(self.by_id.values(), key=lambda x: -x.id):
                        if x.channel is t and (before is None or x.id < before.id):
                            yield x
                return page()
            t.history = history
            self.threads[m.id] = t
            return t
        m.create_thread = AsyncMock(side_effect=create_thread)
        return m

    async def test_it_answers_in_the_thread_and_carries_on_there(self):
        q = self.asked("!nocontext favourite rock?")
        self.assertEqual(await self.answer(q, "Granite."), [])
        thread = self.threads[q.id]
        self.assertEqual(q.create_thread.call_args.kwargs["name"], "favourite rock?")
        q.reply.assert_not_awaited()
        thread.send.assert_awaited_once()
        self.assertEqual(thread.send.call_args.args[0], "Granite.")
        self.assertEqual((self.trace["no_context"], self.trace["channel_id"]), (True, thread.id))
        # the actions find it from the message or in the thread
        self.assertTrue(j.in_channel(self.trace, self.channel.id) and j.in_channel(self.trace, thread.id))
        # in the thread, nobody needs to tag the bot, and it has the thread so far
        follow = self.said("why granite", ping=False, channel=thread)
        self.assertTrue(j.should_respond(follow))
        self.assertEqual(await self.answer(follow, "Very good."), [("pip", "@Testbot favourite rock?"), ("you", "Granite.")])
        self.assertEqual(follow.reply.call_args.args[0], "Very good.")
        # and the channel it came from never sees any of it
        self.assertEqual([e["content"] for e in j.channel_history[self.channel.id]], ["the toaster is on fire again"])
        self.assertFalse(j.should_respond(self.said("anyone?", ping=False)))
        self.assertNotIn("rock", j.recent_chat())  # nor do the statuses, which are public

    async def test_catch_up_leaves_a_message_answered_in_its_thread(self):
        q = self.asked("!nocontext favourite rock?")
        await self.answer(q, "Granite.")

        def history(**kwargs):
            async def page():
                yield q
            return page()
        self.channel.history = history
        with patch.object(j, "DM_USERS", set()), patch.object(j, "respond", new_callable=AsyncMock) as respond:
            await j.catch_up()
        respond.assert_not_awaited()

    async def test_catch_up_answers_a_thread_opened_but_not_answered(self):
        q = self.asked("!nocontext favourite rock?")
        thread = await j.open_thread(q)  # and then a restart, before it answered
        rh.flush()
        rh.logs.clear()

        def history(**kwargs):
            async def page():
                yield q
            return page()
        self.channel.history = history
        with patch.object(j, "DM_USERS", set()), patch.object(j.bot, "fetch_channel", AsyncMock(return_value=thread)), \
                patch.object(j, "llm_message", AsyncMock(return_value=answer("Granite."))):
            await j.catch_up()
        q.create_thread.assert_awaited_once()  # the thread it opened, not another
        self.assertEqual(thread.send.call_args.args[0], "Granite.")

    async def test_editing_the_question_is_seen_in_its_thread(self):
        q = self.asked("!nocontext favourite rock?")
        await self.answer(q, "Granite.")
        q.content = f"<@{self.user.id}> !nocontext favourite colour?"
        q.clean_content = discord.Message.clean_content.function(q)
        await j.on_raw_message_edit(SimpleNamespace(message=q, cached_message=q))
        self.assertEqual([e["content"] for e in j.channel_history[q.id]], ["@Testbot favourite colour?"])
        edit = rh.logs[q.id].events[-1]
        self.assertEqual((edit["type"], edit["text"]), ("edit", "@Testbot favourite colour?"))

    async def test_a_question_edited_into_a_nocontext_one_is_answered_in_its_thread_alone(self):
        rh.log_for(self.channel)
        q = self.asked("favourite colour?")
        q.content, q.mentions = "favourite colour?", []  # chatter when it's sent
        q.clean_content = discord.Message.clean_content.function(q)
        await j.on_message(q)
        q.content, q.mentions = f"<@{self.user.id}> !nocontext favourite colour, secretly?", [self.user]
        q.clean_content = discord.Message.clean_content.function(q)
        with patch.object(j, "llm_message", AsyncMock(return_value=answer("Green."))):
            await j.on_raw_message_edit(SimpleNamespace(message=q, cached_message=None))
        self.assertEqual(self.threads[q.id].send.call_args.args[0], "Green.")
        q.reply.assert_not_awaited()
        # the channel never sees the question it became
        self.assertNotIn("secretly", json.dumps([e.get("text") for e in rh.logs[self.channel.id].events]))
        self.assertNotIn("secretly", json.dumps([e["content"] for e in j.channel_history[self.channel.id]]))

    async def test_replies_under_an_old_side_conversation_stay_out_of_history(self):
        q = self.said("!nocontext favourite rock?")
        a = self.said("Granite. [↳](https://discord.com/channels/400/300/800)", to=q, bot=True)
        b = self.said("why granite", to=a, name="mossy", ping=False)
        c = self.said("Hard.", to=b, bot=True)
        chatter = self.said("anyone seen my keys", name="kettle", ping=False)

        async def history(**kwargs):
            for m in (chatter, c, b, a, q):  # newest first, like Discord
                yield m
        self.channel.history = history
        j.channel_history[self.channel.id] = []
        await self.load_history(SimpleNamespace(channel=self.channel))
        self.assertEqual([e["content"] for e in j.channel_history[self.channel.id]], ["anyone seen my keys"])

    async def test_a_thread_it_cant_post_in_is_answered_in_the_channel(self):
        q = self.asked("!nocontext favourite rock?")
        opened = q.create_thread.side_effect

        async def create_thread(**kwargs):
            t = await opened(**kwargs)
            t.send = AsyncMock(side_effect=discord.Forbidden(Mock(status=403), "no sending in threads"))
            return t
        q.create_thread.side_effect = create_thread
        with patch.object(j, "llm_message", AsyncMock(return_value=answer("Granite."))):
            await j.handle(q)
        self.assertEqual(q.reply.call_args.args[0], "Granite.")

    async def test_nocontext_in_its_thread_carries_it_on(self):
        q = self.asked("!nocontext favourite rock?")
        await self.answer(q, "Granite.")
        thread = self.threads[q.id]
        again = self.said("!nocontext and basalt?", ping=False, channel=thread)
        self.assertFalse(j.no_context(again))
        with patch.object(j, "respond", new_callable=AsyncMock):
            await j.on_message(again)
        self.assertEqual(rh.logs[thread.id].events[-1]["text"], "and basalt?")

    async def test_a_nocontext_reply_shows_nothing_it_replies_to(self):
        earlier = self.said("the secret plan is soup", name="kettle", ping=False)
        q = self.asked("!nocontext what do you think")
        q.reference = SimpleNamespace(message_id=earlier.id, resolved=earlier)
        llm = AsyncMock(return_value=answer("Think good."))
        with patch.object(j, "llm_message", llm):
            await j.handle(q)
        self.assertNotIn("secret plan", str(llm.await_args.args[0]))

    async def test_rocky_in_the_thread_has_no_memories(self):
        rh.use(rh.Log("bot-spam", "400", self.channel.id),
               {"function": {"name": "remember", "arguments": '{"about": "pip", "fact": "likes soup"}'}}, {"n": 1})
        q = self.asked("!nocontext what do you know about me")
        llm = AsyncMock(return_value=answer("Nothing. New friend.", [("react", {"emoji": "🪨"})]))
        with patch.object(j, "llm_message", llm):
            await j.handle(q)
        thread = self.threads[q.id]
        self.assertEqual(thread.send.call_args.args[0], "Nothing. New friend.")
        self.partials[q.id].add_reaction.assert_awaited_once_with("🪨")  # on the message the thread is on
        request, tools = llm.await_args.args[0], llm.await_args.kwargs["tools"]
        self.assertNotIn("soup", str(request))
        self.assertNotIn("<memories>", str(request))
        self.assertNotIn("remember", [t["function"]["name"] for t in tools])
        log = rh.logs[thread.id]
        self.assertEqual([(e["type"], e.get("text")) for e in log.events],
                         [("message", "@Testbot what do you know about me"), ("rocky", "Nothing. New friend."),
                          ("result", None)])
        self.assertFalse(rh.has_log(self.channel.id))  # the channel's own log isn't started, or given it

        rh.flush()
        rh.logs.clear()  # a restart: the thread's log is still one without memories
        follow = self.said("ok remember I like rocks", ping=False, channel=thread)
        llm = AsyncMock(return_value=answer("Rocks good."))
        with patch.object(j, "llm_message", llm):
            await j.on_message(follow)
        self.assertEqual(follow.reply.call_args.args[0], "Rocks good.")
        self.assertNotIn("<memories>", str(llm.await_args.args[0]))
        self.assertIsNone(rh.logs[thread.id].space)
        self.assertEqual(rh.memory("400")["facts"][0]["fact"], "likes soup")

        # the message the thread is on is in the channel: a look or a reaction later goes there, not to the thread
        log = rh.logs[thread.id]
        self.assertEqual((log.channel_of(1), log.channel_of(3)), (self.channel.id, thread.id))
        again = self.said("react to my question", ping=False, channel=thread)
        llm = AsyncMock(return_value=answer(calls=[("react", {"emoji": "👍", "message": 1})]))
        with patch.object(j, "llm_message", llm), \
                patch.object(j.bot, "get_partial_messageable", lambda i: {self.channel.id: self.channel}[i]):
            await j.on_message(again)
        self.partials[q.id].add_reaction.assert_awaited_with("👍")

        # and when it's edited or deleted, the thread's log hears of it, loaded or not
        rh.flush()
        rh.logs.clear()
        q.content = f"<@{self.user.id}> !nocontext what do you know about pip"
        q.clean_content = discord.Message.clean_content.function(q)
        await j.on_raw_message_edit(SimpleNamespace(message=q, cached_message=q))
        await j.on_raw_message_delete(SimpleNamespace(channel_id=self.channel.id, message_id=q.id))
        events = rh.logs[thread.id].events
        self.assertIn({"type": "edit", "on": 1, "text": "@Testbot what do you know about pip"},
                      [{k: e.get(k) for k in ("type", "on", "text")} for e in events])
        self.assertTrue(events[0]["deleted"])
        self.assertEqual(rh.logs[thread.id].channel, "what do you know about me")
        self.assertFalse(rh.has_log(self.channel.id))

    async def test_in_a_dm_its_answered_alone(self):
        q = self.said("!nocontext favourite rock?")
        q.guild = None
        rh.log_for(self.channel)  # a DM Rocky's talked in
        llm = AsyncMock(return_value=answer("Granite."))
        with patch.object(j, "llm_message", llm):
            await j.handle(q)
        self.assertEqual(q.reply.call_args.args[0], "Granite.")
        people = [m for m in llm.await_args.args[0] if m["role"] == "user"]
        self.assertEqual(len(people), 1)  # just the question: no memories, nothing from before
        self.assertIn("favourite rock?", j.text_of(people[0]["content"]))
        self.assertIn("This is #favourite rock? on Discord", j.text_of(llm.await_args.args[0][0]["content"]))  # as a thread is
        self.assertEqual(rh.logs[self.channel.id].events, [])  # nor is it in the DM's log
        rh.flush()
        self.assertEqual(list(Path(rh.LOG_DIR).glob("*")), [])  # and its own is never saved

    async def test_a_bare_nocontext_in_a_dm_is_nothing_to_answer(self):
        q = self.said("!nocontext", ping=False)
        q.guild = None
        llm = AsyncMock()
        with patch.object(j, "llm_message", llm):
            await j.handle(q)
        self.assertNotIn("error", self.trace)
        llm.assert_not_awaited()

    async def test_normal_reply_has_no_marker(self):
        await self.answer(self.said("favourite rock?"), "Granite.")
        self.assertEqual(self.sent.content, "Granite.")

    def test_old_side_conversation_links_stay_out_of_context(self):
        for text in ("Granite. [↳ thread](https://discord.com/channels/400/300/800)",
                     "Granite. [↳](https://discord.com/channels/@me/300/800)"):
            self.assertEqual(j.message_text(self.said(text, bot=True)), "Granite.")

    def test_what_rocky_did_under_its_message_stays_out_of_context(self):
        m = self.said("Stout for Pip.\n-# 🍺 ordered Eridani Stout for pip\n-# 🧠 remembered: pip likes stout", bot=True)
        self.assertEqual(j.message_text(m), "Stout for Pip.")
        self.assertIn("-# 🍺 ordered a lager", j.message_text(self.said("-# 🍺 ordered a lager")))  # only Rocky's


class HistoryForRockyTests(_Handling):
    """The channel's history keeps everything a rebuild read, for Rocky's log."""

    def setUp(self):
        super().setUp()
        self.start = datetime.now(timezone.utc) - timedelta(hours=1)
        j.channel_history[self.channel.id] = []

    def test_the_store_keeps_what_a_rebuild_reads_and_who_asked_from_further_back(self):
        j.add_history(self.channel.id, {"name": "pip", "content": "rocky soup?", "id": 1, "at": self.start,
                                        "to_bot": True, "users": {"501": "pip"}})
        for i in range(j.HISTORY_SCAN + 100):
            j.add_history(self.channel.id, {"name": "kettle", "content": f"message {i}", "id": i + 2,
                                            "at": self.start + timedelta(seconds=i + 1), "to_bot": False})
        stored = [e["content"] for e in j.channel_history[self.channel.id]]
        self.assertEqual(stored, ["rocky soup?"] + [f"message {i}" for i in range(100, j.HISTORY_SCAN + 100)])
        # so Rocky can still ping whoever asked it something before all that chatter
        self.assertEqual(j.mention_targets(j.channel_history[self.channel.id], "Testbot")["pip"], 501)

    async def test_a_rebuild_keeps_everything_it_read(self):
        scanned = []
        for i in range(j.HISTORY_SCAN):
            m = self.said(f"old {i}", name="kettle", ping=False)
            m.created_at = self.start - timedelta(hours=2) + timedelta(seconds=i)
            scanned.append(m)

        async def history(**kwargs):
            for m in reversed(scanned):  # newest first, like Discord
                yield m

        self.channel.history = history
        await self.load_history(SimpleNamespace(channel=self.channel))
        self.assertEqual([e["content"] for e in j.channel_history[self.channel.id]],
                         [f"old {i}" for i in range(j.HISTORY_SCAN)])

if __name__ == "__main__":
    unittest.main()
