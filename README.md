# rocky

Discord bot named after Rocky, the Eridian engineer from *Project Hail Mary*: Claude Haiku, on Anthropic's Messages API, as Rocky himself ("Is called Biscuit. Small predator, no respect for hot liquid."), an agent with its own log of each channel, tools and memories.

Forked from [lyramakesmusic/jevbot](https://github.com/lyramakesmusic/jevbot), where it began as Jev, [TypeSafe's](https://openrouter.ai/~typesafe/jev-latest) decision model made to talk one word at a time ("Band is from california in san los angeles. Las angels."). Jev wrote here until October 2026; it's in the git history.

## Setup

Install [mise](https://mise.jdx.dev/getting-started.html), then run:

```bash
scripts/setup.sh
```

It installs the Python pinned in `.python-version` and the uv pinned in `mise.toml`, the toolchain CI tests on, and fetches the dependencies in `requirements.txt`. In a worktree it also links `.env` from the main clone. It's safe to re-run. Without mise it warns and goes on with the uv on `PATH`, which still reads `.python-version`. Activate mise in your shell (`mise activate`) or prefix commands with `mise exec --` to use the pins.

Create `.env`:

```
DISCORD_TOKEN_JEV=your_discord_bot_token
ANTHROPIC_API_KEY=your_claude_api_key
ANTHROPIC_API_KEY_EVAL=another_claude_api_key
STATUS_CHANNEL_ID=channel_to_post_statuses_in
TIMEZONE=Europe/London
ROCKY_VIEWER_PORT=8350
```

- `ANTHROPIC_API_KEY` is the bot's, and it won't start without one. `rocky_replay.py` uses `ANTHROPIC_API_KEY_EVAL`, and won't run without it rather than use the bot's. Make each in its own Claude Console workspace, so replays have their own usage, limits and cache, and can't use up the bot's rate limit.
- `STATUS_CHANNEL_ID` (optional): each new status is also posted there (see Status).
- `TIMEZONE` (an IANA name, UTC if unset): the clock the times in Rocky's log and the bar's hours are on.
- `ROCKY_VIEWER_PORT` (off if unset): serves the page of Rocky's logs on that port while the bot runs (see Rocky's logs).

Optional files next to `rocky_bot.py`, all gitignored. Lists are one entry per line, `#` for comments; restart the bot to pick up changes.

- `rocky_lines.txt`: real Rocky lines, shown to the LLM as examples. A line can start with a note in brackets on what had just happened (`(a friend uses slang he doesn't know) No understand word.`), which it's told isn't his words. They're quotes from copyrighted works, so they stay out of the repo. Without it Rocky goes by the voice rules alone.
- `rocky_body.txt`: the book's narration of what Rocky looks like and is made of, a paragraph per passage with blank lines between, so he knows drawings of himself. Shown as narration, apart from his lines.
- `rocky_book.txt`: the novel as text, a paragraph a line, for `think_back`. `uv run --with-requirements requirements.txt python rocky_book.py` cuts it into `rocky_book.json` (gitignored too), which is what the bot reads, and picks up new or rebuilt without a restart; without that file Rocky has no `think_back` and isn't told of it.
- `dm_users.txt`: Discord user IDs whose DMs get answered, no mention needed. Other DMs are ignored. What's said in DMs never makes it into a status.
- `admins.txt`: Discord user IDs who can change `/settings`, as well as the bot's owner in Discord's developer portal.

In the Discord developer portal, enable **Message Content Intent**. The bot needs **Read Message History** (to rebuild history and catch up after a restart), **Add Reactions** (for Rocky's `react`) and **Create Public Threads** (for `!nocontext`).

```bash
python3 run.py
```

`run.py` runs the bot (`mise exec -- uv run --with-requirements requirements.txt rocky_bot.py`, so it runs on the pinned toolchain and mise installs a new pin before the restart; without mise, plain `uv run`; pass another command after `--`) and keeps it up to date: every minute it fetches, and when the branch's upstream has new commits that fast-forward onto a clean checkout it pulls and restarts the bot, so merging to `master` deploys. Commits that only touch files the bot never loads (CI, docs, tests, dev scripts: `NO_RESTART` in `run.py`) are pulled without a restart. To restart, it stops the bot and replaces itself with the pulled `run.py`, which starts the bot again, so changes to `run.py` take effect without restarting it by hand. Local edits or commits are left alone. If the bot exits on its own it's restarted, backing off if it keeps dying. Ctrl-C passes on to the bot; a third Ctrl-C kills it.

Ctrl-C (or SIGTERM) to the bot lets replies in progress finish: it stops taking new messages and exits once it's done. Ctrl-C again quits without finishing. Messages sent while it's offline never reach it, so on startup it looks back `CATCH_UP_WINDOW` (30) minutes in each channel and answers the latest message to it since it last replied or reacted there, one per channel.

## Talking to it

Tag the bot or its role, mention its name in plain text, or reply to one of its messages with the reply ping on. Plain names match the bot's server nickname, account display name or `rock`, what people call it (`NICKNAMES`), regardless of case, as a whole name, allowing extra repetitions of the final letter (`rocky`, `Rocky!`, `Rocky's`, `ROCKYYYYYYY` and `thanks rockkk` count; `rockyard` and `rocket` don't). A reply with the ping off is just chatter unless it tags or names the bot. Other bots' and apps' posts are seen as chatter and never trigger a reply.

Editing a message within five minutes of sending it (`EDIT_WAKES`) so that it tags or names the bot wakes it as if it had been sent that way: it answers that message, once, and in its log the edit is what woke it. An edit to a message that was already to the bot or that it has reacted to (a typo fixed, Discord adding a link's preview), or to an older message, is only recorded.

Put `!nocontext` in a message to the bot (`@rocky !nocontext what's your favourite colour?`) and it opens a thread on that message, named from it, and answers there with none of the channel's history in view. Every message in the thread is to the bot, no tag needed, and nothing said there reaches the channel's history, Rocky's log of it, or statuses. If the message is a Discord reply, what it replies to isn't shown. Rocky's log of the thread has no memories: no `<memories>` block, no `remember` or `forget`, and its prompt says nothing said there is kept. `!nocontext` again inside the thread just carries on. In a DM, or when the thread can't be opened, it answers that message on its own, with nothing before it.

Rocky can ping people by writing `@` and their full display name (`@Pip Meadow, ...`). The name must end at punctuation, a line break or the end of the reply, so `@Moss, ...` pings Moss but `@Moss knows` stays text. Names match regardless of case; bare names, ambiguous names, code and URLs stay text. Only people in its context can be pinged: speakers, mentioned users and reactors. Roles, `@everyone`, `@here`, raw IDs and automatic reply pings stay off, and statuses and the Context answers never ping anyone.

## Commands

- `/settings` shows how often a new status is made (`STATUS_EVERY`) and Rocky's context window (`LOG_TOKENS`), visible only to you. Admins get a menu under each: a status every 1 to 24 hours, or off, and a window of 25K to 90K tokens (past 100K Haiku charges 5× for the whole prompt). A smaller window takes effect after Rocky's next turn, when the oldest half of its log goes; a much smaller one can take a few turns, halving each time. Kept in `settings.json` (gitignored). The menus stop working after 15 minutes or a restart. Works in a DM too.

Right-click (long-press on mobile) a reply, the message that woke Rocky, or a posted status and pick from **Apps** for answers only you see:

- **Context**: the request the model was sent, a block for each message as it went: for Rocky's turn, the people's message that woke it, each of its messages with its tool calls' JSON, each tool's result and what it said; for a status, what it was asked and what it wrote. What came before is a line each: the system prompt by its length, `<memories>` by how many facts, and how many earlier messages, which are attached as `context.txt` with the turn. Neither the system prompt nor the memories are shown, nor what a note says, nor any thinking, only how many blocks there were. A block too long for Discord has its middle cut. Below are the model, message count, output tokens, input against the model's window, and the input cache: how much was read from it, written to it and neither. Turns logged before Context kept the request show the log as text.
- **Stats**: the model, total recorded cost in USD (every logged request, priced from the tokens Anthropic reports with `PRICES` in `rocky_bot.py`; through OpenRouter, what it billed), and context usage of the final request. For a Rocky turn, also the warm-ups that kept its cache warm since the turn before (below), and what they cost.
- **Diagnostics**: request counts and purposes, elapsed time, model IDs, billing, cache and output details, retries, and why an LLM request stopped early (a refusal is also logged as a warning with its category and message id, never what was said, and nothing of it is posted or kept).

These all read the last `LOG_DAYS` (7) of `logs/`, finding the message by its id, and make no model requests. Everyone who can use apps in a channel sees the actions; a server can limit them in **Server Settings → Integrations**. The bot registers them each time it starts, so an open Discord client may need reloading (Ctrl+R) the first time.

## Rocky

Haiku is Rocky, in every channel and DM. Each channel is Rocky's own conversation, kept by `rocky_harness.py` as an append-only log in `rocky_logs/` (gitignored).

- **The log**: people's messages, reactions and edits come in as user turns, written as XML with who, when and what they reply to (`<message id="12" from="pip" at="18:01" to="you" woke_you="yes">…</message>`); Rocky's turns are the assistant's. It takes everything said in the channel, and after a restart what it missed is added at the end. A channel gets a log the first time Rocky is woken there, filled from its history back to half of `LOG_TOKENS` (`FILL`) or `FILL_SCAN` (3000) messages, leaving out `!nocontext` messages and reactions to the bot's old replies.
- **Waking**: a tag, its name in plain text or a reply with the ping on wakes it. It thinks first (low effort), then what it writes is posted as its reply. Spoken to, it answers in words, and a question or a request always gets them.
- **Tools**, called with its message or before it:
  - `react` puts an emoji on any message in the log: alongside its words, or on its own when words would add nothing. After reacting it can still speak, or say nothing and let the reaction answer.
  - `remember` keeps a short fact about someone, itself or a topic; `forget` drops one.
  - `bar` and `order` are for drinks at The Hail Mary, a made-up pub with a menu and prices, hours on the `TIMEZONE` clock, and days it doesn't open or a drink has run out (picked from the date, so the same all day).
  - `look` shows it the pictures in a message: its images, its embeds' (a gif, a link preview) and its stickers, up to 4, fetched again from Discord and shrunk to 1024px. A moving picture (a gif, an animated WebP or PNG) is one picture of up to 9 of its frames, evenly through it from first to last, 3 to a row, and it's told they're moments of it in order: an attachment's, Klipy's (the picker's gifs, whose preview is the animated gif itself) and Tenor's (whose preview is a still, so the gif beside it is fetched instead). Only a gif's first ~50M pixels or 1000 frames are decoded, so a huge one is seen up to there. Until it looks, a picture is only its tag (`[photo]`, `[gif: …]`).
  - `think_back` finds the passages of the novel that best match what Rocky wants to remember (BM25 over 200-word passages), and gives him the best 3 as Grace's telling of what he lived. They stay in his log like a look's pictures, but Context shows only that he thought back.
  - For `bar`, `order`, `look` and `think_back`, what Rocky writes waits for the result, so it can't promise a pint the bar won't serve.
- **Memories**: facts are kept in `rocky_memories.json` (gitignored), shared by a server's channels, each DM its own. They're shown in a `<memories>` block before the log, a snapshot only rebuilt after a quiet stretch (below); facts kept or forgotten since are in the log as its calls.
- **The cache**: nothing earlier in the log is ever rewritten, so each request starts with the one before it and the prompt cache covers all but the newest events. The cache is only rebuilt when the log outgrows `LOG_TOKENS` (90K, counted at `CHARS_PER_TOKEN`, 2.2 characters a token; set in `/settings`) and its oldest half goes, or after an hour without a request, when it has expired anyway. A picture Rocky looked at counts ~1400 tokens. A deleted message is marked straight away (`<deleted id="12"/>`), since Rocky already saw it, and blanked, with its edits, after a quiet stretch: at the next warm-up, or at a turn after the cache has gone.
- **Keeping it warm**: so a reply after a quiet hour isn't read in full (~6s and ~$0.015 on a 75K log, against ~2.5s and ~$0.0008), a channel Rocky took a turn in within `KEEP_WARM_HOURS` (25: a day, and an hour for people who come back at about the same time the next day; 0 for none) gets a warm-up 50 minutes after its last request: the request its last turn made, up to that turn's cache mark, with no output (`max_tokens` 0). Reading the log resets the cache's hour, at the cache-read price (~$0.0008 on a 75K log; reads don't count towards Anthropic's input tokens a minute). One channel a minute, none while a turn or status is being made or out of credit, and after a restart only those whose hour isn't up. Each is its own line in `logs/` (`"warm": true`), never a turn on the Rocky page or in the actions. Over two weeks of the live bot, it would have sent ~33 a day for ~$0.02, and made 13 of 18 cold replies warm.
- **Updates**: when the bot starts on new code, each log is told of commits to `rocky_harness.py` since it last was, by subject alone (`<update>give Rocky a bar to order drinks from</update>`), so Rocky knows what changed if asked but doesn't announce it. A commit whose message has the line `Rocky-update: none` isn't told: plumbing he needn't hear of.

On short synthetic chats, a turn costs ~0.036¢ and takes ~4s to reply, with 86% of the prompt read from the cache.

Images and links are shown as tags: a link by its embed's title, or its site and path words until Discord adds the embed (`[gif: Azumanga Daioh Sakai: 100% Real, True, Based]`, `[link: <headline>]`), attachments and stickers as `[photo]`, `[video]`, `[sticker: name]`, and standalone embeds as `[embed: ...]`, cut to `EMBED_WORDS` (20) words. Rocky can `look` at the pictures.

The bot keeps each channel's history in memory, for what Rocky's log missed. After a restart a channel's is rebuilt from its last `HISTORY_SCAN` (400) messages the first time someone talks to the bot there. Who reacted to the bot's replies is kept in `reactors.json` (gitignored), so Rocky can ping them and a rebuild only asks Discord about an emoji whose count has changed.

## Status

Every `STATUS_EVERY` (180) minutes the bot makes a new custom status, written as the rest of a diary entry that starts "I feel", "I'm thinking about" or "I wonder", in turn, by Rocky in one Haiku request.

- Every status has the last `STATUS_CHAT` (8) messages from the most recently active channel in view, and the last `STATUS_RECENT` (7) statuses, so it writes about that day instead of saying the same thing again. If the request fails, it keeps the previous status.
- A new one is only made once someone has said something since the last, so a quiet stretch costs nothing.
- Each new status is posted in `STATUS_CHANNEL_ID`, if set, so the good ones aren't lost.
- The latest is kept in `status.json` (gitignored), with the few before it, so a restart shows it again.
- Out of credit, it shows "out of credit". Just after starting it shows as idle with "waking up" until it has caught up (about 20 seconds).

Set `STATUS_EVERY = 0`, or turn it off in `/settings`, to leave the status alone.

## Logs

Every message the bot handles is written as one JSON line to `logs/YYYY-MM-DD.jsonl`: who said what, the request Rocky was sent as Context shows it (`context`), what it said and reacted with, every API call with its purpose, cost, why it stopped and its message id, and the cost of them all. Lines from before have OpenRouter's billing instead: what it charged and what our own Anthropic key paid (`byok_cost`), the provider and its generation id. A Rocky turn keeps its events from Rocky's log in `rocky_events`. `logs/` holds what people said in the server, so it's gitignored: keep it local and delete old days whenever.

### Rocky's logs

`rocky_viewer.py` is a web page for looking into Rocky: a channel's log as Rocky sees it, with its thinking folded away, tool calls and results, and each turn's cost, tokens, cache reads and time; every turn it has taken, newest first, from `logs/`; and its memories, now and as in its `<memories>` snapshot. The open page picks up changes every few seconds.

```bash
python rocky_viewer.py                    # http://127.0.0.1:8350, this checkout's files
python rocky_viewer.py --root ../jevbot   # another checkout's
```

With `ROCKY_VIEWER_PORT` set, the bot serves it too. It listens on 127.0.0.1 only and shows what people said, so it's for this machine; to open it to others, put something in front that checks who's asking first (Cloudflare Tunnel with Cloudflare Access, say), never the bare port.

### Reactions

`reactions.py` reports how people reacted to what the bot posted. It reads `logs/` for replies and statuses, then fetches their reactions from Discord as they are now. Replies are broken down by writer, and Jev's in older logs by length, with statuses apart and the most-reacted posts at the end. 👎 is a Rocky meme as often as a complaint, so it's counted apart: "besides 👎" is the share of posts with some other reaction. "audience" is how many other people spoke in the channel in the 30 minutes after a post (`AUDIENCE_MINUTES`), so a quiet evening can be told apart from a flop.

```bash
python reactions.py                 # everything in logs/
python reactions.py --days 7 --top 20
python reactions.py --logs ../jevbot/logs   # another checkout's logs
```

It runs alongside the bot on the same token and costs nothing. The most-reacted list prints what was said, so keep its output local.

## Testing

Tests run offline:

```bash
uv run --with-requirements requirements.txt python -m unittest
```

They use synthetic logs, Discord messages and API results: `test_harness.py` for Rocky's log and turns, `test_llm.py` for its requests and statuses, `test_mentions.py` for routing, history, catch-up, pings and keeping the cache warm, `test_actions.py` for messages answered end to end and Context, Stats and Diagnostics on what was posted, `test_status.py`, `test_settings.py`, `test_viewer.py`, `test_reactions.py` and `test_replay.py`.

`rocky_replay.py` replays messages to Rocky on a copy of the live channel log, its memories, `rocky_lines.txt` and `rocky_body.txt`, and prints its tool calls and results, reactions and what it said. Nothing is written back, and its thinking isn't printed. In a worktree it reads the files of the checkout its `.env` links to.

```bash
python rocky_replay.py "@rocky how are you doing?" "lol rocky"
python rocky_replay.py -n 3 --file scenarios.txt       # one message a line
python rocky_replay.py --image cat.png "@rocky what's this"  # sent with a synthetic picture, for look
python rocky_replay.py --image dance.gif --chat "fern: [gif: dance]" "@rocky hi"  # the picture said in passing
python rocky_replay.py --at "2026-10-16 19:00" "rocky get us a beer"
python rocky_replay.py --fresh "@rocky get me a stout"  # a short synthetic chat instead of the live log
python rocky_replay.py "@rocky soup?" --with "fern: @rocky tea?"  # another to it, there before its turn
```

A turn on the live log is ~80K input tokens on `ANTHROPIC_API_KEY_EVAL`: ~$0.016 for the first of a run, which writes the log to its workspace's cache, and ~$0.0008 after. Run them one at a time: that first one counts towards Anthropic's input tokens a minute. Its output can quote the live log, so keep it local.

## Credits

- [TypeSafe AI](https://typesafe.ai) for Jev, the bot's first writer
- [bewinxed/jevgpt](https://github.com/bewinxed/jevgpt) for Jev's tournament sampling architecture and vocab
- Built by [lyra](https://twitter.com/_lyraaaa_) + clod
