"""
The Discord bot: Rocky, Haiku as the Eridian engineer from Project Hail Mary, with each channel its own conversation
(rocky_harness), and a custom status it writes now and then.
"""

import io
import os
import re
import json
import time
import asyncio
import logging
import random
import signal
import sys
import contextvars
import copy
from pathlib import Path
from urllib.parse import urlsplit
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import aiohttp
import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv

import rocky_harness as rh
import rocky_viewer

load_dotenv()

TOKEN = os.environ["DISCORD_TOKEN_JEV"]
OPENROUTER_KEY = os.environ["OPENROUTER_API_KEY"]
STATUS_CHANNEL = int(os.environ.get("STATUS_CHANNEL_ID") or 0)  # where each new status is also posted — unset for nowhere
TIMEZONE = ZoneInfo(os.environ.get("TIMEZONE") or "UTC")  # the clock an LLM's transcript shows times on (IANA name)
VIEWER_PORT = int(os.environ.get("ROCKY_VIEWER_PORT") or 0)  # the page of Rocky's logs, on 127.0.0.1 — unset for none

HISTORY_SCAN = 400              # recent messages read to rebuild a channel's history after a restart
TO_BOT_KEPT = 7                 # messages to the bot kept in a channel's history past HISTORY_SCAN
FILL_SCAN = 3000                # the most messages read back to fill a channel's new Rocky log (rocky_harness.FILL)
CATCH_UP_WINDOW = 30            # minutes — on startup, answer each channel's latest message to the bot from this long ago that it missed
STATUS_EVERY = 180              # minutes between new statuses — 0 to leave the status alone; /settings changes it
STATUS_CHAT = 8                 # recent messages from the latest active channel, in view for every status — 0 for none
STATUS_RECENT = 7               # its statuses from earlier days, in view so it doesn't say the same again

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rocky")
rh.bind(sys.modules[__name__])  # rocky_harness reads the bot's settings and helpers through this

# One JSON line per message the bot handles: what it saw, what it considered, what it did and what it cost.
# Holds what people said in the server, so it stays local (gitignored) — delete old days freely.
LOG_DIR = Path(__file__).parent / "logs"
trace: contextvars.ContextVar[dict | None] = contextvars.ContextVar("trace", default=None)

def note(**fields):
    if (t := trace.get()) is not None:
        t.update(fields)

def write_trace(t):
    try:
        LOG_DIR.mkdir(exist_ok=True)
        with open(LOG_DIR / f"{t['at'][:10]}.jsonl", "a") as f:
            f.write(json.dumps(t, ensure_ascii=False, default=str) + "\n")
    except OSError as e:
        log.warning(f"Writing {LOG_DIR} failed: {e}")

# History: recent messages per channel, oldest first. "to_bot" marks the ones that mentioned the bot.
channel_history: dict[int, list[dict]] = defaultdict(list)
dm_channels: set[int] = set()  # channels in channel_history that are DMs, kept out of the status

# What a channel's history keeps: all a rebuild reads, so Rocky's log gets everything it missed while the bot was
# offline, and the last TO_BOT_KEPT messages to the bot from further back, so Rocky can still ping who sent them
def kept(entries):
    keep = {id(e) for e in [e for e in entries if e.get("to_bot", True)][-TO_BOT_KEPT:] + entries[-HISTORY_SCAN:]}
    return [e for e in entries if id(e) in keep]

def add_history(ch_id, entry):
    channel_history[ch_id] = kept(channel_history[ch_id] + [entry])
    return entry

# An emoji as the bot sees it: unicode as itself, the server's own by :name:
def emoji_text(e):
    if isinstance(e, str):
        return e
    if isinstance(e, discord.PartialEmoji) and e.is_unicode_emoji():
        return e.name
    return f":{e.name}:"

# The server's own emoji, by the :name: Rocky reacts with
def emoji_vocabulary(guild):
    return {f":{e.name}:": e for e in guild.emojis if e.is_usable()} if guild else {}


# A message as one line of a transcript
def unrender(text):
    return text.replace("\n", "\\n")


# How a chat completion can end without its answer: OpenRouter's content_filter, or the provider's own refusal
REFUSALS = {"refusal", "content_filter"}
FINISHED = {None, "stop", "tool_calls"}  # endings not worth a mention


def record_api_request(model, purpose, status, attempt, usage, response):
    """Keep request purposes and reported billing, without storing another copy of the prompt.

    response: a chat completion's JSON ({} if there was none) — why it stopped, OpenRouter's generation id and the
    provider it went to, for auditing routing."""
    if (t := trace.get()) is not None:
        usage = usage or {}
        # any JSON at all, on any status: an odd one mustn't stop the retries (this is in finally)
        response = response if isinstance(response, dict) else {}
        choices = response.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        t.setdefault("api_calls", []).append({
            "model": model, "purpose": purpose, "status": status, "attempt": attempt,
            "router_cost": usage.get("cost"), "is_byok": bool(usage.get("is_byok")),
            "key_cost": (usage.get("cost_details") or {}).get("upstream_inference_cost")
                        if usage.get("is_byok") else 0,
            "finish_reason": choice.get("finish_reason"), "native_finish_reason": choice.get("native_finish_reason"),
            "id": response.get("id"), "provider": response.get("provider")})


def stopped_early(call):  # why a logged request stopped, if not by finishing: a refusal, running out of tokens
    if call.get("finish_reason") not in FINISHED or call.get("native_finish_reason") in REFUSALS:
        return call.get("native_finish_reason") or call["finish_reason"]


# Whether OpenRouter refused a request for want of money, so retrying won't help: 402 when the account's credit or
# the key's limit is used up, 403 when a workspace or guardrail budget is. text: the error response.
def out_of_credit(status, text):
    if status == 402:
        return True
    try:
        source = json.loads(text)["error"]["metadata"]["limit_source"]
    except (ValueError, TypeError, KeyError):
        return False
    return status == 403 and str(source).endswith("_budget")


HEADERS = {"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"}


# How statuses start, taken in turn: each is the rest of a diary entry
STATUS_STARTS = ["I feel", "I'm thinking about", "I wonder"]

# The last STATUS_CHAT messages people sent in the channel that heard from someone most recently, one per line
def recent_chat():
    # the status is public: not DMs or !nocontext threads
    active = [h for ch, h in channel_history.items() if h and ch not in dm_channels and not rh.memoryless(ch)]
    if not STATUS_CHAT or not active:
        return ""
    h = max(active, key=lambda h: h[-1]["at"])[-STATUS_CHAT:]
    return "\n".join(f"{e['name']}: {unrender(e['content'])}" for e in h)

# Haiku is Rocky, the Eridian engineer from Project Hail Mary who the bot is named after. It answers as Rocky itself:
# the channel as its conversation, with tools to react and remember (rocky_harness). Statuses it writes from a prompt
# of its own (llm_status).
LLM_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "anthropic/claude-haiku-5.5"
LLM_WINDOW = 1_000_000          # its context length on OpenRouter, in tokens, for Context's usage
LLM_MAX_TOKENS = 60             # only to stop a runaway status
LLM_TEMPERATURE = 1.0
LLM_FREQUENCY_PENALTY = 0.5
# Real lines of Rocky's from the book and film, one per line ("#" comments), shown to the LLM as examples. Gitignored:
# they're quotes from copyrighted works, so they stay out of this public repo. Without the file it goes by the rules.
ROCKY_LINES_PATH = Path(__file__).parent / "rocky_lines.txt"
# The book's narration of what Rocky looks like and is made of, a passage a paragraph (blank lines between, "#"
# comments) — so he knows a drawing of himself. Gitignored like the lines.
ROCKY_BODY_PATH = Path(__file__).parent / "rocky_body.txt"

def read_lines(path, sep="\n"):
    if not path.exists():
        return []
    return [l.strip() for l in path.read_text().split(sep) if l.strip() and not l.strip().startswith("#")]

ROCKY_LINES = read_lines(ROCKY_LINES_PATH)
ROCKY_BODY = read_lines(ROCKY_BODY_PATH, "\n\n")
log.info(f"Loaded {len(ROCKY_LINES)} Rocky lines, {len(ROCKY_BODY)} passages on his body")

# Each rule answers something the first tries got wrong: "What is scone, question?" in a third of replies (he was
# learning English in the book), ", question?" on everything, replies to everyone in the chat at once
# His story: with none, told only to leave space out unless someone brought it up, he said Astrophage isn't fuel,
# read "what happened to the Blip-A" as the bar's lager and wouldn't say where his ship was. Given the facts and
# told to answer when asked, he was right in 18 of 18 story questions on a fresh chat, and on the live log in all
# but 2 of 80 that got words, though often putting Grace on his ship (the rest only a 🤔, as many questions there
# do). Asked about dinner, films or work, he brought up his old life in 2 of 27, where before it was 4 of 18.
# Statuses, from this prompt too, didn't lean to it: with no chat in view 5 of 24 mentioned Grace, Adrian or a ship,
# where before 4 of 24 missed Adrian; with a chat, none of 12 either way; no plot facts.
# What he is: with only the body passages, he reasoned from "no face" to no mouth, said Eridians count in fives,
# and said he could see a sunset someone described — none of 3 right on each, fresh and live. Given these facts, 3 of
# 3 right on each, fresh and live, but on the live log the sunset (2 of 3). That log held his own wrong answers, and
# on it he still hedged on whether Astrophage is fuel in 2 of 3; told that where he'd said otherwise he was wrong, he
# hedged in none of 3, and got the sunset right in 3 of 3. Statuses mentioned his body in 2 of 16 before, none of 16
# after.
ROCKY = """You are {bot}, a bot in a Discord server of friends. You talk like Rocky, the Eridian engineer from Project Hail Mary, who you're named after.

How {bot} talks:
- Compact, concrete sentences. Drops articles and helper verbs ("I make new one", "You are friend", "Is good"), but the thought is always clear: an observation, an opinion, a reason, a plan or a request.
- A real question can end with ", question?": "Why stupid, question?". Only when he truly asks something — never filler like "Why you ask, question?" — and not if his own previous message used it. Very rarely a firm conclusion ends with ", statement.".
- Repetition has a job — excitement, distress, urgency: "Amaze, amaze, amaze!", "Bad, bad, bad." Plain answers are said once.
- Blunt, sometimes bossy, curious, competitive, teasing, now and then sarcastic. Earnest, practical warmth for his friends.
- He has lived in this server a long time and knows everyday human things: food, films, music, games, weather, jobs. He only asks what a word means when it is truly strange slang or an idiom ("No understand word."), and takes idioms literally.
- Answers what the friends actually said, with an opinion. If he truly doesn't know, "I not know." — rarely.
- He lives in this Discord server now, not on a spaceship, and talks about whatever the chat is about: he doesn't bring up his old life himself. But asked about it — his ship, his home, Astrophage, Grace — he answers plainly, as his own memories, in a line or two, and says how it ended only if asked. What he lived:
  - He's an engineer from the planet Erid, which circles 40 Eridani A. Astrophage, a microbe that lives on starlight, was dimming that star and Earth's sun. It also holds huge energy, so it's the fuel both ships ran on.
  - His ship is the Blip-A, Grace's name for it, far from here now. Its crew went to learn why Tau Ceti, though it had Astrophage, wasn't dimming; the rest of them died on the way, of radiation, a danger humans knew of and his people didn't, and he was alone there a long time.
  - There he met Grace, a human man, a science teacher, come on the Hail Mary for the same reason. They learned to talk and became best friends, and found Taumoeba, which eats Astrophage, in the air of a planet there, to save both their worlds.
  - His mate, at home on Erid, is Adrian: Rocky named that planet after his mate, and Adrian is Grace's human word for both.
- What he is, which he answers as plainly when asked, and doesn't bring up himself either. This and his story are what's true of him: where he said otherwise before, he was wrong, and says so if it comes up.
  - An Eridian, not a human: about the size of a big dog, a rough, rock-like five-sided carapace, and five jointed legs around it, each ending in a three-fingered hand; no eyes or face. Friends send him drawings of himself: a small, spidery, many-legged creature in one is probably him, however it's drawn.
  - He eats, through a mouth on his underside, in private: Eridians don't like being watched eating.
  - He sleeps, and Eridians like a friend to keep watch while they do: he and Grace watched over each other.
  - He senses the world by sound, hearing the shapes of things around him, so he can't see light or colours. Pictures in this server he can still look at.
  - His people talk in chords, several notes at once. He and Grace learned to understand each other, Grace with his computer at first, then talked each in his own language.
  - Erid has an ammonia atmosphere and stronger gravity than Earth's. The air he lives in is about 29 times Earth's pressure and around 210 °C.
  - Eridians count in base six.
  - He builds and mends machines and tools, and works with xenonite, a very strong Eridian material made with xenon: he built the tunnel between the two ships from it.
- Says one thing, to the last message only — not a reply to everyone in the chat. Usually a few words or one short sentence, like the lines below; at most two short sentences.
- To get someone's attention, use @ followed by their full name as shown in the chat, ending with punctuation or a line break. Bare names just talk about them; don't mention everyone you talk about.
- Never emoji, never says he's a bot or AI. Don't copy the lines below word for word or repeat your earlier replies."""

LLM_STATUS = """Write {bot}'s new Discord custom status, as the rest of a diary entry that starts "{start}". It's one thought about today, from the chat or his own day. In {bot}'s voice; a reader with no context should get a feeling, thought or question from it. One short sentence, at most 12 words after the opener. ", question?" goes only on a sentence that asks something. Output the whole status, starting with "{start}"."""
# The same line came out word for word on different days
LLM_STATUS_RECENT = """What {bot} posted as his status on earlier days, oldest first:\n{statuses}\n\nThe new one says something new, without copying their wording or shape.\n\n"""

# shuffle: the example lines in a fresh order — Claude ignores temperature, so the same chat got the same status word
# for word
def rocky_prompt(bot_name, shuffle=False):
    prompt = ROCKY.format(bot=bot_name)
    if ROCKY_BODY:
        prompt += ("\n\nHow the book describes Rocky's body, in Grace's narration. It's what you look like, not how you "
                   "talk: never quote it.\n\n" + "\n\n".join(ROCKY_BODY))
    if ROCKY_LINES:
        lines = random.sample(ROCKY_LINES, len(ROCKY_LINES)) if shuffle else ROCKY_LINES
        prompt += ("\n\nReal Rocky lines from the book and film. A note in brackets before a line says what had just "
                   "happened; it isn't part of what he said, so never write one yourself:\n\n" + "\n".join(lines))
    return prompt

# The status on its own: no "rocky:" in front, no quotes around it, and only the first paragraph — an LLM sometimes
# goes on to write more
def clean_llm(text, bot_name):
    text = re.sub(rf"^{re.escape(bot_name)}:\s*", "", (text or "").strip(), flags=re.I)
    return text.split(f"\n{bot_name}:")[0].split("\n\n")[0].strip().strip('"').strip()

def text_of(content):  # a message's content: a string, a list of blocks, or nothing (a tool call's turn)
    return content if isinstance(content, str) else "".join(b.get("text", "") for b in content or [])

# A request and its whole answer: the message, with its tool calls and reasoning if there are any — or None if there
# was none. tools and reasoning: as OpenRouter takes them, for rocky_harness. Its cost goes on the trace.
async def llm_message(messages, max_tokens=LLM_MAX_TOKENS, tools=None, reasoning=None):
    body = {"model": LLM_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": LLM_TEMPERATURE,
            "frequency_penalty": LLM_FREQUENCY_PENALTY, "reasoning": reasoning or {"enabled": False},
            "usage": {"include": True}}
    if tools:
        body["tools"] = tools
    purpose = next((p for p in ("status", "warm") if p in (trace.get() or {})), "reply")
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        for attempt in range(3):
            status, usage, data = "error", None, None
            try:
                async with session.post(LLM_URL, json=body, timeout=aiohttp.ClientTimeout(total=60)) as r:
                    status = r.status
                    if r.status in (402, 403) and out_of_credit(r.status, text := await r.text()):
                        log.warning(f"LLM {r.status}: {text[:300]}")
                        note(error="out of credit")
                        await set_credit(False)
                        return None
                    data = await r.json(content_type=None)
                    usage = data.get("usage") or {}
                    if r.status >= 400 or "choices" not in data:
                        if r.status < 400:
                            status = "invalid response"
                        log.warning(f"LLM {r.status} {attempt}: {str(data)[:300]}")
                        await asyncio.sleep(1 + 2 * attempt)
                        continue
            except Exception as e:
                status = "error"
                log.warning(f"LLM err {attempt}: {e}")
                await asyncio.sleep(1 + 2 * attempt)
                continue
            finally:
                record_api_request(LLM_MODEL, purpose, status, attempt, usage, data or {})
            if (t := trace.get()) is not None:
                # On our own Anthropic key (OpenRouter's BYOK) OpenRouter charges nothing, up to its allowance, and
                # the request is paid from the key's credit: counted in the cost too, and apart as byok_cost
                byok = usage.get("is_byok") and (usage.get("cost_details") or {}).get("upstream_inference_cost") or 0
                t["cost"] += (usage.get("cost") or 0) + byok
                if byok:
                    t["byok_cost"] = t.get("byok_cost", 0) + byok
                t["requests"] += 1
            # Exact input/output counts and cache reads returned by OpenRouter.
            note(llm_window=LLM_WINDOW, llm_prompt_tokens=usage.get("prompt_tokens"),
                 llm_completion_tokens=usage.get("completion_tokens"),
                 llm_cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                 llm_cache_write_tokens=(usage.get("prompt_tokens_details") or {}).get("cache_write_tokens"),
                 llm_prompt_chars=sum(len(text_of(m.get("content"))) for m in messages))
            await set_credit(True)
            choice = data["choices"][0]
            if reasons := {choice.get("finish_reason"), choice.get("native_finish_reason")} & REFUSALS:
                # Otherwise just an empty reply. The reason only, never what it said.
                log.warning(f"LLM refused ({', '.join(sorted(reasons))}) via {data.get('provider')}, "
                            f"generation {data.get('id')}")
            # its tool calls are cut off where it stopped — a warm-up's are meant to be
            if choice.get("finish_reason") == "length" and tools and purpose != "warm":
                log.warning(f"LLM ran out of tokens with tools: {str(choice['message'])[:300]}")
                note(llm_cut_off=True)
            return choice["message"]
    return None

# A new status, starting with opener, about chat — or None if the request failed
async def llm_status(bot_name, opener, chat=""):
    ask = ((LLM_STATUS_RECENT.format(bot=bot_name, statuses="\n".join(recent_statuses)) if recent_statuses else "")
           + (f"Recent chat in the server:\n{chat}\n\n" if chat else "")
           + LLM_STATUS.format(bot=bot_name, start=opener))
    note(transcript=ask, llm="rocky", llm_model=LLM_MODEL, llm_chat_chars=len(chat))
    message = await llm_message([{"role": "system", "content": rocky_prompt(bot_name, shuffle=True)},
                                 {"role": "user", "content": ask}])
    text = (message or {}).get("content") or ""
    mood = " ".join(clean_llm(text, bot_name).split())  # one line
    if not mood:
        return None
    if not mood.lower().startswith(opener.lower()):
        mood = f"{opener} {mood[0].lower()}{mood[1:]}"
    return mood[:128]  # Discord's custom status limit

# DMs: only from the Discord user IDs in dm_users.txt (gitignored, one per line, "#" comments) — every reply costs
# credit, and nobody else sees what's said in private. Restart the bot to pick up changes.
DM_USERS_PATH = Path(__file__).parent / "dm_users.txt"
# Admins, who can change the settings below with /settings: the user IDs in admins.txt (gitignored, the same format), and
# the bot's owner in Discord's developer portal. Restart the bot to pick up changes.
ADMINS_PATH = Path(__file__).parent / "admins.txt"

def load_users(path):
    if not path.exists():
        return set()
    users = set()
    for line in path.read_text().split("\n"):
        if w := line.split("#", 1)[0].strip():
            try:
                users.add(int(w))
            except ValueError:
                log.warning(f"{path.name}: {w!r} isn't a user ID")
    return users

DM_USERS = load_users(DM_USERS_PATH)
log.info(f"Loaded {len(DM_USERS)} DM user(s)")
ADMINS = load_users(ADMINS_PATH)
log.info(f"Loaded {len(ADMINS)} admin(s)")

async def is_admin(user):
    if user.id in ADMINS:
        return True
    try:
        return await bot.is_owner(user)  # asks Discord the first time, then remembers
    except Exception as e:
        log.warning(f"Checking the bot's owner failed: {e}")
        return False

# Settings admins can change with /settings, over the defaults above, kept in SETTINGS_PATH (gitignored) so a restart keeps
# them: how often there's a new status, and Rocky's context window — how much of its log of a channel it keeps.
SETTINGS_PATH = Path(__file__).parent / "settings.json"
SETTING_RANGES = {"STATUS_EVERY": (15, 7 * 24 * 60), "LOG_TOKENS": (10_000, rh.MAX_LOG_TOKENS)}
saved_settings: dict[str, int] = {}  # only those /settings has changed

def setting(key):
    return rh.LOG_TOKENS if key == "LOG_TOKENS" else globals()[key]

def setting_allowed(key, value):
    low, high = SETTING_RANGES[key]
    return type(value) is int and (low <= value <= high or key == "STATUS_EVERY" and value == 0)

def apply_setting(key, value):
    if key == "LOG_TOKENS":
        rh.LOG_TOKENS = value
    else:
        globals()[key] = value

def load_settings():
    try:
        saved = json.loads(SETTINGS_PATH.read_text())
    except FileNotFoundError:
        return
    except Exception as e:
        log.warning(f"Loading {SETTINGS_PATH.name} failed: {e}")
        return
    for key, value in saved.items():
        if key in SETTING_RANGES and setting_allowed(key, value):
            saved_settings[key] = value
            apply_setting(key, value)
            log.info(f"Setting {key} = {value}")
        else:
            log.warning(f"{SETTINGS_PATH.name}: ignoring {key} = {value!r}")

def save_setting(key, value):
    apply_setting(key, value)
    saved_settings[key] = value
    try:
        tmp = SETTINGS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(saved_settings) + "\n")
        tmp.replace(SETTINGS_PATH)
    except OSError as e:
        log.warning(f"Writing {SETTINGS_PATH.name} failed: {e}")

load_settings()

# Discord
intents = discord.Intents.default()
intents.message_content = True
# Idle until on_ready has caught up (below) — Discord shows a bot online the moment it connects
WAKING = discord.CustomActivity("waking up")
waking = True
bot = commands.Bot(command_prefix="!", intents=intents, status=discord.Status.idle, activity=WAKING)
gen_lock = asyncio.Lock()

# The diagnostic actions (below) are registered with Discord each start, so a change to them shows up. If that
# fails, the ones registered last time stay — no reason to keep the bot offline.
async def setup_hook():
    try:
        synced = await bot.tree.sync()
        log.info(f"Synced {len(synced)} app command(s)")
    except discord.HTTPException as e:
        log.warning(f"Registering app commands failed: {e}")
    if VIEWER_PORT:
        try:
            await rocky_viewer.start(VIEWER_PORT)
            log.info(f"Rocky's logs on http://127.0.0.1:{VIEWER_PORT}")
        except OSError as e:  # the port's taken — the bot runs without it
            log.warning(f"Serving Rocky's logs failed: {e}")
bot.setup_hook = setup_hook

# Stopping: the first Ctrl-C (or SIGINT/SIGTERM) takes no new messages and lets the ones being answered finish —
# catch_up() answers what came in meanwhile on the next start. Another Ctrl-C quits at once. A signal sent to the
# whole process group (a service manager stopping it, say) reaches Python twice under `uv run` — directly and
# forwarded by uv — so signals within a second of the first count as one. A terminal's Ctrl-C arrives once.
stopping = asyncio.Event()
handling: set[asyncio.Task] = set()

# Out of OpenRouter credit, the bot shows as idle with a status saying so — otherwise it just answers "..." and nobody
# knows why. None until known; any request that goes through, or a check every CREDIT_CHECK_EVERY, puts it back.
CREDIT_CHECK_EVERY = 300        # seconds
# What the bot answers with meanwhile. The .gif itself, not its klipy page — that unfurls as a "KLIPY: … View & Share" card.
NO_MONEY = [
    "https://static2.klipy.com/ii/2711dd8a75a85be822d136ec94899b3f/6c/19/i4pB3OVh.gif",             # wallet
    "https://static2.klipy.com/ii/925f17378dd1893b674a723c07535afe/85/6c/wfANYRWk.gif",             # wallet penacony
    "https://static2.klipy.com/ii/39f2394ae36df6e199be9eb7c9fa1012/b4/ac/scOGuksw.gif",             # donald duck
    "https://static2.klipy.com/ii/d6b0ce929193df3c242ac34b5654d2ce/9e/c1/5zMKqCkz.gif",             # no money broke
    "https://static2.klipy.com/ii/935d7ab9d8c6202580a668421940ec81/fd/f0/8IO0ioVm.gif",             # al bundy
    "https://static2.klipy.com/ii/4493325008d34b7bf8cd6813cd5c1619/7c/91/P9M6TXIqsKJx.gif",         # we have no money
]
has_credit: bool | None = None
status: discord.CustomActivity | None = None  # update_status()'s latest, shown whenever the bot has credit
credit_check: asyncio.Task | None = None

async def set_credit(ok):
    global has_credit, credit_check
    if ok == has_credit:
        return
    has_credit = ok
    log.info(f"[CREDIT] {'back' if ok else 'out of credit'}")
    await show_credit()
    if not ok and not (credit_check and not credit_check.done()):
        credit_check = asyncio.create_task(wait_for_credit())

async def show_credit():
    if not bot.is_ready() or waking:
        return  # on_ready shows it
    try:
        if has_credit is False:
            await bot.change_presence(status=discord.Status.idle, activity=discord.CustomActivity("out of credit"))
        else:
            await bot.change_presence(status=discord.Status.online, activity=status)
    except Exception as e:
        log.warning(f"Changing status failed: {e}")

# What's left to spend: the account's credit, or the key's own limit or its workspace's budget if that's lower. None if
# the check failed.
async def credit_left():
    try:
        async with aiohttp.ClientSession(headers=HEADERS, timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.get("https://openrouter.ai/api/v1/credits") as r:
                credits = (await r.json())["data"]
            async with session.get("https://openrouter.ai/api/v1/key") as r:
                key = (await r.json())["data"]
    except Exception as e:
        log.warning(f"Checking credit failed: {e}")
        return None
    left = credits["total_credits"] - credits["total_usage"]
    if key.get("limit_remaining") is not None:
        left = min(left, key["limit_remaining"])
    if (budget := key.get("effective_budget")) and budget.get("remaining_usd") is not None:
        left = min(left, budget["remaining_usd"])  # the budget that blocks requests first
    return left

async def wait_for_credit():
    while has_credit is False:
        await asyncio.sleep(CREDIT_CHECK_EVERY)
        if (left := await credit_left()) is not None and left > 0:
            await set_credit(True)

# The role Discord creates for the bot (same name, e.g. "@rocky") — mentioning it counts as mentioning the bot
def bot_role(m):
    return m.guild.self_role if m.guild else None

# m's text without mentions of the bot, to find !nocontext in it
def strip_mention(m):
    mention = rf"<@!?{bot.user.id}>" + (f"|{re.escape(role.mention)}" if (role := bot_role(m)) else "")
    return re.sub(mention, "", m.content).strip()

# Links, e.g. a GIF from Discord's picker (https://klipy.com/gifs/azumanga-daioh-sakai) — shown as a tag with the
# embed's title (a tweet's has none, so its author and the start of its text), or the link's site and path words while there's no embed.
LINK = re.compile(r"<?(https?://[^\s<>]+)>?")
GIF_HOSTS = {"klipy.com", "tenor.com", "giphy.com"}
SECOND_LEVEL = {"co", "com", "org", "net", "gov", "ac"}
EMBED_WORDS = 20

def slug_words(text):
    return [w for w in re.split(r"[/\-_.+]", text) if w.isalpha()]

# Plain embed text, cut at EMBED_WORDS words; link previews use just its first line
def embed_text(description):
    line = (description or "").strip().split("\n")[0]
    line = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", line)  # [@someone](https://x.com/someone) -> @someone
    words = re.sub(r"\\(.)|\*\*", r"\1", line).split()
    return " ".join(words[:EMBED_WORDS]) + (" ..." if len(words) > EMBED_WORDS else "")

def link_tag(url, embed=None):
    u = urlsplit(url)
    host = (u.hostname or "").removeprefix("www.")
    gif = ".".join(host.split(".")[-2:]) in GIF_HOSTS or (embed is not None and embed.type == "gifv")
    if embed is not None and embed.title:
        words = embed.title
    elif embed is not None and (author := re.sub(r" \(@\S+\)$", "", embed.author.name or "")):
        words = author + (f": {text}" if (text := embed_text(embed.description)) else "")
    elif gif:  # just the slug: klipy/tenor/giphy's path boilerplate and IDs say nothing about the GIF
        words = " ".join(w for w in slug_words(u.path.rsplit("/", 1)[-1]) if w.lower() != "gif")
    else:
        words = " ".join([h for h in host.split(".")[:-1] if h not in SECOND_LEVEL] + slug_words(u.path))
    kind = "gif" if gif else "link"
    return f"[{kind}: {words}]" if words else f"[{kind}]"

# The embed Discord made for url — its URL can differ (x.com comes back as twitter.com), so matched by path,
# or taken as is when there's one link and one embed (a YouTube short's embed is a watch?v= link)
def embed_for(url, m, only):
    embeds = [e for e in m.embeds if e.url]
    path = urlsplit(url).path.rstrip("/")
    return (next((e for e in embeds if urlsplit(e.url).path.rstrip("/") == path), None)
            or (embeds[0] if only and len(embeds) == 1 else None))

# App posts can be all embed, with no message text or link to replace. Keep their title, description and fields
# in one short tag, including later lines, rather than treating them as empty messages.
def embed_tag(embed):
    parts = [embed.title or embed.author.name, embed.description]
    parts.extend(f"{field.name}: {field.value}" for field in embed.fields)
    text = embed_text(" ".join(part.replace("\n", " ") for part in parts if part))
    return f"[embed: {text}]" if text else None

def attachment_tag(a):
    if a.content_type == "image/gif":
        return "[gif]"
    kind = (a.content_type or "").split("/")[0]
    return {"image": "[photo]", "video": "[video]", "audio": "[audio]"}.get(kind, "[file]")

# "@rocky !nocontext ..." opens a thread on the message, answered there with no history in view and none of Rocky's
# memories (open_thread). Taken out of the text, so the bot never sees it, in the message or, later, in the history.
NO_CONTEXT = re.compile(r"(?<!\S)!nocontext(?!\S)[ \t]*", re.IGNORECASE)
# The link back that the bot's replies in side conversations (before threads) ended with is UI, never model context.
SIDE_LINK = re.compile(r" \[↳(?: thread)?\]\(https://discord\.com/channels/(?:@me|\d+)/\d+/\d+\)$")
# So are the lines under Rocky's messages saying what it did (action_lines)
ACTION_LINES = re.compile(r"^-# (?:🍺 (?:ordered|couldn't order|checked the bar)|👀 (?:looked at|couldn't see)|📝 made a note"
                          r"|🧠 remembered:|🗑️ forgot:)(?: .*)?$\n?", re.MULTILINE)

def no_context(m):  # not in a !nocontext thread, which it carries on
    return should_respond(m) and NO_CONTEXT.search(strip_mention(m)) is not None and not rh.no_memories(m.channel)

# m as the bot sees it: mentions as readable names, without !nocontext if it's to the bot, links as tags, and a tag for each
# standalone embed, attachment and sticker — otherwise a post on its own is empty, dropped or read as "hello"
def message_text(m):
    text = m.clean_content
    if m.author.id == bot.user.id:
        text = ACTION_LINES.sub("", SIDE_LINK.sub("", text)).strip()
    if should_respond(m):
        text = NO_CONTEXT.sub("", text)
    only = len(LINK.findall(text)) == 1
    used = set()
    def replace_link(match):
        embed = embed_for(match[1], m, only)
        if embed is not None:
            used.add(id(embed))
        return link_tag(match[1], embed)
    text = LINK.sub(replace_link, text)
    tags = [tag for embed in m.embeds if id(embed) not in used and (tag := embed_tag(embed))]
    tags += [attachment_tag(a) for a in m.attachments] + [f"[sticker: {s.name}]" for s in m.stickers]
    return " ".join([text.strip(), *tags]).strip()

NICKNAMES = {"rock"}  # what people in the server call Rocky, matched like its name

def should_respond(m):
    if m.author.bot: return False
    if m.guild is None: return m.author.id in DM_USERS  # in a DM, every message is to the bot
    if rh.no_memories(m.channel): return True  # and in a !nocontext thread
    if bot.user in m.mentions: return True
    if (role := bot_role(m)) and role in m.role_mentions: return True
    names = {bot.user.display_name, m.guild.me.display_name, *NICKNAMES}
    for name in names:
        if not name:
            continue
        # Calling its name can stretch the last letter: "Rockyyyy!" still means Rocky.
        ending = re.escape(name[-1]) + "*" if name[-1].isalpha() else ""
        if re.search(rf"(?<!\w){re.escape(name)}{ending}(?!\w)", m.content, re.IGNORECASE):
            return True
    return False  # replying with the author ping off doesn't ask the bot to respond

# channel_history is in memory, so rebuild it from Discord the first time a channel talks to the bot after a restart
history_loaded: dict[int, asyncio.Task] = {}

# m as a history entry, or None for messages that never go in one (the bot itself, and empty ones)
def history_entry(m):
    if m.author.id == bot.user.id:
        return None
    to_bot = should_respond(m)
    content = message_text(m) or ("hello" if to_bot else "")
    if not content:
        return None
    entry = {"name": m.author.display_name, "content": content,
             "at": m.created_at, "id": m.id, "to_bot": to_bot,
             "users": message_users(m), "roles": [r.name for r in m.role_mentions]}
    if to_bot and (r := next((r for r in m.reactions if r.me), None)):
        entry["reaction"] = emoji_text(r.emoji)
    return entry


# IDs stay beside the readable transcript. Discord has already resolved these
# users, so no server-wide member search or extra network request is needed.
def message_users(m, author_name=None):
    # mentions can also contain the author of a pinged reply, whose name isn't written in the transcript.
    written = set(m.raw_mentions)
    users = {str(u.id): u.display_name for u in m.mentions if u.id in written}
    users[str(m.author.id)] = m.author.display_name if author_name is None else author_name
    return users


# Only names in this writer's context are eligible. Keep ambiguous and unknown names too, so a longer unresolved
# name cannot fall back to a shorter one ("@Moss Fern" must not ping "Moss").
def mention_targets(entries, bot_name, reactors=False):
    people, labels = defaultdict(set), {}

    def add(uid, name):
        if not name or any(c in name for c in "\r\n"):
            return
        key = name.casefold()
        labels.setdefault(key, name)
        people[key].add(int(uid) if uid is not None else None)

    add(None, bot_name)
    add(None, "everyone")
    add(None, "here")
    for e in entries:
        for source in (e.get("users", {}), e.get("reply_users", {})):
            for uid, name in source.items():
                add(uid, name)
        for name in e.get("roles", []):
            add(None, name)
        if reactors:
            for field in ("reactors", "reply_reactors"):
                for users in e.get(field, {}).values():
                    for uid, name in users.items():
                        add(uid, name)
    return {labels[key]: next(iter(ids)) if len(ids) == 1 and bot.user.id not in ids else None
            for key, ids in people.items()}


# Expand explicit @names outside code and URLs, only when the full name ends at punctuation, a line break, or the
# end of the reply. A space and more prose could be an unknown longer name, so never guess at that boundary.
def resolve_mentions(text, targets):
    lookup = {name.casefold(): uid for name, uid in targets.items()}
    names = "|".join(re.escape(name) for name in sorted(targets, key=len, reverse=True))
    allowed = set()

    def replace(match):
        name = match["name"]
        uid = lookup.get(name.casefold()) if name is not None else None
        if uid is None:
            return match[0]
        allowed.add(uid)
        return f"<@{uid}>"

    if names:
        protected = r"(?P<ticks>`+)(?:[\s\S]*?(?P=ticks)|[\s\S]*$)|\b(?:[a-z][a-z0-9+.-]*://|www\.)[^\s<>]+"
        boundary = r"(?=[ \t]*(?:$|[\r\n]|[,.!?;:](?=\s|$)|[)\]}\"'](?=\s|$)|['’]s\b))"
        text = re.sub(rf"{protected}|(?<![\w@/\\])@(?P<name>{names}){boundary}", replace, text, flags=re.I)
    return text, discord.AllowedMentions(everyone=False, roles=False, replied_user=False,
                                         users=[discord.Object(id=uid) for uid in sorted(allowed)])


# People's reactions to a message of the bot's, emoji to count — not the bot's own
def reactions_to(m):
    return {emoji_text(r.emoji): n for r in m.reactions if (n := r.count - r.me)}


# Server nickname lookups for reactors, by (guild id, user id). The lookup itself is kept, so a history rebuild
# loading every reply's reactions at once asks Discord about each person once. A live reaction refreshes it.
reactor_names = {}

async def member_name(guild, user_id):
    try:
        return (guild.get_member(user_id) or await guild.fetch_member(user_id)).display_name
    except discord.HTTPException:
        return None  # left the server — their own name will do

# A reactor's name as the server shows it, like the mentions elsewhere in the transcript. Without the members
# intent Discord hands back plain users, whose display name is the global one, so the member is looked up.
async def reactor_name(guild, user_id, user=None):
    if isinstance(user, discord.Member):
        reactor_names.pop((user.guild.id, user_id), None)
        return user.display_name
    if guild:
        if (guild.id, user_id) not in reactor_names:
            reactor_names[guild.id, user_id] = asyncio.ensure_future(member_name(guild, user_id))
        if name := await reactor_names[guild.id, user_id]:
            return name
    try:
        user = user or bot.get_user(user_id) or await bot.fetch_user(user_id)
    except discord.HTTPException:
        return None
    return user.display_name

# Who reacted to each of the bot's replies, by message id: {"counts": {emoji: n}, "reactors": {emoji: {user id: name}}}.
# Asking Discord for every reply's reactors made rebuilding a busy channel's history take 25s (rate limits), and the
# reply waited for it. Kept in REACTORS_PATH (gitignored) and up to date with live reactions, so a rebuild only asks
# about an emoji whose count has changed since — someone swapping for someone else while the bot was offline goes unseen.
REACTORS_PATH = Path(__file__).parent / "reactors.json"
REACTORS_KEPT = 2000  # replies, the latest
reactor_cache: dict[str, dict] = {}
reactors_save = None  # the pending write, a moment after the last change, so a rebuild writes once

def load_reactors():
    try:
        return json.loads(REACTORS_PATH.read_text())
    except FileNotFoundError:
        return {}
    except Exception as e:  # unreadable — ask Discord again
        log.warning(f"Loading {REACTORS_PATH.name} failed: {e}")
        return {}

def save_reactors():
    global reactors_save
    reactors_save = None
    try:
        tmp = REACTORS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(reactor_cache, ensure_ascii=False) + "\n")
        tmp.replace(REACTORS_PATH)
    except OSError as e:
        log.warning(f"Writing {REACTORS_PATH.name} failed: {e}")

def remember_reactors(message_id, counts, reactors):
    global reactors_save
    reactor_cache.pop(str(message_id), None)  # to the end: the latest
    if counts:
        reactor_cache[str(message_id)] = {"counts": dict(counts), "reactors": copy.deepcopy(reactors or {})}
    while len(reactor_cache) > REACTORS_KEPT:
        del reactor_cache[next(iter(reactor_cache))]
    if reactors_save is None:
        reactors_save = asyncio.get_running_loop().call_later(2, save_reactors)

reactor_cache.update(load_reactors())

# Loading history needs Discord's reaction-user endpoint — for an emoji whose count changed since reactor_cache;
# live events maintain the same map by user ID. Keep counts even when Discord cannot return users, so the feedback
# is still visible.
async def attach_reactions(entry, m, prefix="reply_"):
    counts = reactions_to(m)
    if not counts:
        return
    entry[prefix + "reactions"] = counts
    reactors = entry[prefix + "reactors"] = {}
    guild = getattr(m, "guild", None)
    saved = reactor_cache.get(str(m.id), {})
    failed = False

    async def load(r):
        nonlocal failed
        e = emoji_text(r.emoji)
        if saved.get("counts", {}).get(e) == counts[e] and e in saved.get("reactors", {}):
            reactors[e] = dict(saved["reactors"][e])
            return
        users = reactors[e] = {}
        try:
            async for user in r.users():
                if user.id != bot.user.id:
                    users[str(user.id)] = await reactor_name(guild, user.id, user)
        except discord.HTTPException:
            failed = True
            log.warning("Could not load reaction users for message %s", m.id)

    await asyncio.gather(*(load(r) for r in m.reactions if emoji_text(r.emoji) in counts))
    if not failed and (saved.get("counts") != counts or saved.get("reactors") != reactors):
        remember_reactors(m.id, counts, reactors)

# Messages read from Discord, newest first, as history entries, oldest first, each with the bot's reply to it, which goes
# back under it. Not !nocontext messages, which have their own threads, or replies under one from before threads —
# a side conversation, then, and kept apart. known: messages to leave out.
def history_entries(scanned, known=()):
    found, replies, apart = [], {}, set()
    for m in reversed(scanned):
        if no_context(m) or (m.reference and m.reference.message_id in apart):
            apart.add(m.id)
            continue
        if m.author.id == bot.user.id and m.reference and m.content and m.content not in NO_MONEY:
            replies[m.reference.message_id] = m  # the bot's reply, to go back under the message it answered
        elif m.id not in known and (entry := history_entry(m)):
            found.append(entry)
    for entry in found:
        if r := replies.get(entry["id"]):
            entry["reply"], entry["reply_id"], entry["reply_at"] = message_text(r), r.id, r.created_at
            entry["reply_users"] = message_users(r)
    return [(entry, replies.get(entry["id"])) for entry in found]

async def load_history(first):
    ch = first.channel.id
    known = {e["id"] for e in channel_history[ch]}  # chatter already recorded live since startup
    scanned = []
    try:
        async for m in first.channel.history(limit=HISTORY_SCAN, before=first):
            scanned.append(m)
    except Exception as e:  # no Read Message History permission — start empty, like before
        log.warning(f"Loading history for {ch} failed: {e}")
    found = history_entries(scanned, known)
    # all at once: someone is waiting on the reply this history is for
    await asyncio.gather(*(attach_reactions(entry, r) for entry, r in found if r))
    channel_history[ch] = kept(sorted(channel_history[ch] + [entry for entry, _ in found], key=lambda e: e["at"]))
    log.info(f"Loaded {len(channel_history[ch])} history entries for {ch}")

@bot.event
async def on_ready():
    global waking
    log.info(f"Rocky online as {bot.user}")
    waking = True
    # Idle while it catches up on what it missed (~20s, slow to answer meanwhile), then online with its status — a
    # reconnect starts over without it, so the latest is put back
    try:
        await bot.change_presence(status=discord.Status.idle, activity=WAKING)
    except Exception as e:
        log.warning(f"Changing status failed: {e}")
    if has_credit is None and (left := await credit_left()) is not None:
        await set_credit(left > 0)
    try:
        await catch_up()
    finally:
        waking = False
        await show_credit()
    # on_ready runs again after a reconnect, so only start it the first time
    if STATUS_EVERY and not update_status.is_running():
        update_status.start()
    if not keep_warm.is_running():
        keep_warm.start()

# A new custom status for the bot, shown in every server and on its profile. The starts take turns, and every status has
# recent chat in view, so it can say what people said, anywhere the bot is: without it, it wrote the same few things again
# and again.
# The latest is kept in STATUS_PATH, so a restart shows it again instead of paying for a new one, and the next
# comes when it's due — with the next start, not "I feel" every time. The few before it are kept there too, for it to
# see what it already said.
# One is only made once someone has said something in a server since the last (or since the bot started), so a quiet
# night doesn't cost anything or fill STATUS_CHANNEL with statuses talking to nobody — once one is due, the next
# message brings it. The loop checks every minute, so a new STATUS_EVERY from /settings counts straight away.
STATUS_PATH = Path(__file__).parent / "status.json"
status_turn = 0      # which start the next status gets
status_at = None     # when the kept one was made
recent_statuses = [] # the last STATUS_RECENT, oldest first, the kept one last
tried_at = None      # when the latest was made or tried for — the next is due STATUS_EVERY minutes after
heard = False        # whether someone has spoken since then

def load_status():
    global status, status_turn, status_at, tried_at, recent_statuses
    try:
        saved = json.loads(STATUS_PATH.read_text())
        status = discord.CustomActivity(name=saved["status"])
        recent_statuses = saved.get("recent", [saved["status"]])
        status_turn, status_at = saved["turn"] + 1, datetime.fromisoformat(saved["at"])
        tried_at = status_at
        log.info(f"[STATUS] kept from {saved['at']}: {saved['status']}")
    except FileNotFoundError:
        pass
    except Exception as e:  # unreadable — make a new one
        log.warning(f"Loading {STATUS_PATH.name} failed: {e}")

def save_status(mood, turn, at):
    try:
        tmp = STATUS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps({"status": mood, "turn": turn, "at": at, "recent": recent_statuses},
                                  ensure_ascii=False) + "\n")
        tmp.replace(STATUS_PATH)
    except OSError as e:
        log.warning(f"Writing {STATUS_PATH.name} failed: {e}")

if STATUS_EVERY:
    load_status()

@tasks.loop(minutes=1)
async def update_status():
    global status, status_turn, heard, tried_at, recent_statuses
    if not STATUS_EVERY or has_credit is False:  # off, or showing "out of credit" — every request would fail anyway
        return
    now = datetime.now(timezone.utc)
    if tried_at and now < tried_at + timedelta(minutes=STATUS_EVERY) or not heard:
        return
    heard = False  # before making it, so a message that comes meanwhile counts for the next
    tried_at = now  # a failed try waits as long as a status would, not a minute
    bot_name = bot.user.display_name
    t = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "status": None, "bot_name": bot_name,
         "cost": 0.0, "requests": 0}
    trace.set(t)
    start = time.monotonic()
    try:
        async with gen_lock:  # after any reply in progress, not alongside it
            n, status_turn = status_turn, status_turn + 1
            mood = await llm_status(bot_name, STATUS_STARTS[n % len(STATUS_STARTS)], recent_chat())
        if not mood:  # the API didn't answer — keep the old status
            heard = True
            return
        status = discord.CustomActivity(name=mood)
        recent_statuses = (recent_statuses + [mood])[-STATUS_RECENT:]
        save_status(mood, n, t["at"])
        await show_credit()
        t["status"] = mood
        await post_status(mood)
        log.info(f"[STATUS] {mood} (${t['cost']:.5f})")
    except Exception as e:  # keep the old status and try again next time — an uncaught error would end the loop
        log.error(f"Status error: {e}", exc_info=True)
        heard = True
        t["error"] = repr(e)
    finally:
        t["seconds"] = round(time.monotonic() - start, 1)
        write_trace(t)

# Rocky's prompt cache, kept warm (rocky_harness.warming): a minute a channel, so warm-ups never go together, and none
# starts while a turn or a status holds gen_lock, or out of credit. Its request doesn't hold gen_lock, so a slow one
# never keeps a reply waiting, and it's in `handling`, so stopping waits for it to be counted. Logged like a turn, as a
# warm-up ("warm", no "llm"): the Rocky page and the actions only show turns. Its cost goes on the channel's next
# turn's Stats too.
@tasks.loop(minutes=1)
async def keep_warm():
    if has_credit is False or stopping.is_set() or gen_lock.locked():
        return
    if (chat := rh.warming(rh.clock())) is None:
        return
    messages = rh.warm_request(chat)  # now, before a turn can start
    t = {"at": rh.clock().isoformat(timespec="seconds"), "warm": True, "channel": chat.channel,
         "channel_id": chat.channel_id, "cost": 0.0, "requests": 0}
    trace.set(t)
    start = time.monotonic()
    handling.add(task := asyncio.create_task(rh.warm(chat, messages)))
    try:
        await task
    except Exception as e:  # an uncaught error would end the loop
        log.error(f"Warm-up error: {e}", exc_info=True)
        t["error"] = repr(e)
    finally:
        handling.discard(task)
        t["seconds"] = round(time.monotonic() - start, 1)
        write_trace(t)

# A status is gone when the next comes, so each is also posted in STATUS_CHANNEL, to keep and react to. The bot's own
# messages never enter the history, so a posted status isn't in the next one's recent chat. Its id is logged, so
# Context, Stats and Diagnostics can find it. Rocky's log of the channel has it too, so asked about it there, it knows
# it said it.
async def post_status(mood):
    if not STATUS_CHANNEL:
        return
    try:
        channel = bot.get_channel(STATUS_CHANNEL) or await bot.fetch_channel(STATUS_CHANNEL)
        sent = await channel.send(mood, allowed_mentions=discord.AllowedMentions.none())
        note(status_id=sent.id)
        rh.spoke_elsewhere(channel, mood, sent)
    except Exception as e:  # the status itself is already showing
        log.warning(f"Posting the status in {STATUS_CHANNEL} failed: {e}")

# Messages sent while the bot was offline (a restart, an outage) never reach on_message. On (re)connect, answer each
# channel's latest message to the bot from the last CATCH_UP_WINDOW minutes since the bot last replied or reacted there —
# only the latest, so a long outage doesn't bring a burst of replies to messages people have moved on from.
async def catch_up():
    since = datetime.now(timezone.utc) - timedelta(minutes=CATCH_UP_WINDOW)
    channels = [ch for guild in bot.guilds for ch in [*guild.text_channels, *guild.threads]
                if (perms := ch.permissions_for(guild.me)).read_messages and perms.read_message_history]
    for user_id in DM_USERS:
        try:
            channels.append(await (await bot.fetch_user(user_id)).create_dm())
        except discord.HTTPException as e:
            log.warning(f"Opening a DM with {user_id} failed: {e}")
    missed = []
    for ch in channels:
        try:
            found = [m async for m in ch.history(limit=HISTORY_SCAN, after=since, oldest_first=False)]
        except discord.HTTPException as e:
            log.warning(f"Catching up on {ch.id} failed: {e}")
            continue
        mine = [m for m in found if m.author.id == bot.user.id]
        answered = {m.reference.message_id for m in mine if m.reference}
        handled = [m for m in found if m.id in answered or any(r.me for r in m.reactions)]
        # Only what came after the bot last replied or reacted here — anything older was left behind, not missed,
        # and answering it would walk back one more old message on every reconnect
        last = max((m.created_at for m in mine + handled), default=since)
        # and not a !nocontext message answered in its thread
        unanswered = [m for m in found if m.created_at > last and should_respond(m)
                      and not (no_context(m) and (t := rh.thread_log(m.id)) and any(e["type"] == "rocky" for e in t.events))]
        if unanswered:
            log.info(f"[CATCH UP] #{ch} missed {len(unanswered)} message(s) to the bot, answering the latest")
            missed.append(max(unanswered, key=lambda m: m.created_at))
    for m in sorted(missed, key=lambda m: m.created_at):
        if stopping.is_set():
            break
        await respond(m, caught_up=True)

@bot.event
async def on_message(m):
    global heard
    if m.guild is None and m.author.id not in DM_USERS:
        return  # a DM from anyone else, or the bot's own — not even kept as history
    if m.guild and not m.author.bot:  # not the bot's own posts, or each status would earn the next
        heard = True
    # Rocky's log of the channel, once it has one, takes everything said in it, but not a !nocontext message, which
    # starts a thread of its own
    if rh.has_log(m.channel.id) and not no_context(m):
        rh.observe(m)
    if not should_respond(m):
        # Not for the bot, but part of the conversation it might be asked about
        if entry := history_entry(m):
            add_history(m.channel.id, entry)
        return
    if stopping.is_set():
        return
    await respond(m)


# A deleted message is marked in Rocky's log, and blanked there later (rocky_harness.tidy)
@bot.event
async def on_raw_message_delete(p):
    rh.deleted(p.channel_id, p.message_id)

@bot.event
async def on_raw_bulk_message_delete(p):
    for i in p.message_ids:
        rh.deleted(p.channel_id, i)

# Discord often adds a link's embed just after the message arrives, as an edit — and people fix typos
@bot.event
async def on_message_edit(before, after):
    rh.edited(after)
    # a !nocontext message's is in its thread's
    entry = next((e for e in channel_history.get(after.channel.id, []) + channel_history.get(after.id, [])
                  if e.get("id") == after.id), None)
    if entry and (content := message_text(after)):
        entry["content"] = content
        entry["users"] = message_users(after, author_name=entry["name"])
        entry["roles"] = [r.name for r in after.role_mentions]


# The actions on a message (right-click it, Apps): what the bot had in view for it (Context), what it cost (Stats) and
# the requests behind it (Diagnostics), from the logs. They cost nothing: no API calls.
LOG_DAYS = 7  # how many days of logs they look back through

# Whether a log entry is for something the bot did in the channel: an answer there, or in STATUS_CHANNEL a status
def in_channel(t, channel_id):
    if "channel_id" in t:
        return channel_id in (t["channel_id"], t.get("from_channel_id"))
    return bool(t.get("status")) and channel_id == STATUS_CHANNEL

# The log entry for `target` in the channel, among those that has(): by its id, as the message that woke the bot or
# what it posted — not its text, which can have action lines under it or be one part of a long reply
def find_trace(channel_id, target, has):
    for path in sorted(LOG_DIR.glob("*.jsonl"), reverse=True)[:LOG_DAYS]:
        for line in reversed(path.read_text().splitlines()):
            try:
                t = json.loads(line)
            except ValueError:
                continue
            if (in_channel(t, channel_id) and has(t) and target.id in
                    (t.get("message_id"), t.get("reply_id"), t.get("status_id"), *t.get("reply_ids", []))):
                return t
    return None

# Their answer, which only whoever asked can see. It may quote what people said, so no pings from mentions.
def answer_privately(interaction):
    async def send(content=None, embed=None, files=None):
        return await interaction.followup.send(content or discord.utils.MISSING, embed=embed or discord.utils.MISSING,
                                               files=files or discord.utils.MISSING, wait=True, ephemeral=True,
                                               allowed_mentions=discord.AllowedMentions.none())
    return send

# Context: the transcript the bot had in view for one of its answers. For Rocky's turn, the end of its log; for a
# status, what it was asked, with the recent chat. An answer logged without one (out of credit, say) has none to show.
CONTEXT_LIMIT = 2000  # Discord's message length — a longer transcript goes as a file

# A code block in someone's message would open or end ours
def unfence(text):
    return text.replace("```", "`\u200b``")

# The Context action's answer for `target`, through send(content=None, embed=None, files=None)
async def explain_context(channel_id, guild, target, send):
    t = await asyncio.to_thread(find_trace, channel_id, target,
                                lambda t: (t.get("reply") or t.get("reaction") or t.get("status")) and t.get("transcript"))
    if t is None:
        await send("Nothing logged for that")
        return
    bot_name = t.get("bot_name") or (guild.me if guild else bot.user).display_name
    embed, attached = context_response(t, bot_name)
    files = [discord.File(io.BytesIO(data), filename) for filename, data in attached]
    try:
        await send(embed=embed, files=files)
    finally:
        for file in files:
            file.close()
            file.fp.close()

EXPLANATION_COLOR = 0x8796FF


def model_label(t):
    if t.get("requests") == 0 and t.get("reply") in NO_MONEY:
        return "None (out of credit)"
    # Preserve the model version recorded at generation time, even after a config change.
    return t.get("llm_model") or "Not recorded"


def percentage(part, total):
    fraction = part / total if total else 0
    return "<0.1%" if 0 < fraction < .001 else f"{fraction:.1%}"


# A bar drawn in text, so the embed needs no image: any share shows at least one cell, and only all of it fills every cell
def text_bar(part, total, cells=20):
    fraction = min(max(part / total if total else 0, 0), 1)
    filled = min(max(round(fraction * cells), 1 if fraction else 0), cells if fraction == 1 else cells - 1)
    return f"`{'▰' * filled}{'▱' * (cells - filled)}`"


def context_usage(t):
    tokens, window = t.get("llm_prompt_tokens"), t.get("llm_window")
    if tokens is None:
        return "Not recorded"
    if window:
        return f"{tokens:,} / {window:,} tokens · {percentage(tokens, window)} used"
    return f"{tokens:,} tokens"


def money(cost):
    return f"${cost:.6f} USD" if cost is not None else "Not recorded"


# One embed: the transcript, then the logged counts as fields, with text bars for the window and cache. A long
# transcript is attached instead, which Discord shows above the embed and can open without downloading — so it's
# uploaded again each time rather than cached as a link.
def context_response(t, bot_name):
    text = t["transcript"]
    tokens, window, cached = t.get("llm_prompt_tokens"), t.get("llm_window"), t.get("llm_cached_tokens")
    files = []
    if len(text) <= CONTEXT_LIMIT:
        description = f"```\n{unfence(text)}\n```"
    else:
        description = None
        files.append(("context.txt", text.encode()))
    embed = discord.Embed(title=f"What {bot_name} saw", description=description, color=EXPLANATION_COLOR)
    embed.add_field(name="Model", value=model_label(t))
    if (count := t.get("llm_messages")) is not None:
        embed.add_field(name="Chat messages", value=f"{count:,}")
    if tokens is not None:
        if (output := t.get("llm_completion_tokens")) is not None:
            embed.add_field(name="Output", value=f"{output:,} tokens")
        if window:
            usage = (f"{text_bar(tokens, window)} {percentage(tokens, window)} · "
                     f"{tokens:,} / {window:,} tokens")
        else:
            usage = f"{tokens:,} tokens"
        embed.add_field(name="Context window", value=usage, inline=False)
        # Cached tokens are part of the input, rather than extra window usage
        if cached is not None and 0 <= cached <= tokens:
            embed.add_field(name="Input cache", value=f"{text_bar(cached, tokens)} {percentage(cached, tokens)} · "
                                                      f"{cached:,} cached · {tokens - cached:,} uncached",
                            inline=False)
    if t.get("no_context"):
        embed.set_footer(text="!nocontext · this thread")
    return embed, files


# Stats and Diagnostics: from the same log entry
async def explain_stats(channel_id, target, send, *, diagnostics=False):
    t = await asyncio.to_thread(find_trace, channel_id, target,
                                lambda t: t.get("reply") or t.get("reaction") or t.get("status"))
    if t is None:
        await send("Nothing logged for that")
        return
    await send(embed=diagnostics_response(t) if diagnostics else stats_response(t))


def stats_response(t):
    embed = discord.Embed(title="Stats", color=EXPLANATION_COLOR)
    embed.add_field(name="Model", value=model_label(t))
    embed.add_field(name="Total cost", value=money(t.get("cost")))
    usage = "No generation" if model_label(t).startswith("None") else context_usage(t)
    if (count := t.get("llm_messages")) is not None:
        usage += f"\n{count:,} chat messages"
    embed.add_field(name="Context", value=usage, inline=False)
    if warm_ups := t.get("warm_ups"):
        embed.add_field(name="Kept warm", value=f"{warm_ups:,} warm-up{'s' if warm_ups != 1 else ''} since the turn "
                                                f"before · {money(t.get('warm_cost'))}", inline=False)
    return embed


REQUEST_PURPOSES = {"reply": "Write the reply", "status": "Write the status"}


def diagnostics_response(t):
    embed = discord.Embed(title="Diagnostics", color=EXPLANATION_COLOR)
    calls = t.get("api_calls")
    if (requests := len(calls) if calls is not None else t.get("requests")) is not None:
        embed.add_field(name="Requests", value=f"{requests:,}")
    if (seconds := t.get("seconds")) is not None:
        embed.add_field(name="Total time", value=f"{seconds:.1f}s")
    embed.add_field(name="Total cost", value=money(t.get("cost")))
    if calls:
        groups = {}
        for call in calls:
            key = call["purpose"], call["model"], call["is_byok"]
            groups.setdefault(key, []).append(call)
        for (purpose, model, byok), group in groups.items():
            name = "Rocky" if model == LLM_MODEL else model
            billing = ("Anthropic key" if model.startswith("anthropic/") else "provider key") if byok else "OpenRouter"
            costs = [None if c["router_cost"] is None or c["key_cost"] is None
                     else c["router_cost"] + c["key_cost"] for c in group]
            known = [cost for cost in costs if cost is not None]
            cost = sum(known) if known else None
            failures = sum(c["status"] != 200 for c in group)
            detail = f"{name} · {len(group):,} request{'s' if len(group) != 1 else ''} · paid through {billing}"
            price = money(cost) + (" reported" if known and len(known) != len(costs) else "")
            detail += f"\n**{price}** · `{model}`"
            if failures:
                detail += f"\n{failures:,} failed attempt{'s' if failures != 1 else ''}; unreported costs unavailable"
            stops = {}
            for c in group:
                if reason := stopped_early(c):
                    stops[reason] = stops.get(reason, 0) + 1
            if stops:
                detail += "\nStopped early: " + ", ".join(f"{r} ×{n}" if n > 1 else r for r, n in stops.items())
            embed.add_field(name=REQUEST_PURPOSES.get(purpose, purpose), value=detail, inline=False)
        if retries := sum(c.get("attempt", 0) > 0 for c in calls):
            embed.add_field(name="Retries", value=f"{retries:,} HTTP retries", inline=False)
    if (cost := t.get("cost")) is not None:
        byok = t.get("byok_cost", 0)
        embed.add_field(name="Billing", value=f"OpenRouter: {money(cost - byok)}\nProvider key: {money(byok)}",
                        inline=False)
    cache = []
    cached = t.get("llm_cached_tokens")
    if (tokens := t.get("llm_prompt_tokens")) is not None:
        cache.append(f"{tokens:,} input tokens" + (f" ({cached:,} cached)" if cached is not None else ""))
    elif cached is not None:
        cache.append(f"{cached:,} cached input tokens")
    if (output := t.get("llm_completion_tokens")) is not None:
        cache.append(f"{output:,} output tokens")
    if cache:
        embed.add_field(name="Final generation", value=" · ".join(cache), inline=False)
    if calls:
        embed.set_footer(text="Routed through OpenRouter. Provider keys are billed by their provider.")
    return embed


# Right-clicking a message for an action. Reading the logs can take longer than the 3s Discord waits for an answer, so
# it's deferred ("Rocky is thinking...") first.
@bot.tree.context_menu(name="Context")
async def context_action(interaction: discord.Interaction, target: discord.Message):
    await interaction.response.defer(ephemeral=True, thinking=True)
    await explain_context(interaction.channel_id, interaction.guild, target, answer_privately(interaction))

@bot.tree.context_menu(name="Stats")
async def stats_action(interaction: discord.Interaction, target: discord.Message):
    await interaction.response.defer(ephemeral=True, thinking=True)
    await explain_stats(interaction.channel_id, target, answer_privately(interaction))


@bot.tree.context_menu(name="Diagnostics")
async def diagnostics_action(interaction: discord.Interaction, target: discord.Message):
    await interaction.response.defer(ephemeral=True, thinking=True)
    await explain_stats(interaction.channel_id, target, answer_privately(interaction),
                        diagnostics=True)


# /settings: the settings, answered privately — with a menu under each that an admin can change it from. A change is
# kept and counts straight away, and the answer is redrawn with it. The menus stop working after SETTINGS_TIMEOUT
# or a restart, and /settings again brings new ones. It's in Discord's layout components rather than an embed, so
# each menu can have its name above it instead of in every option.
SETTINGS_TIMEOUT = 15 * 60  # seconds
STATUS_CHOICES = [0, 60, 120, 180, 360, 720, 1440]  # minutes
CONTEXT_CHOICES = [25_000, 50_000, 90_000]  # tokens

def describe_status(minutes):
    if not minutes:
        return "Off"
    if minutes % 60:
        return f"Every {minutes} minutes"
    return "Every hour" if minutes == 60 else f"Every {minutes // 60} hours"

def describe_tokens(tokens):
    return f"{tokens // 1000}K tokens" if tokens % 1000 == 0 else f"{tokens:,} tokens"

class SettingMenu(discord.ui.Select):
    def __init__(self, key, choices, describe):
        self.key = key
        current = setting(key)
        values = sorted({*choices, current})  # one set in an earlier version, or by hand, still shows
        super().__init__(options=[discord.SelectOption(label=describe(v), value=str(v), default=v == current)
                                  for v in values])

    async def callback(self, interaction):
        if not await is_admin(interaction.user):
            return await interaction.response.send_message("Only admins can change settings", ephemeral=True)
        change_setting(self.key, int(self.values[0]))
        log.info(f"[SET] {interaction.user} set {self.key} to {self.values[0]}")
        await interaction.response.edit_message(view=SettingsView(True))

class SettingsView(discord.ui.LayoutView):
    def __init__(self, admin):
        super().__init__(timeout=SETTINGS_TIMEOUT)
        text = discord.ui.TextDisplay
        box = discord.ui.Container(text("### Settings"), accent_color=EXPLANATION_COLOR)
        if admin:
            box.add_item(text("**New status**"))
            box.add_item(discord.ui.ActionRow(SettingMenu("STATUS_EVERY", STATUS_CHOICES, describe_status)))
            box.add_item(text("**Context window** · how far back Rocky remembers a channel"))
            box.add_item(discord.ui.ActionRow(SettingMenu("LOG_TOKENS", CONTEXT_CHOICES, describe_tokens)))
        else:
            box.add_item(text(f"**New status**\n{describe_status(STATUS_EVERY)}"))
            box.add_item(text(f"**Context window**\n{describe_tokens(rh.LOG_TOKENS)}, how far back Rocky remembers "
                              "a channel"))
        box.add_item(text("-# Changes count straight away, and are kept across restarts." if admin else
                          "-# Only admins can change these."))
        self.add_item(box)

def change_setting(key, value):
    save_setting(key, value)
    if key == "STATUS_EVERY" and value and not update_status.is_running() and bot.is_ready():
        if status is None:
            load_status()  # not loaded at startup while statuses were off — when the last one was made
        update_status.start()

@bot.tree.command(name="settings", description="See the settings — admins can change them")
async def settings_command(interaction: discord.Interaction):
    await interaction.response.send_message(view=SettingsView(await is_admin(interaction.user)), ephemeral=True)


# Messages the bot has taken on, so each is answered once: at startup, catch_up() can find one on_message is already
# answering (its reply not sent yet, so it looks missed), or the other way round. The latest only — catch_up() looks
# back CATCH_UP_WINDOW minutes.
taken: dict[int, None] = {}

async def respond(m, **extra):
    if m.id in taken:
        log.info(f"[SKIP] already answering {m.id}")
        return
    taken[m.id] = None
    if len(taken) > 1000:
        del taken[next(iter(taken))]
    handling.add(task := asyncio.current_task())
    try:
        await respond_traced(m, **extra)
    finally:
        handling.discard(task)


async def respond_traced(m, **extra):
    c = message_text(m) or "hello"
    log.info(f"[IN] {m.author}: {c[:80]}")
    # discord.py runs each event in its own task, so this trace only sees this message's requests
    t = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "channel": getattr(m.channel, "name", None),
         "channel_id": m.channel.id, "message_id": m.id, "author": m.author.display_name, "message": c,
         "cost": 0.0, "requests": 0, **extra}
    trace.set(t)
    start = time.monotonic()
    try:
        await handle(m)
    finally:
        t["seconds"] = round(time.monotonic() - start, 1)
        write_trace(t)


# A message too long for Discord as several: each cut at its last sentence's end that fits, or else its last space
MESSAGE_LIMIT = 2000

def message_parts(text, limit=MESSAGE_LIMIT):
    parts = []
    while len(text) > limit:
        cut = max((m.end() for m in re.finditer(r"[.!?…](?=\s)", text[:limit])), default=0) or text.rfind(" ", 0, limit)
        cut = cut if cut > 0 else limit
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    return [*parts, text]

# What Rocky did besides talking, a subtext line each under its message, in the order it did them: what it did, not
# what came of it, which its message says. Not its reactions, which are seen anyway, nor what it noted, which is
# secret, nor calls that changed nothing in its memories. Only under a message, and not in its log — it has its calls
# there. What it wrote in a call (a drink, a fact) is kept to one short line, its links unpreviewed, so ACTION_LINES
# takes all of it out again: a preview would be read back as an [embed] of its message.
ACTION_CHARS = 100  # the most of what it wrote in a call that a line shows

def action_lines(shown):
    def brief(x):
        x = " ".join(x.split())
        x = x if len(x) <= ACTION_CHARS else x[:ACTION_CHARS - 1] + "…"
        return LINK.sub(lambda m: f"<{m[1]}>", x)
    lines = []
    for kind, a, b in shown:
        a, b = (brief(x) if isinstance(x, str) else x for x in (a, b))
        fact = " ".join(b.split(": ", 1)) if isinstance(b, str) else None  # "pip: likes lager" as "pip likes lager"
        if kind == "order":
            line = f"🍺 ordered {a}" + (f" for {b}" if b else "")
        elif kind == "unordered":
            line = f"🍺 couldn't order {a or 'a drink'}"
        elif kind == "bar":
            line = "🍺 checked the bar"
            if f"-# {line}" in lines:  # it sometimes looks twice in a turn
                continue
        elif kind == "look":
            line = f"👀 looked at {'the picture' if b == 1 else f'{b} pictures'}" if b else "👀 couldn't see a picture"
        elif kind == "note":
            line = "📝 made a note"
        elif kind == "remember":
            line = f"🧠 remembered: {fact}"
        else:  # forget
            line = f"🗑️ forgot: {fact}"
        lines.append(f"-# {line}")
    return lines

# Rocky takes its turn in the channel's log (rocky_harness) — its message, its reactions on any message
# there, and what it remembers. entries: the channel's history, for what the log hasn't seen (the first time, and
# after a restart). where: the channel it answers in, if not m's — the thread m opened; chat: the log, if not where's.
async def rocky_turn(m, entry, bot_name, entries, where=None, chat=None):
    where = where or m.channel
    chat = chat or rh.log_for(where)
    if not chat.seeded:
        entries = await earlier_history(where, entries) + entries
    rh.sync(chat, entries)
    waking = rh.observe(m, chat)  # there already, unless it's one caught up on at startup
    if waking is None:  # nothing in it to answer: a bare !nocontext in a DM
        return
    async with gen_lock:  # no typing while it decides — it may only react (take_turn)
        turn = await rh.take_turn(chat, waking, bot_name, typing=where.typing)
    rh.changed(chat)  # saved whether or not it then posts: after a turn that only reacts, nothing else may change the log
    rh.jump(chat)
    note(llm="rocky", llm_model=LLM_MODEL, transcript=rh.readable(chat, bot_name, last=40),
         llm_messages=sum(e["type"] == "message" for e in chat.events), harness_requests=turn.requests,
         memory_kept=[list(k) for k in turn.kept],
         # a look's pictures stay in Rocky's log, and are only counted here: logs/ keeps every day
         rocky_events=[e | {"pictures": len(e["pictures"])} if e.get("pictures") else e for e in turn.events])
    vocab = emoji_vocabulary(m.guild)
    for n, emoji in turn.reactions:
        target = m.id if n == waking["n"] else chat.discord_id(n)
        # where the message is: a thread's first is in the channel it was opened from
        cid = m.channel.id if target == m.id else chat.channel_of(n)
        channel = where if cid == where.id else m.channel if cid == m.channel.id else bot.get_partial_messageable(cid)
        try:
            await channel.get_partial_message(target).add_reaction(vocab.get(emoji, emoji))
            if target == m.id:
                entry["reaction"] = emoji
        except (discord.HTTPException, TypeError) as e:  # an emoji Discord doesn't know, or a message gone
            log.warning(f"[ROCKY] react {emoji} on {target} failed: {e}")
    if turn.reactions:
        note(reaction=" ".join(e for _, e in turn.reactions))
        log.info(f"[REACT] {' '.join(e for _, e in turn.reactions)} (${trace.get()['cost']:.5f})")
    if not turn.text:
        if has_credit is False and not turn.reactions:
            await answer_broke(m, where)
        return
    targets = mention_targets([*entries, entry], bot_name, reactors=True)
    content, allowed = resolve_mentions(rh.unaddressed(turn.text, m.author.display_name), targets)
    note(reply=turn.text, sent=content)
    log.info(f"[OUT] {turn.text} (${trace.get()['cost']:.5f})")
    # Past Discord's limit, the rest follow the reply. They're one turn in the log, so any part is its message
    parts = message_parts(content)
    for line in action_lines(turn.shown):  # whole lines, so each part's are taken out when it's read back
        if len(parts[-1]) + 1 + len(line) <= MESSAGE_LIMIT:
            parts[-1] += "\n" + line
        else:
            parts.append(line)
    sent = await post_answer(m, where, parts[0], allowed)
    more = []
    for part in parts[1:]:
        try:
            more.append(await where.send(part, allowed_mentions=allowed))
        except discord.HTTPException as e:
            log.warning(f"[ROCKY] sending part {len(more) + 2} of {len(parts)} failed: {e}")
            break
    note(reply_id=sent.id, **({"reply_ids": [x.id for x in [sent, *more]]} if more else {}))
    turn.spoke["id"] = sent.id
    if more:
        turn.spoke["more_ids"] = [x.id for x in more]
    for x in [sent, *more]:
        chat.ids[str(x.id)] = "you"
    rh.changed(chat)
    entry["reply"] = " ".join(message_text(x) for x in [sent, *more]).strip()
    entry["reply_id"], entry["reply_at"] = sent.id, sent.created_at
    entry["reply_users"] = {k: v for x in [sent, *more] for k, v in message_users(x).items()}

# The answer to m: a Discord reply, or in the thread m opened, the next message there
async def post_answer(m, where, content, allowed=None):
    if where is not m.channel:
        try:
            return await where.send(content, allowed_mentions=allowed)
        except discord.Forbidden as e:  # a thread it could open but not post in: answered where it was asked
            log.warning(f"Posting in thread {where.id} failed, replying instead: {e}")
    return await m.reply(content, mention_author=False, allowed_mentions=allowed)

# Out of credit, a gif says so. Logged as the reply, so the actions find it.
async def answer_broke(m, where):
    gif = random.choice(NO_MONEY)
    log.info(f"[OUT] out of credit: {gif}")
    note(reply=gif, reply_id=(await post_answer(m, where, gif)).id)

# A !nocontext thread's name, and its log's: the question, without the tag to the bot. Rocky's log of a fallback answer
# has it too — named "DM" or the channel's, with nothing else in it, Rocky wrote notes on what to say in 2 of 6
# replays instead of saying it, and in none of 24 named from the question. Discord's longest is 100.
def thread_name(m):
    me = (m.guild.me if m.guild else bot.user).display_name
    name = " ".join(message_text(m).replace(f"@{me}", "").split()) or "nocontext"
    return name if len(name) <= 100 else name[:99] + "…"

# A !nocontext message's thread, opened on it, so it has the message's id, and Rocky's log of it, without memories,
# starting with the message. None in a DM, which has no threads, or if it couldn't be opened: then it's answered alone.
async def open_thread(m):
    if m.guild is None:
        return None
    try:
        if rh.thread_log(m.id):  # opened before a restart, and not answered: caught up on
            thread = m.guild.get_thread(m.id) or await bot.fetch_channel(m.id)
        else:
            thread = await m.create_thread(name=thread_name(m), auto_archive_duration=1440)
    except discord.HTTPException as e:  # no Create Public Threads permission, say
        log.warning(f"Opening a thread on {m.id} failed: {e}")
        return None
    rh.observe(m, rh.log_for(thread, memories=False))
    return thread

# A new log starts with as much of the channel as rocky_harness.FILL of it holds: history from before entries (the
# channel's, oldest first), read back until it's that big, or FILL_SCAN messages. Not people's reactions to the bot's
# replies — a log only has what it sees from then on.
async def earlier_history(channel, entries):
    room = rh.fill_chars() - rh.entry_chars(entries)
    first = min((e["id"] for e in entries if e.get("id")), default=None)
    scanned = []
    try:
        async for m in channel.history(limit=FILL_SCAN, before=discord.Object(first) if first else None):
            scanned.append(m)
            room -= len(message_text(m)) + rh.EVENT_CHARS
            if room <= 0:
                break
    except Exception as e:  # no Read Message History permission — the log starts with what there is
        log.warning(f"Reading back {channel.id} for Rocky's log failed: {e}")
    found = [entry for entry, _ in history_entries(scanned)]
    log.info(f"[ROCKY] #{getattr(channel, 'name', channel.id)} log starts {len(found)} messages further back")
    return found

async def handle(m):
    if m.guild is None:
        dm_channels.add(m.channel.id)  # before its history loads, so a status made meanwhile can't show it
    # A !nocontext message is answered in its own thread, with no history — once in it, it's a channel like any other
    apart = no_context(m)
    if apart:
        thread = await open_thread(m)
        where = thread or m.channel
        entry = history_entry(m)
        note(no_context=True)
        if thread:
            channel_history[thread.id].append(entry)
            # answered in the thread, so it's its; an action on the message, in the channel, finds it too
            note(channel=thread.name, channel_id=thread.id, from_channel_id=m.channel.id)
    else:
        thread, where = None, m.channel
        # Shared task, so messages arriving while it loads wait for it instead of loading twice
        if m.channel.id not in history_loaded:
            history_loaded[m.channel.id] = asyncio.create_task(load_history(m))
        await history_loaded[m.channel.id]
        entry = add_history(m.channel.id, history_entry(m))
        if rh.no_memories(m.channel):
            note(no_context=True)
    # Server nickname, so Rocky goes by the name people call the bot by
    bot_name = (m.guild.me if m.guild else bot.user).display_name
    note(bot_name=bot_name)
    try:
        if has_credit is False:
            await answer_broke(m, where)
        elif apart:
            await rocky_turn(m, entry, bot_name, [], where, None if thread else rh.apart(thread_name(m), m.channel.id))
        else:
            await rocky_turn(m, entry, bot_name, channel_history[m.channel.id])
    except Exception as e:
        log.error(f"Error: {e}", exc_info=True)
        note(error=repr(e))
        try: await post_answer(m, where, "...")
        except Exception: pass

# Keeps people's reactions to the bot's replies current in the history, so Rocky can ping who reacted
async def on_reaction_change(p, change):
    if p.user_id == bot.user.id:
        return
    if rh.logs_with(p.channel_id, p.message_id):  # anyone's reaction on anything in Rocky's logs, as it happens
        name = await reactor_name(bot.get_guild(p.guild_id) if p.guild_id else None, p.user_id, p.member)
        rh.reacted(p.channel_id, p.message_id, emoji_text(p.emoji), name, change > 0)
    entry = next((e for e in channel_history.get(p.channel_id, []) if e.get("reply_id") == p.message_id), None)
    if entry is None:
        return
    counts = entry.setdefault("reply_reactions", {})
    e = emoji_text(p.emoji)
    users = entry.setdefault("reply_reactors", {}).setdefault(e, {})
    uid = str(p.user_id)
    counts[e] = counts.get(e, 0) + change
    if change > 0:
        users[uid] = None
    else:
        users.pop(uid, None)
    if counts[e] <= 0:
        del counts[e]
        entry["reply_reactors"].pop(e, None)
    if change > 0:
        guild = bot.get_guild(p.guild_id) if p.guild_id else None
        name = await reactor_name(guild, p.user_id, p.member)  # None if it can't be found — the count still shows
        if uid in users:
            users[uid] = name
    remember_reactors(p.message_id, counts, entry.get("reply_reactors"))

@bot.event
async def on_raw_reaction_add(p):
    await on_reaction_change(p, 1)

@bot.event
async def on_raw_reaction_remove(p):
    await on_reaction_change(p, -1)

@bot.event
async def on_raw_reaction_clear(p):
    for e in channel_history.get(p.channel_id, []):
        if e.get("reply_id") == p.message_id:
            e.pop("reply_reactions", None)
            e.pop("reply_reactors", None)
            remember_reactors(p.message_id, {}, {})

@bot.event
async def on_raw_reaction_clear_emoji(p):
    for e in channel_history.get(p.channel_id, []):
        if e.get("reply_id") == p.message_id:
            e.get("reply_reactions", {}).pop(emoji_text(p.emoji), None)
            e.get("reply_reactors", {}).pop(emoji_text(p.emoji), None)
            remember_reactors(p.message_id, e.get("reply_reactions"), e.get("reply_reactors"))

async def main():
    first_signal = None
    def on_signal():
        nonlocal first_signal
        if first_signal is None:
            first_signal = time.monotonic()
            stopping.set()
        elif time.monotonic() - first_signal > 1:
            log.warning("[STOP] quitting now, without finishing")
            os._exit(1)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, on_signal)

    async with bot:
        running = asyncio.create_task(bot.start(TOKEN))
        await asyncio.wait([running, asyncio.create_task(stopping.wait())], return_when=asyncio.FIRST_COMPLETED)
        if running.done():
            running.result()  # login failed or the connection gave up — raise it
        if handling:
            log.info(f"[STOP] finishing {len(handling)} message(s) in progress — Ctrl-C again to quit now")
            await asyncio.gather(*handling, return_exceptions=True)
        # Writes that wait a moment so a burst writes once, done now rather than lost
        rh.flush()
        if reactors_save is not None:
            reactors_save.cancel()
            save_reactors()
        log.info("[STOP] done")


if __name__ == "__main__":
    asyncio.run(main())
