# Agent instructions

## The live bot

- `run.py` pulls `master` within a minute of a merge and restarts the bot, so merging deploys. It skips the restart when a merge only touches files the bot never loads (`NO_RESTART` in `run.py`). After a merge, check the bot came back up and that the change behaves as expected.
- `run.py` runs on the host's system Python (Ubuntu 24.04's 3.12), not the pinned one, so it has to stay standard-library only and work on that version. CI's `run-py-on-system-python` job checks this on the same Ubuntu release: if the host's release changes, update the job's runner too.
- In a worktree, `.env` is a symlink to the live checkout's. Never write test data to it: it's gitignored and has no copy to restore from. Put test files in `.scratch/` or set env vars inline.

## Privacy

- This repo is public. Never put real chat messages, usernames or server emoji names in PRs, commits, tests or fixtures. Use synthetic conversations, such as `rocky_replay.py --fresh`'s chat. You can validate against real logs locally, but describe what you found without quoting them.
- Rocky's thinking is kept in its logs as Anthropic's thinking blocks (`reasoning_details`, each with a summarised `text` and an encrypted `signature`). Never print or quote a block's `text` in an agent session: an agent doing that was stopped by a `reasoning_extraction` refusal, which Anthropic uses for requests that try to get a model's reasoning into its response text ([docs](https://platform.claude.com/docs/en/build-with-claude/thinking#summarized-thinking)). Lengths, counts and the other fields are fine. Strip the blocks before printing logs: `jq 'del(.. | .reasoning?, .reasoning_details?)'`.
- Never edit a thinking block. The harness sends a turn's blocks back in the requests that finish that turn, and they must go back unaltered: Anthropic rejects a modified block with a 400 ([docs](https://platform.claude.com/docs/en/build-with-claude/thinking#preserving-thinking-blocks)). Earlier turns' blocks stay in the log but aren't sent (`render` in `rocky_harness.py` says why).
- `rocky_lines.txt` holds copyrighted quotes. Never print or commit it.

## Testing

- `scripts/setup.sh` sets up a clone or worktree: it installs the pinned Python and uv with mise and links `.env` in a worktree. Prefix commands with `mise exec --` if `uv --version` doesn't match `mise.toml`.
- Tests run offline: `uv run --with-requirements requirements.txt python -m unittest`.
- Test behaviour, not functions: a few tests that each drive a whole flow, such as a message arriving through to Rocky's reply being posted, with fakes only at the outside edge (Discord, the LLM API, the clock, git). Don't add a test per function or branch. When a flow test covers what unit tests in the area checked, remove them.
- `rocky_replay.py` replays messages to Rocky on the live log, using the eval key (`ANTHROPIC_API_KEY_EVAL`); it won't run on the bot's. The eval key's workspace has its own cache, so the first replay of a log writes it in full. Run replays one at a time: each one writes the whole log to Anthropic's cache, which counts against the account's per-minute input limit.

## Rocky

- When Rocky misbehaves, prefer clearer wording in its prompt over code that inspects a turn and retries it. Replay the new wording, report the results honestly, then ship it and watch how it goes live.

## Pull requests

- Start every PR description with a GitHub note holding a one- or two-sentence tl;dr: what was wrong or missing, and what the PR changes. The rest of the description follows below it.

  ```markdown
  > [!NOTE]
  > **tl;dr:** Rocky's reply was dropped when a second message arrived while it was still replying. Rocky's turn now stays where it began, and messages that arrive during it get their own turn.
  ```
- A PR that touches `rocky_harness.py` with nothing Rocky needs to hear of (plumbing, not a change to him) has the line `Rocky-update: none` in its description, so its commit doesn't reach his logs as an `<update>` (see Updates in `README.md`). It only works if the line stays in the squash commit's message.
