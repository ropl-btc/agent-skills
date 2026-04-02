# Agent Skills

A small collection of practical AI agent skills.

This repo is now a simple multi-skill repository: each top-level folder is a skill with its own `SKILL.md`, `scripts/`, and optional `references/` or other support files.

## Skills

### `persistent-memory`

Local persistent memory workflow for agents.

- SQLite-backed memory storage and retrieval
- helper commands for `init`, `search`, `add`, `recent`, and related maintenance
- best for agents that want a database-backed memory layer inside one workspace

### `telegram-readonly`

Read-only access to a personal Telegram account via Telethon/MTProto.

- list dialogs
- read recent messages from a chat
- search messages
- inspect unread chats and DMs
- uses a skill-local bootstrap flow and stores auth state in `~/.config/telegram-readonly/config.json`

### `ddg-search`

Lightweight DuckDuckGo search as a no-key fallback or second source.

- text, news, image, and video search
- instant-answer lookups
- DuckDuckGo bang resolution
- local script-based skill with a skill-local bootstrap flow

### `twitterapi-io`

Read-only Twitter/X data access via `twitterapi.io`.

- fetch tweets, users, timelines, replies, quote tweets, thread context, mentions, and search results
- local script-based skill, no global CLI required
- stores API key in `~/.config/twitterapi-io/config.json`

## Repo Layout

Each skill is self-contained:

- `SKILL.md`: agent-facing instructions
- `scripts/`: local entrypoints and helpers
- `references/`: optional docs or links

Some skills may also include skill-specific generated local state such as `.venv/`. Those should generally not be committed.

## Using A Skill

Open the skill folder and follow its `SKILL.md`.

Examples:

- [persistent-memory](persistent-memory/SKILL.md)
- [ddg-search](ddg-search/SKILL.md)
- [telegram-readonly](telegram-readonly/SKILL.md)
- [twitterapi-io](twitterapi-io/SKILL.md)

## Notes

- The skills in this repo are optimized for real agent workflows, not polished end-user CLIs.
- Keep changes minimal and keep each skill self-contained.
