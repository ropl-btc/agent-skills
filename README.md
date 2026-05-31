# Agent Skills

A small collection of practical AI agent skills.

Each top-level folder is a self-contained skill.

## Install

Install all skills globally:

```bash
npx skills add ropl-btc/agent-skills -g -y
```

Install a single skill:

```bash
npx skills add ropl-btc/agent-skills@twitterapi-io -g -y
```

## Skills

### `telegram-cli`

Guarded Telegram access to a personal account via Telethon/MTProto.

```bash
npx skills add ropl-btc/agent-skills@telegram-cli -g -y
```

- list dialogs
- read recent messages from a chat
- search messages
- inspect unread chats and DMs
- send, mark-read, archive, and mute only with explicit approval and `--execute`
- uses a cached virtualenv bootstrap flow and stores auth state in `~/.config/telegram-cli/config.json`

### `ddg-search`

Lightweight DuckDuckGo search as a no-key fallback or second source.

```bash
npx skills add ropl-btc/agent-skills@ddg-search -g -y
```

- text, news, image, and video search
- instant-answer lookups
- DuckDuckGo bang resolution
- local script-based skill with a cached virtualenv bootstrap flow

### `twitterapi-io`

Read-only Twitter/X data access via `twitterapi.io`.

```bash
npx skills add ropl-btc/agent-skills@twitterapi-io -g -y
```

- fetch tweets, users, timelines, replies, quote tweets, thread context, mentions, and search results
- local script-based skill, no global CLI required
- stores API key in `~/.config/twitterapi-io/config.json`

### `gsc-cli`

Google Search Console CLI for the official Search Console API.

```bash
npx skills add ropl-btc/agent-skills@gsc-cli -g -y
```

- list Search Console properties
- query Search Analytics performance rows
- inspect Google's indexed status for a URL
- list/get/submit/delete sitemaps with explicit `--execute` for writes
- uses headless-friendly browser URL OAuth and stores tokens in `~/.config/gsc-cli/config.json`

### `markdown-to-pdf`

Local Markdown-to-PDF renderer for clean text-first PDFs.

```bash
npx skills add ropl-btc/agent-skills@markdown-to-pdf -g -y
```

## Repo Layout

Each skill is self-contained:

- `SKILL.md`: agent-facing instructions
- `scripts/`: local entrypoints and helpers
- `references/`: optional docs or links

Generated local state should live outside the repo where practical: credentials in `~/.config/<skill-name>/`, caches and virtualenvs in `~/.cache/<skill-name>/`, and durable generated files in `~/.local/share/<skill-name>/`.

## Skill Docs

- [ddg-search](ddg-search/SKILL.md)
- [gsc-cli](gsc-cli/SKILL.md)
- [telegram-cli](telegram-cli/SKILL.md)
- [twitterapi-io](twitterapi-io/SKILL.md)
- [markdown-to-pdf](markdown-to-pdf/SKILL.md)

## Notes

- The skills in this repo are optimized for real agent workflows, not polished end-user CLIs.
- Keep changes minimal and keep each skill self-contained.
