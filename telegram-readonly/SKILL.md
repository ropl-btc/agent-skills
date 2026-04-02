---
name: telegram-readonly
description: Read the user's personal Telegram account in a controlled, read-only way via Telethon/MTProto. Use when you need to inspect Telegram chats, list dialogs, read recent messages from a specific chat, or search Telegram messages without relying on the Telegram Bot API. Do not use for sending, replying, editing, deleting, or any write action.
allowed-tools: Bash(./scripts/telegram-readonly:*)
---

# Telegram Readonly

Use the local skill script for Telegram reads from the user's personal account.

This skill exists because Telegram Bot API is the wrong tool for reading a real personal account. Use MTProto via Telethon instead.

## Quick rules

- Use this skill only for reads.
- Do not improvise write actions.
- Do not add send/edit/delete logic to the wrapper unless the user explicitly asks.
- Treat the Telethon session like a high-privilege secret.
- Assume unread preservation is best-effort until tested on a real chat.

## Local setup

Prefer the skill-local script and a skill-local virtual environment over any global CLI install.
Prefer saved Telegram config over shell-exported environment variables once setup is complete.
Treat `.venv` as generated local state, not part of the skill itself.

Bootstrap the local environment:

```bash
<skill-path>/scripts/bootstrap_venv.sh
```

After bootstrap, use:

```bash
<skill-path>/scripts/telegram-readonly
```

If `.venv` is missing later, just run the bootstrap script again.

Primary config path:

```bash
~/.config/telegram-readonly/config.json
```

Recommended one-time setup:

1. Make sure `api_id` and `api_hash` are available.
2. Run:

```bash
<skill-path>/scripts/telegram-readonly auth
```

After successful login, the config file stores `api_id`, `api_hash`, and the Telegram session string so future reads do not need exported shell variables.

## Commands

### Show built-in help

```bash
<skill-path>/scripts/telegram-readonly help
```

### Authenticate once

```bash
export TELEGRAM_API_ID='12345678'
export TELEGRAM_API_HASH='your_api_hash'
<skill-path>/scripts/telegram-readonly auth
```

### List chats

`dialogs --query` does token-based matching across `name`, `username`, and `title`, so queries like `petros skynet` work even when the exact full string is not present as one substring.

```bash
<skill-path>/scripts/telegram-readonly dialogs --limit 50
```

### Read recent messages

```bash
<skill-path>/scripts/telegram-readonly messages --chat '@username' --limit 50 --reverse
```

### Search messages

```bash
<skill-path>/scripts/telegram-readonly search 'invoice' --limit 50
```

Restrict search to one chat:

```bash
<skill-path>/scripts/telegram-readonly search 'deadline' --chat '@username' --limit 50
```

### List recent unread chats

Default behavior is opinionated: exclude **muted** and **archived** chats.

```bash
<skill-path>/scripts/telegram-readonly unread-dialogs --limit 10
```

Include muted and/or archived when needed:

```bash
<skill-path>/scripts/telegram-readonly unread-dialogs --limit 10 --include-muted --include-archived
```

### List recent unread DMs only

```bash
<skill-path>/scripts/telegram-readonly unread-dms --limit 10
```

## Workflow

1. Read `references/setup-and-safety.md` if setup, auth, or unread-state behavior matters.
2. Ensure the skill-local virtual environment is bootstrapped.
3. Ensure Telegram API credentials exist.
4. Run `auth` once to create the session and write `~/.config/telegram-readonly/config.json`.
5. Use `dialogs`, `messages`, `search`, `unread-dialogs`, or `unread-dms` as needed.
6. Keep usage narrow and intentional.

## Expected outputs

The wrapper returns JSON. Parse it instead of relying on fragile text scraping.

Dialog objects include:
- `is_user`
- `is_group`
- `is_channel`
- `is_bot`
- `archived`
- `muted`
- unread counters

## Files

- Package repo: `https://github.com/ropl-btc/telegram-readonly-cli`
- Launcher: `scripts/telegram-readonly`
- Python implementation: `scripts/telegram_readonly.py`
- Local bootstrap: `scripts/bootstrap_venv.sh`
- Setup notes: `references/setup-and-safety.md`
- Config storage: `~/.config/telegram-readonly/config.json`
- `.env` is optional fallback only; it is not the preferred long-term setup.
- `.venv/` is generated local state and can be recreated with `scripts/bootstrap_venv.sh`.

## When to stop and ask

Stop and ask before:
- adding write capabilities
- enabling any background watcher/daemon
- broad exporting of large chat histories
- changing how secrets/session storage works
