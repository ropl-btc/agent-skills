---
name: telegram-cli
description: Guarded Telegram CLI for the user's personal account via Telethon/MTProto. Use to inspect chats, list unread dialogs, read/search messages, and perform approved work actions like send, mark-read, archive, and mute with explicit dry-run/execute safety.
allowed-tools: Bash(./scripts/telegram-readonly:*), Bash(./scripts/telegram-cli:*)
---

# Telegram CLI

Use the local skill script for Telegram work on the user's personal account.

This skill exists because Telegram Bot API is the wrong tool for reading a real personal account. Use MTProto via Telethon instead.

## Quick rules

- Prefer reads first, then propose the action queue.
- Write commands are dry-run by default and require `--execute`.
- Never run any write command with `--execute` unless the user explicitly approved that specific action or batch first.
- For `send`, always present a draft message first and ask the user for confirmation before sending.
- Do not run `send --execute` unless the user explicitly approved the final recipient and text.
- Mark-read/archive/mute are still Telegram writes; use them only after the user has approved the batch/action.
- Do not add edit/delete/bulk export/background automation unless the user explicitly asks.
- Treat the Telethon session like a high-privilege secret.
- Assume unread preservation is best-effort until tested on a real chat.

## Local setup

Prefer the skill-local script and a skill-local virtual environment over any global CLI install.
Prefer saved Telegram config over shell-exported environment variables once setup is complete.
Treat `.venv` as generated local state, not part of the skill itself.
If the installer drops skill-local dotfiles, the bootstrap script recreates `.gitignore` automatically.

Bootstrap the local environment:

```bash
<skill-path>/scripts/bootstrap_venv.sh
```

After bootstrap, use:

```bash
<skill-path>/scripts/telegram-cli
```

`scripts/telegram-readonly` remains as a backwards-compatible alias for older workflows.

If `.venv` is missing later, just run the bootstrap script again.

Primary config path:

```bash
~/.config/telegram-cli/config.json
```

Recommended one-time setup:

1. Make sure `api_id` and `api_hash` are available.
2. Save them with:

```bash
<skill-path>/scripts/setup-api-key.sh
```

3. Run:

```bash
<skill-path>/scripts/telegram-cli auth
```

After successful login, the config file stores `api_id`, `api_hash`, and the Telegram session string so future reads do not need exported shell variables.

## Commands

### Show built-in help

```bash
<skill-path>/scripts/telegram-cli help
```

### Authenticate once

```bash
<skill-path>/scripts/setup-api-key.sh
<skill-path>/scripts/telegram-cli auth
```

### List chats

`dialogs --query` does token-based matching across `name`, `username`, and `title`, so queries like `petros skynet` work even when the exact full string is not present as one substring.

```bash
<skill-path>/scripts/telegram-cli dialogs --limit 50
```

### Read recent messages

```bash
<skill-path>/scripts/telegram-cli messages --chat '@username' --limit 50 --reverse
```

### Search messages

```bash
<skill-path>/scripts/telegram-cli search 'invoice' --limit 50
```

Restrict search to one chat:

```bash
<skill-path>/scripts/telegram-cli search 'deadline' --chat '@username' --limit 50
```

### List recent unread chats

Default behavior is opinionated: exclude **muted** and **archived** chats.

```bash
<skill-path>/scripts/telegram-cli unread-dialogs --limit 10
```

Include muted and/or archived when needed:

```bash
<skill-path>/scripts/telegram-cli unread-dialogs --limit 10 --include-muted --include-archived
```

### List recent unread DMs only

```bash
<skill-path>/scripts/telegram-cli unread-dms --limit 10
```

### Send a message

Draft first in chat, ask the user to confirm, then dry-run:

```bash
<skill-path>/scripts/telegram-cli send --chat '@username' --text 'Thanks, will check.'
```

Send only after the user approves final text and recipient:

```bash
<skill-path>/scripts/telegram-cli send --chat '@username' --text 'Thanks, will check.' --execute
```

### Mark read

```bash
<skill-path>/scripts/telegram-cli mark-read --chat 123456789
<skill-path>/scripts/telegram-cli mark-read --chat 123456789 --execute
```

### Archive or unarchive

```bash
<skill-path>/scripts/telegram-cli archive --chat 123456789
<skill-path>/scripts/telegram-cli archive --chat 123456789 --execute
<skill-path>/scripts/telegram-cli archive --chat 123456789 --unarchive --execute
```

### Mute or unmute

```bash
<skill-path>/scripts/telegram-cli mute --chat 123456789 --hours 8
<skill-path>/scripts/telegram-cli mute --chat 123456789 --hours 8 --execute
<skill-path>/scripts/telegram-cli mute --chat 123456789 --unmute --execute
```

## Workflow

1. Read `references/setup-and-safety.md` if setup, auth, or unread-state behavior matters.
2. Ensure the skill-local virtual environment is bootstrapped.
3. Ensure Telegram API credentials exist.
4. Run `auth` once to create the session and write `~/.config/telegram-cli/config.json`.
5. Use `dialogs`, `messages`, `search`, `unread-dialogs`, or `unread-dms` as needed.
6. For writes, get the user's approval first, run the dry-run, check the JSON target/action, then use `--execute`.
7. Keep usage narrow and intentional.

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

- Launcher: `scripts/telegram-cli`
- Launcher: `scripts/telegram-readonly`
- Python implementation: `scripts/telegram_cli.py`
- Local bootstrap: `scripts/bootstrap_venv.sh`
- Credential setup helper: `scripts/setup-api-key.sh`
- Setup notes: `references/setup-and-safety.md`
- Config storage: `~/.config/telegram-cli/config.json`
- `.env` is optional fallback only; it is not the preferred long-term setup.
- `.venv/` is generated local state and can be recreated with `scripts/bootstrap_venv.sh`.

## When to stop and ask

Stop and ask before:
- sending a Telegram message
- enabling any background watcher/daemon
- broad exporting of large chat histories
- changing how secrets/session storage works

## Docs

Fast lookup:
- Telethon client reference: `https://docs.telethon.dev/en/stable/quick-references/client-reference.html`
- Telethon TelegramClient API: `https://docs.telethon.dev/en/stable/modules/client.html`
- Telegram folders/archive API: `https://core.telegram.org/api/folders`
- Telegram notification settings API: `https://core.telegram.org/method/account.updateNotifySettings`
