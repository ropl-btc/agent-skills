# Telegram CLI — Setup and safety

## What this skill is for

Use this skill to read and perform approved Telegram inbox-zero actions from the user's personal account via Telethon/MTProto.

This is not a Telegram bot skill.
It is for local access to a real user account.

## Safety model

The wrapper exposes read commands:
- `auth`
- `dialogs`
- `messages`
- `search`
- `unread-dialogs`
- `unread-dms`
- `help`

The wrapper also exposes guarded write commands:
- send
- mark-read calls
- archive/unarchive
- mute/unmute

Every write command is dry-run by default and requires `--execute`.
Never run any write command with `--execute` unless the user explicitly approved that specific action or batch first.
For `send`, always present a draft message first and ask the user for confirmation before sending.
Do not run `send --execute` unless the user explicitly approved the final recipient and message text.

It still does not expose:
- edit
- delete
- background auto-reply logic

Important: the underlying Telethon session still has high privilege because it is a real Telegram login. The safety comes from the wrapper surface area, not from Telegram granting reduced permissions.

## Files and locations

- Package entrypoint: `telegram-cli`
- Local config: `~/.config/telegram-cli/config.json`

## Prerequisites

1. Telegram API credentials from `https://my.telegram.org`
2. Telethon installed through the skill-local bootstrap
3. One interactive login to create a session string

## Install

From the skill directory:

```bash
scripts/bootstrap_venv.sh
```

## Telegram API credentials

At `https://my.telegram.org`:
1. Log in with the Telegram account phone number.
2. Open API development tools.
3. Create an application.
4. Save `api_id` and `api_hash`.

## First auth flow

Save API credentials:

```bash
scripts/setup-api-key.sh
```

Then authenticate:

```bash
telegram-cli auth
```

The CLI will prompt for:
- phone number
- login code
- 2FA password if enabled

It saves the resulting session string to `~/.config/telegram-cli/config.json`.
Protect that file like a password.

## Read-only usage

Show built-in help:

```bash
telegram-cli help
```

List chats:

`dialogs --query` uses token-based matching across `name`, `username`, and `title`.

```bash
telegram-cli dialogs --limit 50
```

Read one chat:

```bash
telegram-cli messages --chat '@username' --limit 50 --reverse
```

Search globally:

```bash
telegram-cli search 'invoice' --limit 50
```

Search in one chat:

```bash
telegram-cli search 'deadline' --chat '@username' --limit 50
```

List recent unread chats, excluding muted + archived by default:

```bash
telegram-cli unread-dialogs --limit 10
```

List recent unread DMs only, excluding muted + archived by default:

```bash
telegram-cli unread-dms --limit 10
```

Include muted and archived when needed:

```bash
telegram-cli unread-dialogs --limit 10 --include-muted --include-archived
```

## Guarded writes

Draft the message in chat first, ask the user to confirm, then dry-run send:

```bash
telegram-cli send --chat '@username' --text 'Thanks, will check.'
```

Actually send only after the user approves final text and recipient:

```bash
telegram-cli send --chat '@username' --text 'Thanks, will check.' --execute
```

Mark read:

```bash
telegram-cli mark-read --chat 123456789 --execute
```

Archive:

```bash
telegram-cli archive --chat 123456789 --execute
```

Mute:

```bash
telegram-cli mute --chat 123456789 --hours 8 --execute
```

## Unread behavior

Goal: avoid changing unread state.

Read commands never call explicit read acknowledgements.
That should usually avoid marking messages as read, but this must be verified with a live test because Telegram state can be subtle.

Before broad use:
1. pick a sacrificial chat
2. confirm unread badge/state before read
3. fetch messages with the wrapper
4. verify whether unread state changed in official Telegram clients

If unread state changes unexpectedly, stop and adjust workflow before wider rollout.

## Operational guidance

- Prefer narrow reads over broad scraping.
- Start with specific chats or direct need.
- Get the user's approval before every write action or batch.
- Use dry-run before every write.
- Keep sends socially reviewed: present a draft first, then send only after final text and recipient are approved.

## Docs

Fast lookup:
- Telethon client reference: `https://docs.telethon.dev/en/stable/quick-references/client-reference.html`
- Telethon TelegramClient API: `https://docs.telethon.dev/en/stable/modules/client.html`
- Telegram folders/archive API: `https://core.telegram.org/api/folders`
- Telegram notification settings API: `https://core.telegram.org/method/account.updateNotifySettings`
