# Persistent Memory Usage

## Quick Start

```bash
.agents/skills/persistent-memory/scripts/pmem init
.agents/skills/persistent-memory/scripts/pmem sync
.agents/skills/persistent-memory/scripts/pmem search "investor update" --limit 8
```

## Store Durable Memory

```bash
.agents/skills/persistent-memory/scripts/pmem add "Always convert times to CET" --tags "timezone,calendar" --source "assistant"
```

## Inspect and Verify

```bash
.agents/skills/persistent-memory/scripts/pmem recent --limit 10
.agents/skills/persistent-memory/scripts/pmem stats
```

## Notes

- `search` updates `hits` and `last_seen_at` for returned rows.
- `sync` imports bullets from `MEMORY.md` and `memories/*.md`.
- `.memory/memory.db` is the local index database.
