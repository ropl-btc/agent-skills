# Agent Skills

A collection of useful AI agent skills by Robin.

This repository is where I publish practical skills I use in real workflows. The goal is simple: reusable skills that are easy to install and actually useful in day-to-day execution.

## Current Skill

### `persistent-memory`

A lightweight persistent memory skill for local workspace agents.

It provides:
- Durable memory storage in local SQLite (`.memory/memory.db`)
- Fast recall with search and hybrid ranking
- Automatic reinforcement of recalled memories (`hits`, `last_seen_at`)
- Simple CLI workflow for `init`, `sync`, `search`, `add`, `recent`, and `stats`

This skill is inspired by the long-term memory behavior of Clawdbot/OpenClaw, adapted into a simpler, local-first, database-only approach that is easy to run and maintain.

## Install

```bash
npx skills add ropl-btc/agent-skills --skill persistent-memory
```

## Roadmap

I’ll keep adding more skills here as I create and refine them.
