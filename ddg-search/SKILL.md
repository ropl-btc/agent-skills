---
name: ddg-search
description: Search the web with DuckDuckGo as a no-API-key fallback or second source. Use when the normal web search tool or primary search provider is rate-limited, failing, unavailable, not delivering enough useful results, or producing weak results, and you want quick text, news, image, or video search results, instant-answer lookups, or DuckDuckGo bang resolution without browser automation.
allowed-tools: Bash(./scripts/ddg-search:*)
---

# ddg-search

Use the local skill script for lightweight DuckDuckGo search.

This skill exists to make fallback web search simple and low-noise instead of rebuilding ad hoc search helpers each time.

## Quick rules

- Prefer your normal primary search tool when it is healthy.
- Use this skill when you need a fallback or a second source.
- Keep searches narrow and intentional.
- Prefer JSON output by default.
- Use `--text` only when human-readable terminal output is actually more useful.
- Use `references/notes.md` for quick guidance on when this tool fits.

## Local setup

Prefer the skill-local script and a skill-local virtual environment over any global CLI install.
Treat `.venv` as generated local state, not part of the skill itself.

Bootstrap the local environment:

```bash
<skill-path>/scripts/bootstrap_venv.sh
```

After bootstrap, use:

```bash
<skill-path>/scripts/ddg-search
```

If `.venv` is missing later, just run the bootstrap script again.

## Commands

### Show built-in help

```bash
<skill-path>/scripts/ddg-search help
```

### Text search

```bash
<skill-path>/scripts/ddg-search search 'openclaw github'
```

### News search

```bash
<skill-path>/scripts/ddg-search search 'bitcoin etf' --type news --timelimit d --max-results 10
```

### Image search

```bash
<skill-path>/scripts/ddg-search search 'bitcoin logo' --type images --max-results 10
```

### Video search

```bash
<skill-path>/scripts/ddg-search search 'openai launch' --type videos --timelimit w --max-results 10
```

### Restrict to one site

```bash
<skill-path>/scripts/ddg-search search 'python dataclasses' --site docs.python.org
```

### Reverse result order

```bash
<skill-path>/scripts/ddg-search search 'openai launch' --type videos --max-results 10 --reverse
```

### Instant-answer lookup

```bash
<skill-path>/scripts/ddg-search instant 'weather berlin'
```

### Resolve a DuckDuckGo bang

```bash
<skill-path>/scripts/ddg-search bang w 'OpenAI'
```

## Workflow

1. Read `references/notes.md` if you want quick selection guidance.
2. Ensure the skill-local virtual environment is bootstrapped.
3. Use `search` for normal search, `instant` for instant-answer lookups, and `bang` when a DuckDuckGo bang is the cleanest path.
4. Parse JSON output by default.
5. Cross-check with another source when the question is important.

## Expected outputs

The tool returns JSON by default. Parse it instead of scraping text.

Use `--text` when you want a readable terminal view.

- `search` returns query metadata plus `results`
- `instant` returns instant-answer style structured fields
- `bang` returns the DuckDuckGo URL plus the resolved redirect/final URL

## Files

- Source repo: `https://github.com/ropl-btc/ddg-search-cli`
- Launcher: `scripts/ddg-search`
- Python implementation: `scripts/ddg_search.py`
- Local bootstrap: `scripts/bootstrap_venv.sh`
- Notes: `references/notes.md`
- `.venv/` is generated local state and can be recreated with `scripts/bootstrap_venv.sh`

## When to stop and ask

Stop and ask before:
- turning this into browser automation
- adding scraping-heavy flows
- adding provider-specific hacks that make the wrapper fragile
- changing default behavior from lightweight fallback search to something much broader
