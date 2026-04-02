# ddg-search notes

Use this tool for lightweight search when:
- your normal primary search tool is rate-limited
- your normal primary search tool is temporarily failing
- the default search path is not delivering enough useful results
- the default search path looks weak, noisy, or inaccurate and you want a second source
- you want a quick no-key search tool for scripts or agent workflows

Use these commands intentionally:
- `search` for normal web/news/image/video results
- `search --site <domain>` when you already know the target site
- `instant` for instant-answer style lookups
- `bang` when DuckDuckGo bang routing is the cleanest path to a target site

Use a browser tool instead when:
- you need to log in
- you need to interact with a site
- you need to scrape page content after search results
- search-result snippets are not enough

Default behavior should stay simple:
- JSON output first
- small result sets
- narrow queries
- no browser automation
