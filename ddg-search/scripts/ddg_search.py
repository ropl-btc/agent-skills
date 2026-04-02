#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
LOCAL_VENV_PYTHON = SKILL_DIR / ".venv" / "bin" / "python"
BOOTSTRAP_SCRIPT = SCRIPT_DIR / "bootstrap_venv.sh"


def maybe_reexec_local_venv() -> None:
    if os.getenv("DDG_SEARCH_VENV_REEXEC") == "1":
        return
    if "VIRTUAL_ENV" in os.environ:
        return
    if not LOCAL_VENV_PYTHON.exists():
        return
    env = dict(os.environ)
    env["DDG_SEARCH_VENV_REEXEC"] = "1"
    raise SystemExit(subprocess.call([str(LOCAL_VENV_PYTHON), __file__, *sys.argv[1:]], env=env))


maybe_reexec_local_venv()


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def build_query(args: argparse.Namespace) -> str:
    query = args.query.strip()
    if args.site:
        query = f"site:{args.site} {query}".strip()
    return query


def result_text_url(row: dict[str, Any]) -> str:
    return row.get("href") or row.get("url") or row.get("content") or row.get("image") or ""


def print_text_results(results: list[dict[str, Any]]) -> None:
    for i, row in enumerate(results, 1):
        title = row.get("title") or row.get("source") or "(no title)"
        url = result_text_url(row)
        body = row.get("body") or row.get("excerpt") or ""
        print(f"{i}. {title}")
        if url:
            print(f"   {url}")
        if body:
            print(f"   {body[:220]}")
        print()


def run_search(args: argparse.Namespace) -> int:
    try:
        from ddgs import DDGS
    except Exception as exc:
        raise SystemExit(
            f"ddgs is not installed. Bootstrap the skill-local environment first: {BOOTSTRAP_SCRIPT}"
        ) from exc

    query = build_query(args)
    safesearch = "off" if args.unsafe else args.safesearch

    with DDGS() as ddgs:
        common: dict[str, Any] = {
            "region": args.region,
            "safesearch": safesearch,
            "timelimit": args.timelimit,
            "max_results": args.max_results,
            "page": args.page,
            "backend": args.backend,
        }
        if args.type == "text":
            results = list(ddgs.text(query, **common))
        elif args.type == "news":
            results = list(ddgs.news(query, **common))
        elif args.type == "images":
            common.pop("timelimit", None)
            results = list(ddgs.images(query, **common))
        elif args.type == "videos":
            results = list(ddgs.videos(query, **common))
        else:
            results = []

    if args.reverse:
        results = list(reversed(results))

    if args.text:
        print_text_results(results)
        return 0

    payload = {
        "query": query,
        "type": args.type,
        "site": args.site,
        "region": args.region,
        "safesearch": safesearch,
        "timelimit": args.timelimit,
        "max_results": args.max_results,
        "page": args.page,
        "backend": args.backend,
        "reverse": args.reverse,
        "results": results,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def fetch_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; ddg-search-skill/0.1.0)",
            "Accept": "application/json,text/plain,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8", "ignore"))


def run_instant(args: argparse.Namespace) -> int:
    query = args.query.strip()
    params = {
        "q": query,
        "format": "json",
        "no_redirect": "1",
        "no_html": "1",
        "skip_disambig": "0",
    }
    data = fetch_json("https://api.duckduckgo.com/?" + urllib.parse.urlencode(params))
    payload = {
        "query": query,
        "heading": data.get("Heading"),
        "abstract": data.get("AbstractText"),
        "abstract_url": data.get("AbstractURL"),
        "answer": data.get("Answer"),
        "answer_type": data.get("AnswerType"),
        "definition": data.get("Definition"),
        "definition_url": data.get("DefinitionURL"),
        "entity": data.get("Entity"),
        "image": data.get("Image"),
        "official_url": data.get("OfficialDomain"),
        "related_topics": data.get("RelatedTopics", []),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def extract_bang_redirect(html: str) -> Optional[str]:
    match = re.search(r"url=([^'\"]+)", html, flags=re.IGNORECASE)
    if not match:
        return None
    target = match.group(1)
    return urllib.parse.urljoin("https://duckduckgo.com/", target)


def resolve_final_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    if uddg := query.get("uddg"):
        return uddg[0]
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; ddg-search-skill/0.1.0)"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.geturl()


def run_bang(args: argparse.Namespace) -> int:
    bang = args.bang.strip()
    if not bang.startswith("!"):
        bang = f"!{bang}"
    bang_query = f"{bang} {args.query.strip()}".strip()
    url = "https://duckduckgo.com/?q=" + urllib.parse.quote(bang_query)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; ddg-search-skill/0.1.0)"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        html = resp.read().decode("utf-8", "ignore")

    intermediate = extract_bang_redirect(html)
    final_url = resolve_final_url(intermediate) if intermediate else None
    payload = {
        "bang": bang,
        "query": args.query.strip(),
        "bang_query": bang_query,
        "duckduckgo_url": url,
        "redirect_url": intermediate,
        "final_url": final_url,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_help(args: argparse.Namespace) -> int:
    payload = {
        "tool": "ddg-search",
        "purpose": "DuckDuckGo search wrapper for fallback web search and secondary-source checking.",
        "defaults": {
            "type": "text",
            "max_results": 5,
            "region": "wt-wt",
            "safesearch": "moderate",
            "output": "json",
            "page": 1,
            "backend": "auto",
        },
        "notes": [
            "JSON is the default output so agents can parse results directly.",
            "Use --text when a readable terminal view is better than JSON.",
            "Use --site for site-scoped search without manually adding site: prefixes.",
            "Use instant for DuckDuckGo instant-answer style lookups.",
            "Use bang to resolve a DuckDuckGo bang into its final destination URL.",
            "This wrapper is intentionally small and focused.",
            "Use it as a fallback or second search source, not as heavy browser automation.",
        ],
        "commands": {
            "search": "Run DuckDuckGo search. Supports text, news, images, and videos.",
            "instant": "Fetch DuckDuckGo instant-answer style data for a query.",
            "bang": "Resolve a DuckDuckGo bang query to its final destination URL.",
            "help": "Show this command summary.",
        },
        "examples": [
            "./scripts/ddg-search search 'openclaw github'",
            "./scripts/ddg-search search 'bitcoin etf' --type news --timelimit d --max-results 10",
            "./scripts/ddg-search search 'python dataclasses' --site docs.python.org",
            "./scripts/ddg-search search 'openai launch' --type videos --timelimit w --max-results 10 --reverse",
            "./scripts/ddg-search instant 'weather berlin'",
            "./scripts/ddg-search bang w 'OpenAI'",
            "./scripts/ddg-search search 'site:docs.openclaw.ai browser' --text",
        ],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Small DuckDuckGo search CLI with JSON output by default."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="Run DuckDuckGo search")
    search.add_argument("query", help="Search query")
    search.add_argument("--type", choices=["text", "news", "images", "videos"], default="text")
    search.add_argument("--max-results", type=int, default=5)
    search.add_argument("--region", default="wt-wt")
    search.add_argument("--safesearch", default="moderate")
    search.add_argument("--unsafe", action="store_true", help="Shortcut for --safesearch off")
    search.add_argument("--timelimit", default=None)
    search.add_argument("--site", help="Restrict search to one site, like docs.python.org")
    search.add_argument("--page", type=int, default=1)
    search.add_argument("--backend", default="auto", help="ddgs backend selection, default auto")
    search.add_argument("--reverse", action="store_true", help="Reverse the result order after fetch")
    search.add_argument("--text", action="store_true", help="Return readable text output instead of JSON")

    instant = sub.add_parser("instant", help="Fetch DuckDuckGo instant-answer style data")
    instant.add_argument("query", help="Instant-answer query")

    bang = sub.add_parser("bang", help="Resolve a DuckDuckGo bang query to its final URL")
    bang.add_argument("bang", help="Bang token like w, gh, so, yt or !w")
    bang.add_argument("query", help="Bang search query")

    sub.add_parser("help", help="Show a descriptive summary of commands, defaults, and examples")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "search":
            return run_search(args)
        if args.command == "instant":
            return run_instant(args)
        if args.command == "bang":
            return run_bang(args)
        if args.command == "help":
            return cmd_help(args)
        parser.error("unknown command")
        return 2
    except KeyboardInterrupt:
        eprint("Interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
