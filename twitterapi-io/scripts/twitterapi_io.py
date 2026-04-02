#!/usr/bin/env python3
import argparse
import json
import os
import re
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

CONFIG_DIR = Path.home() / ".config" / "twitterapi-io"
CONFIG_PATH = CONFIG_DIR / "config.json"
BASE_URL = "https://api.twitterapi.io"
DEFAULT_UA = "twitterapi-io-cli/0.1.0 (+https://github.com/ropl-btc/twitterapi-io-cli)"


def chmod_600(path: Path) -> None:
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def load_raw_config() -> dict[str, Any]:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


def save_config(data: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n")
    chmod_600(CONFIG_PATH)


def get_api_key() -> str:
    data = load_raw_config()
    key = os.getenv("TWITTERAPI_IO_KEY") or data.get("api_key")
    if not key:
        raise SystemExit("Missing TWITTERAPI_IO_KEY. Set the env var or run the local auth command with --api-key.")
    return key


def curl_json(path: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    api_key = get_api_key()
    url = BASE_URL + path
    if params:
        encoded = urlencode({k: v for k, v in params.items() if v is not None and v != ""})
        if encoded:
            url += "?" + encoded
    cmd = [
        "curl",
        "-sS",
        "--fail-with-body",
        "-H",
        f"x-api-key: {api_key}",
        "-H",
        "Accept: application/json",
        "-H",
        f"User-Agent: {DEFAULT_UA}",
        url,
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=60)
    return json.loads(out.decode("utf-8"))


def extract_tweet_id(value: str) -> str:
    value = value.strip()
    if value.isdigit():
        return value
    m = re.search(r"/status/(\d+)", value)
    if m:
        return m.group(1)
    raise SystemExit("Could not extract tweet id from input")


def extract_user_id(value: str) -> str:
    value = value.strip()
    if value.isdigit():
        return value
    raise SystemExit("Could not extract numeric user id from input")


def pick_photo_url(tweet: dict[str, Any]) -> Optional[str]:
    ext = tweet.get("extendedEntities") or {}
    media = ext.get("media") or []
    for item in media:
        if item.get("type") == "photo" and item.get("media_url_https"):
            return item.get("media_url_https")
    return None


def minimal_tweet(tweet: dict[str, Any]) -> dict[str, Any]:
    author = tweet.get("author") or {}
    return {
        "tweet_id": tweet.get("id"),
        "author": author.get("userName"),
        "author_name": author.get("name"),
        "content": tweet.get("text"),
        "created_at": tweet.get("createdAt"),
        "like_count": tweet.get("likeCount"),
        "reply_count": tweet.get("replyCount"),
        "retweet_count": tweet.get("retweetCount"),
        "quote_count": tweet.get("quoteCount"),
        "view_count": tweet.get("viewCount"),
        "image": pick_photo_url(tweet),
        "url": tweet.get("url"),
    }


def minimal_user(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": user.get("id"),
        "username": user.get("userName"),
        "name": user.get("name"),
        "description": user.get("description"),
        "location": user.get("location"),
        "followers": user.get("followers"),
        "following": user.get("following"),
        "statuses_count": user.get("statusesCount"),
        "media_count": user.get("mediaCount"),
        "is_blue_verified": user.get("isBlueVerified"),
        "verified_type": user.get("verifiedType"),
        "url": user.get("url"),
    }


def collect_tweets_from_page(page: dict[str, Any], *keys: str) -> list[dict[str, Any]]:
    page_data = page.get("data") or {}
    for key in keys:
        if page.get(key):
            return page.get(key) or []
        if page_data.get(key):
            return page_data.get(key) or []
    return []


def collect_users_from_page(page: dict[str, Any], *keys: str) -> list[dict[str, Any]]:
    page_data = page.get("data") or {}
    for key in keys:
        if page.get(key):
            return page.get(key) or []
        if page_data.get(key):
            return page_data.get(key) or []
    return []


def compose_search_query(args: argparse.Namespace) -> str:
    parts = [args.query.strip()]
    if getattr(args, "from_user", None):
        parts.append(f"from:{args.from_user.lstrip('@')}")
    if getattr(args, "within_time", None):
        parts.append(f"within_time:{args.within_time}")
    if getattr(args, "since_time", None):
        parts.append(f"since_time:{args.since_time}")
    if getattr(args, "until_time", None):
        parts.append(f"until_time:{args.until_time}")
    return " ".join(part for part in parts if part).strip()


def print_json(payload: Any) -> int:
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_help(args: argparse.Namespace) -> int:
    payload = {
        "tool": "twitterapi-io",
        "purpose": "Read-only CLI for twitterapi.io to fetch tweets, users, timelines, replies, quotes, thread context, mentions, and advanced search results.",
        "defaults": {
            "user-tweets_limit": 20,
            "replies_limit": 20,
            "quotes_limit": 20,
            "thread-context_limit": 40,
            "mentions_limit": 20,
            "search_queryType": "Latest",
            "search_max_pages": 3,
            "search_max_tweets": 200,
        },
        "notes": [
            "Read-only CLI: no posting, liking, replying, deleting, or write actions are exposed.",
            "API auth uses the x-api-key header, per twitterapi.io docs.",
            "The CLI can read TWITTERAPI_IO_KEY from env or from ~/.config/twitterapi-io/config.json after auth.",
            "Advanced search uses cursor pagination via next_cursor until limits or end of results.",
            "For date filters, prefer working Twitter operators like since_time:, until_time:, and within_time:.",
        ],
        "commands": {
            "auth": "Save API key locally in ~/.config/twitterapi-io/config.json.",
            "tweet": "Fetch one tweet by id or URL.",
            "user": "Fetch one user by username.",
            "user-tweets": "Fetch recent tweets for a user by username or user id.",
            "replies": "Fetch replies for a tweet id or URL.",
            "quotes": "Fetch quote tweets for a tweet id or URL.",
            "thread-context": "Fetch surrounding thread context for a tweet id or URL.",
            "mentions": "Fetch mentions for a username.",
            "search": "Run twitterapi.io advanced tweet search with pagination and optional helper time filters.",
            "help": "Show this command summary.",
        },
        "examples": [
            "./scripts/twitterapi-io auth --api-key YOUR_KEY",
            "./scripts/twitterapi-io tweet --url 'https://x.com/jack/status/20'",
            "./scripts/twitterapi-io user --username OpenAI",
            "./scripts/twitterapi-io user-tweets --username OpenAI --limit 10",
            "./scripts/twitterapi-io replies --url 'https://x.com/jack/status/20' --limit 20",
            "./scripts/twitterapi-io quotes --id 20 --limit 20",
            "./scripts/twitterapi-io thread-context --id 20 --limit 40",
            "./scripts/twitterapi-io mentions --username OpenAI --limit 20",
            "./scripts/twitterapi-io search --query 'AI agents -filter:replies' --from-user OpenAI --within-time 24h --max-tweets 50",
        ],
        "docs": [
            "https://docs.twitterapi.io/introduction",
            "https://docs.twitterapi.io/authentication",
            "https://docs.twitterapi.io/api-reference/endpoint/tweet_advanced_search",
        ],
    }
    return print_json(payload)


def cmd_auth(args: argparse.Namespace) -> int:
    if not args.api_key:
        raise SystemExit("Provide --api-key")
    save_config({"api_key": args.api_key})
    return print_json({"ok": True, "saved_to": str(CONFIG_PATH)})


def cmd_tweet(args: argparse.Namespace) -> int:
    raw = args.id or args.url
    if not raw:
        raise SystemExit("Provide --id or --url")
    tweet_id = extract_tweet_id(raw)
    data = curl_json("/twitter/tweets", {"tweet_ids": tweet_id})
    tweets = data.get("tweets") or []
    if not tweets:
        raise SystemExit("No tweet returned")
    payload = tweets[0] if args.raw else minimal_tweet(tweets[0])
    return print_json(payload)


def cmd_user(args: argparse.Namespace) -> int:
    data = curl_json("/twitter/user/info", {"userName": args.username.lstrip("@")})
    user = data.get("data") or {}
    if not user:
        raise SystemExit("No user returned")
    payload = user if args.raw else minimal_user(user)
    return print_json(payload)


def cmd_user_tweets(args: argparse.Namespace) -> int:
    params = {
        "userName": args.username.lstrip("@") if args.username else None,
        "userId": args.user_id,
        "includeReplies": "true" if args.include_replies else "false",
    }
    all_tweets = []
    seen_ids = set()
    cursor = None
    pages = 0
    while len(all_tweets) < args.limit:
        page_params = dict(params)
        if cursor:
            page_params["cursor"] = cursor
        page = curl_json("/twitter/user/last_tweets", page_params)
        pages += 1
        tweets = collect_tweets_from_page(page, "tweets")
        new_count = 0
        for tweet in tweets:
            tid = tweet.get("id")
            if not tid or tid in seen_ids:
                continue
            seen_ids.add(tid)
            all_tweets.append(tweet if args.raw else minimal_tweet(tweet))
            new_count += 1
            if len(all_tweets) >= args.limit:
                break
        if not page.get("has_next_page"):
            break
        nxt = page.get("next_cursor")
        if not nxt or new_count == 0:
            break
        cursor = nxt
    out = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "username": args.username.lstrip("@") if args.username else None,
        "user_id": args.user_id,
        "pages": pages,
        "count": len(all_tweets),
        "tweets": all_tweets,
    }
    return print_json(out)


def cmd_search(args: argparse.Namespace) -> int:
    all_tweets = []
    seen_ids = set()
    cursor = None
    pages = 0
    last_page = None
    final_query = compose_search_query(args)
    while pages < args.max_pages and len(all_tweets) < args.max_tweets:
        params = {"query": final_query, "queryType": args.queryType}
        if cursor:
            params["cursor"] = cursor
        page = curl_json("/twitter/tweet/advanced_search", params)
        last_page = page
        pages += 1
        tweets = collect_tweets_from_page(page, "tweets")
        new_count = 0
        for tweet in tweets:
            tid = tweet.get("id")
            if not tid or tid in seen_ids:
                continue
            seen_ids.add(tid)
            all_tweets.append(tweet if args.raw else minimal_tweet(tweet))
            new_count += 1
            if len(all_tweets) >= args.max_tweets:
                break
        if not page.get("has_next_page"):
            break
        nxt = page.get("next_cursor")
        if not nxt or new_count == 0:
            break
        cursor = nxt
    out = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "query": final_query,
        "queryType": args.queryType,
        "pages": pages,
        "count": len(all_tweets),
        "tweets": all_tweets,
    }
    if args.include_raw_last_page:
        out["raw"] = {"last_page": last_page}
    return print_json(out)


def cmd_replies(args: argparse.Namespace) -> int:
    tweet_id = extract_tweet_id(args.id or args.url)
    all_tweets = []
    seen_ids = set()
    cursor = None
    pages = 0
    while len(all_tweets) < args.limit:
        params = {"tweetId": tweet_id}
        if args.since_time:
            params["sinceTime"] = args.since_time
        if args.until_time:
            params["untilTime"] = args.until_time
        if cursor:
            params["cursor"] = cursor
        page = curl_json("/twitter/tweet/replies", params)
        pages += 1
        tweets = collect_tweets_from_page(page, "replies", "tweets")
        new_count = 0
        for tweet in tweets:
            tid = tweet.get("id")
            if not tid or tid in seen_ids:
                continue
            seen_ids.add(tid)
            all_tweets.append(tweet if args.raw else minimal_tweet(tweet))
            new_count += 1
            if len(all_tweets) >= args.limit:
                break
        if not page.get("has_next_page"):
            break
        nxt = page.get("next_cursor")
        if not nxt or new_count == 0:
            break
        cursor = nxt
    return print_json({
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "tweet_id": tweet_id,
        "pages": pages,
        "count": len(all_tweets),
        "replies": all_tweets,
    })


def cmd_quotes(args: argparse.Namespace) -> int:
    tweet_id = extract_tweet_id(args.id or args.url)
    all_tweets = []
    seen_ids = set()
    cursor = None
    pages = 0
    while len(all_tweets) < args.limit:
        params = {"tweetId": tweet_id}
        if cursor:
            params["cursor"] = cursor
        page = curl_json("/twitter/tweet/quotes", params)
        pages += 1
        tweets = collect_tweets_from_page(page, "tweets", "quotes")
        new_count = 0
        for tweet in tweets:
            tid = tweet.get("id")
            if not tid or tid in seen_ids:
                continue
            seen_ids.add(tid)
            all_tweets.append(tweet if args.raw else minimal_tweet(tweet))
            new_count += 1
            if len(all_tweets) >= args.limit:
                break
        if not page.get("has_next_page"):
            break
        nxt = page.get("next_cursor")
        if not nxt or new_count == 0:
            break
        cursor = nxt
    return print_json({
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "tweet_id": tweet_id,
        "pages": pages,
        "count": len(all_tweets),
        "quotes": all_tweets,
    })


def cmd_thread_context(args: argparse.Namespace) -> int:
    tweet_id = extract_tweet_id(args.id or args.url)
    all_tweets = []
    seen_ids = set()
    cursor = None
    pages = 0
    while len(all_tweets) < args.limit:
        params = {"tweetId": tweet_id}
        if cursor:
            params["cursor"] = cursor
        page = curl_json("/twitter/tweet/thread_context", params)
        pages += 1
        tweets = collect_tweets_from_page(page, "tweets", "thread", "thread_context")
        new_count = 0
        for tweet in tweets:
            tid = tweet.get("id")
            if not tid or tid in seen_ids:
                continue
            seen_ids.add(tid)
            all_tweets.append(tweet if args.raw else minimal_tweet(tweet))
            new_count += 1
            if len(all_tweets) >= args.limit:
                break
        if not page.get("has_next_page"):
            break
        nxt = page.get("next_cursor")
        if not nxt or new_count == 0:
            break
        cursor = nxt
    return print_json({
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "tweet_id": tweet_id,
        "pages": pages,
        "count": len(all_tweets),
        "tweets": all_tweets,
    })


def cmd_mentions(args: argparse.Namespace) -> int:
    username = args.username.lstrip("@")
    all_tweets = []
    seen_ids = set()
    cursor = None
    pages = 0
    while len(all_tweets) < args.limit:
        params = {"userName": username}
        if cursor:
            params["cursor"] = cursor
        page = curl_json("/twitter/user/mentions", params)
        pages += 1
        tweets = collect_tweets_from_page(page, "tweets", "mentions")
        new_count = 0
        for tweet in tweets:
            tid = tweet.get("id")
            if not tid or tid in seen_ids:
                continue
            seen_ids.add(tid)
            all_tweets.append(tweet if args.raw else minimal_tweet(tweet))
            new_count += 1
            if len(all_tweets) >= args.limit:
                break
        if not page.get("has_next_page"):
            break
        nxt = page.get("next_cursor")
        if not nxt or new_count == 0:
            break
        cursor = nxt
    return print_json({
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "username": username,
        "pages": pages,
        "count": len(all_tweets),
        "tweets": all_tweets,
    })


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Read-only CLI for twitterapi.io")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("help", help="Show a descriptive summary of commands, defaults, and examples")

    auth = sub.add_parser("auth", help="Save API key locally")
    auth.add_argument("--api-key", required=True)

    tweet = sub.add_parser("tweet", help="Fetch one tweet by id or URL")
    tweet.add_argument("--id")
    tweet.add_argument("--url")
    tweet.add_argument("--raw", action="store_true")

    user = sub.add_parser("user", help="Fetch one user by username")
    user.add_argument("--username", required=True)
    user.add_argument("--raw", action="store_true")

    user_tweets = sub.add_parser("user-tweets", help="Fetch recent tweets for a user by username or user id")
    user_tweets.add_argument("--username")
    user_tweets.add_argument("--user-id")
    user_tweets.add_argument("--limit", type=int, default=20)
    user_tweets.add_argument("--include-replies", action="store_true")
    user_tweets.add_argument("--raw", action="store_true")

    replies = sub.add_parser("replies", help="Fetch replies for a tweet by id or URL")
    replies.add_argument("--id")
    replies.add_argument("--url")
    replies.add_argument("--limit", type=int, default=20)
    replies.add_argument("--since-time", type=int)
    replies.add_argument("--until-time", type=int)
    replies.add_argument("--raw", action="store_true")

    quotes = sub.add_parser("quotes", help="Fetch quote tweets for a tweet by id or URL")
    quotes.add_argument("--id")
    quotes.add_argument("--url")
    quotes.add_argument("--limit", type=int, default=20)
    quotes.add_argument("--raw", action="store_true")

    thread = sub.add_parser("thread-context", help="Fetch surrounding thread context for a tweet by id or URL")
    thread.add_argument("--id")
    thread.add_argument("--url")
    thread.add_argument("--limit", type=int, default=40)
    thread.add_argument("--raw", action="store_true")

    mentions = sub.add_parser("mentions", help="Fetch mentions for a username")
    mentions.add_argument("--username", required=True)
    mentions.add_argument("--limit", type=int, default=20)
    mentions.add_argument("--raw", action="store_true")

    search = sub.add_parser("search", help="Run twitterapi.io advanced tweet search with pagination")
    search.add_argument("--query", required=True)
    search.add_argument("--queryType", default="Latest", choices=["Latest", "Top"])
    search.add_argument("--from-user")
    search.add_argument("--within-time")
    search.add_argument("--since-time", type=int)
    search.add_argument("--until-time", type=int)
    search.add_argument("--max-pages", type=int, default=3)
    search.add_argument("--max-tweets", type=int, default=200)
    search.add_argument("--include-raw-last-page", action="store_true")
    search.add_argument("--raw", action="store_true")

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "help":
        return cmd_help(args)
    if args.command == "auth":
        return cmd_auth(args)
    if args.command == "tweet":
        return cmd_tweet(args)
    if args.command == "user":
        return cmd_user(args)
    if args.command == "user-tweets":
        if not args.username and not args.user_id:
            raise SystemExit("Provide --username or --user-id")
        return cmd_user_tweets(args)
    if args.command == "replies":
        if not args.id and not args.url:
            raise SystemExit("Provide --id or --url")
        return cmd_replies(args)
    if args.command == "quotes":
        if not args.id and not args.url:
            raise SystemExit("Provide --id or --url")
        return cmd_quotes(args)
    if args.command == "thread-context":
        if not args.id and not args.url:
            raise SystemExit("Provide --id or --url")
        return cmd_thread_context(args)
    if args.command == "mentions":
        return cmd_mentions(args)
    if args.command == "search":
        return cmd_search(args)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
