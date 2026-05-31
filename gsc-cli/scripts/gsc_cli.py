#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


APP_NAME = "Google Search Console CLI (gsc-cli)"
APP_DESCRIPTION = (
    "Headless-friendly CLI wrapper for the official Google Search Console API: "
    "sites, search analytics, sitemaps, and URL Inspection."
)
CONFIG_DIR = os.path.expanduser(os.environ.get("GSC_CLI_CONFIG_DIR", "~/.config/gsc-cli"))
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")
TOKEN_URL = "https://oauth2.googleapis.com/token"
DEVICE_CODE_URL = "https://oauth2.googleapis.com/device/code"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
WEBMASTERS_BASE = "https://www.googleapis.com/webmasters/v3"
INSPECTION_URL = "https://searchconsole.googleapis.com/v1/urlInspection/index:inspect"
READONLY_SCOPE = "https://www.googleapis.com/auth/webmasters.readonly"
FULL_SCOPE = "https://www.googleapis.com/auth/webmasters"


class CliError(Exception):
    pass


def print_json(data):
    print(json.dumps(data, indent=2, sort_keys=True))


def read_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_config(config):
    os.makedirs(CONFIG_DIR, mode=0o700, exist_ok=True)
    tmp_path = CONFIG_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.chmod(tmp_path, 0o600)
    os.replace(tmp_path, CONFIG_PATH)


def redact_config(config):
    redacted = dict(config)
    for key in ("client_secret", "access_token", "refresh_token"):
        if redacted.get(key):
            redacted[key] = "***"
    return redacted


def http_json(method, url, token=None, body=None, form=None):
    headers = {"Accept": "application/json"}
    data = None
    if token:
        headers["Authorization"] = "Bearer " + token
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif form is not None:
        data = urllib.parse.urlencode(form).encode("utf-8")
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"error": raw}
        raise CliError(json.dumps(payload, indent=2, sort_keys=True))
    except urllib.error.URLError as exc:
        raise CliError(str(exc))


def scope_value(scope_name):
    if scope_name == "readonly":
        return READONLY_SCOPE
    if scope_name == "full":
        return FULL_SCOPE
    if scope_name.startswith("https://"):
        return scope_name
    raise CliError("scope must be 'readonly', 'full', or a full OAuth scope URL")


def encode_path(value):
    return urllib.parse.quote(value, safe="")


def require_config(config, *keys):
    missing = [key for key in keys if not config.get(key)]
    if missing:
        raise CliError(
            "Missing config value(s): "
            + ", ".join(missing)
            + ". Run: gsc-cli auth device --client-id <id>"
        )


def refresh_access_token(config):
    require_config(config, "client_id", "refresh_token")
    form = {
        "client_id": config["client_id"],
        "refresh_token": config["refresh_token"],
        "grant_type": "refresh_token",
    }
    if config.get("client_secret"):
        form["client_secret"] = config["client_secret"]
    token = http_json("POST", TOKEN_URL, form=form)
    config["access_token"] = token["access_token"]
    expires_in = int(token.get("expires_in", 3600))
    config["expires_at"] = int(time.time()) + expires_in - 60
    if token.get("scope"):
        config["granted_scope"] = token["scope"]
    write_config(config)
    return config["access_token"]


def get_access_token():
    config = read_config()
    if os.environ.get("GSC_CLI_ACCESS_TOKEN"):
        return os.environ["GSC_CLI_ACCESS_TOKEN"]
    if config.get("access_token") and int(config.get("expires_at", 0)) > int(time.time()):
        return config["access_token"]
    return refresh_access_token(config)


def auth_device(args):
    config = read_config()
    client_id = args.client_id or os.environ.get("GSC_CLI_CLIENT_ID") or config.get("client_id")
    client_secret = (
        args.client_secret
        or os.environ.get("GSC_CLI_CLIENT_SECRET")
        or config.get("client_secret")
    )
    if not client_id:
        raise CliError("Missing --client-id. Create an OAuth client, then run auth device again.")
    requested_scope = scope_value(args.scope)
    device = http_json(
        "POST",
        DEVICE_CODE_URL,
        form={"client_id": client_id, "scope": requested_scope},
    )
    print_json(
        {
            "message": "Open the verification URL in your browser and enter the user code.",
            "verification_url": device.get("verification_url"),
            "user_code": device.get("user_code"),
            "expires_in": device.get("expires_in"),
            "scope": requested_scope,
        }
    )
    interval = int(device.get("interval", 5))
    deadline = time.time() + int(device.get("expires_in", 1800))
    form = {
        "client_id": client_id,
        "device_code": device["device_code"],
        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
    }
    if client_secret:
        form["client_secret"] = client_secret
    while time.time() < deadline:
        time.sleep(interval)
        try:
            token = http_json("POST", TOKEN_URL, form=form)
            break
        except CliError as exc:
            text = str(exc)
            if "authorization_pending" in text:
                continue
            if "slow_down" in text:
                interval += 5
                continue
            raise
    else:
        raise CliError("OAuth device code expired before authorization completed.")
    config.update(
        {
            "client_id": client_id,
            "scope": requested_scope,
            "access_token": token["access_token"],
            "refresh_token": token.get("refresh_token", config.get("refresh_token")),
            "expires_at": int(time.time()) + int(token.get("expires_in", 3600)) - 60,
        }
    )
    if client_secret:
        config["client_secret"] = client_secret
    if token.get("scope"):
        config["granted_scope"] = token["scope"]
    write_config(config)
    print_json({"ok": True, "config_path": CONFIG_PATH, "scope": requested_scope})


def extract_code(value):
    value = value.strip()
    if value.startswith("http://") or value.startswith("https://"):
        parsed = urllib.parse.urlparse(value)
        params = urllib.parse.parse_qs(parsed.query)
        if params.get("error"):
            raise CliError("OAuth returned error: " + params["error"][0])
        codes = params.get("code")
        if not codes:
            raise CliError("No code= parameter found in pasted redirect URL.")
        return codes[0]
    return value


def auth_url(args):
    config = read_config()
    client_id = args.client_id or os.environ.get("GSC_CLI_CLIENT_ID") or config.get("client_id")
    client_secret = (
        args.client_secret
        or os.environ.get("GSC_CLI_CLIENT_SECRET")
        or config.get("client_secret")
    )
    if not client_id:
        raise CliError("Missing --client-id. Create a Desktop OAuth client, then run auth url again.")
    requested_scope = scope_value(args.scope)
    redirect_uri = args.redirect_uri
    state = "gsc-cli"
    query = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": requested_scope,
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
        }
    )
    url = AUTH_URL + "?" + query
    print_json(
        {
            "message": (
                "Open auth_url in your browser. It will likely redirect to localhost and fail; "
                "paste the final redirected URL or just the code value back here."
            ),
            "auth_url": url,
            "redirect_uri": redirect_uri,
            "scope": requested_scope,
        }
    )
    if args.code:
        code = extract_code(args.code)
    else:
        print("Paste final redirected URL or code: ", end="", file=sys.stderr, flush=True)
        code = extract_code(sys.stdin.readline())
    form = {
        "client_id": client_id,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
    }
    if client_secret:
        form["client_secret"] = client_secret
    token = http_json("POST", TOKEN_URL, form=form)
    config.update(
        {
            "client_id": client_id,
            "scope": requested_scope,
            "access_token": token["access_token"],
            "refresh_token": token.get("refresh_token", config.get("refresh_token")),
            "expires_at": int(time.time()) + int(token.get("expires_in", 3600)) - 60,
            "redirect_uri": redirect_uri,
        }
    )
    if client_secret:
        config["client_secret"] = client_secret
    if token.get("scope"):
        config["granted_scope"] = token["scope"]
    write_config(config)
    print_json({"ok": True, "config_path": CONFIG_PATH, "scope": requested_scope})


def auth_status(_args):
    config = read_config()
    out = {
        "config_path": CONFIG_PATH,
        "configured": bool(config),
        "has_client_id": bool(config.get("client_id")),
        "has_client_secret": bool(config.get("client_secret")),
        "has_refresh_token": bool(config.get("refresh_token")),
        "has_access_token": bool(config.get("access_token")),
        "expires_at": config.get("expires_at"),
        "scope": config.get("scope"),
        "granted_scope": config.get("granted_scope"),
    }
    if args_bool(_args, "show_config"):
        out["config"] = redact_config(config)
    print_json(out)


def api_get(path):
    return http_json("GET", WEBMASTERS_BASE + path, token=get_access_token())


def api_post(path, body):
    return http_json("POST", WEBMASTERS_BASE + path, token=get_access_token(), body=body)


def api_put(path):
    return http_json("PUT", WEBMASTERS_BASE + path, token=get_access_token())


def api_delete(path):
    return http_json("DELETE", WEBMASTERS_BASE + path, token=get_access_token())


def parse_json_body(value):
    if not value:
        return {}
    if value.startswith("@"):
        with open(value[1:], "r", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(value)


def parse_csv(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_filter(value):
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise CliError(
            "Filter must be dimension:operator:expression, e.g. page:contains:/blog/"
        )
    return {"dimension": parts[0], "operator": parts[1], "expression": parts[2]}


def sites_list(_args):
    print_json(api_get("/sites"))


def sites_get(args):
    print_json(api_get("/sites/" + encode_path(args.site)))


def dry_run_or_execute(args, action, payload, func):
    if not args.execute:
        print_json({"dry_run": True, "action": action, "request": payload})
        return
    result = func()
    print_json({"ok": True, "action": action, "response": result})


def sites_add(args):
    path = "/sites/" + encode_path(args.site)
    dry_run_or_execute(args, "sites.add", {"site": args.site}, lambda: api_put(path))


def sites_delete(args):
    path = "/sites/" + encode_path(args.site)
    dry_run_or_execute(args, "sites.delete", {"site": args.site}, lambda: api_delete(path))


def analytics_query(args):
    body = parse_json_body(args.body_json)
    body.setdefault("startDate", args.start_date)
    body.setdefault("endDate", args.end_date)
    dimensions = parse_csv(args.dimensions)
    if dimensions:
        body.setdefault("dimensions", dimensions)
    if args.type:
        body.setdefault("type", args.type)
    if args.aggregation_type:
        body.setdefault("aggregationType", args.aggregation_type)
    if args.row_limit is not None:
        body.setdefault("rowLimit", args.row_limit)
    if args.start_row is not None:
        body.setdefault("startRow", args.start_row)
    filters = [parse_filter(item) for item in args.filter]
    if filters and "dimensionFilterGroups" not in body:
        body["dimensionFilterGroups"] = [{"groupType": "and", "filters": filters}]
    path = "/sites/" + encode_path(args.site) + "/searchAnalytics/query"
    print_json(api_post(path, body))


def sitemaps_list(args):
    path = "/sites/" + encode_path(args.site) + "/sitemaps"
    if args.sitemap_index:
        path += "?sitemapIndex=" + urllib.parse.quote(args.sitemap_index, safe="")
    print_json(api_get(path))


def sitemaps_get(args):
    path = "/sites/" + encode_path(args.site) + "/sitemaps/" + encode_path(args.sitemap)
    print_json(api_get(path))


def sitemaps_submit(args):
    path = "/sites/" + encode_path(args.site) + "/sitemaps/" + encode_path(args.sitemap)
    payload = {"site": args.site, "sitemap": args.sitemap}
    dry_run_or_execute(args, "sitemaps.submit", payload, lambda: api_put(path))


def sitemaps_delete(args):
    path = "/sites/" + encode_path(args.site) + "/sitemaps/" + encode_path(args.sitemap)
    payload = {"site": args.site, "sitemap": args.sitemap}
    dry_run_or_execute(args, "sitemaps.delete", payload, lambda: api_delete(path))


def url_inspect(args):
    body = {
        "inspectionUrl": args.url,
        "siteUrl": args.site,
        "languageCode": args.language_code,
    }
    print_json(http_json("POST", INSPECTION_URL, token=get_access_token(), body=body))


def doctor(_args):
    config = read_config()
    checks = {
        "config_path": CONFIG_PATH,
        "config_exists": os.path.exists(CONFIG_PATH),
        "has_client_id": bool(config.get("client_id")),
        "has_refresh_token": bool(config.get("refresh_token")),
        "scope": config.get("scope"),
    }
    try:
        token = get_access_token()
        checks["can_refresh_or_use_token"] = bool(token)
        sites = http_json("GET", WEBMASTERS_BASE + "/sites", token=token)
        checks["sites_list_ok"] = True
        checks["site_count"] = len(sites.get("siteEntry", []))
    except Exception as exc:
        checks["sites_list_ok"] = False
        checks["error"] = str(exc)
    print_json(checks)


def info(_args):
    print_json({"name": APP_NAME, "description": APP_DESCRIPTION})


def help_command(args):
    args.parser.print_help()


def args_bool(args, key):
    return bool(getattr(args, key, False))


def valid_date(value):
    try:
        dt.date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError("date must be YYYY-MM-DD")
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        prog="gsc-cli",
        description=APP_NAME + " - " + APP_DESCRIPTION,
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("info", help="Print CLI name and description")
    p.set_defaults(func=info)

    p = sub.add_parser("doctor", help="Check config, token refresh, and sites.list")
    p.set_defaults(func=doctor)

    auth = sub.add_parser("auth", help="Authenticate and inspect auth state")
    auth_sub = auth.add_subparsers(dest="auth_command")
    p = auth_sub.add_parser("device", help="Run Google OAuth device flow")
    p.add_argument("--client-id")
    p.add_argument("--client-secret")
    p.add_argument("--scope", default="readonly", help="readonly, full, or a scope URL")
    p.set_defaults(func=auth_device)
    p = auth_sub.add_parser("url", help="Run browser URL OAuth flow for Desktop clients")
    p.add_argument("--client-id")
    p.add_argument("--client-secret")
    p.add_argument("--scope", default="readonly", help="readonly, full, or a scope URL")
    p.add_argument("--redirect-uri", default="http://localhost")
    p.add_argument("--code", help="Authorization code or full redirected URL")
    p.set_defaults(func=auth_url)
    p = auth_sub.add_parser("status", help="Show redacted auth status")
    p.add_argument("--show-config", action="store_true")
    p.set_defaults(func=auth_status)

    sites = sub.add_parser("sites", help="List, get, add, or delete GSC properties")
    sites_sub = sites.add_subparsers(dest="sites_command")
    p = sites_sub.add_parser("list")
    p.set_defaults(func=sites_list)
    p = sites_sub.add_parser("get")
    p.add_argument("--site", required=True)
    p.set_defaults(func=sites_get)
    p = sites_sub.add_parser("add")
    p.add_argument("--site", required=True)
    p.add_argument("--execute", action="store_true")
    p.set_defaults(func=sites_add)
    p = sites_sub.add_parser("delete")
    p.add_argument("--site", required=True)
    p.add_argument("--execute", action="store_true")
    p.set_defaults(func=sites_delete)

    analytics = sub.add_parser("analytics", help="Query Search Analytics")
    analytics_sub = analytics.add_subparsers(dest="analytics_command")
    p = analytics_sub.add_parser("query")
    p.add_argument("--site", required=True)
    p.add_argument("--start-date", required=True, type=valid_date)
    p.add_argument("--end-date", required=True, type=valid_date)
    p.add_argument("--dimensions", default="")
    p.add_argument("--type", default="web")
    p.add_argument("--filter", action="append", default=[])
    p.add_argument("--aggregation-type")
    p.add_argument("--row-limit", type=int)
    p.add_argument("--start-row", type=int)
    p.add_argument("--body-json", default="")
    p.set_defaults(func=analytics_query)

    sitemaps = sub.add_parser("sitemaps", help="List, get, submit, or delete sitemaps")
    sitemaps_sub = sitemaps.add_subparsers(dest="sitemaps_command")
    p = sitemaps_sub.add_parser("list")
    p.add_argument("--site", required=True)
    p.add_argument("--sitemap-index")
    p.set_defaults(func=sitemaps_list)
    p = sitemaps_sub.add_parser("get")
    p.add_argument("--site", required=True)
    p.add_argument("--sitemap", required=True)
    p.set_defaults(func=sitemaps_get)
    p = sitemaps_sub.add_parser("submit")
    p.add_argument("--site", required=True)
    p.add_argument("--sitemap", required=True)
    p.add_argument("--execute", action="store_true")
    p.set_defaults(func=sitemaps_submit)
    p = sitemaps_sub.add_parser("delete")
    p.add_argument("--site", required=True)
    p.add_argument("--sitemap", required=True)
    p.add_argument("--execute", action="store_true")
    p.set_defaults(func=sitemaps_delete)

    url = sub.add_parser("url", help="Inspect URL index status")
    url_sub = url.add_subparsers(dest="url_command")
    p = url_sub.add_parser("inspect")
    p.add_argument("--site", required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--language-code", default="en-US")
    p.set_defaults(func=url_inspect)

    p = sub.add_parser("help", help="Show help")
    p.set_defaults(func=help_command, parser=parser)
    return parser


def main(argv):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    try:
        args.func(args)
        return 0
    except CliError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
