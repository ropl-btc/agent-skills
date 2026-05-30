#!/usr/bin/env bash
set -euo pipefail

config_dir="${HOME}/.config/telegram-cli"
config_file="${config_dir}/config.json"

mkdir -p "$config_dir"
chmod 700 "$config_dir"

printf "Telegram API ID: "
IFS= read -r api_id
if [ -z "$api_id" ]; then
  printf "No Telegram API ID entered. Nothing changed.\n" >&2
  exit 1
fi

printf "Telegram API hash: "
IFS= read -r api_hash
if [ -z "$api_hash" ]; then
  printf "No Telegram API hash entered. Nothing changed.\n" >&2
  exit 1
fi

python3 - "$config_file" "$api_id" "$api_hash" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
api_id = int(sys.argv[2])
api_hash = sys.argv[3]

data = {}
if path.exists():
    data = json.loads(path.read_text())

data["api_id"] = api_id
data["api_hash"] = api_hash

path.write_text(json.dumps(data, indent=2) + "\n")
path.chmod(0o600)
PY

printf "Saved Telegram API credentials to %s\n" "$config_file"
printf "Next run: %s/telegram-cli auth\n" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
