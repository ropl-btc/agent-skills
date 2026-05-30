#!/usr/bin/env bash
set -euo pipefail

config_dir="${HOME}/.config/twitterapi-io"
config_file="${config_dir}/config.json"

mkdir -p "$config_dir"
chmod 700 "$config_dir"

printf "twitterapi.io API key: "
IFS= read -r api_key
if [ -z "$api_key" ]; then
  printf "No API key entered. Nothing changed.\n" >&2
  exit 1
fi

python3 - "$config_file" "$api_key" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
api_key = sys.argv[2]

data = {}
if path.exists():
    data = json.loads(path.read_text())

data["api_key"] = api_key

path.write_text(json.dumps(data, indent=2) + "\n")
path.chmod(0o600)
PY

printf "Saved twitterapi.io API key to %s\n" "$config_file"
printf "Next run: %s/twitterapi-io help\n" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
