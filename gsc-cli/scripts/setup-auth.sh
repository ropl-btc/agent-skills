#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

printf 'Google Search Console CLI OAuth setup\n'
printf 'Create/choose a Google OAuth Desktop client, then paste its values here.\n'
printf 'Client ID: '
IFS= read -r CLIENT_ID
printf 'Client secret (optional, press enter to skip): '
IFS= read -rs CLIENT_SECRET
printf '\n'
printf 'Scope [readonly/full] (default readonly): '
IFS= read -r SCOPE

SCOPE="${SCOPE:-readonly}"

if [ -n "$CLIENT_SECRET" ]; then
  export GSC_CLI_CLIENT_SECRET="$CLIENT_SECRET"
  exec "$SCRIPT_DIR/gsc-cli" auth url --client-id "$CLIENT_ID" --scope "$SCOPE"
fi

exec "$SCRIPT_DIR/gsc-cli" auth url --client-id "$CLIENT_ID" --scope "$SCOPE"
