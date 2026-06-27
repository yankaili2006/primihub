#!/usr/bin/env bash
#
# Verify the GitHub PAT currently stored in Vault (secret/pcloud/github). Run ON
# the Vault host (.50). Read-only: confirms identity, scopes (repo, not broader),
# and a real authenticated repo read. With --check-old, also confirms the old
# token is dead (paste it hidden when prompted).
set -euo pipefail

export VAULT_ADDR="${VAULT_ADDR:-http://127.0.0.1:8200}"
SECRET_PATH="secret/pcloud/github"
EXPECT_USER="yankaili2006"
REPO="yankaili2006/primihub"
API="https://api.github.com"

for c in vault curl jq; do command -v "$c" >/dev/null || { echo "error: $c not found" >&2; exit 1; }; done

TOK="$(vault kv get -field=token "$SECRET_PATH")" || { echo "error: cannot read token from Vault" >&2; exit 1; }
hdrs="$(curl -fsS -D - -o /dev/null -H "Authorization: Bearer $TOK" "$API/user")" \
  || { echo "FAIL: stored token rejected by GitHub" >&2; unset TOK; exit 1; }
login="$(curl -fsS -H "Authorization: Bearer $TOK" "$API/user" | jq -r .login)"
scopes="$(printf '%s' "$hdrs" | tr -d '\r' | awk -F': ' 'tolower($1)=="x-oauth-scopes"{print $2}')"
code="$(curl -fsS -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $TOK" "$API/repos/$REPO")"
unset TOK

echo "identity:  ${login}   (expected ${EXPECT_USER})"
echo "scopes:    ${scopes:-<none>}"
echo "repo read: HTTP ${code}   (${REPO})"

extra="$(printf '%s' "$scopes" | tr ',' '\n' | sed 's/ //g' | grep -vE '^(repo(:.*)?)?$' || true)"
[ -z "$extra" ] || { echo "WARNING: over-scoped beyond 'repo':"; printf '  - %s\n' $extra; }

ok=1
[ "$login" = "$EXPECT_USER" ] || { echo "  -> identity mismatch"; ok=0; }
[ "$code" = 200 ] || { echo "  -> repo read not 200"; ok=0; }

if [ "${1:-}" = "--check-old" ]; then
  printf 'Paste the OLD token to confirm it is revoked (hidden), Enter: ' >&2
  read -rs OLD; echo >&2
  if [ -n "$OLD" ]; then
    oc="$(curl -fsS -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $OLD" "$API/user" || true)"
    unset OLD
    if [ "$oc" = 401 ]; then echo "old token: revoked (HTTP 401) ✓"
    else echo "WARNING: old token still returns HTTP ${oc} — REVOKE IT at https://github.com/settings/tokens"; ok=0; fi
  fi
fi

[ "$ok" = 1 ] && echo "PASS" || { echo "CHECK FAILED" >&2; exit 1; }
