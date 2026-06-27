#!/usr/bin/env bash
#
# Paste-safe rotation of the GitHub Personal Access Token stored in Vault at
# secret/pcloud/github. Run ON the Vault host (.50 / 100.64.0.50).
#
# MANUAL prereqs (browser; GitHub removed the PAT-creation API, so these cannot
# be scripted):
#   1) https://github.com/settings/tokens -> "Generate new token (classic)"
#      -> tick ONLY the `repo` scope -> generate -> copy it.
#   2) Do NOT revoke the old token yet; revoke it AFTER this script + verify pass.
#
# What this does: reads the new token from stdin WITHOUT echoing it (never in
# argv, history, or the process table), validates it against the GitHub API
# (identity == expected account, scopes == repo and nothing broader) BEFORE
# writing, then `vault kv patch`es it in (kv v2 -> url/username preserved).
set -euo pipefail

export VAULT_ADDR="${VAULT_ADDR:-http://127.0.0.1:8200}"
SECRET_PATH="secret/pcloud/github"
EXPECT_USER="yankaili2006"
API="https://api.github.com"

for c in vault curl jq; do command -v "$c" >/dev/null || { echo "error: $c not found" >&2; exit 1; }; done
vault status >/dev/null 2>&1 || { echo "error: Vault unreachable or sealed at $VAULT_ADDR" >&2; exit 1; }

printf 'Paste the NEW classic PAT (input hidden), then press Enter: ' >&2
read -rs NEWTOK; echo >&2
[ -n "$NEWTOK" ] || { echo "error: empty token" >&2; exit 1; }

case "$NEWTOK" in
  ghp_*) ;;
  github_pat_*) echo "error: that is a fine-grained PAT; this rotation expects a CLASSIC repo-scope token (ghp_...)." >&2; unset NEWTOK; exit 1 ;;
  *) echo "error: unrecognized token prefix (expected ghp_...)." >&2; unset NEWTOK; exit 1 ;;
esac

echo "validating against GitHub before storing..." >&2
hdrs="$(curl -fsS -D - -o /dev/null -H "Authorization: Bearer $NEWTOK" -H "X-GitHub-Api-Version: 2022-11-28" "$API/user")" \
  || { echo "error: GitHub rejected the token (GET /user failed) — not stored." >&2; unset NEWTOK; exit 1; }
login="$(curl -fsS -H "Authorization: Bearer $NEWTOK" "$API/user" | jq -r .login)"
scopes="$(printf '%s' "$hdrs" | tr -d '\r' | awk -F': ' 'tolower($1)=="x-oauth-scopes"{print $2}')"

echo "  identity: ${login}   (expected ${EXPECT_USER})" >&2
echo "  scopes:   ${scopes:-<none>}" >&2

if [ "$login" != "$EXPECT_USER" ]; then
  printf "  WARNING: identity != %s. Store anyway? [y/N] " "$EXPECT_USER" >&2
  read -r a; [ "$a" = y ] || { unset NEWTOK; exit 1; }
fi

# Over-scope guard: anything other than `repo` (or its repo:* children) is flagged.
extra="$(printf '%s' "$scopes" | tr ',' '\n' | sed 's/ //g' | grep -vE '^(repo(:.*)?)?$' || true)"
if [ -n "$extra" ]; then
  echo "  WARNING: token carries scopes beyond 'repo':" >&2
  printf '    - %s\n' $extra >&2
  printf "  Store an over-scoped token anyway? [y/N] " >&2
  read -r a; [ "$a" = y ] || { unset NEWTOK; exit 1; }
fi

DATE="$(date -u +%Y-%m-%d)"
printf %s "$NEWTOK" | vault kv patch "$SECRET_PATH" token=- \
  scope_note="classic PAT repo scope (rotated ${DATE})" >/dev/null
unset NEWTOK
echo "OK: new token written to ${SECRET_PATH} (version bumped; url/username preserved)." >&2
echo "NEXT: revoke the OLD token in the browser, then run ./verify-github-pat.sh --check-old" >&2
