#!/usr/bin/env bash
# Placeholder uploader. Reads the bucket and prefix, then stops.
# It does not call aws, and it does not read or print access keys.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

strip_quotes() {
  local value="$1"
  value="${value%\"}"
  value="${value#\"}"
  value="${value%\'}"
  value="${value#\'}"
  printf '%s' "$value"
}

bucket="$(awk '/^s3_bucket:/ {print $2; exit}' config/storage.yaml)"
prefix="$(awk '/^s3_prefix:/ {print $2; exit}' config/storage.yaml)"
bucket="$(strip_quotes "${S3_BUCKET:-$bucket}")"
prefix="$(strip_quotes "${S3_PREFIX:-$prefix}")"

if [[ -z "$bucket" || -z "$prefix" || "$bucket" == "TODO" || "$prefix" == "TODO" ]]; then
  echo "S3 is not configured. Set s3_bucket and s3_prefix in config/storage.yaml, or export S3_BUCKET and S3_PREFIX. Nothing was uploaded." >&2
  exit 1
fi

echo "S3 upload is not implemented yet. Nothing was uploaded." >&2
echo "Configured destination: s3://${bucket}/${prefix}/" >&2
echo "Spark path would be: s3a://${bucket}/${prefix}/" >&2
echo "Local data directory: ${ROOT}/data/" >&2
exit 1
