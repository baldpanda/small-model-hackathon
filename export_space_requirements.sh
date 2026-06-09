#!/usr/bin/env bash

set -euo pipefail

tmp_file="$(mktemp)"
trap 'rm -f "$tmp_file"' EXIT

uv export \
  --format requirements-txt \
  --no-hashes \
  --no-annotate \
  --no-header \
  --output-file "$tmp_file"

grep -v '^torch==' "$tmp_file" > requirements.txt
printf 'torch==2.10.0\n' >> requirements.txt
