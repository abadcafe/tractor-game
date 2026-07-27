#!/usr/bin/env bash
set -euo pipefail

exec python -m server.web --host 127.0.0.1 --port 8000
