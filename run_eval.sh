#!/usr/bin/env bash
# Run eval.py against a named answering model.
#
# The two backends differ only in base_url / model / key, so the corpus, retrieval
# settings and prompts stay identical — which is what makes the numbers comparable.
#
#   ./run_eval.sh deepseek                    # scope=covered (44 questions)
#   ./run_eval.sh gemma --scope labor          # original 9-question subset
#   ./run_eval.sh gemma --limit 2              # smoke test
#
# gemma is self-hosted vLLM on spark-alpha, reached through an SSH tunnel:
#   ssh -N -L 18000:localhost:8000 spark-alpha
# It is unauthenticated, but the OpenAI client rejects an empty key, hence EMPTY.
set -euo pipefail

target="${1:?usage: ./run_eval.sh [deepseek|gemma] [extra eval.py args...]}"
shift || true

case "$target" in
  deepseek)
    export OPENLAW_BASE_URL="https://openrouter.ai/api/v1"
    export OPENLAW_MODEL="deepseek/deepseek-v4-pro"
    ;;
  gemma)
    export OPENLAW_BASE_URL="http://localhost:18000/v1"
    export OPENLAW_MODEL="google/gemma-4-26B-A4B-it"   # case-sensitive: vLLM matches exactly
    export OPENLAW_API_KEY="EMPTY"
    if ! curl -sf -m 5 "${OPENLAW_BASE_URL}/models" >/dev/null; then
      echo "No vLLM on ${OPENLAW_BASE_URL} — is the tunnel up?" >&2
      echo "  ssh -N -L 18000:localhost:8000 spark-alpha" >&2
      exit 1
    fi
    ;;
  *)
    echo "unknown target '$target' (expected deepseek or gemma)" >&2
    exit 1
    ;;
esac

echo "==> ${OPENLAW_MODEL} via ${OPENLAW_BASE_URL}"
exec uv run python eval.py --model "$OPENLAW_MODEL" --base-url "$OPENLAW_BASE_URL" "$@"
