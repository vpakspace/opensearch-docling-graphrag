#!/usr/bin/env bash
# Pull required Ollama models for the RAG pipeline
set -euo pipefail

OLLAMA_HOST="${OLLAMA_BASE_URL:-http://localhost:11434}"

echo "Pulling LLM model: qwen2.5:7b ..."
curl -sf "${OLLAMA_HOST}/api/pull" -d '{"name":"qwen2.5:7b"}' | while read -r line; do
    status=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || true)
    [ -n "$status" ] && echo "  $status"
done

echo "Pulling embedding model: nomic-embed-text-v2-moe ..."
curl -sf "${OLLAMA_HOST}/api/pull" -d '{"name":"nomic-embed-text-v2-moe"}' | while read -r line; do
    status=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || true)
    [ -n "$status" ] && echo "  $status"
done

echo "Done. Models available:"
curl -sf "${OLLAMA_HOST}/api/tags" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for m in data.get('models', []):
    print(f\"  - {m['name']} ({m.get('size', 0) // 1_000_000} MB)\")
"
