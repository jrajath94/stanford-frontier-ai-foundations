#!/bin/bash
for i in $(seq 1 15); do
  echo "=== attempt $i $(date -u +%H:%M:%S)"
  out=$(python3 ~/workspace/skills/github/bin/gh_publish.py push-dir stanford-frontier-ai-foundations ~/workspace/companions/push-staging/f1 2>&1)
  code=$?
  echo "$out" | tail -5
  if [ $code -eq 0 ]; then echo "PUSH SUCCEEDED on attempt $i"; exit 0; fi
  sleep 8
done
echo "ALL ATTEMPTS FAILED"; exit 1
