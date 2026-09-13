#!/usr/bin/env bash
echo "=== Recent Checkpoints ==="
grep -A 3 "Checkpoint Saved" marathon.log || echo "No checkpoints found yet."
echo ""
echo "=== Last 15 Lines of Log ==="
tail -n 15 marathon.log
