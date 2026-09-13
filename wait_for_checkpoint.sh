#!/bin/bash
while true; do
  if grep -q "--- Checkpoint Saved: manoid_grandmaster_ckpt_25.pt ---" marathon.log; then
    echo "Checkpoint 25 reached!"
    grep -A 4 "--- Checkpoint Saved: manoid_grandmaster_ckpt_25.pt ---" marathon.log
    exit 0
  fi
  sleep 10
done
