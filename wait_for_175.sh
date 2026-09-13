#!/usr/bin/env bash
while true; do
  if grep -q "manoid_grandmaster_ckpt_175.pt" marathon.log; then
    echo "Checkpoint 175 reached!"
    exit 0
  fi
  sleep 5
done
