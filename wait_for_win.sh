#!/usr/bin/env bash
while true; do
  if grep -q "Checkmate (Won)" marathon.log; then
    echo "First Checkmate (Won) reached!"
    tail -n 25 marathon.log
    exit 0
  fi
  
  # Fail-safe condition just in case it runs out of episodes
  if grep -q "Episode 110" marathon.log; then
      echo "Reached episode 110. Exiting wait script."
      tail -n 10 marathon.log
      exit 0
  fi
  sleep 5
done
