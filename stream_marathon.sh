#!/bin/bash
echo "=== Live Telemetry Stream Started (Press Ctrl+C to exit) ==="
tail -n 20 -f marathon.log | sed --unbuffered \
  -e 's/Checkmate (Won)/\x1b[1;32mCheckmate (Won)\x1b[0m/g' \
  -e 's/Draw\/Stalemate/\x1b[1;33mDraw\/Stalemate\x1b[0m/g' \
  -e 's/TRAIN/\x1b[1;31mTRAIN\x1b[0m/g'
