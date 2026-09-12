#!/usr/bin/env bash
# Mock script to simulate colab execution

RESUME_FROM=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --resume-from)
      RESUME_FROM="$2"
      shift 2
      ;;
    *)
      COMMAND="$1"
      shift
      ;;
  esac
done

if [ -n "$RESUME_FROM" ] && [ -f "$RESUME_FROM" ]; then
    echo "Resuming job from checkpoint: $RESUME_FROM"
    START_STEP=$(cat "$RESUME_FROM")
else
    echo "Starting new job..."
    START_STEP=0
fi

echo "Provisioning Colab instance..."
sleep 1
echo "Executing job..."

for i in $(seq $((START_STEP + 1)) $((START_STEP + 3))); do
    echo "Processing step $i..."
    sleep 1
    if [ -n "$RESUME_FROM" ]; then
        echo "$i" > "$RESUME_FROM"
        echo "Saved checkpoint at step $i to $RESUME_FROM"
    fi
done

echo "Job completed successfully."
