echo "1. Repo state vs GitHub:"
git status
git log --oneline -15
git diff --stat HEAD

echo ""
echo "2. Checkpoint files actually on disk:"
find . -iname "*.pt" -exec ls -la {} \;
for f in $(find . -iname "*.pt"); do
  python3 -c "import torch; sd=torch.load('$f', map_location='cpu'); print('$f'); [print(k, tuple(v.shape)) for k,v in sd.items()]"
done

echo ""
echo "3. Which checkpoint uci_manoid.py actually points to, and whether it exists:"
grep -n CHECKPOINT_PATH uci_manoid.py
ls -la manoid_grandmaster_ckpt_500.pt manoid_grandmaster_ckpt_600.pt 2>&1

echo ""
echo "4. Dependencies actually installed:"
python3 -c "import torch, chess; print(torch.__version__, chess.__version__)"
which stockfish
stockfish --version 2>&1 | head -2

echo ""
echo "5. A real, live UCI smoke test (not a log file — run it now):"
printf "uci\nisready\nposition startpos\ngo\nquit\n" | timeout 60 python3 uci_manoid.py

echo ""
echo "6. Most recent real training log, freshest first:"
ls -lat *.log | head -5
LATEST_LOG=$(ls -t *.log | head -1)
tail -40 "$LATEST_LOG"

echo ""
echo "7. Any marathon/background process currently or previously running:"
ps aux | grep -i marathon
cat state.json 2>&1
