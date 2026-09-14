import re, math

log = open("/data/data/com.termux/files/home/manoid/smoke_test_adaptive_weights.log").read()

raw_pat = re.compile(r"Loss \[raw\]\s+\| V: ([\d.]+) \| P: ([\d.]+) \| D: ([\d.]+) \| Dn: ([\d.]+) \| PV: ([\d.]+)")
elo_pat = re.compile(r"Elo: (\d+)")

# Parse properly
rows = []
for m in raw_pat.finditer(log):
    rows.append(tuple(float(m.group(i)) for i in range(1,6)))

elos = [int(m.group(1)) for m in elo_pat.finditer(log)]

avg = lambda xs: sum(xs)/len(xs) if xs else 0

h1 = rows[:25]; h2 = rows[25:]
rand = math.log(64)*2

print("=" * 62)
print("  BEFORE vs AFTER: Policy/PV Loss")
print("=" * 62)
print(f"\n  Random-init baseline (ln64×2): {rand:.4f}")
print()
print(f"  BEFORE (fixed weights, stability smoke, 45 eps):")
print(f"    policy_loss avg = 8.0315   dist from random = -0.2863")
print(f"    pv_loss     avg = 8.1660   dist from random = -0.1518")
print()
if rows:
    pl_all  = [r[1] for r in rows]
    pvl_all = [r[4] for r in rows]
    pl_h1   = [r[1] for r in h1]; pl_h2  = [r[1] for r in h2]
    pvl_h1  = [r[4] for r in h1]; pvl_h2 = [r[4] for r in h2]
    print(f"  AFTER (adaptive weights, 50 eps):")
    print(f"    policy_loss avg = {avg(pl_all):.4f}   dist from random = {avg(pl_all)-rand:+.4f}")
    print(f"    pv_loss     avg = {avg(pvl_all):.4f}   dist from random = {avg(pvl_all)-rand:+.4f}")
    print()
    print(f"  Trend (first 25 eps vs last 25 eps):")
    print(f"    policy: {avg(pl_h1):.4f} → {avg(pl_h2):.4f}   Δ = {avg(pl_h2)-avg(pl_h1):+.4f}")
    print(f"    pv:     {avg(pvl_h1):.4f} → {avg(pvl_h2):.4f}   Δ = {avg(pvl_h2)-avg(pvl_h1):+.4f}")
    print()
    print(f"  Policy improvement vs before baseline:")
    print(f"    8.0315 → {avg(pl_h2):.4f} (using 2nd half)   Δ = {avg(pl_h2)-8.0315:+.4f}")
    print()

print("=" * 62)
print("  Elo Trend")
print("=" * 62)
if elos:
    for i, e in enumerate(elos[::10], 0):
        print(f"    ep {(i)*10+1}: {e}")
    print(f"    ep 50: {elos[-1]}")
    print(f"    Start: {elos[0]}  →  End: {elos[-1]}   Δ = {elos[-1]-elos[0]:+d}")

