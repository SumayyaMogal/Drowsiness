"""
Quick sanity check of extracted features (run after the 2-video benchmark).

    python check_features.py                 # checks every .npz in ./features
    python check_features.py features/01_alert.npz --start 60 --seconds 30

Prints face-found rate and EAR statistics, and saves ear_check.png (open it and look for blinks as dips).
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="*")
ap.add_argument("--dir", default="features")
ap.add_argument("--start", type=float, default=60.0, help="start of the plotted segment (s)")
ap.add_argument("--seconds", type=float, default=30.0)
a = ap.parse_args()

files = a.files or sorted(glob.glob(os.path.join(a.dir, "*.npz")))
if not files:
    raise SystemExit("No .npz files found. Run extract_features.py first.")

fig, axes = plt.subplots(len(files), 1, figsize=(12, 3 * len(files)), squeeze=False)
for f, ax in zip(files, axes[:, 0]):
    d = np.load(f)
    cols = list(d["cols"]); X = d["X"]; t = d["t"]
    found = X[:, cols.index("face_found")].mean()
    ear = np.nanmean(X[:, [cols.index("ear_l"), cols.index("ear_r")]], axis=1)
    ok = np.isfinite(ear)
    name = os.path.basename(f)
    print(f"{name}: {len(t)} frames, {t[-1] / 60:.1f} min, face found {found:.1%}")
    if ok.sum() == 0:
        print("   -> NO FACE DETECTED in this video. Do not run the full extraction yet.")
        continue
    p10, p50, p90 = np.nanpercentile(ear, [10, 50, 90])
    print(f"   EAR p10/median/p90 = {p10:.3f} / {p50:.3f} / {p90:.3f}   (open eyes ~0.25-0.35, closed ~0.05-0.15)")
    m = (t >= a.start) & (t < a.start + a.seconds)
    ax.plot(t[m], ear[m]); ax.set_title(f"{name}  (t = {a.start:.0f}-{a.start + a.seconds:.0f} s)")
    ax.set_ylabel("EAR"); ax.set_xlabel("time (s)")
plt.tight_layout()
plt.savefig("ear_check.png", dpi=110)
print("\nSaved ear_check.png - open it and check that blinks appear as sharp dips.")
