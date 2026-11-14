"""
Quick sanity check of extracted features (run after the 2-video benchmark, and again after the full run).

    python check_features.py                              # stats for every .npz in ./features
    python check_features.py features/01_alert.npz --start 60 --seconds 30
    python check_features.py --max_plots 3 --out ear_check.png

For every file it prints: duration, face-found rate, share of NaN values per signal, EAR percentiles,
head-pose ranges, timestamp gaps and a rough blink-dip count. It plots the EAR trace of the first
--max_plots files and saves it to --out (open it and look for blinks as sharp dips).
Exit code is 1 if any file shows a problem, so it can also be used in a script.
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SIGNALS = ["ear_l", "ear_r", "mar", "pitch", "yaw", "roll", "brow_l", "brow_r"]
MIN_FACE_FOUND = 0.80      # warn below this share of frames with a detected face
MAX_GAP_FACTOR = 3.0       # warn if a timestamp gap is this many times the median step


def check_file(path):
    """Print statistics for one .npz file. Returns (problems list, ear array, time array)."""
    problems = []
    name = os.path.basename(path)
    with np.load(path) as d:
        cols = [str(c) for c in d["cols"]]
        X, t = d["X"], d["t"]
    if len(t) == 0:
        print(f"{name}: EMPTY file")
        return [f"{name}: empty"], None, None

    col = {c: i for i, c in enumerate(cols)}
    found = X[:, col["face_found"]].mean()
    ear = np.nanmean(X[:, [col["ear_l"], col["ear_r"]]], axis=1)
    print(f"{name}: {len(t)} frames, {t[-1] / 60:.1f} min, face found {found:.1%}")

    if found < MIN_FACE_FOUND:
        problems.append(f"{name}: face found in only {found:.1%} of frames")
    if not np.isfinite(ear).any():
        print("   -> NO FACE DETECTED in this video. Do not run the full extraction yet.")
        problems.append(f"{name}: no face detected")
        return problems, None, None

    nan_share = {s: float(np.isnan(X[:, col[s]]).mean()) for s in SIGNALS if s in col}
    worst = max(nan_share, key=nan_share.get)
    print(f"   NaN share: worst signal = {worst} ({nan_share[worst]:.1%})")

    p10, p50, p90 = np.nanpercentile(ear, [10, 50, 90])
    print(f"   EAR p10/median/p90 = {p10:.3f} / {p50:.3f} / {p90:.3f}"
          "   (typically open ~0.25-0.35, closed ~0.05-0.15; depends on camera and subject)")

    for ang in ("pitch", "yaw", "roll"):
        v = X[:, col[ang]]
        if np.isfinite(v).any():
            lo, hi = np.nanpercentile(v, [2, 98])
            print(f"   {ang:5s} 2-98% range: {lo:7.1f} .. {hi:7.1f} deg")
            if hi - lo > 200:   # a range this wide usually means angle wrap-around (+-180)
                problems.append(f"{name}: {ang} range {lo:.0f}..{hi:.0f} looks like angle wrap-around")

    dt = np.diff(t)
    if len(dt):
        step = np.median(dt)
        gaps = int((dt > MAX_GAP_FACTOR * step).sum()) if step > 0 else 0
        if gaps:
            problems.append(f"{name}: {gaps} timestamp gap(s) larger than {MAX_GAP_FACTOR:.0f}x the median step")
            print(f"   WARNING: {gaps} large timestamp gap(s)")
        # Rough blink-dip count: frames where EAR falls below 70% of this video's median.
        # This is only a plausibility check, not the blink definition used in the pipeline.
        low = np.isfinite(ear) & (ear < 0.7 * p50)
        dips = int((low & ~np.roll(low, 1)).sum())
        minutes = max((t[-1] - t[0]) / 60.0, 1e-9)
        print(f"   rough EAR-dip count: {dips} ({dips / minutes:.1f} per min)")
    return problems, ear, t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--dir", default="features")
    ap.add_argument("--start", type=float, default=60.0, help="start of the plotted segment (s)")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--max_plots", type=int, default=6, help="plot at most this many files (stats are printed for all)")
    ap.add_argument("--out", default="ear_check.png")
    a = ap.parse_args()

    files = a.files or sorted(f for f in glob.glob(os.path.join(a.dir, "*.npz"))
                              if not f.endswith(".tmp.npz"))   # ignore unfinished temp files
    if not files:
        sys.exit("No .npz files found. Run extract_features.py first.")

    all_problems, plots = [], []
    for f in files:
        try:
            problems, ear, t = check_file(f)
        except Exception as e:
            print(f"{os.path.basename(f)}: could not be read ({type(e).__name__}: {e})")
            all_problems.append(f"{os.path.basename(f)}: unreadable")
            continue
        all_problems += problems
        if ear is not None and len(plots) < a.max_plots:
            plots.append((os.path.basename(f), ear, t))

    if plots:
        fig, axes = plt.subplots(len(plots), 1, figsize=(12, 3 * len(plots)), squeeze=False)
        for (name, ear, t), ax in zip(plots, axes[:, 0]):
            start = a.start
            if start >= t[-1]:                      # video shorter than --start: plot from the beginning
                start = t[0]
            m = (t >= start) & (t < start + a.seconds)
            ax.plot(t[m], ear[m])
            ax.set_title(f"{name}  (t = {start:.0f}-{start + a.seconds:.0f} s)")
            ax.set_ylabel("EAR")
            ax.set_xlabel("time (s)")
        plt.tight_layout()
        plt.savefig(a.out, dpi=110)
        plt.close(fig)
        print(f"\nSaved {a.out} ({len(plots)} plot(s)) - check that blinks appear as sharp dips.")

    print(f"\nChecked {len(files)} file(s).")
    if all_problems:
        print(f"{len(all_problems)} problem(s) found:")
        for p in all_problems:
            print("  -", p)
        sys.exit(1)
    print("No problems found.")


if __name__ == "__main__":
    main()
