"""
Grab a few frames from a video into ONE image so you can see orientation, framing and face visibility.

    python peek_video.py "C:\\RLDD\\Fold1_part1\\Fold1_part1\\01\\0.mov"
    python peek_video.py <video> --times 10 120 300 500

Prints resolution, fps, duration and any rotation metadata, and saves peek_<name>.jpg
"""
import argparse
import os

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--times", type=float, nargs="+", default=[10, 120, 300, 500], help="seconds into the video")
ap.add_argument("--height", type=int, default=360, help="height of each thumbnail")
a = ap.parse_args()

cap = cv2.VideoCapture(a.video)
if not cap.isOpened():
    raise SystemExit(f"Could not open {a.video}")
fps = cap.get(cv2.CAP_PROP_FPS)
n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
rot = cap.get(cv2.CAP_PROP_ORIENTATION_META) if hasattr(cv2, "CAP_PROP_ORIENTATION_META") else float("nan")
dur = n / fps if fps else float("nan")
print(f"{os.path.basename(a.video)}: {int(w)}x{int(h)} px, {fps:.2f} fps, {dur / 60:.1f} min, rotation metadata = {rot}")
print("PORTRAIT frame (taller than wide)" if h > w else "landscape frame")

thumbs = []
for t in a.times:
    if t >= dur:
        continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, f = cap.read()
    if not ok:
        continue
    f = cv2.resize(f, (int(f.shape[1] * a.height / f.shape[0]), a.height))
    cv2.putText(f, f"t={t:.0f}s", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    thumbs.append(f)
cap.release()
if not thumbs:
    raise SystemExit("No frames could be read at those times.")
out = f"peek_{os.path.splitext(os.path.basename(a.video))[0]}.jpg"
cv2.imwrite(out, np.hstack(thumbs))
print("Saved", out)
