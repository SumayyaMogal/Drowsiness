"""
Grab a few frames from a video into ONE image so you can see orientation, framing and face visibility.

    python peek_video.py "C:\\RLDD\\Fold1_part1\\Fold1_part1\\01\\0.mov"
    python peek_video.py <video> --times 10 120 300 500 --cols 2
    python peek_video.py <video> --detect            # also report whether MediaPipe finds a face

Prints resolution, fps, duration and rotation metadata, and saves the image as
peek_<subject>_<video>.jpg (the parent folder name is included because every subject's videos are
called 0, 5 and 10), or to the path given with --out.
"""
import argparse
import math
import os

import cv2
import numpy as np


def build_name(video):
    parent = os.path.basename(os.path.dirname(os.path.abspath(video)))
    stem = os.path.splitext(os.path.basename(video))[0]
    return f"peek_{parent}_{stem}.jpg" if parent else f"peek_{stem}.jpg"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--times", type=float, nargs="+", default=[10, 120, 300, 500], help="seconds into the video")
    ap.add_argument("--height", type=int, default=360, help="height of each thumbnail")
    ap.add_argument("--cols", type=int, default=4, help="thumbnails per row")
    ap.add_argument("--detect", action="store_true", help="run MediaPipe FaceMesh on each frame (needs mediapipe)")
    ap.add_argument("--out", default=None, help="output image path (default: peek_<subject>_<video>.jpg)")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or not math.isfinite(fps) or fps <= 0:
        print("WARNING: fps not reported by the file; assuming 30 (extract_features.py makes the same assumption)")
        fps = 30.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    rot = cap.get(cv2.CAP_PROP_ORIENTATION_META) if hasattr(cv2, "CAP_PROP_ORIENTATION_META") else float("nan")
    auto = cap.get(cv2.CAP_PROP_ORIENTATION_AUTO) if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO") else float("nan")
    dur = n / fps if n and n > 0 else float("nan")   # frame count can be unreliable for some .mov files
    print(f"{os.path.basename(a.video)}: {int(w)}x{int(h)} px, {fps:.2f} fps, {dur / 60:.1f} min, "
          f"rotation metadata = {rot}, auto-rotate = {auto}")
    print("PORTRAIT frame (taller than wide)" if h > w else "landscape frame")

    mesh = None
    if a.detect:
        try:
            import mediapipe as mp
            mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                                   min_detection_confidence=0.5)
        except Exception as e:
            print(f"--detect disabled ({type(e).__name__}: {e})")

    thumbs, skipped = [], []
    for t in a.times:
        if math.isfinite(dur) and t >= dur:
            skipped.append(t)
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, f = cap.read()
        if not ok:
            skipped.append(t)
            continue
        label = f"t={t:.0f}s"
        if mesh is not None:
            found = mesh.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).multi_face_landmarks is not None
            label += "  face: yes" if found else "  face: NO"
        f = cv2.resize(f, (int(f.shape[1] * a.height / f.shape[0]), a.height))
        cv2.putText(f, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        thumbs.append(f)
    cap.release()
    if mesh is not None:
        mesh.close()
    if skipped:
        print("Skipped times (beyond the video or unreadable): " + ", ".join(f"{t:.0f}s" for t in skipped))
    if not thumbs:
        raise SystemExit("No frames could be read at those times.")

    # Arrange thumbnails in a grid; pad rows with black so every row has the same width.
    cols = max(1, a.cols)
    rows = [thumbs[i:i + cols] for i in range(0, len(thumbs), cols)]
    width = max(sum(im.shape[1] for im in r) for r in rows)
    grid = []
    for r in rows:
        row = np.hstack(r)
        if row.shape[1] < width:
            row = np.hstack([row, np.zeros((row.shape[0], width - row.shape[1], 3), np.uint8)])
        grid.append(row)
    out = a.out or build_name(a.video)
    if not cv2.imwrite(out, np.vstack(grid)):
        raise SystemExit(f"Could not write {out}")
    print("Saved", out)


if __name__ == "__main__":
    main()
