"""
Stage 1 (run ONCE): UTA-RLDD videos -> per-frame facial signals, cached as small .npz files.
Everything after this (windowing, calibration, RF/XGB/MLP, LOSO) runs on the cached
signals, so it is fast and can be re-run as often as you like.

Setup (Python 3.10-3.12):
    pip install mediapipe==0.10.14 opencv-python numpy

Usage:
    python extract_features.py --videos D:/RLDD --out features --stride 2 --workers 3
    (benchmark first: add --limit 2; re-extract existing files: add --overwrite)

Assumed layout (verify against your download): <videos>/**/<subject_id>/{0,5,10}.mov|mp4
    0 = alert, 5 = low vigilant, 10 = drowsy
Split videos (10_1.mp4, 10_2.mp4) are processed separately.

Output: features/<subject>_<label>[_p<part>].npz with arrays
    t (s), X (T x 10), cols, fps
Feature computation is unchanged from the previous version, so existing cached
.npz files remain valid.
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

cv2.setNumThreads(1)  # one thread per worker; parallelism comes from the Pool

# MediaPipe FaceMesh landmark indices
R_EYE = [33, 160, 158, 133, 153, 144]       # p1..p6 (outer corner first)
L_EYE = [362, 385, 387, 263, 373, 380]      # p1..p6 (inner corner first)
MOUTH = [78, 308, 81, 178, 13, 14, 311, 402]  # corners, then 3 inner-lip vertical pairs
POSE_IDX = [1, 152, 263, 33, 291, 61]       # nose, chin, 263, 33, 291, 61 (same order as POSE_3D)

# Generic 3D face model in OpenCV camera convention (x right, y down, z away from camera), mm.
POSE_3D = np.array([[0, 0, 0],               # nose tip
                    [0, 63.6, 12.5],         # chin
                    [43.3, -32.7, 26],       # landmark 263 (appears on image right)
                    [-43.3, -32.7, 26],      # landmark 33 (appears on image left)
                    [28.9, 28.9, 24.1],      # landmark 291 (mouth corner, image right)
                    [-28.9, 28.9, 24.1]], dtype=np.float64)  # landmark 61 (mouth corner, image left)

BROW_R, LID_R = 105, 159
BROW_L, LID_L = 334, 386
COLS = ["ear_l", "ear_r", "mar", "pitch", "yaw", "roll",
        "brow_l", "brow_r", "interocular", "face_found"]
LABELS = {"0": "alert", "5": "low", "10": "drowsy"}


def _d(a, b):
    return float(np.linalg.norm(a - b))


def ear(p):  # p: 6x2 in order p1..p6
    return (_d(p[1], p[5]) + _d(p[2], p[4])) / (2.0 * _d(p[0], p[3]) + 1e-9)


def mar(p):  # p: 8x2 -> corners(0,1), pairs (2,3),(4,5),(6,7)
    return (_d(p[2], p[3]) + _d(p[4], p[5]) + _d(p[6], p[7])) / (3.0 * _d(p[0], p[1]) + 1e-9)


def head_pose(pts_px, w, h):
    """Approximate pitch/yaw/roll in degrees (generic 3D model, focal length ~ image width)."""
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(POSE_3D, pts_px.astype(np.float64), cam, np.zeros(4),
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return [np.nan] * 3
    return list(cv2.RQDecomp3x3(cv2.Rodrigues(rvec)[0])[0])  # pitch, yaw, roll


def output_path(path, out_dir):
    subject = os.path.basename(os.path.dirname(path))
    code, *part = os.path.splitext(os.path.basename(path))[0].split("_")  # "10_1" -> "10", ["1"]
    label = LABELS.get(code, "unknown")
    suffix = f"_p{part[0]}" if part else ""
    return os.path.join(out_dir, f"{subject}_{label}{suffix}.npz")


def process_video(args):
    path, out_dir, stride, max_w, overwrite = args
    out = output_path(path, out_dir)
    if os.path.exists(out) and not overwrite:  # resume support
        return out, "skipped", True
    try:
        import mediapipe as mp  # import inside worker (Windows spawn)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return out, f"FAILED: cannot open {path}", False
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        rows, times, i = [], [], 0
        try:
            while True:
                if i % stride:  # skipped frame: advance without decoding where possible
                    if not cap.grab():
                        break
                    i += 1
                    continue
                ok, frame = cap.read()
                if not ok:
                    break
                h0, w0 = frame.shape[:2]
                if w0 > max_w:
                    frame = cv2.resize(frame, (max_w, int(h0 * max_w / w0)))
                h, w = frame.shape[:2]
                res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if res.multi_face_landmarks:
                    lm = res.multi_face_landmarks[0].landmark
                    xy = np.array([[l.x * w, l.y * h] for l in lm])
                    io = _d(xy[33], xy[263])
                    pitch, yaw, roll = head_pose(xy[POSE_IDX], w, h)
                    rows.append([ear(xy[L_EYE]), ear(xy[R_EYE]), mar(xy[MOUTH]),
                                 pitch, yaw, roll,
                                 _d(xy[BROW_L], xy[LID_L]) / io, _d(xy[BROW_R], xy[LID_R]) / io,
                                 io, 1.0])
                else:
                    rows.append([np.nan] * (len(COLS) - 1) + [0.0])
                times.append(i / fps)  # real timestamps -> handles differing frame rates
                i += 1
        finally:
            cap.release()
            mesh.close()
        if not rows:
            return out, "FAILED: no frames decoded", False
        # Write to a temp file first, then rename, so an interrupted run never leaves a
        # half-written .npz that the resume check would treat as finished.
        tmp = out[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, t=np.array(times, np.float32), X=np.array(rows, np.float32),
                            cols=np.array(COLS), fps=fps)
        os.replace(tmp, out)
        found = float(np.mean([r[-1] for r in rows]))
        return out, f"{len(rows)} frames @ {fps:.1f} fps (stride {stride}), face found {found:.0%}", True
    except Exception as e:  # one bad video must not kill the whole pool
        return out, f"FAILED: {type(e).__name__}: {e}", False


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, help="root folder of the UTA-RLDD videos")
    ap.add_argument("--out", default="features")
    ap.add_argument("--stride", type=int, default=2, help="keep every Nth frame (2 => ~15 fps)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max_width", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0, help="process only first N videos (benchmark)")
    ap.add_argument("--overwrite", action="store_true", help="re-extract videos that already have an .npz")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    vids = sorted(p for ext in ("mov", "mp4", "m4v", "avi", "mkv")
                  for p in glob.glob(os.path.join(a.videos, "**", f"*.{ext}"), recursive=True))
    vids = [v for v in vids if os.path.splitext(os.path.basename(v))[0].split("_")[0] in LABELS]
    if a.limit:
        vids = vids[:a.limit]
    print(f"{len(vids)} videos found")
    if not vids:
        sys.exit("No videos matched. Expected <videos>/**/<subject_id>/{0,5,10}[_part].mov|mp4")

    # Two different videos mapping to the same output name would silently overwrite/skip each other.
    names = [output_path(v, a.out) for v in vids]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        sys.exit("Output name collision (check folder layout): " + ", ".join(sorted(dup)[:5]))

    failed = []
    with Pool(a.workers) as pool:
        jobs = [(v, a.out, a.stride, a.max_width, a.overwrite) for v in vids]
        for k, (o, msg, ok) in enumerate(pool.imap_unordered(process_video, jobs), 1):
            print(f"[{k}/{len(vids)}] {os.path.basename(o)}: {msg}", flush=True)
            if not ok:
                failed.append((o, msg))
    if failed:
        print(f"\n{len(failed)} video(s) failed:")
        for o, msg in failed:
            print(f"  {os.path.basename(o)}: {msg}")
        sys.exit(1)
    print("\nAll videos processed.")
