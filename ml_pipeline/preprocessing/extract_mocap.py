"""Extract 3D motion capture from the FSL-105 clips, for the avatar pipeline.

This is NOT the classification extractor and must not be confused with it.

  extract_fsl105_v2.py  -> features for the LSTM. Wrist-normalised,
                           canonicalised, single hands mirrored into one block,
                           resampled to exactly 30 frames. Absolute position is
                           deliberately destroyed. Useless for animation.

  this script           -> MediaPipe *world* landmarks: real 3D in metres,
                           hip-centred for pose and wrist-centred for hands.
                           No normalisation, no canonicalisation, no mirroring.
                           This is what a rig should be driven from.

MediaPipe returns both coordinate sets from the same detection. The image
landmarks are normalised to the frame with a weak relative z, which is fine for
a classifier and wrong for a skeleton -- a rig driven from them only looks
correct head-on.

One clean exemplar per sign, not an average: averaging 20 takes produces mush.
Candidate clips are tried in order and the first with good detection coverage
wins.

Output: avatar/mocap/<id>_<LABEL>.json  (see avatar/MOCAP_FORMAT.md)
"""

import os
import json
import glob
import argparse
from multiprocessing import Pool

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "../data/raw/fsl_105/fsl_105_dataset/clips")
LABELS_CSV = os.path.join(HERE, "../data/raw/fsl_105/fsl_105_dataset/labels.csv")
OUT = os.path.join(HERE, "../../avatar/mocap")
HAND_MODEL = os.path.join(HERE, "hand_landmarker.task")
POSE_MODEL = os.path.join(HERE, "pose_landmarker_full.task")

TARGET_FPS = 30          # animation playback rate
SCAN_FRAMES = 40         # first pass, to locate the active span
MIN_COVERAGE = 0.80      # fraction of frames needing a hand, to accept a clip
MAX_CANDIDATES = 3       # clips to try per sign before taking the best seen
CANVAS_W, CANVAS_H = 640, 480

_det = {}


def _init_detectors():
    """One detector pair per worker process; they are not picklable."""
    import mediapipe as mp
    if _det:
        return _det
    base = mp.tasks.BaseOptions
    _det["mp"] = mp
    _det["hand"] = mp.tasks.vision.HandLandmarker.create_from_options(
        mp.tasks.vision.HandLandmarkerOptions(
            base_options=base(model_asset_path=HAND_MODEL,
                              delegate=base.Delegate.CPU),
            running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=2))
    _det["pose"] = mp.tasks.vision.PoseLandmarker.create_from_options(
        mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base(model_asset_path=POSE_MODEL,
                              delegate=base.Delegate.CPU),
            running_mode=mp.tasks.vision.RunningMode.IMAGE))
    return _det


def letterbox(img):
    """Same framing the classifier sees, so detection behaves identically."""
    h, w = img.shape[:2]
    s = min(CANVAS_W / w, CANVAS_H / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
    y0, x0 = (CANVAS_H - nh) // 2, (CANVAS_W - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh),
                                                interpolation=cv2.INTER_AREA)
    return canvas


def detect(frame):
    d = _init_detectors()
    mp = d["mp"]
    image = mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(letterbox(frame), cv2.COLOR_BGR2RGB))

    out = {"pose_world": None, "pose_image": None,
           "left_hand_world": None, "right_hand_world": None}

    pr = d["pose"].detect(image)
    if pr.pose_world_landmarks:
        out["pose_world"] = [[l.x, l.y, l.z, getattr(l, "visibility", 1.0)]
                             for l in pr.pose_world_landmarks[0]]
    if pr.pose_landmarks:
        out["pose_image"] = [[l.x, l.y, l.z, getattr(l, "visibility", 1.0)]
                             for l in pr.pose_landmarks[0]]

    hr = d["hand"].detect(image)
    if hr.hand_landmarks and hr.hand_world_landmarks:
        for i, handed in enumerate(hr.handedness):
            if i >= len(hr.hand_world_landmarks):
                break
            pts = [[l.x, l.y, l.z] for l in hr.hand_world_landmarks[i]]
            key = ("left_hand_world" if handed[0].category_name == "Left"
                   else "right_hand_world")
            out[key] = pts
    return out


def frames_at(cap, indices):
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        yield (fr if ok else None)


def process_clip(path):
    """Returns (frames, coverage, src_fps) or None."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    if n <= 0:
        cap.release()
        return None

    # Pass 1 -- locate the span where the hands are actually up.
    scan_idx = np.linspace(0, n - 1, SCAN_FRAMES).round().astype(int)
    present = []
    for fr in frames_at(cap, scan_idx):
        if fr is None:
            present.append(False)
            continue
        r = detect(fr)
        present.append(r["left_hand_world"] is not None or
                       r["right_hand_world"] is not None)
    present = np.array(present)
    if present.sum() < 4:
        cap.release()
        return None
    first, last = scan_idx[np.where(present)[0][[0, -1]]]

    # Pass 2 -- sample that span at the animation frame rate.
    span = max(1, last - first)
    count = max(6, int(round(span / fps * TARGET_FPS)))
    idx = np.linspace(first, last, count).round().astype(int)

    frames, hits = [], 0
    for fr in frames_at(cap, idx):
        if fr is None:
            frames.append({"pose_world": None, "pose_image": None,
                           "left_hand_world": None, "right_hand_world": None})
            continue
        r = detect(fr)
        if r["left_hand_world"] is not None or r["right_hand_world"] is not None:
            hits += 1
        frames.append(r)
    cap.release()
    return frames, hits / len(frames), fps


def build_sign(job):
    cid, label = job
    vids = sorted(p for p in glob.glob(os.path.join(RAW, str(cid), "*"))
                  if p.lower().endswith((".mov", ".mp4")))
    best = None
    for path in vids[:MAX_CANDIDATES]:
        try:
            res = process_clip(path)
        except Exception as e:
            print(f"  [{cid}] {os.path.basename(path)}: {e}", flush=True)
            continue
        if res is None:
            continue
        frames, coverage, fps = res
        if best is None or coverage > best[1]:
            best = (frames, coverage, fps, path)
        if coverage >= MIN_COVERAGE:
            break

    if best is None:
        return (cid, label, None, 0.0)

    frames, coverage, fps, path = best
    doc = {
        "label": label,
        "class_id": cid,
        "source_clip": os.path.basename(path),
        "fps": TARGET_FPS,
        "frame_count": len(frames),
        "hand_coverage": round(coverage, 3),
        "frames": frames,
    }
    safe = label.replace(" ", "_").replace("/", "_")
    out_path = os.path.join(OUT, f"{cid:03d}_{safe}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    return (cid, label, len(frames), coverage)


def load_labels():
    labels = {}
    with open(LABELS_CSV, encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            p = line.strip().split(",")
            if len(p) >= 2:
                labels[int(p[0])] = p[1]
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="",
                    help="comma-separated class ids, e.g. 0,3,20. default: all")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    labels = load_labels()
    ids = ([int(x) for x in args.classes.split(",") if x.strip()]
           if args.classes else sorted(labels))
    jobs = [(i, labels[i]) for i in ids if i in labels]

    print(f"extracting {len(jobs)} signs -> avatar/mocap", flush=True)
    with Pool(args.workers) as pool:
        for cid, label, frames, cov in pool.imap_unordered(build_sign, jobs):
            if frames is None:
                print(f"  FAILED {cid:3d} {label}", flush=True)
            else:
                print(f"  ok {cid:3d} {label:<22} {frames:3d} frames  "
                      f"coverage {cov*100:.0f}%", flush=True)


if __name__ == "__main__":
    main()
