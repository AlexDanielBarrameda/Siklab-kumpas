"""Does swapping pose_landmarker_full for _lite cost accuracy?

Raising the live frame rate is worth real accuracy: the ASL model scores 59.9%
at 5 distinct frames against 66.7% at 30, so every extra frame per sign helps.
The lite pose model is the obvious lever -- it is ~40% smaller and materially
faster -- but it is not a free swap.

Both extractors built their training features with pose_landmarker_full
(extract_asl_citizen.py:55, extract_fsl105_v2.py:59). The pose block is never
wrist-normalised and carries roughly 34 points of the model's accuracy, so
running a DIFFERENT pose model on device puts those 100 dims in a distribution
the model was never trained on -- the same class of mismatch that made ASL fail
while FSL worked.

So this extracts the same clips twice, once with each pose model, and reports:

  AGREEMENT   how far the two models place the same landmarks. Large divergence
              means the swap is a domain shift, not an optimisation.
  ACCURACY    the trained model's accuracy on lite-extracted features vs
              full-extracted ones. This is the number that decides it.

    python pose_model_swap.py --n 150 --model ../models/asl_525frame_best.keras
"""

import os
import sys
import csv
import json
import zipfile
import tempfile
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRE = os.path.join(HERE, "../preprocessing")
sys.path.insert(0, PRE)

ZIP = os.path.join(HERE, "../data/raw/asl_citizen/ASL_Citizen.zip")
SPLITS = os.path.join(HERE, "../data/raw/asl_citizen/ASL_Citizen/splits")
DATA = os.path.join(HERE, "../data/processed/asl_citizen")

HAND_MODEL = os.path.join(PRE, "hand_landmarker.task")
POSE_FULL = os.path.join(PRE, "pose_landmarker_full.task")
POSE_LITE = os.path.join(PRE, "pose_landmarker_lite.task")

CANVAS_W, CANVAS_H = 640, 480
SEQ = 30
PROBE_FRAMES = 45
MIN_HAND_FRAMES = 8
POSE0 = 126


def letterbox(img):
    h, w = img.shape[:2]
    s = min(CANVAS_W / w, CANVAS_H / h)
    nw, nh = int(w * s), int(h * s)
    c = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
    y0, x0 = (CANVAS_H - nh) // 2, (CANVAS_W - nw) // 2
    c[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh),
                                           interpolation=cv2.INTER_AREA)
    return c


def make_detectors(pose_path):
    import mediapipe as mp
    from mediapipe.tasks.python.core.base_options import BaseOptions
    hand = mp.tasks.vision.HandLandmarker.create_from_options(
        mp.tasks.vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL),
            running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=2))
    pose = mp.tasks.vision.PoseLandmarker.create_from_options(
        mp.tasks.vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE))
    return mp, hand, pose


def landmarks_for(mp, hand, pose, frame):
    image = mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(letterbox(frame), cv2.COLOR_BGR2RGB))
    lh, rh = np.zeros(63), np.zeros(63)
    try:
        res = hand.detect(image)
        if res.hand_landmarks:
            for i, handed in enumerate(res.handedness):
                c = np.array([[l.x, l.y, l.z]
                              for l in res.hand_landmarks[i]]).flatten()
                if handed[0].category_name == "Left":
                    lh = c
                elif handed[0].category_name == "Right":
                    rh = c
    except Exception:
        pass
    p = np.zeros(100)
    try:
        res = pose.detect(image)
        if res.pose_landmarks:
            p = np.array([[q.x, q.y, q.z, q.visibility]
                          for q in res.pose_landmarks[0][:25]]).flatten()
    except Exception:
        pass
    v = np.concatenate([lh, rh, p])
    wrist = v[63:66].copy()
    if np.any(wrist != 0):
        for o in (0, 63):
            v[o:o + 63:3] -= wrist[0]
            v[o + 1:o + 63:3] -= wrist[1]
            v[o + 2:o + 63:3] -= wrist[2]
    return v


def extract(zf, member, mp, hand, pose):
    tmp = None
    try:
        data = zf.read(member)
        fd, tmp = tempfile.mkstemp(suffix=".mp4")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        cap = cv2.VideoCapture(tmp)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            cap.release()
            return None
        vecs = []
        for idx in np.linspace(0, n - 1, PROBE_FRAMES).round().astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, fr = cap.read()
            if ok and fr is not None:
                vecs.append(landmarks_for(mp, hand, pose, fr))
        cap.release()
        kept = [v for v in vecs if np.any(v[0:126] != 0)]
        if len(kept) < MIN_HAND_FRAMES:
            return None
        sel = np.linspace(0, len(kept) - 1, SEQ).round().astype(int)
        return np.array([kept[i] for i in sel], dtype=np.float32)
    except Exception:
        return None
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    if not os.path.exists(POSE_LITE):
        raise SystemExit(f"missing {POSE_LITE}")

    labels = {}
    for line in open(os.path.join(DATA, "labels.txt"), encoding="utf-8"):
        p = line.strip().split(",", 1)
        if len(p) == 2:
            labels[p[1]] = int(p[0])

    rows = []
    with open(os.path.join(SPLITS, "test.csv"), newline="",
              encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            g = r["Gloss"].strip().upper()
            if g in labels:
                rows.append((r["Video file"].strip(), labels[g]))
    rng = np.random.default_rng(7)
    rng.shuffle(rows)
    rows = rows[:args.n]

    zf = zipfile.ZipFile(ZIP)
    out = {}
    for tag, path in (("full", POSE_FULL), ("lite", POSE_LITE)):
        print(f"extracting {len(rows)} clips with pose_landmarker_{tag}...",
              flush=True)
        mp, hand, pose = make_detectors(path)
        seqs, ys = [], []
        for i, (member, cls) in enumerate(rows):
            s = extract(zf, f"ASL_Citizen/videos/{member}", mp, hand, pose)
            if s is not None:
                seqs.append(s)
                ys.append(cls)
            if (i + 1) % 25 == 0:
                print(f"   {i+1}/{len(rows)}", flush=True)
        out[tag] = (np.stack(seqs), np.array(ys), [m for m, _ in rows])
        hand.close()
        pose.close()

    # Only clips both models produced, so the comparison is paired.
    nfull, nlite = len(out["full"][0]), len(out["lite"][0])
    n = min(nfull, nlite)
    Xf, yf = out["full"][0][:n], out["full"][1][:n]
    Xl, yl = out["lite"][0][:n], out["lite"][1][:n]
    print(f"\nusable clips: full {nfull}, lite {nlite}, paired {n}")

    print(f"\n{'='*60}\nAGREEMENT\n{'='*60}")
    d = np.abs(Xf[:, :, POSE0:] - Xl[:, :, POSE0:])
    live = (Xf[:, :, POSE0:] != 0) & (Xl[:, :, POSE0:] != 0)
    dd = d[live]
    print(f"  pose landmark |difference|  median {np.median(dd):.4f}   "
          f"p90 {np.percentile(dd, 90):.4f}   max {dd.max():.4f}")
    both = np.any(Xf[:, :, POSE0:] != 0, axis=2)
    onlyf = both & ~np.any(Xl[:, :, POSE0:] != 0, axis=2)
    print(f"  frames where lite found NO pose but full did: "
          f"{onlyf.sum()} / {both.sum()}  ({100*onlyf.mean():.2f}%)")

    if args.model:
        import tensorflow as tf
        sys.path.insert(0, PRE)
        from canonicalize import canonicalise_frame, denormalise_frame
        model = tf.keras.models.load_model(os.path.abspath(args.model))

        def canon(X):
            o = np.empty_like(X)
            for i in range(len(X)):
                for t in range(X.shape[1]):
                    o[i, t] = canonicalise_frame(denormalise_frame(X[i, t]))
            return o

        print(f"\n{'='*60}\nACCURACY\n{'='*60}")
        for tag, X, y in (("full (matches training)", Xf, yf),
                          ("lite (device would use)", Xl, yl)):
            p = model.predict(canon(X), verbose=0, batch_size=128)
            acc = np.mean(np.argmax(p, axis=1) == y) * 100
            print(f"  {tag:<26} {acc:6.2f}%   (n={len(y)})")
        print("\n  A large drop means lite must not be swapped in without "
              "re-extracting\n  the corpus and retraining on lite features.")


if __name__ == "__main__":
    main()
