"""Extract FSL features from a second signer's own recordings.

FSL-105 has one signer, and its 98.54% is a random split in which that signer
appears in both train and test. It is not evidence the model works for anyone
else, and a tester session confirmed it does not — predictions collapsed onto
BREAD, the signature of input outside the trained distribution.

A second signer is what makes cross-signer accuracy MEASURABLE on FSL for the
first time. This ingests videos filmed per docs/FSL_RECORDING.md, using the same
extraction path as extract_fsl105_v2.py so the features are directly comparable,
and writes a `signer.npy` alongside so the splits can be made signer-disjoint
rather than random.

Naming: <LABEL_WITH_UNDERSCORES>_<take>.mp4, e.g. GOOD_MORNING_2.mp4. Anything
that cannot be matched to a label in fsl105_labels.txt is reported and skipped,
never silently mislabelled.

    python extract_fsl_signer2.py --videos path/to/fsl_recordings --out signer2
"""

import os
import re
import glob
import argparse

import cv2
import numpy as np

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
import mediapipe as mp

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(HERE, "../../flutter_app/assets/labels/fsl105_labels.txt")
OUT_ROOT = os.path.join(HERE, "../data/processed/fsl_105_v2")

HAND_MODEL = os.path.join(HERE, "hand_landmarker.task")
POSE_MODEL = os.path.join(HERE, "pose_landmarker_full.task")

# Identical to extract_fsl105_v2.py. These must not drift: a different canvas or
# probe count puts the new signer in a different feature space from the corpus
# it is meant to be compared against.
SEQUENCE_LENGTH = 30
PROBE_FRAMES = 45
MIN_HAND_FRAMES = 8
CANVAS_W, CANVAS_H = 640, 480
MIRROR = False

VIDEO_EXT = ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.mkv")


def load_labels():
    out = {}
    with open(LABELS, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(",", 1)
            if len(p) == 2 and p[1]:
                out[p[1].strip().upper()] = int(p[0])
    return out


def key(s):
    """Collapse to letters and digits so GOOD_MORNING, good morning and
    GoodMorning all match the same label."""
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def letterbox(img):
    h, w = img.shape[:2]
    s = min(CANVAS_W / w, CANVAS_H / h)
    nw, nh = int(w * s), int(h * s)
    c = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
    y0, x0 = (CANVAS_H - nh) // 2, (CANVAS_W - nw) // 2
    c[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh),
                                           interpolation=cv2.INTER_AREA)
    return c


class Detectors:
    def __init__(self):
        from mediapipe.tasks.python.core.base_options import BaseOptions
        self.hand = mp.tasks.vision.HandLandmarker.create_from_options(
            mp.tasks.vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=HAND_MODEL),
                running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=2))
        self.pose = mp.tasks.vision.PoseLandmarker.create_from_options(
            mp.tasks.vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=POSE_MODEL),
                running_mode=mp.tasks.vision.RunningMode.IMAGE))

    def frame_vector(self, frame):
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(letterbox(frame), cv2.COLOR_BGR2RGB))
        lh, rh = np.zeros(63), np.zeros(63)
        try:
            res = self.hand.detect(image)
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

        pose = np.zeros(100)
        try:
            res = self.pose.detect(image)
            if res.pose_landmarks:
                pose = np.array([[p.x, p.y, p.z, p.visibility]
                                 for p in res.pose_landmarks[0][:25]]).flatten()
        except Exception:
            pass

        v = np.concatenate([lh, rh, pose])
        wrist = v[63:66].copy()
        if np.any(wrist != 0):
            for o in (0, 63):
                v[o:o + 63:3] -= wrist[0]
                v[o + 1:o + 63:3] -= wrist[1]
                v[o + 2:o + 63:3] -= wrist[2]
        return v

    def close(self):
        self.hand.close()
        self.pose.close()


def extract(path, det):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return None, "unreadable"
    vecs = []
    for idx in np.linspace(0, n - 1, PROBE_FRAMES).round().astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if not ok or fr is None:
            continue
        if MIRROR:
            fr = cv2.flip(fr, 1)
        vecs.append(det.frame_vector(fr))
    cap.release()

    kept = [v for v in vecs if np.any(v[0:126] != 0)]
    if len(kept) < MIN_HAND_FRAMES:
        return None, f"only {len(kept)} frames with a hand"
    sel = np.linspace(0, len(kept) - 1, SEQUENCE_LENGTH).round().astype(int)
    return np.array([kept[i] for i in sel], dtype=np.float32), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="folder of recordings")
    ap.add_argument("--out", default="signer2",
                    help="suffix for the written tensors")
    ap.add_argument("--signer-id", type=int, default=1,
                    help="0 is the original FSL-105 signer")
    args = ap.parse_args()

    labels = load_labels()
    by_key = {key(name): idx for name, idx in labels.items()}

    files = []
    for pat in VIDEO_EXT:
        files += glob.glob(os.path.join(args.videos, "**", pat), recursive=True)
    files = sorted(set(files))
    if not files:
        raise SystemExit(f"no videos found under {args.videos}")
    print(f"found {len(files)} videos")

    det = Detectors()
    seqs, ys, skipped, unmatched = [], [], [], []

    for i, path in enumerate(files):
        stem = os.path.splitext(os.path.basename(path))[0]
        # Strip a trailing take number: GOOD_MORNING_2 -> GOOD_MORNING
        base = re.sub(r"[_\-\s]*\d+$", "", stem)
        k = key(base)
        if k not in by_key:
            unmatched.append(os.path.basename(path))
            continue

        seq, why = extract(path, det)
        if seq is None:
            skipped.append(f"{os.path.basename(path)}: {why}")
            continue
        seqs.append(seq)
        ys.append(by_key[k])

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(files)}  kept {len(seqs)}", flush=True)

    det.close()

    if unmatched:
        print(f"\n{len(unmatched)} file(s) matched no label — RENAME AND RERUN:")
        for u in unmatched[:20]:
            print(f"   {u}")
        if len(unmatched) > 20:
            print(f"   ... and {len(unmatched)-20} more")

    if skipped:
        print(f"\n{len(skipped)} clip(s) skipped (no usable hand):")
        for s in skipped[:15]:
            print(f"   {s}")
        if len(skipped) > 15:
            print(f"   ... and {len(skipped)-15} more")

    if not seqs:
        raise SystemExit("nothing extracted")

    X = np.stack(seqs)
    y = np.array(ys)
    signer = np.full(len(y), args.signer_id, dtype=np.int32)

    os.makedirs(OUT_ROOT, exist_ok=True)
    np.save(os.path.join(OUT_ROOT, f"X_{args.out}.npy"), X)
    np.save(os.path.join(OUT_ROOT, f"y_{args.out}.npy"), y)
    np.save(os.path.join(OUT_ROOT, f"signer_{args.out}.npy"), signer)

    covered = len(set(ys))
    print(f"\nwrote X_{args.out}.npy {X.shape}")
    print(f"words covered: {covered}/{len(labels)}")
    if covered < len(labels):
        have = {by_key[key(n)] for n in [] } or set(ys)
        missing = [n for n, idx in sorted(labels.items(), key=lambda p: p[1])
                   if idx not in have]
        print(f"MISSING {len(missing)} word(s): {', '.join(missing[:25])}"
              f"{' ...' if len(missing) > 25 else ''}")
    print("\nNext: train with a signer-disjoint split to get the first honest "
          "cross-signer number for FSL.")


if __name__ == "__main__":
    main()
