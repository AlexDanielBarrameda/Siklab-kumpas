"""Extract ASL Citizen into the same feature form as FSL-105.

Design constraints:

1. Output must be interchangeable with fsl_105_v2 so the app reuses its whole
   preprocessing path — same 226-d layout, same hand-present sampling, same
   right-wrist normalisation, and canonicalize.py applied afterwards. Only the
   model file and the label list differ between languages.

2. ASL and FSL stay SEPARATE models. Merging would force one classifier to
   choose between two languages' meanings for an identical sign, which no
   amount of data resolves, and would drag FSL accuracy down with it.

3. Videos are read straight out of the 42.8 GB zip and never unpacked to disk.
   Only ~15k of the 83,399 clips are needed, so extracting the archive would
   waste ~40 GB and a lot of time.

4. The official splits are signer-disjoint (train 35 signers, test 11, zero
   overlap), so they are preserved verbatim. That is what yields a genuine
   cross-signer accuracy figure — the thing FSL-105 cannot give, being a
   single signer.

Variant handling: ASL-LEX carries regional variants as WHAT1 / WHAT2. Keeping
both would give the app two classes that mean the same word, so one variant per
concept is kept (the better-recorded one) and displayed under the plain name.

Usage:
    python extract_asl_citizen.py --plan            # choose words, extract nothing
    python extract_asl_citizen.py --concepts 500
"""

import os
import re
import csv
import json
import time
import zipfile
import argparse
import tempfile
from collections import Counter, defaultdict
from multiprocessing import Pool

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "../data/raw/asl_citizen")
ZIP = os.path.join(ROOT, "ASL_Citizen.zip")
SPLITS = os.path.join(ROOT, "ASL_Citizen", "splits")
OUT_DIR = os.path.join(HERE, "../data/processed/asl_citizen")
HAND_MODEL = os.path.join(HERE, "hand_landmarker.task")
POSE_MODEL = os.path.join(HERE, "pose_landmarker_full.task")

SEQUENCE_LENGTH = 30
PROBE_FRAMES = 45
MIN_HAND_FRAMES = 8
CANVAS_W, CANVAS_H = 640, 480
MIRROR = False                      # matches extract_fsl105_v2.py

# Everyday vocabulary, prioritised ahead of the frequency top-up.
EVERYDAY = """
HELLO PLEASE SORRY YES NO NAME WHO WHY WHERE WHEN WHAT HOW
GOOD BAD HAPPY SAD LOVE LIKE WANT NEED HELP STOP GO COME EAT DRINK WATER
MOTHER FATHER SISTER BROTHER FAMILY FRIEND MAN WOMAN BOY GIRL BABY PEOPLE
HOME SCHOOL WORK BATHROOM HOSPITAL TODAY TOMORROW YESTERDAY NOW MORNING
AFTERNOON NIGHT TIME DAY WEEK YEAR UNDERSTAND KNOW LEARN TEACH READ WRITE
DEAF HEARING SIGN LISTEN HOT COLD BIG SMALL MORE FINISH AGAIN THANKYOU
MONEY BUY CAR PHONE DOCTOR SICK PAIN
""".split()

_state = {}


def _init():
    import mediapipe as mp
    if _state:
        return _state
    base = mp.tasks.BaseOptions
    _state["mp"] = mp
    _state["hand"] = mp.tasks.vision.HandLandmarker.create_from_options(
        mp.tasks.vision.HandLandmarkerOptions(
            base_options=base(model_asset_path=HAND_MODEL,
                              delegate=base.Delegate.CPU),
            running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=2))
    _state["pose"] = mp.tasks.vision.PoseLandmarker.create_from_options(
        mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base(model_asset_path=POSE_MODEL,
                              delegate=base.Delegate.CPU),
            running_mode=mp.tasks.vision.RunningMode.IMAGE))
    _state["zip"] = zipfile.ZipFile(ZIP)
    return _state


def letterbox(img):
    h, w = img.shape[:2]
    s = min(CANVAS_W / w, CANVAS_H / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    c = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
    y0, x0 = (CANVAS_H - nh) // 2, (CANVAS_W - nw) // 2
    c[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh),
                                           interpolation=cv2.INTER_AREA)
    return c


def landmarks_for(frame):
    """226-d vector, identical layout and normalisation to extract_fsl105_v2."""
    d = _init()
    mp = d["mp"]
    image = mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(letterbox(frame), cv2.COLOR_BGR2RGB))

    lh, rh = np.zeros(63), np.zeros(63)
    try:
        res = d["hand"].detect(image)
        if res.hand_landmarks:
            for i, handed in enumerate(res.handedness):
                coords = np.array([[l.x, l.y, l.z]
                                   for l in res.hand_landmarks[i]]).flatten()
                if handed[0].category_name == "Left":
                    lh = coords
                elif handed[0].category_name == "Right":
                    rh = coords
    except Exception:
        pass

    pose = np.zeros(100)
    try:
        res = d["pose"].detect(image)
        if res.pose_landmarks:
            pose = np.array([[p.x, p.y, p.z, p.visibility]
                             for p in res.pose_landmarks[0][:25]]).flatten()
    except Exception:
        pass

    v = np.concatenate([lh, rh, pose])
    wrist = v[63:66].copy()
    if np.any(wrist != 0):
        v[0:63:3] -= wrist[0]; v[1:63:3] -= wrist[1]; v[2:63:3] -= wrist[2]
        v[63:126:3] -= wrist[0]; v[64:126:3] -= wrist[1]; v[65:126:3] -= wrist[2]
    return v


def _job(args):
    """Stream one clip out of the zip, process it, return the sequence.

    The clip is written to a temp file because cv2.VideoCapture needs a path;
    it is removed immediately. Disk use stays negligible instead of unpacking
    40 GB.
    """
    member, cls, split = args
    d = _init()
    tmp = None
    try:
        data = d["zip"].read(member)
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
            if not ok or fr is None:
                continue
            if MIRROR:
                fr = cv2.flip(fr, 1)
            vecs.append(landmarks_for(fr))
        cap.release()

        kept = [v for v in vecs if np.any(v[0:126] != 0)]
        if len(kept) < MIN_HAND_FRAMES:
            return None
        sel = np.linspace(0, len(kept) - 1, SEQUENCE_LENGTH).round().astype(int)
        return (np.array([kept[i] for i in sel], dtype=np.float32), cls, split)
    except Exception:
        return None
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# --------------------------------------------------------------- vocabulary
def load_index():
    rows = []
    for split in ("train", "val", "test"):
        with open(os.path.join(SPLITS, f"{split}.csv"),
                  newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                rows.append({
                    "gloss": r["Gloss"].strip().upper(),
                    "video": r["Video file"].strip(),
                    "signer": r["Participant ID"].strip(),
                    "split": split,
                })
    return rows


def base_concept(gloss):
    """WHAT1 / WHAT2 -> WHAT. Leaves genuinely distinct glosses alone."""
    return re.sub(r"\d+$", "", gloss) or gloss


def choose_concepts(rows, n_concepts):
    """One gloss per concept, everyday words first, then best-recorded."""
    by_gloss = Counter(r["gloss"] for r in rows)
    signers = defaultdict(set)
    for r in rows:
        signers[r["gloss"]].add(r["signer"])

    variants = defaultdict(list)
    for g in by_gloss:
        variants[base_concept(g)].append(g)

    # Within a concept keep the variant with the most videos, then most signers.
    chosen_variant = {
        base: max(vs, key=lambda g: (by_gloss[g], len(signers[g])))
        for base, vs in variants.items()
    }

    everyday = [w for w in EVERYDAY if w in chosen_variant]
    # Prefer plain alphabetic glosses for the top-up: numeric and compound
    # entries skew technical, and the aim is everyday communication.
    rest = [b for b in chosen_variant
            if b not in everyday and b.isalpha() and 2 <= len(b) <= 12]
    rest.sort(key=lambda b: (-len(signers[chosen_variant[b]]),
                             -by_gloss[chosen_variant[b]], b))

    picked = everyday + rest
    picked = picked[:n_concepts]
    return [(b, chosen_variant[b]) for b in sorted(picked)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", type=int, default=500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--plan", action="store_true",
                    help="pick the vocabulary and report, extract nothing")
    args = ap.parse_args()

    rows = load_index()
    concepts = choose_concepts(rows, args.concepts)
    gloss_of = {b: g for b, g in concepts}
    idx_of = {b: i for i, (b, _) in enumerate(concepts)}
    wanted = {g: b for b, g in concepts}

    jobs = []
    per_split = Counter()
    for r in rows:
        b = wanted.get(r["gloss"])
        if b is None:
            continue
        jobs.append((f"ASL_Citizen/videos/{r['video']}", idx_of[b], r["split"]))
        per_split[r["split"]] += 1

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "labels.txt"), "w", encoding="utf-8") as f:
        for i, (b, g) in enumerate(concepts):
            f.write(f"{i},{b}\n")
    with open(os.path.join(OUT_DIR, "concepts.json"), "w", encoding="utf-8") as f:
        json.dump([{"index": i, "display": b, "gloss": g}
                   for i, (b, g) in enumerate(concepts)], f, indent=1)

    everyday_hits = sum(1 for b, _ in concepts if b in EVERYDAY)
    print(f"concepts: {len(concepts)}   "
          f"({everyday_hits} from the everyday list, rest by signer coverage)")
    print(f"clips:    {len(jobs):,}   {dict(per_split)}")
    merged = [(b, g) for b, g in concepts if b != g]
    print(f"variants merged: {len(merged)}  e.g. "
          + ", ".join(f"{g}->{b}" for b, g in merged[:6]))
    print(f"labels -> {os.path.relpath(os.path.join(OUT_DIR,'labels.txt'))}")

    if args.plan:
        print("\n--plan given, nothing extracted")
        return

    t0 = time.time()
    X, y, sp, done, kept = [], [], [], 0, 0
    with Pool(args.workers) as pool:
        for res in pool.imap_unordered(_job, jobs, chunksize=4):
            done += 1
            if res is not None:
                X.append(res[0]); y.append(res[1]); sp.append(res[2]); kept += 1
            if done % 250 == 0:
                el = time.time() - t0
                eta = el / done * (len(jobs) - done)
                print(f"  {done:,}/{len(jobs):,}  kept {kept:,}  "
                      f"elapsed {el/60:.0f}m  eta {eta/60:.0f}m", flush=True)
            if done % 2000 == 0:
                np.save(os.path.join(OUT_DIR, "X_partial.npy"),
                        np.array(X, dtype=np.float32))
                np.save(os.path.join(OUT_DIR, "y_partial.npy"), np.array(y))

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    sp = np.array(sp)
    np.save(os.path.join(OUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUT_DIR, "y.npy"), y)
    np.save(os.path.join(OUT_DIR, "split.npy"), sp)
    for f in ("X_partial.npy", "y_partial.npy"):
        p = os.path.join(OUT_DIR, f)
        if os.path.exists(p):
            os.remove(p)

    print(f"\nX {X.shape}  y {y.shape}")
    print(f"kept {kept:,}/{len(jobs):,}  in {(time.time()-t0)/60:.0f} min")
    for s in ("train", "val", "test"):
        print(f"  {s:<6} {(sp == s).sum():,}")
    print("next: canonicalize, then train")


if __name__ == "__main__":
    main()
