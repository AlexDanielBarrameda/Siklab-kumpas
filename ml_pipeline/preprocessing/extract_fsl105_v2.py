"""FSL-105 landmark extraction, v2.

Rewritten against three defects measured in the v1 pipeline
(extract_fsl105.py), all of which contributed to the low offline ceiling and
the live-camera collapse:

1. TEMPORAL COVERAGE (the big one).
   v1 kept 30 *consecutive* frames from the middle of each clip:

       start = (len(frames) - SEQUENCE_LENGTH) // 2
       frames = frames[start : start + SEQUENCE_LENGTH]

   The source clips are 60 fps / ~244 frames / ~4.07 s, so those 30 frames are
   0.50 s -- about 12% of each sign. The LSTM never saw a complete sign. v2
   samples 30 frames spread uniformly across the whole clip.

2. ASPECT DISTORTION.
   Clips are 640x360 (16:9). v1 did cv2.resize(img, (640, 480)), stretching the
   signer vertically by 1.33x. MediaPipe normalises x by width and y by height,
   so every training landmark carries that stretch while the live app -- which
   letterboxes preserving aspect -- does not. v2 letterboxes to match the app.

3. MIRRORING -- deliberately NOT applied here.
   The live pipeline mirrors the frame for fingerspelling (MainActivity.kt,
   mirror=true), which inverts MediaPipe's Left/Right label. Mirroring the
   training clips to match was tried and measured worse: it pushed essentially
   every hand into the lh block (rh non-zero fell 77.7% -> 1.1%), which disables
   the right-wrist normalisation below and leaves raw absolute coordinates.
   Mirroring also mirrors the geometry itself -- a mirrored right hand has
   left-hand shape -- so it only achieves parity if the app mirrors too.

   Resolution: training stays unmirrored, and the app sends mirror=false in
   WORDS mode while keeping mirror=true for LETTERS (which is already accurate
   and must not be disturbed). See InferenceService.mirrorInput.

Also ~8x faster than v1: it seeks to the 30 frames it needs instead of decoding
and running MediaPipe over all 244.

Writes to data/processed/fsl_105_v2/ so the v1 tensors stay intact.
"""

import os
import sys
import json
import glob
import numpy as np
import cv2
import mediapipe as mp

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "../data/raw/fsl_105/fsl_105_dataset/clips")
LABELS_CSV = os.path.join(HERE, "../data/raw/fsl_105/fsl_105_dataset/labels.csv")
OUTPUT_DIR = os.path.join(HERE, "../data/processed/fsl_105_v2")
CHECKPOINT = os.path.join(OUTPUT_DIR, "checkpoint.json")

HAND_MODEL = os.path.join(HERE, "hand_landmarker.task")
POSE_MODEL = os.path.join(HERE, "pose_landmarker_full.task")

SEQUENCE_LENGTH = 30
LANDMARK_DIM = 226

# Frames probed per clip before filtering. Uniform sampling across the full clip
# lands ~55% of frames on the signer at rest (hands down) at the head and tail,
# and the live app never buffers a no-hand frame -- inference_service.dart
# returns early when no hand is detected. So we probe wider, keep only the
# hand-present frames, and resample those to SEQUENCE_LENGTH, which is exactly
# what the live buffer does.
PROBE_FRAMES = 45
MIN_HAND_FRAMES = 8

# Must match the live canvas in MainActivity.kt (4:3, letterboxed not stretched).
CANVAS_W = 640
CANVAS_H = 480

# Left unmirrored to match v1's handedness convention and keep the right-wrist
# normalisation active. The app compensates by sending mirror=false for words.
MIRROR = False

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def load_labels():
    labels = {}
    with open(LABELS_CSV, "r", encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                labels[parts[0]] = parts[1]
    return labels


def letterbox(img):
    """Scale to fit CANVAS_W x CANVAS_H preserving aspect, pad with black.

    Mirrors buildCanvas() in MainActivity.kt so normalised landmark coordinates
    mean the same thing in training and at inference.
    """
    h, w = img.shape[:2]
    scale = min(CANVAS_W / w, CANVAS_H / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
    x0 = (CANVAS_W - nw) // 2
    y0 = (CANVAS_H - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def sample_frames(video_path, count=SEQUENCE_LENGTH):
    """Returns `count` frames spread uniformly across the whole clip."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    idxs = np.linspace(0, total - 1, count).round().astype(int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            # Hold the previous frame rather than dropping a timestep, so the
            # sequence keeps its fixed length and even spacing.
            if frames:
                frames.append(frames[-1])
            continue
        if MIRROR:
            frame = cv2.flip(frame, 1)
        frames.append(letterbox(frame))
    cap.release()
    return frames


def landmarks_for(frame, hand_det, pose_det):
    """226-d vector: 63 left hand + 63 right hand + 100 pose, then normalised."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    lh = np.zeros(63)
    rh = np.zeros(63)
    try:
        res = hand_det.detect(mp_image)
        if res.hand_landmarks:
            for idx, handedness in enumerate(res.handedness):
                label = handedness[0].category_name
                coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in res.hand_landmarks[idx]]
                ).flatten()
                if label == "Left":
                    lh = coords
                elif label == "Right":
                    rh = coords
    except Exception:
        pass

    pose = np.zeros(100)
    try:
        res = pose_det.detect(mp_image)
        if res.pose_landmarks:
            pose = np.array(
                [[p.x, p.y, p.z, p.visibility]
                 for p in res.pose_landmarks[0][:25]]
            ).flatten()
    except Exception:
        pass

    v = np.concatenate([lh, rh, pose])

    # Identical to v1 and to inference_service.dart: right wrist only, no
    # fallback. Kept deliberately so the app needs no normalisation change.
    wrist = v[63:66].copy()
    if np.any(wrist != 0):
        v[0:63:3] -= wrist[0]
        v[1:63:3] -= wrist[1]
        v[2:63:3] -= wrist[2]
        v[63:126:3] -= wrist[0]
        v[64:126:3] -= wrist[1]
        v[65:126:3] -= wrist[2]
    return v


def process_video(path, hand_det, pose_det):
    frames = sample_frames(path, PROBE_FRAMES)
    if len(frames) < MIN_HAND_FRAMES:
        return None

    vecs = [landmarks_for(f, hand_det, pose_det) for f in frames]

    # Keep only frames where a hand was actually found, matching the live buffer.
    kept = [v for v in vecs if np.any(v[0:126] != 0)]
    if len(kept) < MIN_HAND_FRAMES:
        return None

    # Resample the hand-present run to exactly SEQUENCE_LENGTH, nearest-neighbour
    # -- the same thing _resampleWindow() does on device, and safe across frames
    # where a hand appears or vanishes (blending those would invent poses).
    idx = np.linspace(0, len(kept) - 1, SEQUENCE_LENGTH).round().astype(int)
    return np.array([kept[i] for i in idx], dtype=np.float32)


def main():
    limit = None
    for a in sys.argv[1:]:
        if a.startswith("--limit="):
            limit = int(a.split("=")[1])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    labels = load_labels()

    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL,
                                 delegate=BaseOptions.Delegate.CPU),
        running_mode=VisionRunningMode.IMAGE, num_hands=2)
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL,
                                 delegate=BaseOptions.Delegate.CPU),
        running_mode=VisionRunningMode.IMAGE)

    sequences, targets = [], []
    skipped = 0
    class_folders = sorted(os.listdir(RAW_DIR), key=lambda x: int(x))
    if limit:
        class_folders = class_folders[:limit]

    with HandLandmarker.create_from_options(hand_options) as hand_det, \
         PoseLandmarker.create_from_options(pose_options) as pose_det:

        for folder in class_folders:
            label_id = int(folder)
            folder_path = os.path.join(RAW_DIR, folder)
            vids = [p for p in glob.glob(os.path.join(folder_path, "*"))
                    if p.lower().endswith((".mov", ".mp4"))]
            name = labels.get(folder, f"unknown_{folder}")
            print(f"class {label_id:3d} | {name:28s} | {len(vids)} clips",
                  flush=True)

            for v in vids:
                seq = process_video(v, hand_det, pose_det)
                if seq is None:
                    skipped += 1
                    print(f"  skipped {os.path.basename(v)}", flush=True)
                    continue
                sequences.append(seq)
                targets.append(label_id)

            if label_id % 10 == 9:
                np.save(os.path.join(OUTPUT_DIR, "X_partial.npy"),
                        np.array(sequences, dtype=np.float32))
                np.save(os.path.join(OUTPUT_DIR, "y_partial.npy"),
                        np.array(targets))
                with open(CHECKPOINT, "w") as f:
                    json.dump({"last_class": label_id,
                               "count": len(sequences)}, f)
                print(f"  [checkpoint] {len(sequences)} sequences", flush=True)

    X = np.array(sequences, dtype=np.float32)
    y = np.array(targets)
    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)

    for f in ["X_partial.npy", "y_partial.npy"]:
        p = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(p):
            os.remove(p)
    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    print(f"\nDone. {len(sequences)} sequences, {skipped} skipped")
    print(f"X {X.shape}  y {y.shape}")


if __name__ == "__main__":
    main()
