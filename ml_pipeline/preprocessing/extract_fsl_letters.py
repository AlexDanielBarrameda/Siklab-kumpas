import os
import numpy as np
import mediapipe as mp
import cv2
import json
import signal

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["DISPLAY"] = ""

# Paths
RAW_DIR = "../data/raw/fsl_letters/Collated"
OUTPUT_DIR = "../data/processed/fsl_letters"
CHECKPOINT_FILE = "../data/processed/fsl_letters/checkpoint.json"

LANDMARK_DIM = 226
HAND_MODEL = "hand_landmarker.task"
POSE_MODEL = "pose_landmarker_full.task"

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LABEL_MAP = {letter: idx for idx, letter in enumerate(LETTERS)}

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def timeout_handler(signum, frame):
    raise TimeoutError("Image processing timed out")


def process_single_image(image_path):
    """Process one image with a 30-second timeout."""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)

    try:
        frame = cv2.imread(image_path)
        if frame is None:
            signal.alarm(0)
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=HAND_MODEL, delegate=BaseOptions.Delegate.CPU
            ),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2,
        )
        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=POSE_MODEL, delegate=BaseOptions.Delegate.CPU
            ),
            running_mode=VisionRunningMode.IMAGE,
        )

        lh = np.zeros(63)
        rh = np.zeros(63)

        with HandLandmarker.create_from_options(hand_options) as hand_det:
            try:
                hand_result = hand_det.detect(mp_image)
                if hand_result.hand_landmarks:
                    for idx, handedness in enumerate(hand_result.handedness):
                        hand_label = handedness[0].category_name
                        coords = np.array(
                            [
                                [lm.x, lm.y, lm.z]
                                for lm in hand_result.hand_landmarks[idx]
                            ]
                        ).flatten()
                        if hand_label == "Left":
                            lh = coords
                        elif hand_label == "Right":
                            rh = coords
            except Exception:
                pass

        pose = np.zeros(100)
        with PoseLandmarker.create_from_options(pose_options) as pose_det:
            try:
                pose_result = pose_det.detect(mp_image)
                if pose_result.pose_landmarks:
                    pose = np.array(
                        [
                            [lm.x, lm.y, lm.z, lm.visibility]
                            for lm in pose_result.pose_landmarks[0][:25]
                        ]
                    ).flatten()
            except Exception:
                pass

        landmarks = np.concatenate([lh, rh, pose])

        # Normalize relative to wrist
        wrist = landmarks[63:66].copy()
        if np.any(wrist != 0):
            landmarks = landmarks.copy()
            landmarks[0:63:3] -= wrist[0]
            landmarks[1:63:3] -= wrist[1]
            landmarks[2:63:3] -= wrist[2]
            landmarks[63:126:3] -= wrist[0]
            landmarks[64:126:3] -= wrist[1]
            landmarks[65:126:3] -= wrist[2]

        signal.alarm(0)

        # Only return if at least hands were detected
        if np.any(landmarks[:126] != 0):
            return landmarks
        return None

    except TimeoutError:
        signal.alarm(0)
        return None
    except Exception as e:
        signal.alarm(0)
        print(f"  ERROR: {image_path}: {e}", flush=True)
        return None


def save_checkpoint(sequences, targets, last_letter):
    X = np.array(sequences)
    y = np.array(targets)
    np.save(os.path.join(OUTPUT_DIR, "X_partial.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y_partial.npy"), y)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"last_letter": last_letter, "count": len(sequences)}, f)
    print(
        f"  [CHECKPOINT] Saved {len(sequences)} samples up to letter {last_letter}",
        flush=True,
    )


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return [], [], ""
    with open(CHECKPOINT_FILE, "r") as f:
        info = json.load(f)
    last_letter = info["last_letter"]
    X = np.load(os.path.join(OUTPUT_DIR, "X_partial.npy"))
    y = np.load(os.path.join(OUTPUT_DIR, "y_partial.npy"))
    print(
        f"  [RESUME] Resuming after letter {last_letter} with {len(X)} samples",
        flush=True,
    )
    return list(X), list(y), last_letter


def process_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sequences, targets, last_letter = load_checkpoint()
    skipped = 0

    # Find where to resume
    start_idx = 0
    if last_letter:
        if last_letter in LETTERS:
            start_idx = LETTERS.index(last_letter) + 1

    for letter in LETTERS[start_idx:]:
        letter_dir = os.path.join(RAW_DIR, letter)

        if not os.path.exists(letter_dir):
            print(f"WARNING: folder not found for letter {letter}", flush=True)
            continue

        image_files = [
            f
            for f in os.listdir(letter_dir)
            if f.endswith(".jpg") or f.endswith(".png")
        ]

        label_id = LABEL_MAP[letter]
        print(
            f"Processing letter {letter} (id={label_id:2d}) | {len(image_files)} images",
            flush=True,
        )

        for image_file in image_files:
            image_path = os.path.join(letter_dir, image_file)
            landmarks = process_single_image(image_path)

            if landmarks is not None:
                sequences.append(landmarks)
                targets.append(label_id)
            else:
                skipped += 1

        print(f"  Done letter {letter} — total so far: {len(sequences)}", flush=True)

        # Save checkpoint every 5 letters
        if (LETTERS.index(letter) + 1) % 5 == 0:
            save_checkpoint(sequences, targets, letter)

    # Final save
    X = np.array(sequences)
    y = np.array(targets)

    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)

    # Save label map
    with open(os.path.join(OUTPUT_DIR, "labels.txt"), "w") as f:
        for letter, idx in LABEL_MAP.items():
            f.write(f"{idx},{letter}\n")

    # Clean up checkpoints
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    for f in ["X_partial.npy", "y_partial.npy"]:
        path = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(path):
            os.remove(path)

    print(f"\nDone! Saved {len(sequences)} samples, skipped {skipped}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")


if __name__ == "__main__":
    process_all()
