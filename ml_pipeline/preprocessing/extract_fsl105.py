import os
import sys
import numpy as np
import mediapipe as mp
import cv2
import av
import json
import signal

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["DISPLAY"] = ""

# Paths
RAW_DIR = "../data/raw/fsl_105/fsl_105_dataset/clips"
LABELS_CSV = "../data/raw/fsl_105/fsl_105_dataset/labels.csv"
OUTPUT_DIR = "../data/processed/fsl_105"
CHECKPOINT_FILE = "../data/processed/fsl_105/checkpoint.json"

SEQUENCE_LENGTH = 30
LANDMARK_DIM = 226
HAND_MODEL = "hand_landmarker.task"
POSE_MODEL = "pose_landmarker_full.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def load_labels(labels_csv):
    labels = {}
    with open(labels_csv, "r", encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            labels[parts[0]] = parts[1]
    return labels


def read_video_frames(video_path):
    frames = []
    try:
        container = av.open(video_path)
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="bgr24")
            img = cv2.resize(img, (640, 480))
            frames.append(img)
        container.close()
    except Exception as e:
        print(f"  ERROR reading {video_path}: {e}", flush=True)
    return frames


def timeout_handler(signum, frame):
    raise TimeoutError("Video processing timed out")


def process_single_video(video_path):
    """Process one video with a 60-second timeout."""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)  # 60 second timeout per video

    try:
        frames_raw = read_video_frames(video_path)
        if len(frames_raw) == 0:
            signal.alarm(0)
            return None

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

        frames = []
        with HandLandmarker.create_from_options(
            hand_options
        ) as hand_det, PoseLandmarker.create_from_options(pose_options) as pose_det:

            for frame in frames_raw:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                lh = np.zeros(63)
                rh = np.zeros(63)
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

                # Normalize
                wrist = landmarks[63:66].copy()
                if np.any(wrist != 0):
                    landmarks = landmarks.copy()
                    landmarks[0:63:3] -= wrist[0]
                    landmarks[1:63:3] -= wrist[1]
                    landmarks[2:63:3] -= wrist[2]
                    landmarks[63:126:3] -= wrist[0]
                    landmarks[64:126:3] -= wrist[1]
                    landmarks[65:126:3] -= wrist[2]

                frames.append(landmarks)

        signal.alarm(0)  # cancel timeout

        if len(frames) == 0:
            return None

        if len(frames) < SEQUENCE_LENGTH:
            padding = [np.zeros(LANDMARK_DIM)] * (SEQUENCE_LENGTH - len(frames))
            frames = padding + frames
        else:
            start = (len(frames) - SEQUENCE_LENGTH) // 2
            frames = frames[start : start + SEQUENCE_LENGTH]

        return np.array(frames)

    except TimeoutError:
        signal.alarm(0)
        print(f"  TIMEOUT: {video_path}", flush=True)
        return None
    except Exception as e:
        signal.alarm(0)
        print(f"  ERROR: {video_path}: {e}", flush=True)
        return None


def save_checkpoint(sequences, targets, last_class):
    """Save progress so we can resume if interrupted."""
    X = np.array(sequences)
    y = np.array(targets)
    np.save(os.path.join(OUTPUT_DIR, "X_partial.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y_partial.npy"), y)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"last_class": last_class, "count": len(sequences)}, f)
    print(
        f"  [CHECKPOINT] Saved {len(sequences)} sequences up to class {last_class}",
        flush=True,
    )


def load_checkpoint():
    """Resume from last checkpoint if available."""
    if not os.path.exists(CHECKPOINT_FILE):
        return [], [], -1
    with open(CHECKPOINT_FILE, "r") as f:
        info = json.load(f)
    last_class = info["last_class"]
    X = np.load(os.path.join(OUTPUT_DIR, "X_partial.npy"))
    y = np.load(os.path.join(OUTPUT_DIR, "y_partial.npy"))
    print(
        f"  [RESUME] Resuming from class {last_class + 1} with {len(X)} sequences",
        flush=True,
    )
    return list(X), list(y), last_class


def process_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    labels = load_labels(LABELS_CSV)

    # Resume from checkpoint if available
    sequences, targets, last_class = load_checkpoint()
    skipped = 0

    class_folders = sorted(os.listdir(RAW_DIR), key=lambda x: int(x))

    for folder in class_folders:
        label_id = int(folder)

        # Skip already processed classes
        if label_id <= last_class:
            continue

        folder_path = os.path.join(RAW_DIR, folder)
        label_name = labels.get(folder, f"unknown_{folder}")

        video_files = [
            f
            for f in os.listdir(folder_path)
            if f.endswith(".MOV") or f.endswith(".mp4")
        ]

        print(
            f"Processing class {label_id:3d} | {label_name:30s} | {len(video_files)} videos",
            flush=True,
        )

        for video_file in video_files:
            video_path = os.path.join(folder_path, video_file)
            sequence = process_single_video(video_path)

            if sequence is not None:
                sequences.append(sequence)
                targets.append(label_id)
            else:
                skipped += 1
                print(f"  WARNING: skipped {video_file}", flush=True)

        print(f"  Done class {label_id} — total so far: {len(sequences)}", flush=True)

        # Save checkpoint every 10 classes
        if label_id % 10 == 9:
            save_checkpoint(sequences, targets, label_id)

    # Final save
    X = np.array(sequences)
    y = np.array(targets)
    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)

    # Clean up checkpoint files
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    for f in ["X_partial.npy", "y_partial.npy"]:
        path = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(path):
            os.remove(path)

    print(f"\nDone! Saved {len(sequences)} sequences, skipped {skipped}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")


if __name__ == "__main__":
    process_all()
