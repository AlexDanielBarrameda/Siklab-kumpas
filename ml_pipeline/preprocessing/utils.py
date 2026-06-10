import os

# Force CPU usage
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

import mediapipe as mp
import numpy as np
import cv2
import av

# =========================
# CONSTANTS
# =========================

SEQUENCE_LENGTH = 30
LANDMARK_DIM = 226  # 63 LH + 63 RH + 100 Pose

# Model paths
HAND_MODEL = "hand_landmarker.task"
POSE_MODEL = "pose_landmarker_full.task"

# =========================
# MEDIAPIPE BASE
# =========================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# =========================
# HAND DETECTOR
# =========================

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=HAND_MODEL,
        delegate=BaseOptions.Delegate.CPU,
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
)

# Create ONCE globally
hand_detector = HandLandmarker.create_from_options(hand_options)

# =========================
# POSE DETECTOR
# =========================

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=POSE_MODEL,
        delegate=BaseOptions.Delegate.CPU,
    ),
    running_mode=VisionRunningMode.IMAGE,
)

# Create ONCE globally
pose_detector = PoseLandmarker.create_from_options(pose_options)


# =========================
# VIDEO READER
# =========================


def read_video_frames(video_path):
    """
    Reads frames from a video file using PyAV.
    Returns a list of BGR numpy arrays.
    """

    frames = []

    try:
        container = av.open(video_path)

        for frame in container.decode(video=0):

            img = frame.to_ndarray(format="bgr24")

            # Resize for stability + speed
            img = cv2.resize(img, (640, 480))

            frames.append(img)

        container.close()

    except Exception as e:
        print(f"ERROR reading {video_path}: {e}")

    return frames


# =========================
# LANDMARK EXTRACTION
# =========================


def extract_landmarks(frame):
    """
    Takes a single BGR frame.
    Returns a 226-dimensional landmark vector.
    """

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # -------------------------
    # HANDS
    # -------------------------

    lh = np.zeros(63)
    rh = np.zeros(63)

    try:

        hand_result = hand_detector.detect(mp_image)

        if hand_result.hand_landmarks:

            for idx, handedness in enumerate(hand_result.handedness):

                hand_label = handedness[0].category_name

                landmarks = hand_result.hand_landmarks[idx]

                coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()

                if hand_label == "Left":
                    lh = coords

                elif hand_label == "Right":
                    rh = coords

    except Exception as e:
        print(f"HAND DETECTION ERROR: {e}")

    # -------------------------
    # POSE
    # -------------------------

    pose = np.zeros(100)

    try:

        pose_result = pose_detector.detect(mp_image)

        if pose_result.pose_landmarks:

            landmarks = pose_result.pose_landmarks[0][:25]

            pose = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]
            ).flatten()

    except Exception as e:
        print(f"POSE DETECTION ERROR: {e}")

    return np.concatenate([lh, rh, pose])


# =========================
# NORMALIZATION
# =========================


def normalize_landmarks(landmarks):
    """
    Normalize landmarks relative to wrist position.
    Makes features position-invariant.
    """

    landmarks = landmarks.copy()

    # Left wrist exists
    if np.any(landmarks[0:3] != 0):
        wrist = landmarks[0:3]

    # Right wrist exists
    elif np.any(landmarks[63:66] != 0):
        wrist = landmarks[63:66]

    else:
        return landmarks

    # -------------------------
    # LEFT HAND
    # -------------------------

    landmarks[0:63:3] -= wrist[0]
    landmarks[1:63:3] -= wrist[1]
    landmarks[2:63:3] -= wrist[2]

    # -------------------------
    # RIGHT HAND
    # -------------------------

    landmarks[63:126:3] -= wrist[0]
    landmarks[64:126:3] -= wrist[1]
    landmarks[65:126:3] -= wrist[2]

    return landmarks
