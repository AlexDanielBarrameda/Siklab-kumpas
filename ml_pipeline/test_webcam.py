import os

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
from collections import deque

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_DIR  = r"E:\Thesis\Siklab-kumpas\ml_pipeline\models"
LABELS_DIR = r"E:\Thesis\Siklab-kumpas\flutter_app\assets\labels"
ASSETS_DIR = r"E:\Thesis\Siklab-kumpas\flutter_app\assets\models"

FSL105_TFLITE  = os.path.join(ASSETS_DIR, "fsl105.tflite")
LETTERS_TFLITE = os.path.join(ASSETS_DIR, "fsl_letters.tflite")
FSL105_LABELS  = os.path.join(LABELS_DIR, "fsl105_labels.txt")
LETTERS_LABELS = os.path.join(LABELS_DIR, "fsl_letters_labels.txt")
HAND_MODEL     = r"E:\Thesis\Siklab-kumpas\ml_pipeline\preprocessing\hand_landmarker.task"
POSE_MODEL     = r"E:\Thesis\Siklab-kumpas\ml_pipeline\preprocessing\pose_landmarker_full.task"

# ── Settings ───────────────────────────────────────────────────────────────
SEQUENCE_LENGTH     = 30
LANDMARK_DIM        = 226
CONFIDENCE_THRESHOLD = 0.60
MODE                = "fsl105"  # start in word mode


# ── Load labels ────────────────────────────────────────────────────────────
def load_labels(path):
    labels = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                labels[int(parts[0])] = ",".join(parts[1:]).strip()
    return labels

fsl105_labels  = load_labels(FSL105_LABELS)
letters_labels = load_labels(LETTERS_LABELS)
print(f"Loaded {len(fsl105_labels)} FSL-105 labels")
print(f"Loaded {len(letters_labels)} FSL Letters labels")


# ── Load TFLite models ─────────────────────────────────────────────────────
fsl105_interp = tf.lite.Interpreter(model_path=FSL105_TFLITE)
fsl105_interp.allocate_tensors()
fsl105_in  = fsl105_interp.get_input_details()
fsl105_out = fsl105_interp.get_output_details()
print(f"FSL-105 model loaded | input: {fsl105_in[0]['shape']}")

letters_interp = tf.lite.Interpreter(model_path=LETTERS_TFLITE)
letters_interp.allocate_tensors()
letters_in  = letters_interp.get_input_details()
letters_out = letters_interp.get_output_details()
print(f"Letters model loaded | input: {letters_in[0]['shape']}")


# ── MediaPipe setup ────────────────────────────────────────────────────────
BaseOptions          = mp.tasks.BaseOptions
HandLandmarker       = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
PoseLandmarker       = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode    = mp.tasks.vision.RunningMode

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=HAND_MODEL,
        delegate=BaseOptions.Delegate.CPU
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
)
pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=POSE_MODEL,
        delegate=BaseOptions.Delegate.CPU
    ),
    running_mode=VisionRunningMode.IMAGE,
)


# ── Landmark extraction ────────────────────────────────────────────────────
def extract_landmarks(frame, hand_det, pose_det):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    lh = np.zeros(63)
    rh = np.zeros(63)

    hand_result = hand_det.detect(mp_image)
    if hand_result.hand_landmarks:
        for idx, handedness in enumerate(hand_result.handedness):
            label = handedness[0].category_name
            lms = hand_result.hand_landmarks[idx]
            coords = np.array([[lm.x, lm.y, lm.z] for lm in lms]).flatten()
            if label == "Left":
                lh = coords
            else:
                rh = coords

    pose = np.zeros(100)
    pose_result = pose_det.detect(mp_image)
    if pose_result.pose_landmarks and pose_result.pose_landmarks:
        pts = pose_result.pose_landmarks[0][:25]
        pose = np.array([[p.x, p.y, p.z,
                          p.visibility if hasattr(p, "visibility") else 1.0]
                         for p in pts]).flatten()

    landmarks = np.concatenate([lh, rh, pose])

    # Normalize relative to right wrist
    wrist = landmarks[63:66].copy()
    if np.any(wrist != 0):
        landmarks = landmarks.copy()
        landmarks[0:63:3]   -= wrist[0]
        landmarks[1:63:3]   -= wrist[1]
        landmarks[2:63:3]   -= wrist[2]
        landmarks[63:126:3] -= wrist[0]
        landmarks[64:126:3] -= wrist[1]
        landmarks[65:126:3] -= wrist[2]

    return landmarks


# ── Inference ──────────────────────────────────────────────────────────────
def run_fsl105(frame_buffer):
    if len(frame_buffer) < SEQUENCE_LENGTH:
        return None, 0.0
    seq = np.array(list(frame_buffer), dtype=np.float32)[np.newaxis]  # (1,30,226)
    fsl105_interp.set_tensor(fsl105_in[0]["index"], seq)
    fsl105_interp.invoke()
    scores = fsl105_interp.get_tensor(fsl105_out[0]["index"])[0]
    idx = int(np.argmax(scores))
    return fsl105_labels.get(idx, f"Unknown({idx})"), float(scores[idx])


def run_letters(landmarks):
    inp = landmarks.astype(np.float32)[np.newaxis]  # (1,226)
    letters_interp.set_tensor(letters_in[0]["index"], inp)
    letters_interp.invoke()
    scores = letters_interp.get_tensor(letters_out[0]["index"])[0]
    idx = int(np.argmax(scores))
    return letters_labels.get(idx, f"Unknown({idx})"), float(scores[idx])


# ── Draw overlay ───────────────────────────────────────────────────────────
def draw_overlay(frame, mode, result_label, confidence,
                 has_hands, fps, frame_count, sequence_len):
    h, w = frame.shape[:2]

    # Dark overlay bar at top
    cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)

    # Mode indicator
    mode_text = "MODE: FSL-105 WORDS" if mode == "fsl105" else "MODE: FSL LETTERS"
    mode_color = (0, 200, 255) if mode == "fsl105" else (0, 255, 150)
    cv2.putText(frame, mode_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, mode_color, 2)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # Hand detection status
    hand_color = (0, 255, 0) if has_hands else (0, 0, 255)
    hand_text  = "HAND: DETECTED" if has_hands else "HAND: NOT FOUND"
    cv2.putText(frame, hand_text, (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, hand_color, 1)

    # Buffer progress (only for FSL-105)
    if mode == "fsl105":
        bar_w = int((sequence_len / SEQUENCE_LENGTH) * 200)
        cv2.rectangle(frame, (w - 220, 40), (w - 20, 60), (50, 50, 50), -1)
        cv2.rectangle(frame, (w - 220, 40), (w - 220 + bar_w, 60),
                      (0, 200, 100), -1)
        cv2.putText(frame, f"BUF:{sequence_len}/{SEQUENCE_LENGTH}",
                    (w - 215, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (200, 200, 200), 1)

    # Result display at bottom
    if result_label and confidence >= CONFIDENCE_THRESHOLD:
        cv2.rectangle(frame, (0, h - 90), (w, h), (20, 20, 20), -1)
        conf_pct = f"{confidence * 100:.1f}%"

        # Confidence bar
        bar_len = int(confidence * (w - 40))
        bar_color = (0, 200, 100) if confidence >= 0.75 else \
                    (0, 165, 255) if confidence >= 0.60 else (0, 0, 255)
        cv2.rectangle(frame, (20, h - 85), (20 + bar_len, h - 70),
                      bar_color, -1)

        cv2.putText(frame, result_label, (20, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(frame, conf_pct, (20, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, bar_color, 2)
    elif result_label:
        cv2.rectangle(frame, (0, h - 50), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, f"Low confidence: {result_label} ({confidence*100:.1f}%)",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (100, 100, 255), 1)

    # Controls hint
    cv2.putText(frame,
                "M: switch mode | C: clear buffer | Q: quit",
                (10, h - (100 if result_label else 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    return frame


# ── Main loop ──────────────────────────────────────────────────────────────
def main():
    global MODE

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_buffer  = deque(maxlen=SEQUENCE_LENGTH)
    result_label  = ""
    confidence    = 0.0
    last_result_t = 0
    fps           = 0.0
    prev_t        = time.time()
    frame_count   = 0
    skip          = 0  # frame skip counter

    print("\n=== SIKLAB-KUMPAS Webcam Test ===")
    print("M — switch between FSL-105 / FSL Letters")
    print("C — clear frame buffer")
    print("Q — quit")
    print("=================================\n")

    with HandLandmarker.create_from_options(hand_options) as hand_det, \
         PoseLandmarker.create_from_options(pose_options) as pose_det:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame")
                break

            frame = cv2.flip(frame, 1)  # mirror for selfie-view
            frame_count += 1
            skip += 1

            # FPS calculation
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-5))
            prev_t = now

            # Process every 2nd frame to reduce CPU load
            has_hands = False
            if skip % 2 == 0:
                landmarks = extract_landmarks(frame, hand_det, pose_det)
                has_hands = np.any(landmarks[:126] != 0)

                if has_hands:
                    frame_buffer.append(landmarks)

                    if MODE == "fsl105":
                        if len(frame_buffer) == SEQUENCE_LENGTH:
                            lbl, conf = run_fsl105(frame_buffer)
                            now2 = time.time()
                            if lbl != result_label or \
                               now2 - last_result_t > 2.0:
                                result_label  = lbl
                                confidence    = conf
                                last_result_t = now2
                                if conf >= CONFIDENCE_THRESHOLD:
                                    print(f"[FSL-105]  {lbl}  ({conf*100:.1f}%)")

                    else:  # letters mode — single frame
                        lbl, conf = run_letters(landmarks)
                        now2 = time.time()
                        if lbl != result_label or \
                           now2 - last_result_t > 1.0:
                            result_label  = lbl
                            confidence    = conf
                            last_result_t = now2
                            if conf >= CONFIDENCE_THRESHOLD:
                                print(f"[LETTERS]  {lbl}  ({conf*100:.1f}%)")
                else:
                    # No hands — slowly drain buffer
                    if frame_buffer:
                        frame_buffer.popleft()

            # Draw UI
            frame = draw_overlay(
                frame, MODE, result_label, confidence,
                has_hands, fps, frame_count, len(frame_buffer)
            )
            cv2.imshow("SIKLAB-KUMPAS | Webcam Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                MODE = "letters" if MODE == "fsl105" else "fsl105"
                frame_buffer.clear()
                result_label = ""
                confidence   = 0.0
                print(f"Switched to: {MODE.upper()}")
            elif key == ord("c"):
                frame_buffer.clear()
                result_label = ""
                confidence   = 0.0
                print("Buffer cleared")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()