"""FSL-105 word model, v2: trained on full-sign sequences with augmentation.

Differences from train_fsl105_unroll.py (which produced the 72.5% model):

1. Consumes data/processed/fsl_105_v2 (extract_fsl105_v2.py) -- sequences that
   span the whole sign instead of 0.50s from its middle.

2. Adds landmark-space augmentation. The v1 script had none at all: load,
   train_test_split, fit. Augmentation is cheap on landmarks and targets a gap
   measured directly against the device:

     - SCALE. Training hand span is ~0.106 (median) while the phone measures
       0.20-0.29 -- the FSL-105 signer stands much further back than a user sits
       from a handheld phone. These features are wrist-relative but never
       scale-normalised, so apparent size multiplies every coordinate.
     - TRANSLATION. The signer is centred in the clips; a handheld phone is not.
     - ROTATION. Small camera tilt.
     - TEMPORAL. Live frames arrive at ~4 fps and are resampled to 30, so real
       sequences are jittery and partly duplicated. Frame dropout and time warp
       simulate that.
     - NOISE. Landmark jitter from a lower-quality live detection.

   Augmentation is applied to the hand blocks and the pose x/y/z, never to pose
   visibility (index 3 of each pose landmark), and never to all-zero frames --
   a zero block means "no hand detected" and must stay zero.

Architecture is unchanged from the v1 model: unroll=True is required for TFLite
conversion without the Flex delegate.
"""

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data/processed/fsl_105_v2")
OUT_KERAS = os.path.join(HERE, "fsl105_v2aug_best.keras")
OUT_TFLITE = os.path.join(HERE, "fsl105_v2aug.tflite")

SEQ, DIM, NCLASS = 30, 226, 105
EPOCHS = 150
BATCH = 16
AUG_PER_SAMPLE = 3          # extra augmented copies of each training sequence

rng = np.random.default_rng(42)

# ── augmentation ──────────────────────────────────────────────────────────
HAND = slice(0, 126)
POSE_XYZ = [126 + 4 * k + c for k in range(25) for c in (0, 1, 2)]


def _augment_one(seq):
    """seq: (30, 226) -> augmented copy."""
    s = seq.copy()

    # Which timesteps actually carry a detected hand.
    live = np.any(s[:, HAND] != 0, axis=1)

    # --- spatial: scale about the wrist anchor, translate, rotate ---
    # Training hand span is ~0.10 median; the device measures 0.20-0.33. A
    # 0.75-1.9 range was tried first and left only 21% of augmented samples
    # inside the live band, because its median scale (~1.3) lands at 0.13.
    # This range spans "as filmed" through "3x closer", so the augmented set
    # covers both the clean test distribution and the phone's.
    scale = rng.uniform(0.8, 3.0)
    theta = np.deg2rad(rng.uniform(-12, 12))
    tx, ty = rng.uniform(-0.05, 0.05, 2)
    ca, sa = np.cos(theta), np.sin(theta)

    for t in range(SEQ):
        if not live[t]:
            continue                     # keep "no hand" frames exactly zero
        for base in (0, 63):
            blk = s[t, base:base + 63].reshape(21, 3)
            if not np.any(blk):
                continue
            anchor = blk[0].copy()
            off = blk - anchor
            off[:, :2] *= scale
            off[:, 2] *= scale
            x, y = off[:, 0].copy(), off[:, 1].copy()
            off[:, 0] = x * ca - y * sa
            off[:, 1] = x * sa + y * ca
            blk = anchor + off
            blk[:, 0] += tx
            blk[:, 1] += ty
            s[t, base:base + 63] = blk.reshape(-1)

    # Pose gets the same translation and a milder scale about its own centroid,
    # so hands and body stay geometrically consistent. Visibility is untouched.
    pose = s[:, POSE_XYZ].reshape(SEQ, 25, 3)
    nz = np.any(pose != 0, axis=(1, 2))
    if nz.any():
        centre = pose[nz].reshape(-1, 3).mean(axis=0)
        for t in range(SEQ):
            if not nz[t]:
                continue
            p = pose[t] - centre
            p[:, :2] *= scale
            p = p + centre
            p[:, 0] += tx
            p[:, 1] += ty
            pose[t] = p
        s[:, POSE_XYZ] = pose.reshape(SEQ, -1)

    # --- landmark jitter ---
    noise = rng.normal(0, 0.004, size=s[:, HAND].shape)
    mask = (s[:, HAND] != 0)
    s[:, HAND] += noise * mask

    # --- temporal: drop frames and re-stretch, mimicking ~4 fps live capture ---
    if rng.random() < 0.7:
        keep = np.where(rng.random(SEQ) > rng.uniform(0.1, 0.4))[0]
        if len(keep) >= 6:
            idx = np.linspace(0, len(keep) - 1, SEQ).round().astype(int)
            s = s[keep[idx]]

    return s.astype(np.float32)


def augment_set(X, y, copies):
    outX, outY = [X], [y]
    for _ in range(copies):
        outX.append(np.stack([_augment_one(v) for v in X]))
        outY.append(y)
    return np.concatenate(outX), np.concatenate(outY)


def build_model():
    """Architecture identical to v1. unroll=True is required for TFLite
    conversion without the Flex delegate."""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True, unroll=True,
                             input_shape=(SEQ, DIM)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64, return_sequences=False, unroll=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(NCLASS, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def main():
    # --canon trains on the canonicalised tensors (canonicalize.py), where every
    # single-hand frame sits in the rh block with one chirality. Without it the
    # model splits into two regimes and anything the device sends (always
    # lh-only) collapses onto the few classes that own the rare regime: measured
    # 99.0% -> 13.5% when clean clips are fed through the lh-only path.
    canon = "--canon" in sys.argv
    xf, yf = ("X_canon.npy", "y_canon.npy") if canon else ("X.npy", "y.npy")
    global OUT_KERAS, OUT_TFLITE
    if canon:
        OUT_KERAS = os.path.join(HERE, "fsl105_v2canon_best.keras")
        OUT_TFLITE = os.path.join(HERE, "fsl105_v2canon.tflite")

    X = np.load(os.path.join(DATA, xf)).astype(np.float32)
    y = np.load(os.path.join(DATA, yf))
    print(f"loaded {xf} {X.shape}, {len(np.unique(y))} classes"
          f"{'  [canonicalised]' if canon else ''}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Augment ONLY the training split - val/test stay clean so the reported
    # numbers remain comparable with the 72.5% baseline.
    X_train, y_train = augment_set(X_train, y_train, AUG_PER_SAMPLE)
    print(f"train after augmentation: {X_train.shape}")
    print(f"val {X_val.shape}  test {X_test.shape}")

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=7, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(OUT_KERAS, monitor="val_accuracy",
                                           save_best_only=True),
    ]

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH, callbacks=callbacks, verbose=2)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy (clean, unaugmented): {acc:.4f}")
    print(f"v1 baseline for comparison        : 0.7250")

    # TFLite (no Flex delegate, thanks to unroll=True)
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    conv.target_spec.supported_types = [tf.float16]
    tfl = conv.convert()
    with open(OUT_TFLITE, "wb") as f:
        f.write(tfl)
    print(f"TFLite written: {OUT_TFLITE}  ({len(tfl)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
