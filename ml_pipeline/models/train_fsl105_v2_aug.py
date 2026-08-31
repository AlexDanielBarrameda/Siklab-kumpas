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


# Per-corpus calibration. These MUST be passed explicitly for a new corpus:
# ASL inherited the FSL numbers by default and both were wrong for it.
#
#   scale_range   FSL training hand span is ~0.10 median against a device that
#                 measures 0.20-0.33, so FSL needs scaling UP. ASL's span is
#                 already 0.214 and matches the device, so the same range
#                 stretched ASL hands to 0.64 — physically implausible, and a
#                 large share of its augmented set was junk.
#
#   keep_frames   how many DISTINCT frames to subsample to before stretching
#                 back to SEQ, simulating a low capture rate. The device
#                 delivers 4-6 distinct frames per sign; the original 0.1-0.4
#                 drop rate simulated 18-27, so the model never saw its own
#                 operating condition.
#   frame_range   how much of the canvas the SIGNER'S BODY fills, scaled about
#                 the canvas centre. The pose block is stored as absolute
#                 letterboxed image coordinates — the wrist normalisation only
#                 touches indices 0..126 — so framing feeds straight into 100 of
#                 the 226 dims, and substituting an uninformative pose costs 34
#                 accuracy points. Measured median shoulder width:
#
#                     FSL-105 v2   0.222      ASL Citizen  0.410
#
#                 Both corpora are landscape (FSL-105 source is 640x360, so its
#                 extractor letterboxes it at scale 1.0). The difference is
#                 filming DISTANCE, not orientation: the FSL signer stands far
#                 enough back to occupy roughly half the apparent width of an
#                 ASL Citizen signer.
#
#                 That gap happens to land FSL in the same regime the device
#                 produces by a different route — MainActivity rotates the
#                 sensor frame upright to portrait and letterboxes it into the
#                 480x360 canvas at scale 0.5625, so the signer fills 56% of the
#                 width. Measured device signing posture (-1.14) sits inside
#                 FSL's range (p5 -1.18) and outside ASL's (p5 -0.78), which is
#                 why the shared pipeline works for FSL words and fails for ASL
#                 words. Simulating the device letterbox on ASL test clips costs
#                 9.6 points on its own. FSL already sits in the right regime,
#                 so its range stays tight around 1.0.
FSL_AUG = dict(scale_range=(0.8, 3.0), keep_frames=(18, 27), drop_prob=0.7,
               frame_range=(0.92, 1.08))
ASL_AUG = dict(scale_range=(0.7, 1.4), keep_frames=(4, 15), drop_prob=0.85,
               frame_range=(0.50, 1.05))

# Measured under the device condition (device framing + 5 distinct frames),
# top-1 on the signer-disjoint test split:
#
#     baseline: wide scale, NO framing aug     48.06%
#     narrow scale, NO framing aug             43.46%
#     narrow scale + framing aug (ASL_AUG)     59.86%
#     wide scale   + framing aug (this)        59.13%
#
# Read the first two rows alone and the wide range looks essential — narrowing
# it to a physically plausible (0.7, 1.4) cost 4.6 points. The last two rows
# show why that reading was wrong: once framing is augmented explicitly the two
# ranges tie. The wide range was never buying scale robustness on its own, it
# was partially compensating for the framing gap, because scaling the whole
# body up and down incidentally covers some of the same variation.
#
# So ASL_AUG (plausible scale + explicit framing) is the one to use; this
# variant is kept only to document that the wide range adds nothing once the
# real cause is addressed.
ASL_AUG_WIDE = dict(scale_range=(0.8, 3.0), keep_frames=(4, 15), drop_prob=0.85,
                    frame_range=(0.50, 1.05))

# For body-normalised features (body_normalize.py), where every coordinate is
# already expressed in shoulder widths.
#
# frame_range MUST be disabled here. The framing transform scales pose about the
# canvas centre (0.5, 0.5), which is only meaningful for absolute image
# coordinates; body-normalised pose is centred on 0, so the same transform would
# translate the whole body by 0.5*(1-s) and corrupt it. Framing invariance is
# structural now rather than learned, so the augmentation is redundant as well as
# wrong.
#
# scale_range stays mild for regularisation only — apparent size is already
# normalised out, so a wide range would reintroduce the nuisance the transform
# exists to remove.
ASL_AUG_BODYNORM = dict(scale_range=(0.9, 1.15), keep_frames=(4, 15),
                        drop_prob=0.85, frame_range=(1.0, 1.0))


def _augment_one(seq, scale_range=(0.8, 3.0), keep_frames=(18, 27),
                 drop_prob=0.7, frame_range=(0.92, 1.08)):
    """seq: (30, 226) -> augmented copy."""
    s = seq.copy()

    # Which timesteps actually carry a detected hand.
    live = np.any(s[:, HAND] != 0, axis=1)

    # --- spatial: scale about the wrist anchor, translate, rotate ---
    scale = rng.uniform(*scale_range)
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

    # --- framing: how much of the canvas the body fills ---
    #
    # Simulates the device letterbox. MainActivity rotates the sensor frame
    # upright to portrait, then fits it into the 480x360 training canvas at
    # scale min(480/480, 360/640) = 0.5625, leaving the signer in a narrow
    # central column. Pose is read from that full canvas and is never
    # wrist-normalised, so the compression lands directly in the features.
    #
    # Deliberately POSE ONLY. The hand blocks escape this on device because the
    # second landmarker pass re-crops around the hand, which is why measured
    # device hand span (0.18-0.28) already matches the corpus (0.218) while the
    # body does not.
    frame_s = rng.uniform(*frame_range)
    if abs(frame_s - 1.0) > 1e-6:
        live_pose = np.any(s[:, 126:] != 0, axis=1)
        for k in range(25):
            xi, yi, zi = 126 + 4 * k, 126 + 4 * k + 1, 126 + 4 * k + 2
            s[live_pose, xi] = (s[live_pose, xi] - 0.5) * frame_s + 0.5
            s[live_pose, yi] = (s[live_pose, yi] - 0.5) * frame_s + 0.5
            s[live_pose, zi] *= frame_s

    # --- landmark jitter ---
    noise = rng.normal(0, 0.004, size=s[:, HAND].shape)
    mask = (s[:, HAND] != 0)
    s[:, HAND] += noise * mask

    # --- temporal: subsample to N distinct frames, then stretch back to SEQ ---
    #
    # This mirrors the live pipeline exactly: it collects however many frames
    # the camera manages, then resamples that run to SEQ before inference. Doing
    # the same here is what teaches the model to read a sparsely sampled sign,
    # rather than only the densely sampled ones in the corpus.
    if rng.random() < drop_prob:
        lo, hi = keep_frames
        n_keep = int(rng.integers(lo, hi + 1))
        if n_keep < SEQ:
            pick = np.linspace(0, SEQ - 1, n_keep).round().astype(int)
            back = np.linspace(0, n_keep - 1, SEQ).round().astype(int)
            s = s[pick][back]

    return s.astype(np.float32)


def augment_set(X, y, copies, **aug):
    """`aug` selects the per-corpus calibration; see FSL_AUG / ASL_AUG."""
    outX, outY = [X], [y]
    for _ in range(copies):
        outX.append(np.stack([_augment_one(v, **aug) for v in X]))
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
