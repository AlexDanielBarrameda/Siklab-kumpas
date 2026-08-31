"""Train the ASL word model on ASL Citizen.

Kept deliberately parallel to train_fsl105_v2_aug.py: same architecture, same
augmentation, same canonicalised 226-d input. Only the data and the label list
differ, so the app reuses its entire preprocessing path and swaps the model.

The one important difference is the evaluation. FSL-105 is a single signer, so
its held-out accuracy measures "recognises this person". ASL Citizen ships
signer-disjoint splits — 35 signers in train, 11 in test, zero overlap — so the
test number here is a genuine CROSS-SIGNER result on people the model has never
seen. That is a materially harder and more honest measurement, and the two
numbers should not be compared directly.

Usage:
    python train_asl.py --everyday      # ~80 common words, fast sanity check
    python train_asl.py                 # all 500
"""

import os
import sys
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_fsl105_v2_aug import (augment_set, build_model,  # noqa: E402
                                 ASL_AUG, ASL_AUG_WIDE, ASL_AUG_BODYNORM)

DATA = os.path.join(HERE, "../data/processed/asl_citizen")

EVERYDAY = set("""
HELLO PLEASE SORRY YES NO NAME WHO WHY WHERE WHEN WHAT HOW
GOOD BAD HAPPY SAD LOVE LIKE WANT NEED HELP STOP GO COME EAT DRINK WATER
MOTHER FATHER SISTER BROTHER FAMILY FRIEND MAN WOMAN BOY GIRL BABY PEOPLE
HOME SCHOOL WORK BATHROOM HOSPITAL TODAY TOMORROW YESTERDAY NOW MORNING
AFTERNOON NIGHT TIME DAY WEEK YEAR UNDERSTAND KNOW LEARN TEACH READ WRITE
DEAF HEARING SIGN LISTEN HOT COLD BIG SMALL MORE FINISH AGAIN THANKYOU
MONEY BUY CAR PHONE DOCTOR SICK PAIN
""".split())

AUG_COPIES = 2          # lower than FSL: there is far more real data here
EPOCHS = 120
BATCH = 32


def subsample_frames(X, n):
    """Simulate capturing only `n` distinct frames of each sign.

    Identical to what the app does: take the frames the camera managed, then
    resample that run back to 30 before inference. Deterministic — no rng — so
    the numbers are reproducible for the write-up.
    """
    if n >= X.shape[1]:
        return X
    pick = np.linspace(0, X.shape[1] - 1, n).round().astype(int)
    back = np.linspace(0, n - 1, X.shape[1]).round().astype(int)
    return X[:, pick][:, back]


def load_labels():
    out = {}
    with open(os.path.join(DATA, "labels.txt"), encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(",", 1)
            if len(p) == 2:
                out[int(p[0])] = p[1]
    return out



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--everyday", action="store_true",
                    help="train only on the common-word subset")
    ap.add_argument("--core200", action="store_true",
                    help="train on the 200-word shipping vocabulary "
                         "(asl_core200.py)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--bodynorm", action="store_true",
                    help="train on X_bodynorm.npy — every coordinate expressed "
                         "in shoulder widths, so framing and apparent size are "
                         "removed structurally rather than augmented around")
    ap.add_argument("--wide", action="store_true",
                    help="keep the wide scale range, which measured better "
                         "under the device condition despite being less "
                         "physically plausible")
    args = ap.parse_args()
    if args.bodynorm:
        aug_cfg = ASL_AUG_BODYNORM
    elif args.wide:
        aug_cfg = ASL_AUG_WIDE
    else:
        aug_cfg = ASL_AUG

    labels = load_labels()
    feat = "X_bodynorm.npy" if args.bodynorm else "X_canon.npy"
    X = np.load(os.path.join(DATA, feat)).astype(np.float32)
    print(f"features: {feat}")
    y = np.load(os.path.join(DATA, "y.npy"))
    split = np.load(os.path.join(DATA, "split.npy"))

    if args.everyday or args.core200:
        if args.core200:
            from asl_core200 import CORE200
            wanted, default_tag = set(CORE200), "core200"
        else:
            wanted, default_tag = EVERYDAY, "everyday"
        keep_cls = sorted(c for c, name in labels.items() if name in wanted)
        mask = np.isin(y, keep_cls)
        X, y, split = X[mask], y[mask], split[mask]
        remap = {c: i for i, c in enumerate(keep_cls)}
        y = np.array([remap[c] for c in y])
        labels = {remap[c]: labels[c] for c in keep_cls}
        tag = args.tag or default_tag
    else:
        tag = args.tag or "500"

    n_classes = len(labels)
    tr, va, te = split == "train", split == "val", split == "test"
    print(f"classes {n_classes}   train {tr.sum():,}  val {va.sum():,}  "
          f"test {te.sum():,}  (test signers are disjoint from train)")

    # ASL_AUG, not the FSL defaults. Two things were wrong before:
    #   - scale was uniform(0.8, 3.0), tuned for FSL's 0.10 median span. ASL's
    #     is already 0.214 and matches the device, so that stretched hands to
    #     0.64 and much of the augmented set was physically implausible.
    #   - temporal dropping simulated 18-27 distinct frames when the device
    #     delivers 4-6, so the model never trained on its own operating point.
    Xtr, ytr = augment_set(X[tr], y[tr], AUG_COPIES, **aug_cfg)
    print(f"train after augmentation: {Xtr.shape}  (ASL calibration: {aug_cfg})",
          flush=True)

    model = build_model_n(n_classes)
    model.fit(
        Xtr, ytr, validation_data=(X[va], y[va]),
        epochs=EPOCHS, batch_size=BATCH, verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12,
                                             restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                 patience=6, min_lr=1e-6),
        ])

    loss, acc = model.evaluate(X[te], y[te], verbose=0)
    probs = model.predict(X[te], verbose=0)
    top5 = np.mean([y[te][i] in np.argsort(-probs[i])[:5] for i in range(te.sum())])

    print(f"\n=== ASL {tag} ===")
    print(f"CROSS-SIGNER test accuracy : {acc*100:.2f}%   (top-5 {top5*100:.2f}%)")
    print(f"classes                    : {n_classes}")
    print(f"published SOTA, 2731 signs : ~62%  (different task size)")

    # The 30-frame number above is NOT what the app experiences. The device
    # captures 4-6 distinct frames per sign and stretches them to 30, so that is
    # the row to read when predicting live behaviour and the one to quote in the
    # evaluation chapter.
    print(f"\naccuracy vs capture rate (test split, signer-disjoint)")
    print(f"{'distinct frames':<18}{'top-1':>9}{'top-5':>9}")
    for n in (4, 5, 6, 8, 12, 30):
        Xn = subsample_frames(X[te], n)
        p = model.predict(Xn, verbose=0)
        t1 = np.mean(np.argmax(p, axis=1) == y[te])
        t5 = np.mean([y[te][i] in np.argsort(-p[i])[:5] for i in range(len(p))])
        note = "  <- device operating point" if n in (4, 5, 6) else ""
        print(f"{n:<18}{t1*100:>8.2f}%{t5*100:>8.2f}%{note}")

    keras_path = os.path.join(HERE, f"asl_{tag}_best.keras")
    tfl_path = os.path.join(HERE, f"asl_{tag}.tflite")
    model.save(keras_path)

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    conv.target_spec.supported_types = [tf.float16]
    tfl = conv.convert()
    with open(tfl_path, "wb") as f:
        f.write(tfl)
    print(f"TFLite: {os.path.basename(tfl_path)}  ({len(tfl)/1024:.0f} KB)")

    with open(os.path.join(HERE, f"asl_{tag}_labels.txt"), "w",
              encoding="utf-8") as f:
        for i in range(n_classes):
            f.write(f"{i},{labels[i]}\n")


def build_model_n(n_classes):
    """Same shape as the FSL model; unroll=True is required for TFLite
    conversion without the Flex delegate."""
    m = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True, unroll=True,
                             input_shape=(30, 226)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64, return_sequences=False, unroll=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


if __name__ == "__main__":
    main()
