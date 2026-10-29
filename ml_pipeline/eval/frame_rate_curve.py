"""Accuracy as a function of how many frames the camera actually caught.

The number normally reported for a sign model — "67.4% cross-signer" — is
measured on 30 evenly sampled frames of each clip. The phone does not deliver
30. It delivers 4-6 distinct frames per sign and stretches them to 30 before
inference, so the headline number describes a condition the app never operates
in. This measures the curve instead, which is what predicts live behaviour.

Use it to compare a retrained model against the one already shipped:

    python frame_rate_curve.py --model ../models/asl_525_best.keras
    python frame_rate_curve.py --model ../models/asl_525cal_best.keras
"""

import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data/processed/asl_citizen")

FRAME_COUNTS = (3, 4, 5, 6, 8, 12, 20, 30)


def subsample_frames(X, n):
    """Exactly what the app does: keep n distinct frames, resample back to 30.

    Deterministic, so the table is reproducible for the write-up.
    """
    if n >= X.shape[1]:
        return X
    pick = np.linspace(0, X.shape[1] - 1, n).round().astype(int)
    back = np.linspace(0, n - 1, X.shape[1]).round().astype(int)
    return X[:, pick][:, back]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test",
                    help="signer-disjoint by construction in ASL Citizen")
    args = ap.parse_args()

    X = np.load(os.path.join(DATA, "X_canon.npy")).astype(np.float32)
    y = np.load(os.path.join(DATA, "y.npy"))
    split = np.load(os.path.join(DATA, "split.npy"))

    m = split == args.split
    X, y = X[m], y[m]

    model = tf.keras.models.load_model(os.path.abspath(args.model))
    n_out = model.output_shape[-1]

    # A model trained on a subset would misalign against the full label vector.
    keep = y < n_out
    if not keep.all():
        print(f"note: dropping {(~keep).sum()} samples outside this model's "
              f"{n_out} classes")
        X, y = X[keep], y[keep]

    print(f"\nmodel   {os.path.basename(args.model)}   ({n_out} classes)")
    print(f"split   {args.split}   {len(y):,} clips\n")
    print(f"{'distinct frames':<18}{'top-1':>9}{'top-5':>9}   note")
    print("-" * 52)

    for n in FRAME_COUNTS:
        p = model.predict(subsample_frames(X, n), verbose=0, batch_size=256)
        t1 = np.mean(np.argmax(p, axis=1) == y)
        t5 = np.mean([y[i] in np.argsort(-p[i])[:5] for i in range(len(p))])
        note = ("device operating point" if n in (4, 5, 6)
                else "as-published" if n == 30 else "")
        print(f"{n:<18}{t1*100:>8.2f}%{t5*100:>8.2f}%   {note}")


if __name__ == "__main__":
    main()
