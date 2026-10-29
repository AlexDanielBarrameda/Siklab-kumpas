"""Is the unnormalised pose block what breaks live ASL recognition?

Neither frame rate (-7 pts) nor mis-segmentation (-9 to -24 pts) accounts for a
collapse from 67.7% to near-zero live. This tests the remaining structural
suspect.

The wrist normalisation in every extractor touches indices 0..126 only:

    v[0:63:3] -= wrist[0] ... v[63:126:3] -= wrist[0] ...

Indices 126..226 — the 25 pose landmarks — are left as ABSOLUTE letterboxed
image coordinates. So the model reads where the signer's body sits in the frame,
not just how they moved. ASL Citizen signers are framed consistently: webcam,
centred, a fixed distance back. A phone held at arm's length frames a person
quite differently, and every one of those 100 dims shifts with it.

Two measurements:

  SENSITIVITY  translate and scale the pose block on clean test clips. If modest
               reframing collapses accuracy, the model is keyed to framing.

  RELIANCE     zero the pose block entirely. If accuracy barely moves, pose is
               contributing little and can be normalised or dropped cheaply; if
               it collapses, the model depends on the very dims most likely to
               be out of distribution live.

Also reports body geometry so the corpus can be compared against the device's
own measured numbers.

    python pose_domain_probe.py --model ../models/asl_525_best.keras
"""

import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data/processed/asl_citizen")

# Pose landmark layout: 25 landmarks x (x, y, z, visibility) from index 126.
POSE0 = 126
XI = [POSE0 + 4 * k + 0 for k in range(25)]
YI = [POSE0 + 4 * k + 1 for k in range(25)]
ZI = [POSE0 + 4 * k + 2 for k in range(25)]
L_SHOULDER, R_SHOULDER = 11, 12


def shift_pose(X, dx, dy):
    X = X.copy()
    live = np.any(X[:, :, POSE0:] != 0, axis=2)
    for idx, d in ((XI, dx), (YI, dy)):
        if d:
            blk = X[:, :, idx]
            blk[live] += d
            X[:, :, idx] = blk
    return X


def scale_pose(X, s):
    """Scale about the shoulder midpoint — the signer standing nearer/further."""
    X = X.copy()
    live = np.any(X[:, :, POSE0:] != 0, axis=2)
    cx = (X[:, :, POSE0 + 4 * L_SHOULDER] + X[:, :, POSE0 + 4 * R_SHOULDER]) / 2
    cy = (X[:, :, POSE0 + 4 * L_SHOULDER + 1] +
          X[:, :, POSE0 + 4 * R_SHOULDER + 1]) / 2
    for idx, c in ((XI, cx), (YI, cy)):
        blk = X[:, :, idx]
        cc = c[:, :, None]
        blk = np.where(live[:, :, None], (blk - cc) * s + cc, blk)
        X[:, :, idx] = blk
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    X = np.load(os.path.join(DATA, "X_canon.npy")).astype(np.float32)
    y = np.load(os.path.join(DATA, "y.npy"))
    split = np.load(os.path.join(DATA, "split.npy"))
    m = split == "test"
    X, y = X[m], y[m]

    model = tf.keras.models.load_model(os.path.abspath(args.model))
    keep = y < model.output_shape[-1]
    X, y = X[keep], y[keep]

    def acc(Xv):
        p = model.predict(Xv, verbose=0, batch_size=256)
        return np.mean(np.argmax(p, axis=1) == y) * 100

    base = acc(X)
    print(f"\nmodel {os.path.basename(args.model)}   {len(y):,} test clips")
    print(f"clean                                 {base:6.2f}%   <- reference\n")

    print("RELIANCE")
    Z = X.copy()
    Z[:, :, POSE0:] = 0.0
    print(f"  pose block zeroed                   {acc(Z):6.2f}%")

    print("\nSENSITIVITY — translation (normalised image units)")
    for d in (0.05, 0.10, 0.20):
        print(f"  shifted right {d:.2f}                  {acc(shift_pose(X, d, 0)):6.2f}%")
        print(f"  shifted down  {d:.2f}                  {acc(shift_pose(X, 0, d)):6.2f}%")

    print("\nSENSITIVITY — scale about shoulders (signer nearer / further)")
    for s in (0.8, 1.25, 1.5):
        print(f"  pose scaled x{s:<4}                   {acc(scale_pose(X, s)):6.2f}%")

    # Body geometry, for comparison against the device's own logged numbers.
    print("\nCORPUS BODY GEOMETRY (test split, frames with pose)")
    lx = X[:, :, POSE0 + 4 * L_SHOULDER]
    rx = X[:, :, POSE0 + 4 * R_SHOULDER]
    ly = X[:, :, POSE0 + 4 * L_SHOULDER + 1]
    live = np.any(X[:, :, POSE0:] != 0, axis=2)
    sw = np.abs(lx - rx)[live]
    print(f"  shoulder width   median {np.median(sw):.3f}   "
          f"p10 {np.percentile(sw, 10):.3f}   p90 {np.percentile(sw, 90):.3f}")
    print(f"  shoulder y       median {np.median(ly[live]):.3f}")
    cx = ((lx + rx) / 2)[live]
    print(f"  body centre x    median {np.median(cx):.3f}   "
          f"p10 {np.percentile(cx, 10):.3f}   p90 {np.percentile(cx, 90):.3f}")


if __name__ == "__main__":
    main()
