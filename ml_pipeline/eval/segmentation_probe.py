"""How much does mis-segmenting a sign cost?

frame_rate_curve.py showed the shipped ASL model still scores ~61% when given
only 5 distinct frames — yet live recognition is near-zero. Frame rate therefore
explains about 7 points of a ~60 point gap, and something else dominates.

The leading suspect is segmentation. Every clip in ASL Citizen is trimmed to the
sign itself. The app has to decide where the sign starts and ends from a live
stream, and it gets that boundary wrong in specific, predictable ways:

  TRUNCATED    the posture gate opens late / closes early, so the window holds
               only the middle of the sign.
  PADDED       the window opens on the hands rising and closes after they fall,
               so real movement occupies only the centre and the model sees
               approach and retraction it was never trained on.
  SHIFTED      the window is the right length but offset in time.

Each is simulated here against the clean test split, so the cost is measured
independently of frame rate, lighting, or any device difference.

    python segmentation_probe.py --model ../models/asl_525_best.keras
"""

import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data/processed/asl_citizen")
SEQ = 30


def _restretch(seq_block, out=SEQ):
    """Resample a (n, 226) run back to `out` timesteps — what the app does."""
    n = seq_block.shape[1]
    idx = np.linspace(0, n - 1, out).round().astype(int)
    return seq_block[:, idx]


def truncated(X, keep_frac):
    """Window holds only the central `keep_frac` of the sign."""
    n = int(round(SEQ * keep_frac))
    start = (SEQ - n) // 2
    return _restretch(X[:, start:start + n])


def padded(X, pad_frac):
    """Sign occupies only the middle; the rest is held rest-posture.

    The first and last real frames stand in for the approach and retraction —
    the hands are present but static, which is what the camera sees when the
    gate opens early and closes late.
    """
    n = int(round(SEQ * (1 - pad_frac)))
    n = max(n, 2)
    core = _restretch(X, n)
    pad = (SEQ - n)
    head = pad // 2
    tail = pad - head
    return np.concatenate([
        np.repeat(core[:, :1], head, axis=1),
        core,
        np.repeat(core[:, -1:], tail, axis=1),
    ], axis=1)


def shifted(X, frames):
    """Right length, wrong offset — edge frames repeat at the boundary."""
    if frames == 0:
        return X
    if frames > 0:
        return np.concatenate(
            [np.repeat(X[:, :1], frames, axis=1), X[:, :-frames]], axis=1)
    f = -frames
    return np.concatenate(
        [X[:, f:], np.repeat(X[:, -1:], f, axis=1)], axis=1)


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
    print(f"clean, correctly segmented          {base:6.2f}%   <- reference\n")

    print("TRUNCATED  (gate opens late / closes early)")
    for f in (0.9, 0.75, 0.6, 0.5, 0.35):
        print(f"  keeps central {f*100:3.0f}% of the sign      "
              f"{acc(truncated(X, f)):6.2f}%")

    print("\nPADDED  (gate opens early / closes late; sign is centred)")
    for f in (0.2, 0.35, 0.5, 0.65):
        print(f"  {f*100:3.0f}% of window is static padding  "
              f"{acc(padded(X, f)):6.2f}%")

    print("\nSHIFTED  (right length, wrong offset)")
    for s in (-6, -3, 3, 6):
        print(f"  offset {s:+d} frames                  {acc(shifted(X, s)):6.2f}%")


if __name__ == "__main__":
    main()
