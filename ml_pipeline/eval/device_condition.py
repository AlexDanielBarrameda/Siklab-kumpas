"""Compare models under the condition the app actually runs in.

The published number for a sign model is measured on 30 evenly sampled frames of
a clip framed the way its corpus was filmed. The phone matches neither:

  FRAME RATE   it delivers 4-6 distinct frames per sign, stretched back to 30.
  FRAMING      MainActivity rotates the sensor frame upright to portrait, then
               letterboxes it into the 480x360 canvas at scale 0.5625, so the
               signer fills 56% of the width. Measured median shoulder width is
               0.222 in FSL-105 against 0.410 in ASL Citizen — both corpora are
               landscape, so this is filming distance, not orientation, but it
               lands FSL in the same regime the device letterbox produces.
               Device signing posture (-1.14) falls inside FSL's range (p5
               -1.18) and outside ASL's (p5 -0.78), which is why the shared
               pipeline works for FSL words and fails for ASL words.
  CONFIDENCE   predictions below the app's threshold are never shown, so the
               user-visible rate is (correct AND confident), not accuracy.

This reports all four combinations so a change can be attributed rather than
guessed at.

    python device_condition.py ../models/asl_525_best.keras ../models/asl_525frame_best.keras
"""

import os
import sys
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data/processed/asl_citizen")

SEQ = 30
POSE0 = 126
DEVICE_LETTERBOX = 0.5625      # min(480/480, 360/640) in MainActivity
DEVICE_FRAMES = 5              # measured live capture, ~5 fps
CONF = 0.6                     # InferenceService.confidenceThreshold


def letterbox_pose(X, s=DEVICE_LETTERBOX):
    """Scale the pose block about the canvas centre — the device letterbox.

    Pose only: on device the hand blocks are re-cropped by the second
    landmarker pass, which is why measured device hand span (0.18-0.28) already
    matches the corpus (0.218) while the body does not.
    """
    Z = X.copy()
    live = np.any(Z[:, :, POSE0:] != 0, axis=2)
    for k in range(25):
        for c, centred in ((0, True), (1, True), (2, False)):
            i = POSE0 + 4 * k + c
            b = Z[:, :, i]
            newb = (b - 0.5) * s + 0.5 if centred else b * s
            Z[:, :, i] = np.where(live, newb, b)
    return Z


def subsample(X, n):
    if n >= X.shape[1]:
        return X
    pick = np.linspace(0, X.shape[1] - 1, n).round().astype(int)
    back = np.linspace(0, n - 1, X.shape[1]).round().astype(int)
    return X[:, pick][:, back]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--bodynorm", action="store_true",
                    help="model expects body_normalize.py features. The device "
                         "letterbox is applied to the RAW canonical vector and "
                         "the normalisation runs after it, which is the real "
                         "order of operations — normalising first would hide "
                         "exactly the failure this is meant to catch.")
    args = ap.parse_args()

    X = np.load(os.path.join(DATA, "X_canon.npy")).astype(np.float32)
    y = np.load(os.path.join(DATA, "y.npy"))
    split = np.load(os.path.join(DATA, "split.npy"))
    m = split == "test"
    X, y = X[m], y[m]

    conditions = [
        ("corpus framing, 30 frames   (as published)", X),
        ("corpus framing, 5 frames", subsample(X, DEVICE_FRAMES)),
        ("DEVICE framing, 30 frames", letterbox_pose(X)),
        ("DEVICE framing, 5 frames    (live condition)",
         letterbox_pose(subsample(X, DEVICE_FRAMES))),
    ]

    # Order matters and is the whole point. The letterbox happens in the camera
    # pipeline, before features are built, so the normalisation must run on the
    # already-letterboxed vector. Normalising the corpus first and letterboxing
    # after would report invariance the device does not actually have — the
    # first attempt at this transform looked fine on corpus features and
    # collapsed from 65.36% to 30.77% once the order was correct.
    if args.bodynorm:
        sys.path.insert(0, os.path.join(HERE, "../preprocessing"))
        from body_normalize import normalize_dataset
        print("features: body-normalised (applied after each condition)")
        conditions = [(n, normalize_dataset(Z)) for n, Z in conditions]

    # Corpus name -> corpus index, so a model trained on a subset can be scored
    # against the right classes. A subset model re-indexes its outputs 0..n-1,
    # so comparing raw corpus indices against it silently scores the wrong
    # words rather than failing.
    corpus_idx = {}
    for line in open(os.path.join(DATA, "labels.txt"), encoding="utf-8"):
        p = line.strip().split(",", 1)
        if len(p) == 2 and p[1]:
            corpus_idx[p[1]] = int(p[0])

    for path in args.models:
        model = tf.keras.models.load_model(os.path.abspath(path))
        n_out = model.output_shape[-1]

        lbl_path = os.path.abspath(path).replace("_best.keras", "_labels.txt")
        if os.path.exists(lbl_path) and n_out < len(corpus_idx):
            names = {}
            for line in open(lbl_path, encoding="utf-8"):
                p = line.strip().split(",", 1)
                if len(p) == 2 and p[1]:
                    names[int(p[0])] = p[1]
            # model output k  ->  corpus class id
            back = {corpus_idx[v]: k for k, v in names.items()
                    if v in corpus_idx}
            keep = np.isin(y, list(back))
            yy = np.array([back[c] for c in y[keep]])
            print(f"\n[{os.path.basename(path)}: {n_out}-class subset, "
                  f"scored on {keep.sum():,} of {len(y):,} test clips]")
        else:
            keep = y < n_out
            yy = y[keep]
        print(f"\n=== {os.path.basename(path)} ===")
        print(f"{'condition':<44}{'top-1':>8}{'top-5':>8}{'shown':>8}")
        print("-" * 68)
        for name, Xc in conditions:
            p = model.predict(Xc[keep], verbose=0, batch_size=256)
            t1 = np.mean(np.argmax(p, axis=1) == yy) * 100
            t5 = np.mean([yy[i] in np.argsort(-p[i])[:5]
                          for i in range(len(p))]) * 100
            shown = np.mean((np.argmax(p, axis=1) == yy) &
                            (p.max(axis=1) >= CONF)) * 100
            print(f"{name:<44}{t1:>7.2f}%{t5:>7.2f}%{shown:>7.2f}%")
        print(f"{'':<44}{'':>8}{'':>8}   ^ correct AND above {CONF}")


if __name__ == "__main__":
    main()
