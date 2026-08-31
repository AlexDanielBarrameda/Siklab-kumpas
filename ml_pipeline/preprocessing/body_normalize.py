"""Re-express the 226-d vector in shoulder widths instead of image units.

WHY

Two sources of person- and camera-dependence survive canonicalisation, and both
are nuisance variation rather than signal:

  HAND SCALE  the hand blocks are wrist-relative but never scale-normalised, so
              apparent hand size multiplies every coordinate. A different hand
              at a different distance is a different feature vector for the same
              sign.

  POSE FRAME  indices 126..226 are absolute letterboxed image coordinates. Where
              the signer stands is a model input. Measured: substituting an
              uninformative pose block costs 34 accuracy points, and simulating
              the device letterbox costs 9.6.

Dividing everything by shoulder width and centring pose on the shoulder midpoint
removes camera distance, image scale and body size together, in one consistent
unit. Hand-to-body proportion is preserved (it is roughly constant across
people), and where the hand sits relative to the body still lives in the pose
block, so nothing the classifier needs is discarded.

WHY IT IS TESTED ON ASL

FSL-105 has a single signer, so it cannot measure whether anything generalises
to a new person -- its 98.54% is a random split in which the same signer appears
in train and test. ASL Citizen has 52 signers with signer-disjoint splits, so it
CAN. If this normalisation raises ASL's cross-signer accuracy, that is direct
evidence it transfers to unseen people, and the transform applies unchanged to
FSL. We borrow ASL's signer diversity to validate a fix we then apply to both.

PORTING

Any change here must be mirrored in inference_service.dart, applied after
canonicalisation and the wrist normalisation. The two implementations must stay
byte-equivalent or the model sees a different feature space at inference than it
saw in training -- the exact failure that made words decode as WHEELCHAIR PERSON.
"""

import numpy as np

POSE0 = 126
L_SHOULDER, R_SHOULDER = 11, 12

# Below this the shoulders are too close together to divide by -- the signer is
# turned side-on, or the detection is bad. Dividing anyway amplifies noise into
# the whole vector.
MIN_WIDTH = 0.04


def body_frame(v):
    """(cx, cy, w) for one 226-d frame, or None when pose is unusable."""
    lsx = v[POSE0 + 4 * L_SHOULDER]
    lsy = v[POSE0 + 4 * L_SHOULDER + 1]
    rsx = v[POSE0 + 4 * R_SHOULDER]
    rsy = v[POSE0 + 4 * R_SHOULDER + 1]
    if (lsx == 0 and lsy == 0) or (rsx == 0 and rsy == 0):
        return None
    w = float(np.hypot(lsx - rsx, lsy - rsy))
    if w < MIN_WIDTH:
        return None
    return (lsx + rsx) / 2.0, (lsy + rsy) / 2.0, w


def normalize_frame(v):
    """v: (226,) canonicalised + wrist-normalised. Returns a scaled copy.

    Frames with no usable pose are returned unchanged rather than zeroed: a
    zeroed frame is far more out-of-distribution than an un-scaled one, and the
    corpus contains a pose in 99.99% of frames so this is a rare path.
    """
    bf = body_frame(v)
    if bf is None:
        return v.copy()
    cx, cy, w = bf
    out = v.copy()

    # Hands are scaled by their OWN span, not by shoulder width.
    #
    # Dividing hands by shoulder width was tried first and is catastrophic on
    # device: 65.36% -> 30.77% under the device letterbox. It couples hand scale
    # to body scale, and the device breaks that coupling — the second landmarker
    # pass re-crops around the hand, restoring its apparent size, while pose
    # stays compressed by the letterbox. Hands then get divided by a shoulder
    # width 0.5625x too small and inflate by 1.78x.
    #
    # Scaling each hand by its own span decouples the two exactly as the device
    # does, and costs only the hand-size-relative-to-body cue, which is the
    # nuisance this transform exists to remove.
    for base in (0, 63):
        blk = out[base:base + 63].reshape(21, 3)
        if not np.any(blk):
            continue
        span = float(np.linalg.norm(blk[:, :2] - blk[0, :2], axis=1).max())
        if span > 1e-6:
            out[base:base + 63] = (blk / span).reshape(-1)

    # Pose is absolute: centre on the shoulders, then scale. Visibility (index 3
    # of each landmark) is a probability and must not be touched.
    for k in range(25):
        xi, yi, zi = POSE0 + 4 * k, POSE0 + 4 * k + 1, POSE0 + 4 * k + 2
        if out[xi] == 0 and out[yi] == 0:
            continue                      # landmark not detected; leave as zero
        out[xi] = (out[xi] - cx) / w
        out[yi] = (out[yi] - cy) / w
        out[zi] = out[zi] / w
    return out


def normalize_dataset(X):
    """X: (N, T, 226) -> body-normalised copy."""
    out = np.empty_like(X)
    for i in range(len(X)):
        for t in range(X.shape[1]):
            out[i, t] = normalize_frame(X[i, t])
    return out


if __name__ == "__main__":
    import os
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="asl_citizen",
                    help="folder under ml_pipeline/data/processed")
    ap.add_argument("--src", default="X_canon.npy")
    ap.add_argument("--out", default="X_bodynorm.npy")
    args = ap.parse_args()

    HERE = os.path.dirname(os.path.abspath(__file__))
    P = os.path.join(HERE, "../data/processed", args.data)

    X = np.load(os.path.join(P, args.src)).astype(np.float32)
    print(f"{args.src} {X.shape}")

    usable = sum(body_frame(X[i, t]) is not None
                 for i in range(min(len(X), 400)) for t in range(X.shape[1]))
    total = min(len(X), 400) * X.shape[1]
    print(f"frames with a usable body frame: {100*usable/total:.2f}%")

    N = normalize_dataset(X)
    np.save(os.path.join(P, args.out), N)

    # Scale invariance is the whole point, so show that it happened.
    def span(A):
        v = []
        for i in range(0, len(A), max(1, len(A) // 300)):
            for t in range(A.shape[1]):
                b = A[i, t, 63:126].reshape(21, 3)
                if np.any(b):
                    v.append(np.linalg.norm(b[:, :2] - b[0, :2], axis=1).max())
        return np.array(v)

    a, b = span(X), span(N)
    print(f"hand span before  median {np.median(a):.3f}  "
          f"p10 {np.percentile(a,10):.3f}  p90 {np.percentile(a,90):.3f}")
    print(f"hand span after   median {np.median(b):.3f}  "
          f"p10 {np.percentile(b,10):.3f}  p90 {np.percentile(b,90):.3f}")
    print(f"spread (p90/p10)  before {np.percentile(a,90)/max(np.percentile(a,10),1e-6):.2f}x"
          f"   after {np.percentile(b,90)/max(np.percentile(b,10),1e-6):.2f}x")
    print(f"saved {args.out}")
