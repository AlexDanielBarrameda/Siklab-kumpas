"""v3 feature representation: dual-channel, with motion and presence flags.

The v2 vector is 226-d: 63 left-hand + 63 right-hand + 100 pose, wrist-relative
but never scale-normalised, and pose left as absolute image coordinates. It has
three measured weaknesses:

  ONE CHANNEL   the same numbers have to carry both WHERE the hand is and WHAT
                shape it makes. Body-normalising fixes location and destroys
                handshape scale; hand-normalising does the reverse. Tested
                directly: body normalisation alone left cross-signer accuracy
                unchanged at 67.75%.

  NO MOTION     movement is one of the five phonological parameters of a sign,
                and the model has to infer it from raw position across
                timesteps. At ~15 training clips per class that is a lot to ask.

  NO PRESENCE   a zeroed hand block is indistinguishable from a hand that
                genuinely sits at the origin.

v3 carries both channels explicitly, plus velocity, plus a handful of distances
that encode the location contrasts ASL actually uses (FATHER vs MOTHER is the
same handshape and movement at forehead vs chin), plus a detection flag per
hand.

    LAYOUT                                            dims
    body-normalised pose xyz                            75   location
    hand blocks / shoulder width                       126   inter-hand geometry
    hand-local shapes (own wrist, own span)            126   handshape
    distances: wrist-nose x2, wrist-chest x2, wrist-wrist 5   location contrasts
    hand-detected flags                                  2   presence
    velocity of (pose + hand blocks)                   201   movement
                                                      ----
                                                       535

Everything is derived from X_canon.npy, so canonicalisation is preserved -- that
fix is load-bearing (feeding clean clips through the un-canonicalised path
measured 99.0% -> 13.5%) and is not being relitigated here.
"""

import numpy as np

POSE0 = 126
NOSE, L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 0, 11, 12, 15, 16
L_ELBOW, R_ELBOW = 13, 14
# Hand landmarks: wrist, index knuckle, middle knuckle, pinky knuckle.
H_WRIST, H_INDEX_MCP, H_MIDDLE_MCP, H_PINKY_MCP = 0, 5, 9, 17
MIN_WIDTH = 0.04
EPS = 1e-6

# MediaPipe normalises x to image WIDTH and y to image HEIGHT independently, so
# on a 4:3 canvas one x-unit is 4/3 as many pixels as one y-unit. Any distance,
# span or rotation computed from raw coords therefore mixes units. Both our
# training canvas (640x480) and the device canvas (480x360) are 4:3, so this is
# not a train/inference mismatch -- but it does distort every geometric quantity
# the features are built from, so x is rescaled to square pixels first.
ASPECT = 640.0 / 480.0

DIM_V3 = 754


def build(X, with_velocity=True):
    """X: (N, T, 226) canonicalised -> (N, T, 535) float32."""
    X = np.asarray(X, dtype=np.float32).copy()
    N, T = X.shape[0], X.shape[1]

    # Square-pixel coordinates before anything geometric is computed.
    X[:, :, 0:126:3] *= ASPECT                       # hand x (wrist-relative)
    px_idx = [POSE0 + 4 * k for k in range(25)]
    X[:, :, px_idx] *= ASPECT                        # pose x (absolute)

    pose = X[:, :, POSE0:].reshape(N, T, 25, 4)
    px, py, pz = pose[..., 0], pose[..., 1], pose[..., 2]

    lsx, lsy = px[:, :, L_SHOULDER], py[:, :, L_SHOULDER]
    rsx, rsy = px[:, :, R_SHOULDER], py[:, :, R_SHOULDER]
    cx, cy = (lsx + rsx) / 2.0, (lsy + rsy) / 2.0
    w = np.hypot(lsx - rsx, lsy - rsy)

    # Frames without a usable body frame get zeros rather than a division that
    # amplifies detector noise across all 535 dims.
    ok = w > MIN_WIDTH
    wsafe = np.where(ok, w, 1.0)[..., None]
    okp = ok[..., None]

    bx = np.where(okp, (px - cx[..., None]) / wsafe, 0.0)
    by = np.where(okp, (py - cy[..., None]) / wsafe, 0.0)
    bz = np.where(okp, pz / wsafe, 0.0)
    body_pose = np.stack([bx, by, bz], axis=-1).reshape(N, T, 75)

    # --- hands -----------------------------------------------------------
    hands = X[:, :, :126].reshape(N, T, 2, 21, 3)
    present = np.any(hands != 0, axis=(3, 4))            # (N, T, 2)

    wrist = hands[:, :, :, :1, :]
    centred = hands - wrist
    span = np.linalg.norm(centred[..., :2], axis=-1).max(axis=-1)     # (N,T,2)
    span_ok = span > EPS

    # NOTHING derived from the hands may be scaled by shoulder width.
    #
    # The device breaks the coupling between hand scale and body scale: the
    # second landmarker pass re-crops around the hand, restoring its apparent
    # size, while the letterbox leaves the body compressed by 0.5625. Any
    # feature that divides a hand quantity by a body quantity is therefore
    # 1.78x wrong on device while looking perfectly healthy on the corpus.
    #
    # This was measured twice before it was believed:
    #   whole-vector body normalisation   65.36% -> 30.77% under the letterbox
    #   hand_loc scaled by shoulder width 73.59% -> 33.24% under the letterbox
    #
    # So hands are scaled by a HAND quantity (the primary hand's span) and pose
    # by a BODY quantity, and the two regimes never mix. Location relative to
    # the body is not lost: it lives in body_pose, via pose wrists 15/16.
    ref = span[:, :, 1]                                    # primary hand span
    ref = np.where(ref > EPS, ref, np.where(span[:, :, 0] > EPS,
                                            span[:, :, 0], 1.0))

    # Channel A: inter-hand geometry, in hand units.
    hand_loc = (hands / ref[..., None, None, None]).reshape(N, T, 126)
    hand_loc = np.where(np.repeat(present, 63, axis=2) != 0, hand_loc, 0.0)

    # Channel B: pure handshape — each hand against its own wrist and own span,
    # so hand size and camera distance drop out entirely.
    shape = np.where(span_ok[..., None, None],
                     centred / np.where(span_ok, span, 1.0)[..., None, None],
                     0.0)
    shape = np.where(present[..., None, None], shape, 0.0)
    hand_shape = shape.reshape(N, T, 126)

    # --- explicit location contrasts -------------------------------------
    # The shoulder midpoint is the origin after normalisation, so distance to
    # chest is simply the norm of the wrist.
    def pt(i):
        return np.stack([bx[:, :, i], by[:, :, i]], axis=-1)

    nose, lw, rw = pt(NOSE), pt(L_WRIST), pt(R_WRIST)
    d = np.stack([
        np.linalg.norm(lw - nose, axis=-1),
        np.linalg.norm(rw - nose, axis=-1),
        np.linalg.norm(lw, axis=-1),
        np.linalg.norm(rw, axis=-1),
        np.linalg.norm(lw - rw, axis=-1),
    ], axis=-1)
    dists = np.where(okp, d, 0.0)

    flags = present.astype(np.float32)

    parts = [body_pose, hand_loc, hand_shape, dists, flags]

    if with_velocity:
        # Two motion timescales, following the ISLR 1st-place solution: dx over
        # one frame and dx2 over two. At ~5 captured frames per sign a single
        # first difference is a very short baseline, and the wider one survives
        # frame duplication where the narrow one collapses to zero.
        pos = np.concatenate([body_pose, hand_loc], axis=2)
        vel = np.zeros_like(pos)
        vel[:, 1:] = pos[:, 1:] - pos[:, :-1]
        acc = np.zeros_like(pos)
        acc[:, 2:] = pos[:, 2:] - pos[:, :-2]
        parts += [vel, acc]

    # --- limb and palm orientation (appended, so earlier offsets are stable) ---
    #
    # Arm geometry is already implicit in body_pose, but the model has to infer
    # an ANGLE from raw coordinates, and at ~15 clips per class that is a lot to
    # ask. Signs that share a trajectory and differ only in forearm slant are
    # exactly what gets confused -- HELLO and FUTURE travel the same path from
    # the head outward, and differ in whether the forearm is angled or straight.
    # Reported from device testing as HELLO decoding to FUTURE.
    #
    # Unit vectors rather than raw angles: no wraparound at +/-pi, and a
    # near-zero-length limb degrades to zeros instead of a random direction.
    def unit(dx, dy):
        n = np.sqrt(dx * dx + dy * dy)
        good = n > EPS
        ns = np.where(good, n, 1.0)
        return np.where(good, dx / ns, 0.0), np.where(good, dy / ns, 0.0), n

    arm = []
    for sh, el, wr in ((L_SHOULDER, L_ELBOW, L_WRIST),
                       (R_SHOULDER, R_ELBOW, R_WRIST)):
        ux, uy, _ = unit(bx[:, :, el] - bx[:, :, sh], by[:, :, el] - by[:, :, sh])
        fx, fy, fl = unit(bx[:, :, wr] - bx[:, :, el], by[:, :, wr] - by[:, :, el])
        # Elbow flexion as a dot product: +1 straight, -1 fully folded.
        flex = ux * fx + uy * fy
        arm += [ux, uy, fx, fy, flex, np.where(ok, fl, 0.0)]

    for h in range(2):
        blk = shape[:, :, h]                     # (N, T, 21, 3), hand-local
        px_, py_, _ = unit(
            blk[:, :, H_MIDDLE_MCP, 0] - blk[:, :, H_WRIST, 0],
            blk[:, :, H_MIDDLE_MCP, 1] - blk[:, :, H_WRIST, 1])
        # Knuckle spread separates an open hand from a fist at the same angle.
        spread = np.linalg.norm(
            blk[:, :, H_INDEX_MCP, :2] - blk[:, :, H_PINKY_MCP, :2], axis=-1)
        arm += [px_, py_, np.where(present[:, :, h], spread, 0.0)]

    parts.append(np.stack(arm, axis=-1))

    out = np.concatenate(parts, axis=2).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def device_letterbox(X, s=0.5625):
    """The camera pipeline's framing, as a transform on raw canonical vectors.

    MainActivity rotates the sensor frame upright to portrait and fits it into
    the 480x360 canvas at scale min(480/480, 360/640) = 0.5625. It is a uniform
    image scale, so pose x, y AND z all scale; the hand blocks do not, because
    the second landmarker pass re-crops around the hand and restores its size.
    That asymmetry is the whole reason hand quantities must never be divided by
    body quantities.
    """
    Z = X.copy()
    live = np.any(Z[:, :, POSE0:] != 0, axis=2)
    for k in range(25):
        for c, centred in ((0, True), (1, True), (2, False)):
            i = POSE0 + 4 * k + c
            b = Z[:, :, i]
            Z[:, :, i] = np.where(live, (b - 0.5) * s + 0.5 if centred else b * s, b)
    return Z


def check_invariance(X, tol=1e-4, verbose=True):
    """Assert the representation is unchanged by the device letterbox.

    Cheap, and it catches the failure mode that has bitten this pipeline twice:
    a feature that mixes hand scale with body scale looks perfectly healthy on
    corpus data and collapses on device.

        whole-vector body normalisation   65.36% -> 30.77%
        hand_loc scaled by shoulder width 73.59% -> 33.24%

    Both were found only after training. Run this first instead.
    """
    A, B = build(X), build(device_letterbox(X))
    blocks = [("body_pose", 0, 75), ("hand_loc", 75, 201),
              ("hand_shape", 201, 327), ("distances", 327, 332),
              ("flags", 332, 334), ("velocity", 334, 535),
              ("accel", 535, 736), ("arm", 736, 754)]
    worst, bad = 0.0, []
    for name, a, b in blocks:
        d = float(np.abs(A[:, :, a:b] - B[:, :, a:b]).max())
        worst = max(worst, d)
        if d > tol:
            bad.append(name)
        if verbose:
            print(f"  {name:<12}{d:.3e}{'   NOT INVARIANT' if d > tol else ''}")
    if verbose:
        print(f"  {'whole vector':<12}{worst:.3e}")
    if bad:
        raise AssertionError(
            f"not letterbox-invariant: {', '.join(bad)} — some feature is "
            f"mixing hand scale with body scale")
    return worst


def motion_energy(X):
    """Per-frame movement of the hands, for idle trimming. X: (N, T, 226)."""
    h = X[:, :, :126]
    e = np.zeros(X.shape[:2], dtype=np.float32)
    e[:, 1:] = np.linalg.norm(h[:, 1:] - h[:, :-1], axis=2)
    e[:, 0] = e[:, 1]
    return e


def trim_idle(X, frac=0.15, min_keep=8):
    """Drop leading/trailing frames whose motion is below `frac` of the clip's
    peak, then resample the remainder back to T.

    ASL Citizen clips carry noticeable dead air at both ends -- the signer
    settling before starting and holding after finishing. Those frames spend
    sequence budget on stillness and, worse, teach the model that a sign begins
    with a static hold, which is exactly what the app's segmentation does NOT
    deliver.
    """
    T = X.shape[1]
    e = motion_energy(X)
    out = np.empty_like(X)
    for i in range(len(X)):
        peak = e[i].max()
        if peak <= EPS:
            out[i] = X[i]
            continue
        live = np.where(e[i] >= frac * peak)[0]
        if len(live) < min_keep:
            out[i] = X[i]
            continue
        a, b = live[0], live[-1]
        seg = X[i, a:b + 1]
        idx = np.linspace(0, len(seg) - 1, T).round().astype(int)
        out[i] = seg[idx]
    return out


if __name__ == "__main__":
    import os
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="asl_citizen")
    ap.add_argument("--src", default="X_canon.npy")
    ap.add_argument("--out", default="X_v3.npy")
    ap.add_argument("--trim", action="store_true", help="trim idle frames first")
    args = ap.parse_args()

    HERE = os.path.dirname(os.path.abspath(__file__))
    P = os.path.join(HERE, "../data/processed", args.data)

    X = np.load(os.path.join(P, args.src)).astype(np.float32)
    print(f"{args.src} {X.shape}")

    # Gate the build on invariance rather than discovering the problem after a
    # training run and a device-condition evaluation.
    print("\ndevice-letterbox invariance:")
    check_invariance(X[:200])
    print()

    if args.trim:
        e = motion_energy(X)
        X = trim_idle(X)
        print(f"idle-trimmed (median peak motion {np.median(e.max(axis=1)):.4f})")

    V = build(X)
    np.save(os.path.join(P, args.out), V)
    print(f"v3 {V.shape}  -> {args.out}")
    print(f"  finite: {np.isfinite(V).all()}   "
          f"mean |v| {np.abs(V).mean():.4f}   max |v| {np.abs(V).max():.2f}")
