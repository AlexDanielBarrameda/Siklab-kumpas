"""Collapse the two hand-block regimes into one.

MEASURED PROBLEM
The 226-d layout is 63 left-hand + 63 right-hand + 100 pose, and the wrist
normalisation keys off the right-hand block with no fallback. That creates two
incompatible feature spaces:

  * hand labelled "Right" -> lands in rh -> gets wrist-normalised
  * hand labelled "Left"  -> lands in lh -> rh stays 0 -> NEVER normalised

The model learns them separately. Feeding v2's own training clips through the
lh-only path drops accuracy from 99.0% to 13.5%, collapsing onto the handful of
classes that happen to own that regime (NICE TO MEET YOU, BREAD, AUNTIE...).

On device, MediaPipe labels the signer's hand "Left" essentially always, so
100% of live frames hit the rare path -- while only 11.9% of training frames do.
That, not domain shift, is why words decoded as WHEELCHAIR PERSON at 99%.

FIX
Canonicalise every frame so the primary hand is always in rh with a consistent
chirality, then normalise. Applied identically in training (this script) and at
inference (canonicaliseHands in inference_service.dart), the regime split
disappears.
"""

import numpy as np

LH = slice(0, 63)
RH = slice(63, 126)

# MediaPipe hand landmark indices
WRIST, INDEX_MCP, PINKY_MCP = 0, 5, 17


def hand_chirality(block):
    """Which way round does this hand APPEAR in the image? +1 or -1.

    Uses the signed area of the palm triangle wrist -> index knuckle -> pinky
    knuckle. Mirroring the image flips the sign; rotating it in-plane does not.
    So this recovers apparent handedness from the geometry alone, with no
    dependence on MediaPipe's Left/Right label -- which flips between
    consecutive frames and is what made fingerspelling depend on a specific
    combination of mirror setting and which hand the user signs with.

    Returns 0 when the palm is too close to edge-on for the sign to be reliable.
    """
    p = block.reshape(21, 3)
    w, i, k = p[WRIST], p[INDEX_MCP], p[PINKY_MCP]
    cross = (i[0] - w[0]) * (k[1] - w[1]) - (i[1] - w[1]) * (k[0] - w[0])
    # Scale the threshold to the hand so it is resolution independent.
    span = np.linalg.norm(p[:, :2] - w[:2], axis=1).max()
    if span < 1e-6 or abs(cross) < 0.02 * span * span:
        return 0
    return 1 if cross > 0 else -1


def mirror_block(block):
    """Mirror a hand about its own wrist in x, flipping apparent chirality."""
    p = block.reshape(21, 3).copy()
    p[:, 0] = 2.0 * p[0, 0] - p[:, 0]
    return p.reshape(-1)


def canonicalise_frame(v):
    """v: (226,) raw landmark vector, NOT yet wrist-normalised. Returns a copy.

    - one hand only  -> placed in rh; if it came from lh its x is mirrored about
      the wrist so both hands share the same chirality
    - two hands      -> ordered by POSITION IN THE IMAGE, leftmost wrist to lh,
      rather than by MediaPipe's label
    - then wrist-normalised against rh, exactly as the extractors did

    The two-handed ordering matters as much as the single-hand case. Swapping the
    blocks on two-handed clips drops accuracy from 98.9% to 2.3%, and MediaPipe's
    Left/Right label flips between consecutive frames on the same hand -- so any
    two-handed sign was a coin flip. Wrist x-position is a property of the image,
    not of a classifier, so it is stable.
    """
    v = v.copy()
    lh = v[LH]
    rh = v[RH]
    has_l = np.any(lh != 0)
    has_r = np.any(rh != 0)

    if has_l and has_r:
        # Leftmost wrist (smaller x) belongs in the left block.
        if lh[0] > rh[0]:
            v[LH], v[RH] = rh.copy(), lh.copy()

    elif has_l and not has_r:
        blk = lh.reshape(21, 3).copy()
        wrist_x = blk[0, 0]
        # Mirror x about the wrist: turns a left-hand shape into a right-hand
        # one, so a single hand always looks the same regardless of which hand
        # MediaPipe thought it saw.
        blk[:, 0] = 2.0 * wrist_x - blk[:, 0]
        v[RH] = blk.reshape(-1)
        v[LH] = 0.0

    # Normalise against the right wrist (unchanged from the extractors).
    wrist = v[63:66].copy()
    if np.any(wrist != 0):
        v[0:63:3] -= wrist[0]
        v[1:63:3] -= wrist[1]
        v[2:63:3] -= wrist[2]
        v[63:126:3] -= wrist[0]
        v[64:126:3] -= wrist[1]
        v[65:126:3] -= wrist[2]
    return v


def denormalise_frame(v):
    """Undo the extractors' wrist normalisation so a stored vector can be
    re-canonicalised. Only valid for frames where rh was the anchor: the wrist
    itself sits at the origin, and lh (if empty originally) holds -wrist."""
    v = v.copy()
    if not np.any(v[RH] != 0):
        return v          # never normalised in the first place
    # lh[0:3] == lh_wrist - rh_wrist. When lh was empty it is exactly -rh_wrist.
    off = -v[0:3]
    if np.allclose(v[LH], np.tile(off, 21), atol=1e-6):
        v[LH] = 0.0       # restore "no left hand"
        v[63:126:3] += off[0]
        v[64:126:3] += off[1]
        v[65:126:3] += off[2]
    return v


def canonicalise_letter_frame(v):
    """Fingerspelling canonicalisation, driven by GEOMETRY not by the label.

    Whichever block holds the hand, its apparent chirality is measured from the
    palm triangle, mirrored to a single convention, and placed in rh. The result
    is identical whether the signer used their left or right hand and whether
    the frame was mirrored -- which removes the constraint that fingerspelling
    only worked for specific mirror/block/hand combinations.

    Input is a stored (already wrist-normalised) letters vector.
    """
    v = denormalise_frame(v.copy())

    def real(b):
        p = b.reshape(21, 3)
        return (p.max(axis=0) - p.min(axis=0)).max() > 1e-6

    if real(v[RH]):
        blk = v[RH].copy()
    elif real(v[LH]):
        blk = v[LH].copy()
    else:
        return v

    if hand_chirality(blk) == -1:
        blk = mirror_block(blk)

    v[RH] = blk
    v[LH] = 0.0

    wrist = v[63:66].copy()
    if np.any(wrist != 0):
        v[63:126:3] -= wrist[0]
        v[64:126:3] -= wrist[1]
        v[65:126:3] -= wrist[2]
    return v


def canonicalise_dataset(X):
    """X: (N, T, 226) already-normalised sequences -> canonicalised copy."""
    out = np.empty_like(X)
    for i in range(len(X)):
        for t in range(X.shape[1]):
            out[i, t] = canonicalise_frame(denormalise_frame(X[i, t]))
    return out


if __name__ == "__main__":
    import os
    P = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "../data/processed/fsl_105_v2")
    X = np.load(os.path.join(P, "X.npy")).astype(np.float32)
    y = np.load(os.path.join(P, "y.npy"))

    before_lh = (np.all(X[:, :, RH] == 0, axis=2) &
                 np.any(X[:, :, LH] != 0, axis=2)).mean()
    C = canonicalise_dataset(X)
    after_lh = (np.all(C[:, :, RH] == 0, axis=2) &
                np.any(C[:, :, LH] != 0, axis=2)).mean()
    has_hand = np.any(C[:, :, 0:126] != 0, axis=2)

    print(f"lh-only frames : {before_lh*100:.1f}%  ->  {after_lh*100:.1f}%")
    print(f"frames with a hand preserved: {has_hand.mean()*100:.1f}%")
    np.save(os.path.join(P, "X_canon.npy"), C.astype(np.float32))
    np.save(os.path.join(P, "y_canon.npy"), y)
    print(f"wrote X_canon.npy {C.shape}")
