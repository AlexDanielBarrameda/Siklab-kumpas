"""v3: separable-conv + shallow-transformer encoder over v3 features.

The reference point is the Google Isolated Sign Language Recognition
competition -- 250 signs, landmark input, TFLite on-device inference, which is
very nearly this problem. Winning entries were small models built from stacked
1D depthwise-separable convolutions with a shallow Transformer encoder on top,
and they cleared 80% where our LSTM(128)->LSTM(64) reaches 67.75% cross-signer
on 200 classes.

Two reasons that architecture suits this better than the LSTM:

  * separable conv over time captures the short movement patterns a sign is made
    of at a fraction of the parameters of a recurrent layer, which matters at
    ~15 training clips per class.
  * self-attention lets the classifier weigh the informative part of a sign
    directly rather than carrying it through 30 recurrent steps.

It also removes a deployment constraint rather than adding one. The LSTM needs
unroll=True to convert to TFLite without the Flex delegate; conv and attention
have no recurrence to unroll.

Ablations are selectable so each piece of the rebuild can be attributed:

    python train_v3.py --arch conv --features v3     # everything
    python train_v3.py --arch lstm --features v3     # features only
    python train_v3.py --arch conv --features v2     # architecture only
"""

import os
import sys
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import keras

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "../preprocessing"))

DATA = os.path.join(HERE, "../data/processed/asl_citizen")
SEQ = 30
EPOCHS = 140
BATCH = 32

rng = np.random.default_rng(11)

# MediaPipe hand landmarks are symmetric, so a horizontal flip needs no index
# permutation within a hand -- but the two hand BLOCKS must swap, or a flipped
# two-handed sign lands in the wrong feature space.
HAND_A = slice(0, 63)
HAND_B = slice(63, 126)
POSE_BASE = 126


# ── augmentation ──────────────────────────────────────────────────────────
def temporal_subsample(X, keep_frames=(4, 15), prob=0.85):
    """Simulate the device's capture rate on RAW sequences, before features.

    This has to run before feature construction, not after. v3 carries 201
    velocity dims computed as first differences; when only 5 distinct frames are
    stretched to 30, most consecutive frames are duplicates and those dims
    collapse to near-zero. Subsampling the finished feature tensor would
    resample a velocity channel that was computed at full rate, which is not
    what the app produces — it builds features from whatever frames it actually
    caught.

    Measured cost of omitting this: conv+v3 reached 70.26% at 30 frames but only
    60.84% at 5, below the v2 baseline's 62.53%, i.e. it won where it did not
    matter and lost where it did.
    """
    out = X.copy()
    T = X.shape[1]
    lo, hi = keep_frames
    sel = rng.random(len(X)) < prob
    for i in np.where(sel)[0]:
        n = int(rng.integers(lo, hi + 1))
        if n >= T:
            continue
        pick = np.linspace(0, T - 1, n).round().astype(int)
        back = np.linspace(0, n - 1, T).round().astype(int)
        out[i] = X[i][pick][back]
    return out


def speed_warp(X, rate=(0.5, 1.5), prob=0.8):
    """Resample the clip to a different duration, then back to T.

    Signing tempo varies between people far more than it does between takes by
    the same person, and FSL-105 taught us how badly a single-signer corpus
    generalises. The ISLR 1st-place solution warps 0.5-1.5x; the same range is
    used here.
    """
    out = X.copy()
    T = X.shape[1]
    for i in np.where(rng.random(len(X)) < prob)[0]:
        n = max(4, int(round(T * rng.uniform(*rate))))
        a = np.linspace(0, T - 1, n)
        b = np.linspace(0, n - 1, T).round().astype(int)
        out[i] = X[i][a.round().astype(int)][b]
    return out


def temporal_crop(X, keep=(0.7, 1.0), prob=0.5):
    """Take a contiguous sub-span and stretch it back to T.

    The app's segmentation rarely captures a sign end-to-end; this is the
    training-side counterpart of the truncation measured in segmentation_probe
    (58.44% at a central 50%).
    """
    out = X.copy()
    T = X.shape[1]
    for i in np.where(rng.random(len(X)) < prob)[0]:
        n = int(T * rng.uniform(*keep))
        if n >= T or n < 4:
            continue
        a = int(rng.integers(0, T - n))
        seg = X[i, a:a + n]
        out[i] = seg[np.linspace(0, n - 1, T).round().astype(int)]
    return out


def spatial_affine(X, shear=(-0.15, 0.15), degree=(-15, 15), prob=0.75):
    """Shear and rotate in the image plane, on RAW coordinates.

    Scale and shift are deliberately absent: the v3 features are invariant to
    both by construction, so augmenting them would be a no-op. Shear and
    rotation are NOT removed by that normalisation — they correspond to camera
    tilt and off-axis viewing, which a handheld phone produces constantly.
    """
    out = X.copy()
    T = X.shape[1]
    px = [POSE_BASE + 4 * k for k in range(25)]
    py = [POSE_BASE + 4 * k + 1 for k in range(25)]

    for i in np.where(rng.random(len(X)) < prob)[0]:
        sh = rng.uniform(*shear)
        th = np.deg2rad(rng.uniform(*degree))
        c, s_ = np.cos(th), np.sin(th)
        shear_m = (np.array([[1.0, sh], [0.0, 1.0]], np.float32)
                   if rng.random() < 0.5 else
                   np.array([[1.0, 0.0], [sh, 1.0]], np.float32))
        M = shear_m @ np.array([[c, s_], [-s_, c]], np.float32)

        # Hands are wrist-relative offsets, so they rotate about their own
        # origin with no recentring.
        h = out[i, :, :126].reshape(T, 21 * 2, 3)
        h[:, :, :2] = h[:, :, :2] @ M
        out[i, :, :126] = h.reshape(T, 126)

        # Pose is absolute, so it rotates about the canvas centre. Undetected
        # landmarks stay at exactly zero.
        live = np.any(out[i, :, POSE_BASE:] != 0, axis=1)
        if not live.any():
            continue
        xy = np.stack([out[i][:, px], out[i][:, py]], axis=-1)      # (T, 25, 2)
        xy = (xy - 0.5) @ M + 0.5
        for k in range(25):
            out[i][live, px[k]] = xy[live, k, 0]
            out[i][live, py[k]] = xy[live, k, 1]
    return out


def make_idle(Xraw, n, seq=SEQ):
    """Synthesise NO-SIGN examples: hands present in frame but not signing.

    The app currently has no way to say "nothing". Segmentation fires on hands
    entering and leaving frame, so every hand that appears produces a guess from
    the 200-word vocabulary — which is why a tester sees confident nonsense
    rather than silence. A dedicated class lets the model decline.

    Three kinds of negative, all built from real frames so the landmark
    statistics stay honest:

      FROZEN     one frame held for the whole window — hands up, no movement.
      DRIFT      a very short span stretched out, i.e. slow aimless motion.
      TRANSITION the tail of one clip spliced to the head of another, which is
                 what the camera catches between two signs.
    """
    out = np.empty((n, seq, Xraw.shape[2]), dtype=np.float32)
    for i in range(n):
        kind = rng.integers(0, 3)
        a = int(rng.integers(0, len(Xraw)))
        if kind == 0:
            t = int(rng.integers(0, seq))
            out[i] = np.repeat(Xraw[a, t:t + 1], seq, axis=0)
        elif kind == 1:
            t = int(rng.integers(0, seq - 3))
            seg = Xraw[a, t:t + 3]
            out[i] = seg[np.linspace(0, 2, seq).round().astype(int)]
        else:
            b = int(rng.integers(0, len(Xraw)))
            half = seq // 2
            out[i] = np.concatenate([
                np.repeat(Xraw[a, -1:], half, axis=0),
                np.repeat(Xraw[b, :1], seq - half, axis=0)], axis=0)
    return out


def augment_batch(Xraw, y, n_classes, build_fn, copies=2, flip=True, mask=True,
                  mixup=True, temporal=True):
    """Augment RAW sequences, then build features from each augmented copy.

    Order matters. The geometric augmentations that mattered for v2 (scale
    0.8-3.0, canvas framing 0.5-1.05) simulate nuisances the v3 channels remove
    by construction, so they are gone. What replaces them is capture-rate
    simulation, and that has to happen before feature construction so velocity
    is recomputed on the sparse sequence exactly as the app computes it.
    """
    outX, outY = [build_fn(Xraw)], [y]
    for _ in range(copies):
        R = Xraw
        if temporal:
            R = speed_warp(R)
            R = temporal_crop(R)
            R = temporal_subsample(R)
        R = spatial_affine(R)
        A = build_fn(R)

        if flip:
            # Left-handed signers. Half the batch gets its two hand channels
            # swapped and the x components negated. Without this the model only
            # ever sees one handedness of two-handed signs.
            sel = rng.random(len(A)) < 0.5
            A[sel] = _flip(A[sel])

        if mask:
            # Drop one hand, or a span of frames — the two things that actually
            # happen on device when detection stutters.
            for i in np.where(rng.random(len(A)) < 0.3)[0]:
                if rng.random() < 0.5:
                    _mask_hand(A[i], int(rng.integers(0, 2)))
                else:
                    a = int(rng.integers(0, SEQ - 4))
                    A[i, a:a + int(rng.integers(2, 6))] = 0.0

        # Landmark jitter, scaled to the feature magnitude.
        A += rng.normal(0, 0.01, A.shape).astype(np.float32)
        outX.append(A.astype(np.float32))
        outY.append(y)

    Xa = np.concatenate(outX)
    ya = np.concatenate(outY)

    if mixup:
        Xa, ya = _mixup(Xa, ya, n_classes)
    return Xa, ya


# Mirroring is not just a sign change. MediaPipe pose landmarks are anatomically
# left/right, so a mirrored body has its left shoulder where the right one was.
# Negating x WITHOUT swapping these pairs produces an impossible skeleton, and an
# involution test does not catch it — flipping twice still returns the original.
POSE_MIRROR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12),
                     (13, 14), (15, 16), (17, 18), (19, 20), (21, 22), (23, 24)]


def _swap_pose_pairs(B, base, stride, ncomp):
    """Swap left/right pose landmarks in a block laid out landmark-major."""
    for a, b in POSE_MIRROR_PAIRS:
        ia, ib = base + a * stride, base + b * stride
        t = B[:, :, ia:ia + ncomp].copy()
        B[:, :, ia:ia + ncomp] = B[:, :, ib:ib + ncomp]
        B[:, :, ib:ib + ncomp] = t


def _swap_blocks(B, i, j, n):
    t = B[:, :, i:i + n].copy()
    B[:, :, i:i + n] = B[:, :, j:j + n]
    B[:, :, j:j + n] = t


def _flip(A):
    """Mirror horizontally, for either feature layout.

    v3 (535): pose is shoulder-centred so mirroring is x -> -x.
    v2 (226): pose is absolute in [0,1] so mirroring is x -> 1 - x.
    Both swap the two hand blocks and the anatomical pose pairs.
    """
    B = A.copy()
    dim = B.shape[-1]

    if dim == 226:
        # hands are wrist-relative offsets: negate x, swap the two blocks
        B[:, :, 0:126:3] *= -1.0
        _swap_blocks(B, 0, 63, 63)
        # pose x is absolute; mirror about the canvas centre, keep zeros zero
        px = B[:, :, 126::4]
        B[:, :, 126::4] = np.where(px != 0, 1.0 - px, 0.0)
        _swap_pose_pairs(B, 126, 4, 4)
        return B

    # ---- v3 ----
    B[:, :, 0:75:3] *= -1.0
    _swap_pose_pairs(B, 0, 3, 3)

    for base in (75, 201):                       # hand_loc, hand_shape
        _swap_blocks(B, base, base + 63, 63)
        B[:, :, base:base + 126:3] *= -1.0

    for i, j in ((327, 328), (329, 330)):        # per-hand distances
        t = B[:, :, i].copy()
        B[:, :, i] = B[:, :, j]
        B[:, :, j] = t

    t = B[:, :, 332].copy()                      # detection flags
    B[:, :, 332] = B[:, :, 333]
    B[:, :, 333] = t

    # velocity (334:535) and acceleration (535:736) are differences of
    # (body_pose ++ hand_loc), so each mirrors exactly as the positions do.
    for v0 in (334, 535):
        if v0 + 201 > dim:
            break
        B[:, :, v0:v0 + 75:3] *= -1.0
        _swap_pose_pairs(B, v0, 3, 3)
        _swap_blocks(B, v0 + 75, v0 + 138, 63)
        B[:, :, v0 + 75:v0 + 201:3] *= -1.0

    # arm/palm orientation (736:754): 2 arms x (ux, uy, fx, fy, flex, len)
    # then 2 hands x (palm_x, palm_y, spread). Mirroring negates every x
    # component and swaps left arm with right; flex, length and spread are
    # scalars that survive a mirror unchanged.
    if dim >= 754:
        _swap_blocks(B, 736, 742, 6)          # left arm <-> right arm
        for base in (736, 742):
            B[:, :, base] *= -1.0             # upper-arm x
            B[:, :, base + 2] *= -1.0         # forearm x
        _swap_blocks(B, 748, 751, 3)          # palm A <-> palm B
        for base in (748, 751):
            B[:, :, base] *= -1.0             # palm axis x
    return B


def _mask_hand(seq, which):
    """Drop one hand entirely, in whichever layout this is.

    Every channel that describes that hand has to go together — including its
    velocity — or the model is handed a hand with no position but a coherent
    motion trace, which never occurs on device.
    """
    if seq.shape[-1] == 226:
        seq[:, 63 * which:63 * (which + 1)] = 0.0
        return
    # hand_loc, hand_shape, hand velocity, hand acceleration
    for base in (75, 201, 409, 610):
        if base + 63 * (which + 1) <= seq.shape[-1]:
            seq[:, base + 63 * which:base + 63 * (which + 1)] = 0.0
    seq[:, 332 + which] = 0.0


def _mixup(X, y, n_classes, alpha=0.2, frac=0.5):
    """Convex combinations of sequence pairs, with soft labels.

    Returns one-hot / mixed targets, so the loss must be categorical
    crossentropy rather than the sparse variant.
    """
    Y = np.eye(n_classes, dtype=np.float32)[y]
    n = int(len(X) * frac)
    if n < 2:
        return X, Y
    idx = rng.permutation(len(X))[:n]
    partner = rng.permutation(len(X))[:n]
    lam = rng.beta(alpha, alpha, size=(n, 1, 1)).astype(np.float32)
    X[idx] = lam * X[idx] + (1.0 - lam) * X[partner]
    Y[idx] = lam[:, :, 0] * Y[idx] + (1.0 - lam[:, :, 0]) * Y[partner]
    return X, Y


# ── model ─────────────────────────────────────────────────────────────────
def sep_block(x, filters, kernel=17, drop=0.2):
    """Depthwise-separable conv over time, pre-activation, with a residual.

    kernel=17 follows the ISLR 1st-place solution. It looked oversized for a
    30-step sequence next to the 5 used here originally, but a sign is a whole
    gesture rather than a local edge: a wide kernel sees most of it at once,
    which is exactly what a 5-tap filter cannot do.
    """
    skip = x
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("swish")(x)
    x = tf.keras.layers.SeparableConv1D(filters, kernel, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("swish")(x)
    x = tf.keras.layers.SeparableConv1D(filters, kernel, padding="same")(x)
    x = tf.keras.layers.Dropout(drop)(x)
    if skip.shape[-1] == filters:
        x = tf.keras.layers.Add()([skip, x])
    return x


def transformer_block(x, heads=4, ff=2.0, drop=0.2):
    d = x.shape[-1]
    h = tf.keras.layers.LayerNormalization()(x)
    h = tf.keras.layers.MultiHeadAttention(
        num_heads=heads, key_dim=max(d // heads, 8), dropout=drop)(h, h)
    x = tf.keras.layers.Add()([x, h])
    h = tf.keras.layers.LayerNormalization()(x)
    h = tf.keras.layers.Dense(int(d * ff), activation="swish")(h)
    h = tf.keras.layers.Dropout(drop)(h)
    h = tf.keras.layers.Dense(d)(h)
    return tf.keras.layers.Add()([x, h])


@keras.saving.register_keras_serializable()
class LateDropout(tf.keras.layers.Layer):
    """Dropout that switches on partway through training.

    From the ISLR 1st-place solution, which uses a rate of 0.8 before the
    classifier. Applying it from step 0 stops a small model converging at all;
    switching it on once features have formed acts as a strong late regulariser,
    which is what a ~15-clips-per-class regime needs.
    """

    def __init__(self, rate=0.8, start_step=0, **kw):
        super().__init__(**kw)
        self.rate = rate
        self.start_step = start_step
        self.drop = tf.keras.layers.Dropout(rate)

    def build(self, input_shape):
        super().build(input_shape)
        self._step = self.add_weight(
            name="step", shape=(), dtype=tf.int64,
            initializer=tf.keras.initializers.Constant(0), trainable=False)

    def call(self, x, training=False):
        if not training:
            return x
        self._step.assign_add(1)
        return tf.cond(self._step < self.start_step,
                       lambda: x, lambda: self.drop(x, training=True))

    def get_config(self):
        c = super().get_config()
        c.update(rate=self.rate, start_step=self.start_step)
        return c


def build_conv(n_classes, dim, width=192, dropout_step=0):
    inp = tf.keras.Input(shape=(SEQ, dim))
    x = tf.keras.layers.Dense(width)(inp)
    for _ in range(3):
        x = sep_block(x, width)
    x = transformer_block(x)
    x = transformer_block(x)
    x = tf.keras.layers.LayerNormalization()(x)
    # Mean and max pooling together: mean carries the whole sign, max catches
    # the brief distinctive moment that separates similar signs.
    x = tf.keras.layers.Concatenate()([
        tf.keras.layers.GlobalAveragePooling1D()(x),
        tf.keras.layers.GlobalMaxPooling1D()(x),
    ])
    x = LateDropout(0.8, start_step=dropout_step)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out)


def build_lstm(n_classes, dim):
    """The v2 architecture, for the features-only ablation."""
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQ, dim)),
        tf.keras.layers.LSTM(128, return_sequences=True, unroll=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64, unroll=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])


def subsample(X, n):
    if n >= X.shape[1]:
        return X
    pick = np.linspace(0, X.shape[1] - 1, n).round().astype(int)
    back = np.linspace(0, n - 1, X.shape[1]).round().astype(int)
    return X[:, pick][:, back]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["conv", "lstm"], default="conv")
    ap.add_argument("--features", choices=["v3", "v2"], default="v3")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--no-mixup", action="store_true")
    ap.add_argument("--idle-class", action="store_true",
                    help="add a NO-SIGN class so the model can decline")
    ap.add_argument("--vocab", default=None,
                    help="file with one word per line, restricting the class "
                         "set. Fewer classes means fewer confusable "
                         "competitors, which is the cheapest accuracy lever "
                         "available for a fixed demo vocabulary.")
    ap.add_argument("--no-temporal", action="store_true",
                    help="disable capture-rate simulation "
                         "(reproduces the first v3 run)")
    args = ap.parse_args()

    tag = args.tag or f"{args.arch}_{args.features}"

    if args.vocab:
        with open(args.vocab, encoding="utf-8") as f:
            CORE200 = [w.strip() for w in f if w.strip()]
        print(f"vocabulary: {args.vocab} ({len(CORE200)} words)")
    else:
        from asl_core200 import CORE200
    labels = {}
    for line in open(os.path.join(DATA, "labels.txt"), encoding="utf-8"):
        p = line.strip().split(",", 1)
        if len(p) == 2 and p[1]:
            labels[int(p[0])] = p[1]

    # Always load the RAW canonical tensors; features are built after
    # augmentation so capture-rate simulation reaches the velocity channels.
    X = np.load(os.path.join(DATA, "X_canon.npy")).astype(np.float32)
    y = np.load(os.path.join(DATA, "y.npy"))
    split = np.load(os.path.join(DATA, "split.npy"))

    keep_cls = sorted(c for c, n in labels.items() if n in set(CORE200))
    mask = np.isin(y, keep_cls)
    X, y, split = X[mask], y[mask], split[mask]
    remap = {c: i for i, c in enumerate(keep_cls)}
    y = np.array([remap[c] for c in y])
    labels = {remap[c]: labels[c] for c in keep_cls}
    n_classes = len(labels)

    tr, va, te = split == "train", split == "val", split == "test"

    if args.features == "v3":
        from features_v3 import build as build_v3, trim_idle
        # Idle trimming is part of the representation, so it applies to every
        # split — the app never sees the dead air at a corpus clip's ends.
        X = trim_idle(X)
        build_fn = build_v3
    else:
        build_fn = lambda Z: Z            # noqa: E731

    print(f"features {args.features}   arch={args.arch}")
    print(f"classes {n_classes}   train {tr.sum():,}  val {va.sum():,}  "
          f"test {te.sum():,}  (signer-disjoint)", flush=True)

    idle_id = None
    if args.idle_class:
        # One extra class, sized like a large word class so it neither
        # dominates nor gets ignored.
        idle_id = n_classes
        n_idle = int(np.median(np.bincount(y[tr])) * 12)
        Xi = make_idle(X[tr], n_idle)
        X = np.concatenate([X, Xi])
        y = np.concatenate([y, np.full(len(Xi), idle_id)])
        tr = np.concatenate([tr, np.ones(len(Xi), bool)])
        va = np.concatenate([va, np.zeros(len(Xi), bool)])
        te = np.concatenate([te, np.zeros(len(Xi), bool)])
        n_classes += 1
        labels[idle_id] = ""          # empty label = never displayed by the app
        print(f"idle/no-sign class: {len(Xi)} synthetic negatives "
              f"(class {idle_id})", flush=True)

    Xtr, Ytr = augment_batch(
        X[tr], y[tr], n_classes, build_fn, copies=args.copies,
        flip=not args.no_flip, mask=not args.no_mask,
        mixup=not args.no_mixup, temporal=not args.no_temporal)
    Xva, Xte = build_fn(X[va]), build_fn(X[te])
    print(f"train after augmentation: {Xtr.shape}", flush=True)

    steps_per_epoch = max(1, len(Xtr) // BATCH)
    model = (build_conv(n_classes, Xtr.shape[-1],
                        dropout_step=steps_per_epoch * 15)
             if args.arch == "conv"
             else build_lstm(n_classes, Xtr.shape[-1]))
    # Mixup produces soft labels, so val targets must be one-hot too.
    Yva = np.eye(n_classes, dtype=np.float32)[y[va]]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"])
    print(f"parameters: {model.count_params():,}", flush=True)

    model.fit(
        Xtr, Ytr, validation_data=(Xva, Yva),
        epochs=EPOCHS, batch_size=BATCH, verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                             restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                 patience=7, min_lr=1e-6),
        ])

    probs = model.predict(Xte, verbose=0, batch_size=256)
    pred = np.argmax(probs, axis=1)
    acc = (pred == y[te]).mean() * 100
    top5 = np.mean([y[te][i] in np.argsort(-probs[i])[:5]
                    for i in range(len(probs))]) * 100
    shown = ((pred == y[te]) & (probs.max(axis=1) >= 0.6)).mean() * 100

    print(f"\n=== v3 {tag} ===")
    print(f"CROSS-SIGNER top-1 : {acc:.2f}%   top-5 {top5:.2f}%   "
          f"shown {shown:.2f}%")
    print(f"v2 baseline (core200): 67.75% / 90.05% / 60.67%")

    print(f"\naccuracy vs capture rate")
    for n in (4, 5, 6, 8, 30):
        # Subsample the RAW sequence and build features after, exactly as the
        # app does. Subsampling the finished feature tensor instead resamples a
        # velocity channel that was computed at full rate, which understates
        # v3 at low capture rates (measured 62.94% vs a true 65.12% at 5).
        p = model.predict(build_fn(subsample(X[te], n)), verbose=0,
                          batch_size=256)
        t1 = (np.argmax(p, axis=1) == y[te]).mean() * 100
        sh = ((np.argmax(p, axis=1) == y[te]) &
              (p.max(axis=1) >= 0.6)).mean() * 100
        note = "  <- device operating point" if n in (4, 5, 6) else ""
        print(f"  {n:<4}{t1:>7.2f}%   shown {sh:>6.2f}%{note}")

    model.save(os.path.join(HERE, f"asl_v3_{tag}_best.keras"))
    # Without this, evaluation tools fall back to a y < n_classes heuristic and
    # silently score against the wrong 200 classes (measured once as 0.77%).
    with open(os.path.join(HERE, f"asl_v3_{tag}_labels.txt"), "w",
              encoding="utf-8") as f:
        for i in range(n_classes):
            f.write("%d,%s\n" % (i, labels.get(i, "")))
    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        conv.target_spec.supported_types = [tf.float16]
        tfl = conv.convert()
        with open(os.path.join(HERE, f"asl_v3_{tag}.tflite"), "wb") as f:
            f.write(tfl)
        print(f"\nTFLite: asl_v3_{tag}.tflite  ({len(tfl)/1024:.0f} KB)  "
              f"— no Flex delegate needed")
    except Exception as e:
        print(f"\nTFLite conversion FAILED: {str(e)[:300]}")


if __name__ == "__main__":
    main()
