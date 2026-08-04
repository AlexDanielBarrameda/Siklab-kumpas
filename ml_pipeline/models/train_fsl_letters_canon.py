"""FSL letters model, canonicalised by geometric chirality.

WHY: fingerspelling currently only works for four specific combinations of
mirror setting / hand block / which hand the user signs with. Those four are
exactly the cases where the three chirality flips compose to one canonical form;
the other four fail. The cause is that the hand block is chosen from MediaPipe's
Left/Right label, which agrees with the actual apparent chirality only 28.6% of
the time on this dataset.

FIX: measure apparent chirality from the palm geometry (signed area of the
wrist / index-knuckle / pinky-knuckle triangle), mirror every hand to one
convention, and always place it in the right-hand block. Applied identically
here and in inference_service.dart, the mirror and hand-block settings stop
mattering for letters.

Keeps the original architecture so the numbers stay comparable to the 97.80%
baseline of fsl_letters.tflite.
"""

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "preprocessing"))
from canonicalize import canonicalise_letter_frame, hand_chirality, mirror_block

DATA = os.path.join(HERE, "../data/processed/fsl_letters")
OUT_KERAS = os.path.join(HERE, "fsl_letters_canon_best.keras")
OUT_TFLITE = os.path.join(HERE, "fsl_letters_canon.tflite")

DIM, NCLASS = 226, 26
EPOCHS, BATCH = 100, 32
AUG_COPIES = 2

rng = np.random.default_rng(42)
HAND = slice(0, 126)


def augment(v):
    """Scale / rotate / jitter the hand. No mirror augmentation: chirality is
    now canonical, and mirroring would reintroduce the ambiguity we removed."""
    s = v.copy()
    blk = s[63:126].reshape(21, 3)
    if not np.any(blk):
        return s
    anchor = blk[0].copy()
    off = blk - anchor

    off *= rng.uniform(0.8, 1.25)
    th = np.deg2rad(rng.uniform(-15, 15))
    ca, sa = np.cos(th), np.sin(th)
    x, y = off[:, 0].copy(), off[:, 1].copy()
    off[:, 0] = x * ca - y * sa
    off[:, 1] = x * sa + y * ca

    blk = anchor + off
    blk[:, 0] += rng.uniform(-0.03, 0.03)
    blk[:, 1] += rng.uniform(-0.03, 0.03)
    s[63:126] = blk.reshape(-1)
    s[HAND] += rng.normal(0, 0.004, 126) * (s[HAND] != 0)
    return s.astype(np.float32)


def build_model():
    m = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="relu", input_shape=(DIM,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(NCLASS, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def main():
    X = np.load(os.path.join(DATA, "X.npy")).astype(np.float32)
    y = np.load(os.path.join(DATA, "y.npy"))
    print(f"loaded {X.shape}", flush=True)

    C = np.stack([canonicalise_letter_frame(v) for v in X])
    chir = np.array([hand_chirality(v[63:126]) for v in C])
    print(f"canonicalised: chirality +1 {np.mean(chir==1)*100:.1f}%  "
          f"-1 {np.mean(chir==-1)*100:.1f}%  undecidable {np.mean(chir==0)*100:.1f}%",
          flush=True)

    Xtr, Xtmp, ytr, ytmp = train_test_split(C, y, test_size=0.3,
                                            random_state=42, stratify=y)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5,
                                          random_state=42, stratify=ytmp)

    aug = [Xtr] + [np.stack([augment(v) for v in Xtr]) for _ in range(AUG_COPIES)]
    Xtr_a = np.concatenate(aug)
    ytr_a = np.concatenate([ytr] * (AUG_COPIES + 1))
    print(f"train {Xtr_a.shape}  val {Xva.shape}  test {Xte.shape}", flush=True)

    model = build_model()
    model.fit(Xtr_a, ytr_a, validation_data=(Xva, yva),
              epochs=EPOCHS, batch_size=BATCH, verbose=2,
              callbacks=[
                  tf.keras.callbacks.EarlyStopping(
                      monitor="val_loss", patience=12,
                      restore_best_weights=True),
                  tf.keras.callbacks.ReduceLROnPlateau(
                      monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6),
                  tf.keras.callbacks.ModelCheckpoint(
                      OUT_KERAS, monitor="val_accuracy", save_best_only=True),
              ])

    loss, acc = model.evaluate(Xte, yte, verbose=0)
    print(f"\nTest accuracy (canonicalised) : {acc:.4f}")
    print(f"baseline fsl_letters.tflite    : 0.9780")

    # THE GATE: the same hand, mirrored, must give the same answer.
    mirrored = Xte.copy()
    for i in range(len(mirrored)):
        blk = mirrored[i, 63:126]
        if np.any(blk):
            mirrored[i, 63:126] = mirror_block(blk)
    mirrored = np.stack([canonicalise_letter_frame(v) for v in mirrored])
    p0 = model.predict(Xte, verbose=0).argmax(axis=1)
    p1 = model.predict(mirrored, verbose=0).argmax(axis=1)
    print(f"mirrored-input accuracy        : {(p1 == yte).mean():.4f}")
    print(f"predictions unchanged          : {(p0 == p1).mean()*100:.1f}%")

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    tfl = conv.convert()
    with open(OUT_TFLITE, "wb") as f:
        f.write(tfl)
    print(f"TFLite written: {OUT_TFLITE} ({len(tfl)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
