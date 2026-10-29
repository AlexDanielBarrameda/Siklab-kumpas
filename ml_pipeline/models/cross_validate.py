"""5-fold cross-validation of the canonicalised FSL-105 word model.

WHY: the single train/val/test split leaves only ~3 test clips per class
(320 test clips / 105 classes), so per-class figures like "THREE: 100%" rest on
three samples and cannot be trusted. Cross-validation tests every clip exactly
once, giving ~20 test results per class and a confusion matrix built from all
2129 predictions.

This measures; it does not produce a shipping model. The five fold models are
discarded. The deployed model stays fsl105_v2canon.tflite.

Comparability with train_fsl105_v2_aug.py is deliberate: same architecture, same
augmentation, augmentation applied to the training portion only so held-out
clips stay clean.
"""

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_fsl105_v2_aug import augment_set, build_model, AUG_PER_SAMPLE

DATA = os.path.join(HERE, "../data/processed/fsl_105_v2")
LBL = os.path.join(HERE, "../../flutter_app/assets/labels/fsl105_labels.txt")
REPORT = os.path.join(HERE, "cv_report.txt")
NPZ = os.path.join(HERE, "cv_predictions.npz")

N_FOLDS = 5
EPOCHS = 150
BATCH = 16


def load_labels():
    out = {}
    for line in open(LBL, encoding="utf-8"):
        q = line.strip().split(",")
        if len(q) >= 2:
            out[int(q[0])] = ",".join(q[1:]).strip()
    return out


def main():
    labels = load_labels()
    X = np.load(os.path.join(DATA, "X_canon.npy")).astype(np.float32)
    y = np.load(os.path.join(DATA, "y_canon.npy"))
    print(f"{X.shape[0]} clips, {len(np.unique(y))} classes", flush=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_pred = np.full(len(y), -1, dtype=np.int64)
    oof_conf = np.zeros(len(y), dtype=np.float32)
    fold_acc = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        # Carve a validation slice out of the training portion for early stopping,
        # so the held-out fold is never seen in any capacity.
        tr, va = train_test_split(tr_idx, test_size=0.15, random_state=42,
                                  stratify=y[tr_idx])
        Xtr, ytr = augment_set(X[tr], y[tr], AUG_PER_SAMPLE)

        model = build_model()
        model.fit(
            Xtr, ytr,
            validation_data=(X[va], y[va]),
            epochs=EPOCHS, batch_size=BATCH, verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=15, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6),
            ],
        )

        probs = model.predict(X[te_idx], verbose=0)
        oof_pred[te_idx] = probs.argmax(axis=1)
        oof_conf[te_idx] = probs.max(axis=1)
        acc = (oof_pred[te_idx] == y[te_idx]).mean()
        fold_acc.append(acc)
        print(f"fold {fold}/{N_FOLDS}: {acc*100:.2f}%  "
              f"(train {len(tr)}+aug={len(Xtr)}, test {len(te_idx)})", flush=True)
        tf.keras.backend.clear_session()

    fold_acc = np.array(fold_acc)
    overall = (oof_pred == y).mean()

    lines = []
    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    out("")
    out("=" * 66)
    out("5-FOLD CROSS-VALIDATION - FSL-105 canonicalised word model")
    out("=" * 66)
    out(f"per-fold accuracy : "
        f"{', '.join(f'{a*100:.2f}%' for a in fold_acc)}")
    out(f"mean +/- sd       : {fold_acc.mean()*100:.2f}% +/- "
        f"{fold_acc.std()*100:.2f}%")
    out(f"pooled accuracy   : {overall*100:.2f}%  "
        f"({(oof_pred==y).sum()}/{len(y)} clips)")
    out(f"single-split figure for reference: 97.19% on 320 clips")

    # Per-class, now with ~20 samples each
    out("")
    out("WEAKEST CLASSES (recall, n = clips of that class)")
    rows = []
    for c in np.unique(y):
        m = y == c
        rows.append((labels.get(int(c), str(c)), (oof_pred[m] == c).mean(),
                     int(m.sum()), c))
    rows.sort(key=lambda r: r[1])
    for name, acc, n, c in rows[:15]:
        m = y == c
        wrong = oof_pred[m][oof_pred[m] != c]
        if len(wrong):
            v, k = np.unique(wrong, return_counts=True)
            top = ", ".join(f"{labels.get(int(v[i]),'?')} x{k[i]}"
                            for i in np.argsort(-k)[:3])
        else:
            top = "-"
        out(f"  {name:<22} {acc*100:5.1f}%  (n={n:2d})  -> {top}")

    out("")
    out(f"classes at 100%: {sum(1 for r in rows if r[1] == 1.0)}/105")

    # Most common confusions across the whole dataset
    out("")
    out("MOST COMMON CONFUSIONS (true -> predicted, count)")
    pairs = {}
    for t, p in zip(y, oof_pred):
        if t != p:
            pairs[(int(t), int(p))] = pairs.get((int(t), int(p)), 0) + 1
    for (t, p), n in sorted(pairs.items(), key=lambda kv: -kv[1])[:20]:
        out(f"  {labels.get(t,'?'):<22} -> {labels.get(p,'?'):<22} {n}")

    # The pairs the device showed
    out("")
    out("PAIRS OBSERVED ON DEVICE")
    rev = {v: k for k, v in labels.items()}
    for a, b in [("TWO", "WEDNESDAY"), ("THREE", "TWO"), ("YES", "YELLOW"),
                 ("ONE", "ONE")]:
        ca, cb = rev.get(a), rev.get(b)
        if ca is None or cb is None:
            continue
        m = y == ca
        out(f"  {a:<10} -> {b:<12} {(oof_pred[m]==cb).sum()}/{m.sum()} clips "
            f"({(oof_pred[m]==cb).mean()*100:.0f}%)")

    # Confidence calibration at the app's gate
    out("")
    gate = oof_conf >= 0.6
    out(f"at the app's 0.60 confidence gate:")
    out(f"  passing the gate     : {gate.mean()*100:.1f}% of clips")
    out(f"  accuracy among those : {(oof_pred[gate]==y[gate]).mean()*100:.1f}%")
    out(f"  mean conf when right : {oof_conf[oof_pred==y].mean():.3f}")
    out(f"  mean conf when wrong : {oof_conf[oof_pred!=y].mean():.3f}")

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    np.savez(NPZ, y_true=y, y_pred=oof_pred, conf=oof_conf,
             fold_acc=fold_acc)
    out("")
    out(f"report -> {REPORT}")
    out(f"raw predictions -> {NPZ}")


if __name__ == "__main__":
    main()
