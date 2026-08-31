"""Diff the tensors the device actually builds against the training corpus.

Offline simulation of every device condition we could identify -- device
framing, 5 distinct frames, the confidence gate -- predicts roughly 48% of ASL
signs should be recognised and shown. Live, almost none are. Frame rate,
segmentation, canonicalisation and zero-hand frames were each measured and ruled
out, so the remainder is something the corpus cannot be made to reproduce.

This compares real device tensors, block by block, against the distribution the
model was trained on, and reports which dimensions are out of range. It also
runs the model on the dumped tensor so the device's own prediction can be
reproduced offline -- if the two disagree, the problem is in the interpreter or
the TFLite conversion rather than in the features.

Capture dumps by enabling the DUMP chip in a debug build, signing a few words,
then:

    adb pull /sdcard/Android/data/com.siklabkumpas.siklab_kumpas/files/tensor_dumps
    python compare_device_tensor.py tensor_dumps/ --model ../models/asl_525frame_best.keras
"""

import os
import csv
import json
import glob
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data/processed/asl_citizen")

POSE0 = 126
L_SHOULDER, R_SHOULDER = 11, 12
L_WRIST, R_WRIST = 15, 16


def load_dumps(path):
    files = (sorted(glob.glob(os.path.join(path, "*.json")))
             if os.path.isdir(path) else [path])
    out = []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
            d["_file"] = os.path.basename(f)
            d["_seq"] = np.array(d["sequence"], dtype=np.float32)
            if d["_seq"].shape != (30, 226):
                print(f"  skipping {d['_file']}: shape {d['_seq'].shape}")
                continue
            out.append(d)
        except Exception as e:
            print(f"  skipping {os.path.basename(f)}: {e}")
    return out


def hand_spans(seq):
    v = []
    for t in range(seq.shape[0]):
        for b in (0, 63):
            blk = seq[t, b:b + 63].reshape(21, 3)
            if np.any(blk):
                v.append(np.linalg.norm(blk[:, :2] - blk[0, :2], axis=1).max())
    return np.array(v) if v else np.array([np.nan])


def body_stats(seq):
    """Shoulder width and posture, using the app's own formula."""
    sw, post = [], []
    for t in range(seq.shape[0]):
        if not np.any(seq[t, POSE0:]):
            continue

        def px(j, c):
            return seq[t, POSE0 + 4 * j + c]

        lsx, lsy = px(L_SHOULDER, 0), px(L_SHOULDER, 1)
        rsx, rsy = px(R_SHOULDER, 0), px(R_SHOULDER, 1)
        if (lsx == 0 and lsy == 0) or (rsx == 0 and rsy == 0):
            continue
        w = np.hypot(lsx - rsx, lsy - rsy)
        if w < 1e-4:
            continue
        sw.append(w)
        lwy, rwy = px(L_WRIST, 1), px(R_WRIST, 1)
        if lwy == 0 and rwy == 0:
            continue
        wrist = rwy if lwy == 0 else (lwy if rwy == 0 else min(lwy, rwy))
        post.append(((lsy + rsy) / 2 - wrist) / w)
    return (np.array(sw) if sw else np.array([np.nan]),
            np.array(post) if post else np.array([np.nan]))


def corpus_reference():
    X = np.load(os.path.join(DATA, "X_canon.npy"), mmap_mode="r")
    sp = np.load(os.path.join(DATA, "split.npy"))
    idx = np.where(sp == "train")[0][:1500]
    S = np.array(X[idx]).astype(np.float32)
    spans = np.concatenate([hand_spans(s) for s in S])
    sw = np.concatenate([body_stats(s)[0] for s in S])
    po = np.concatenate([body_stats(s)[1] for s in S])
    return S, spans, sw, po


def band(name, dev, ref, unit=""):
    """One comparison line, flagged when the device sits outside p1..p99."""
    dev = dev[~np.isnan(dev)]
    ref = ref[~np.isnan(ref)]
    if dev.size == 0 or ref.size == 0:
        print(f"  {name:<22} no data")
        return
    lo, hi = np.percentile(ref, 1), np.percentile(ref, 99)
    med = np.median(dev)
    flag = "  <-- OUTSIDE corpus p1-p99" if (med < lo or med > hi) else ""
    print(f"  {name:<22} device {med:7.3f}{unit}   "
          f"corpus {np.median(ref):7.3f} [{lo:.3f}, {hi:.3f}]{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", help="directory of dumped .json files, or one file")
    ap.add_argument("--model", default=None,
                    help="reproduce the device's prediction offline")
    args = ap.parse_args()

    dumps = load_dumps(args.dumps)
    if not dumps:
        raise SystemExit(f"no usable dumps in {args.dumps}")
    print(f"loaded {len(dumps)} dump(s)\n")

    print("loading corpus reference (1500 train clips)...")
    S, ref_span, ref_sw, ref_post = corpus_reference()

    dev = np.stack([d["_seq"] for d in dumps])
    print(f"\n{'='*66}\nDISTRIBUTION COMPARISON\n{'='*66}")

    band("hand span", np.concatenate([hand_spans(s) for s in dev]), ref_span)
    band("shoulder width", np.concatenate([body_stats(s)[0] for s in dev]), ref_sw)
    band("posture", np.concatenate([body_stats(s)[1] for s in dev]), ref_post)

    # Whole-vector reach: how far outside the corpus range each block sits.
    print()
    for name, sl in (("left-hand block", slice(0, 63)),
                     ("right-hand block", slice(63, 126)),
                     ("pose block", slice(126, 226))):
        d, r = dev[:, :, sl].ravel(), S[:, :, sl].ravel()
        d, r = d[d != 0], r[r != 0]
        if d.size == 0:
            print(f"  {name:<22} device: ALL ZERO"
                  f"  <-- model never saw this in training")
            continue
        lo, hi = np.percentile(r, 1), np.percentile(r, 99)
        out = ((d < lo) | (d > hi)).mean() * 100
        flag = "  <-- large fraction out of range" if out > 20 else ""
        print(f"  {name:<22} {out:5.1f}% of device values outside "
              f"corpus p1-p99{flag}")

    # Frames actually captured — the app resamples whatever it got up to 30, so
    # the tensor never reveals this on its own.
    print(f"\n{'='*66}\nCAPTURE\n{'='*66}")
    for d in dumps:
        print(f"  {d['_file'][:34]:<36} raw {d.get('raw_frames', '?'):>3} frames"
              f"  {d.get('span_ms', '?'):>5}ms  {d.get('fps', 0):.1f}fps"
              f"  -> {d.get('label') or '(none)'} "
              f"{(d.get('confidence') or 0):.2f}")

    if args.model:
        import tensorflow as tf
        model = tf.keras.models.load_model(os.path.abspath(args.model))
        labels = {}
        lp = os.path.join(DATA, "labels.txt")
        if os.path.exists(lp):
            for line in open(lp, encoding="utf-8"):
                p = line.strip().split(",", 1)
                if len(p) == 2:
                    labels[int(p[0])] = p[1]
        print(f"\n{'='*66}\nOFFLINE REPRODUCTION\n{'='*66}")
        print("  device said            | this model says        | agree")
        probs = model.predict(dev, verbose=0)
        for d, p in zip(dumps, probs):
            k = int(np.argmax(p))
            mine = f"{labels.get(k, k)} {p[k]:.2f}"
            theirs = f"{d.get('label') or '(none)'} {(d.get('confidence') or 0):.2f}"
            same = "yes" if (d.get("label") or "").upper() == \
                labels.get(k, "").upper() else "NO"
            print(f"  {theirs:<22} | {mine:<22} | {same}")
        print("\n  Disagreement here means the TFLite model on the device and "
              "this\n  Keras model differ — a conversion or label problem, not "
              "a feature one.")


if __name__ == "__main__":
    main()
