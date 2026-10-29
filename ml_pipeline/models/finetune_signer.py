"""Adapt the ASL model to the demonstrator, from tensors captured on device.

WHY

The 50-word model scores 90.86% cross-signer on ASL Citizen and produces almost
nothing for this signer on ten specific words. The captured dumps show why: the
predictions are CONSISTENT, not random.

    signed          model says
    HELLO           CAMERA x4, PARENTS x2
    THANK YOU       MOTHER x6
    EAT             SORRY x6
    WATER           MOTHER x6
    HELP            WHY x5

Consistency is the whole argument for adaptation. Scattered predictions would
mean the capture was noise and no amount of training would help; a stable
mapping to the wrong label means the features are separable and only the
decision boundary is misplaced. That is what fine-tuning moves.

This is speaker adaptation, standard in speech recognition, and it MUST be
disclosed: the shipped model is tuned to one demonstrator and its accuracy for
anyone else is the un-adapted number, not the one printed at the end of this
script.

METHOD

The corpus data is mixed back in at every step rather than training on the
device captures alone. With ~8 examples per word, training on them alone would
overfit within an epoch and destroy the 40 words that already work — the
failure mode is catastrophic forgetting, and the cost of it here is the whole
demo. Only the classifier head and the last block are unfrozen for the same
reason.

    python finetune_signer.py --dumps <dir> --order HELLO,THANKYOU,...
"""

import os
import sys
import glob
import json
import argparse
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "../preprocessing"))

from train_v3 import LateDropout, augment_batch          # noqa: E402
from features_v3 import build, trim_idle                 # noqa: E402

DATA = os.path.join(HERE, "../data/processed/asl_citizen")


def load_dumps(d):
    """Device captures, in capture order, with their timestamps.

    Two on-device formats exist. The original inlined the tensor as JSON, which
    turned out to block the camera stream long enough to wedge it, so newer
    dumps write the tensor as a raw float32 sidecar and keep only metadata in
    the JSON. Both are read here — the early captures are still the only
    examples we have for several words.
    """
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            j = json.load(open(f, encoding="utf-8"))
            if "sequence" in j:
                seq = np.array(j["sequence"], dtype=np.float32)
            elif "bin" in j:
                binp = os.path.join(d, j["bin"])
                if not os.path.exists(binp):
                    continue
                seq = np.fromfile(binp, dtype=np.float32).reshape(
                    int(j["rows"]), int(j["cols"]))
            else:
                continue
            if seq.shape != (30, 226):
                continue
            stamp = os.path.basename(f).split("_", 2)[2].replace(".json", "")
            t = datetime.strptime(stamp[:19], "%Y-%m-%dT%H-%M-%S")
            out.append((t, seq, j.get("label")))
        except Exception:
            continue
    # Sort by TIMESTAMP, not filename. The dump filename carries a counter that
    # restarts at 000 every time the app does, so filename order interleaves
    # sessions and puts the pauses in the wrong places — which silently merged
    # and dropped whole words rather than failing.
    out.sort(key=lambda r: r[0])
    return out


def group_by_pauses(recs, gap_s=8.0):
    """Split the capture into per-word runs on the deliberate pauses.

    The protocol was: sign one word N times, rest ~10s, next word. Anything
    shorter than `gap_s` is a gap between repetitions, anything longer is a
    change of word.
    """
    groups, cur = [], []
    for i, (t, seq, lab) in enumerate(recs):
        if cur and (t - recs[i - 1][0]).total_seconds() > gap_s:
            groups.append(cur)
            cur = []
        cur.append((seq, lab))
    if cur:
        groups.append(cur)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True, action="append",
                    help="capture directory; repeat for several sessions")
    ap.add_argument("--order", required=True, action="append",
                    help="comma-separated words for the matching --dumps, in "
                         "the order they were signed; repeat alongside --dumps")
    ap.add_argument("--base", default="asl_v3_demo50_best.keras")
    ap.add_argument("--labels", default="asl_v3_demo50_labels.txt")
    ap.add_argument("--tag", default="demo50adapt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--min-group", type=int, default=6)
    ap.add_argument("--max-group", type=int, default=11)
    ap.add_argument("--unfreeze", type=int, default=6,
                    help="how many trailing layers to train. Too few and the "
                         "adapted words compete for capacity and evict each "
                         "other; too many and the corpus features that make "
                         "the rest of the vocabulary work are overwritten.")
    ap.add_argument("--holdout", type=float, default=0.25,
                    help="fraction of device examples per word kept OUT of "
                         "training, to measure generalisation to the signer's "
                         "next rendition rather than memorisation")
    ap.add_argument("--reps", type=float, default=0.35,
                    help="how heavily device examples are oversampled against "
                         "the corpus")
    ap.add_argument("--gap", type=float, action="append", default=None,
                    help="seconds of silence separating two words; repeat "
                         "per --dumps when sessions used different pauses")
    args = ap.parse_args()
    if not args.gap:
        args.gap = [8.0]

    names = {}
    for line in open(os.path.join(HERE, args.labels), encoding="utf-8"):
        p = line.rstrip("\n").split(",", 1)
        if len(p) == 2:
            names[p[1]] = int(p[0])
    n_classes = max(names.values()) + 1

    # Sessions are segmented independently: pauses only delimit words WITHIN a
    # session, and the gap between two sessions is meaningless.
    groups, order = [], []
    # One gap per session: the pause length was a hand instruction and differed
    # between capture runs (~10s in the first, 15s in the second). A single
    # global threshold either merges words in one session or splits them in the
    # other, and both produce mislabelled examples.
    gaps = args.gap if len(args.gap) == len(args.dumps) else \
        [args.gap[0]] * len(args.dumps)
    for d, o, gp in zip(args.dumps, args.order, gaps):
        recs = load_dumps(d)
        g = group_by_pauses(recs, gap_s=gp)
        words = [w.strip().upper() for w in o.split(",")]
        print(f"{os.path.basename(d.rstrip('/'))}: {len(recs)} captures -> "
              f"{len(g)} groups, {len(words)} words given")
        groups += g
        order += words
    print(f"total {len(groups)} groups, {len(order)} words")

    Xd, yd = [], []
    for gi, g in enumerate(groups):
        if gi >= len(order):
            print(f"  group {gi}: no word given, skipped")
            continue
        w = order[gi]
        if w not in names:
            print(f"  group {gi} {w}: not in the label set, skipped")
            continue
        # A group far off the expected repetition count means the pause
        # boundaries did not line up with the words, so its label is a guess.
        # A wrong label here teaches the model the opposite of what is wanted,
        # which is worse than having no example at all.
        if not (args.min_group <= len(g) <= args.max_group):
            print(f"  group {gi} {w}: n={len(g)} outside "
                  f"[{args.min_group},{args.max_group}], skipped as unreliable")
            continue
        for seq, _ in g:
            Xd.append(seq)
            yd.append(names[w])
        print(f"  group {gi} {w}: {len(g)} examples")

    if not Xd:
        raise SystemExit("no usable device captures")
    Xd = np.stack(Xd)
    yd = np.array(yd)

    # Hold out some device examples per word.
    #
    # Reporting accuracy on the examples the model just trained on says only
    # that fitting happened. It said 97.80% while THANK YOU and YES were about
    # to fail 7 of 8 times live, because the adaptation had memorised those
    # captures rather than learning the signer's rendition. A holdout makes
    # over-fitting visible here instead of on the device.
    rs = np.random.default_rng(0)
    tr_idx, te_idx = [], []
    for c in np.unique(yd):
        idx = np.where(yd == c)[0]
        rs.shuffle(idx)
        k = max(1, int(round(len(idx) * args.holdout)))
        te_idx += list(idx[:k])
        tr_idx += list(idx[k:])
    Xd_te, yd_te = Xd[te_idx], yd[te_idx]
    Xd, yd = Xd[tr_idx], yd[tr_idx]
    print(f"\ndevice examples: {len(yd)} train + {len(yd_te)} held out, "
          f"across {len(set(yd))} words")

    # Corpus data, so the words that already work keep working.
    Xc = np.load(os.path.join(DATA, "X_canon.npy")).astype(np.float32)
    yc_raw = np.load(os.path.join(DATA, "y.npy"))
    split = np.load(os.path.join(DATA, "split.npy"))
    lab = {}
    for line in open(os.path.join(DATA, "labels.txt"), encoding="utf-8"):
        p = line.strip().split(",", 1)
        if len(p) == 2 and p[1]:
            lab[p[1]] = int(p[0])
    keep = {lab[w]: names[w] for w in names if w in lab}
    m = np.isin(yc_raw, list(keep))
    Xtr_c = Xc[m & (split == "train")]
    ytr_c = np.array([keep[c] for c in yc_raw[m & (split == "train")]])
    Xte_c = Xc[m & (split == "test")]
    yte_c = np.array([keep[c] for c in yc_raw[m & (split == "test")]])
    print(f"corpus train {len(ytr_c)}, corpus test {len(yte_c)}")

    model = tf.keras.models.load_model(
        os.path.join(HERE, args.base),
        custom_objects={"LateDropout": LateDropout})

    def evaluate(tag):
        pc = model.predict(build(trim_idle(Xte_c)), verbose=0, batch_size=256)
        ac = (np.argmax(pc, 1) == yte_c).mean() * 100
        ptr = model.predict(build(trim_idle(Xd)), verbose=0, batch_size=256)
        atr = (np.argmax(ptr, 1) == yd).mean() * 100
        pte = model.predict(build(trim_idle(Xd_te)), verbose=0, batch_size=256)
        ate = (np.argmax(pte, 1) == yd_te).mean() * 100
        ste = ((np.argmax(pte, 1) == yd_te) &
               (pte.max(1) >= 0.35)).mean() * 100
        # device-held-out is the only column that predicts live behaviour.
        print(f"  {tag:<10} corpus {ac:6.2f}%   device-train {atr:6.2f}%   "
              f"device-HELD-OUT {ate:6.2f}%  (shown {ste:5.2f}%)")
        return ac, ate

    print("\nbefore adaptation")
    evaluate("base")

    # Freeze everything except the tail. With ~8 examples per word, letting the
    # whole network move would overwrite the corpus features that make the
    # other 40 words work.
    for layer in model.layers[:-args.unfreeze]:
        layer.trainable = False
    trainable = sum(1 for l in model.layers if l.trainable)
    print(f"\nunfrozen layers: {trainable}/{len(model.layers)}")

    # Oversample the device captures so they are not drowned by the corpus,
    # but keep the corpus present so it anchors everything else.
    reps = max(1, int(round(len(ytr_c) / max(len(yd), 1) * args.reps)))
    Xmix = np.concatenate([Xtr_c] + [Xd] * reps)
    ymix = np.concatenate([ytr_c] + [yd] * reps)
    print(f"mix: {len(ytr_c)} corpus + {len(yd)}x{reps} device = {len(ymix)}")

    Xa, Ya = augment_batch(Xmix, ymix, n_classes, build, copies=1)
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-4),
                  loss=tf.keras.losses.CategoricalCrossentropy(
                      label_smoothing=0.1),
                  metrics=["accuracy"])
    model.fit(Xa, Ya, epochs=args.epochs, batch_size=32, verbose=2,
              callbacks=[tf.keras.callbacks.ReduceLROnPlateau(
                  monitor="loss", factor=0.5, patience=4, min_lr=1e-6)])

    print("\nafter adaptation")
    evaluate("adapted")

    out = os.path.join(HERE, f"asl_v3_{args.tag}_best.keras")
    model.save(out)
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    conv.target_spec.supported_types = [tf.float16]
    tfl = conv.convert()
    with open(os.path.join(HERE, f"asl_v3_{args.tag}.tflite"), "wb") as f:
        f.write(tfl)
    with open(os.path.join(HERE, f"asl_v3_{args.tag}_labels.txt"), "w",
              encoding="utf-8") as f:
        inv = {v: k for k, v in names.items()}
        for i in range(n_classes):
            f.write("%d,%s\n" % (i, inv.get(i, "")))
    print(f"\nTFLite: asl_v3_{args.tag}.tflite  ({len(tfl)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
