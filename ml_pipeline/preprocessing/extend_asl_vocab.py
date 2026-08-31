"""Add the grammatical core to the ASL vocabulary.

The 500-word selection ranked candidates by signer coverage, which favours
well-recorded content words and quietly drops function words. The result had
all six question words and 13/13 common verbs but no first-person pronoun at
all — so `YOU WANT WATER` was expressible and `ME WANT WATER` was not, which
makes sign chaining far less useful than it should be.

These words are in ASL Citizen; they simply were not picked. This appends them.

New classes are appended at indices 500+ rather than re-sorting the vocabulary.
Re-sorting would renumber every existing class and invalidate the already
extracted y labels for no benefit.
"""

import os
import re
import csv
import numpy as np
from collections import Counter, defaultdict
from multiprocessing import Pool

import extract_asl_citizen as base

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "../data/processed/asl_citizen")
MODELS = os.path.join(HERE, "../models")

# Grammatical scaffolding: pronouns, negation, conjunctions, modals, quantifiers.
WANTED = [
    "ME", "MY", "MINE", "MYSELF", "SELF", "HE", "HIS", "THEY",
    "NOT", "BUT", "BECAUSE", "IF", "WITH", "FOR", "TO", "OR",
    "ALL", "SOME", "MANY", "MUST", "SHOULD", "WILL",
    "NEVER", "ALWAYS", "SOMETIMES",
]


def main():
    rows = base.load_index()
    counts = Counter(r["gloss"] for r in rows)
    signers = defaultdict(set)
    for r in rows:
        signers[r["gloss"]].add(r["signer"])

    # Resolve each wanted concept to its best-recorded variant (THEY -> THEY1).
    variants = defaultdict(list)
    for g in counts:
        variants[re.sub(r"\d+$", "", g) or g].append(g)
    resolved = {}
    for w in WANTED:
        if w in variants:
            resolved[w] = max(variants[w],
                              key=lambda g: (counts[g], len(signers[g])))

    existing = [l.strip().split(",", 1)[1]
                for l in open(os.path.join(MODELS, "asl_500_labels.txt"),
                              encoding="utf-8") if "," in l]
    new = [w for w in WANTED if w in resolved and w not in existing]
    start = len(existing)
    print(f"existing classes {start}   adding {len(new)}")

    jobs, idx_of = [], {}
    for i, w in enumerate(new):
        idx_of[w] = start + i
    for r in rows:
        for w in new:
            if r["gloss"] == resolved[w]:
                jobs.append((f"ASL_Citizen/videos/{r['video']}",
                             idx_of[w], r["split"]))
    print(f"clips to process: {len(jobs)}", flush=True)

    X, y, sp, done, kept = [], [], [], 0, 0
    with Pool(6) as pool:
        for res in pool.imap_unordered(base._job, jobs, chunksize=4):
            done += 1
            if res is not None:
                X.append(res[0]); y.append(res[1]); sp.append(res[2]); kept += 1
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)}  kept {kept}", flush=True)

    # Append to the existing tensors.
    oldX = np.load(os.path.join(OUT, "X.npy"))
    oldy = np.load(os.path.join(OUT, "y.npy"))
    oldsp = np.load(os.path.join(OUT, "split.npy"))
    X = np.concatenate([oldX, np.array(X, dtype=np.float32)])
    y = np.concatenate([oldy, np.array(y)])
    sp = np.concatenate([oldsp, np.array(sp)])
    np.save(os.path.join(OUT, "X.npy"), X)
    np.save(os.path.join(OUT, "y.npy"), y)
    np.save(os.path.join(OUT, "split.npy"), sp)

    labels = existing + new
    with open(os.path.join(OUT, "labels.txt"), "w", encoding="utf-8") as f:
        for i, w in enumerate(labels):
            f.write(f"{i},{w}\n")

    print(f"\nkept {kept}/{len(jobs)}")
    print(f"X {X.shape}   classes {len(labels)}")
    print("added:", " ".join(new))
    print("next: retrain")


if __name__ == "__main__":
    main()
