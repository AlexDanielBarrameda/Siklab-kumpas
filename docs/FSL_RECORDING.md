# Recording a second FSL signer

## Why this is the one thing that fixes FSL

FSL-105 has a single signer. `train_fsl105_v2_aug.py` splits it with
`train_test_split(X, y, stratify=y)` — a random split, so the same person
appears in both training and test. Its 98.54% therefore means *"recognises this
specific person repeating signs it has already watched them perform."* It is not
evidence the model works for anybody else, and the tester session confirmed it
does not: predictions collapsed onto BREAD, which is the signature of input
falling outside the trained distribution entirely.

No amount of pipeline work changes this. The model has never seen a human other
than the one in those clips. Adding a second signer is what converts "should
generalise" into a number that can be defended — and, for the first time, makes
cross-signer accuracy on FSL *measurable at all*.

## Scope

- **105 words**, the existing FSL-105 vocabulary (`flutter_app/assets/labels/fsl105_labels.txt`).
- **3 takes per word** → 315 clips. Two takes is workable if time is short; the
  third mainly buys robustness.
- Roughly 2–3 hours including setup and breaks.

This is you recording yourself. It is not research on deaf participants, so it
is separate from the ethics approval that gates participant testing.

## Camera setup

The point is to match how the app is actually used, **not** to match the
original FSL-105 videos. The device letterboxes the sensor frame at scale
0.5625, so the signer occupies about 56% of the canvas width — film the way you
will demo.

| | |
|---|---|
| Orientation | **Landscape**, matching FSL-105's 640×360 source |
| Distance | Head and both hands in frame with room to spare; roughly arm's length plus a step back |
| Height | Camera at chest-to-eye level, not looking up from a desk |
| Background | Plain and uncluttered; avoid skin-toned walls — the palm detector leans on chroma |
| Lighting | Front-lit. Never backlit or with a window behind you |
| Clothing | Contrasting with skin and background; short or rolled sleeves |
| Frame rate | 30fps is plenty |

**Vary things deliberately between takes.** Take 1 at your normal distance,
take 2 a step closer, take 3 a step back or slightly off-centre. Variation is
the entire point — a second signer filmed identically 315 times teaches the
model far less than one filmed across a realistic range.

## Signing

- Start with hands **down at rest**, sign once, return hands to rest. The app
  segments on hands entering and leaving frame, so rest-to-rest matches how it
  will actually see you.
- Leave about a second of rest at each end.
- Sign at **natural speed**. Do not slow down to help the model — that produces
  training data that does not match the demo.
- If you fluff one, pause, return to rest, and sign it again. The extractor
  takes the whole clip; a clean retake at the end is fine.

## File naming

One file per take, named for the word so the extractor can label it without a
spreadsheet:

```
fsl_recordings/
  GOOD_MORNING_1.mp4
  GOOD_MORNING_2.mp4
  GOOD_MORNING_3.mp4
  THANK_YOU_1.mp4
  ...
```

Use the label text from `fsl105_labels.txt` with spaces replaced by underscores,
then `_1` / `_2` / `_3`. Case is ignored. Anything the extractor cannot match to
a label is reported and skipped rather than silently mislabelled.

## Extracting

```
python ml_pipeline/preprocessing/extract_fsl_signer2.py \
    --videos path/to/fsl_recordings \
    --out    signer2
```

It writes `X.npy`, `y.npy` and `signer.npy` beside the existing FSL-105 tensors,
using the *same* extraction path as `extract_fsl105_v2.py` so the features are
directly comparable.

## What we get that we cannot get today

With two signers, the split becomes **signer-disjoint**: train on FSL-105's
signer, test on yours. That produces the first honest cross-signer number for
FSL — directly comparable to ASL's 67.7% — instead of a same-signer number that
flatters the model.

Then train on both and test each way. If cross-signer accuracy is poor even
with body normalisation, that is a real finding worth reporting, and it argues
for scoping FSL as a demonstrated prototype. If it holds up, FSL becomes a
second working language.

Either outcome is defensible. The current situation — quoting 98.54% for a model
that fails on the first person who tries it — is not.
