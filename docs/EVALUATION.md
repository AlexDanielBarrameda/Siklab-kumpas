# Evaluation: recognition accuracy at the operating point

## Why the usual number is the wrong number

Sign recognition results are conventionally reported on held-out clips: each
video is sampled to a fixed number of frames — 30 here — and the model is scored
on how many it labels correctly. That figure describes the model. It does not
describe the application, because the application never receives that input.

Two differences separate the benchmark from the deployed system, and both were
measured directly on the device rather than assumed.

**Capture rate.** The phone completes roughly five landmark extractions per
sign, not thirty. The app collects whatever frames it manages between the hands
entering and leaving the frame, then resamples that run to 30 before inference.
A benchmark clip sampled at 30 distinct frames is therefore six times denser
than anything the model sees in use.

**Framing.** The 226-dimensional feature vector is 63 left-hand + 63 right-hand
+ 100 pose values. The wrist normalisation applied by every extractor touches
indices 0–126 only, so the 100 pose dimensions remain *absolute image
coordinates*. Where the signer sits in the frame is therefore a model input.
`MainActivity` rotates the sensor frame upright to portrait and letterboxes it
into the 480×360 canvas at scale 0.5625, leaving the signer occupying 56% of the
width, while ASL Citizen's clips are native landscape and fill the frame.
Measured median shoulder width is 0.410 in ASL Citizen against 0.222 in FSL-105.

The consequence is measurable: substituting a plausible-but-uninformative pose
block (the corpus mean, or the clip's own first frame held static) costs 34
accuracy points. Pose carries roughly half the model's discriminative power, and
it was the half most exposed to the framing difference.

**This also explains an asymmetry that had no other explanation.** The same
pipeline served both languages, yet FSL words worked acceptably on device while
ASL words did not. FSL-105's signer stands far enough back to occupy about half
the apparent width of an ASL Citizen signer, which coincidentally places FSL in
the same regime the phone's letterbox produces. Device signing posture, measured
as `(shoulder_y − wrist_y) / shoulder_width`, is −1.14; that falls inside
FSL-105's range (p5 = −1.18) and outside ASL Citizen's (p5 = −0.78). The ASL
model was being asked to classify frames from outside its training distribution
on essentially every attempt.

## Accuracy as a function of capture rate

ASL model, 525 classes, ASL Citizen signer-disjoint test split (6,378 clips;
the 11 test signers appear in no training clip).

| distinct frames | top-1 | top-5 |
|---|---|---|
| 3 | 38.71% | 65.49% |
| 4 | 54.47% | 79.32% |
| 5 | 60.98% | 84.09% |
| 6 | 63.50% | 85.43% |
| 8 | 65.21% | 87.05% |
| 12 | 66.73% | 87.99% |
| 20 | 67.04% | 88.08% |
| 30 *(as published)* | 67.69% | 88.44% |

Capture rate costs about 7 points between the published figure and the operating
point. It is a real effect but a secondary one — which contradicted the initial
hypothesis that frame rate was the dominant cause of poor live performance.

## Accuracy under mis-segmentation

The app must infer where a sign starts and ends from a live stream; corpus clips
are pre-trimmed. Measured against the 67.69% clean reference:

| perturbation | top-1 |
|---|---|
| truncated to central 75% | 64.94% |
| truncated to central 50% | 58.44% |
| truncated to central 35% | 49.56% |
| 35% of window static padding | 63.09% |
| 50% of window static padding | 55.44% |
| 65% of window static padding | 43.51% |
| shifted ±3 frames | ~65% |
| shifted ±6 frames | ~58% |

Also secondary: 9–24 points depending on severity.

## The compounded operating point

Applying the measured device conditions together, and then the app's 0.6
confidence threshold — below which nothing is displayed, so the user sees
silence:

| condition | top-1 | user-visible |
|---|---|---|
| as published (corpus framing, 30 frames) | 67.69% | 59.02% |
| + 5 distinct frames | 60.98% | 51.13% |
| + device framing | 50.80% | — |
| + imperfect segmentation | 44.51% | 35.50% |

"User-visible" is the fraction both correct *and* above the confidence gate. It
is the honest figure for what the application delivers, and it is the one this
chapter reports.

## Interventions

Each was measured under the device condition (device framing, 5 distinct
frames), not at the published operating point.

| model | top-1 | user-visible |
|---|---|---|
| `asl_525` — original | 48.06% | 39.35% |
| `asl_525cal` — recalibrated augmentation | 43.46% | 33.43% |
| `asl_525frame` — + pose-framing augmentation | 59.86% | 48.12% |
| `asl_core200` — + 200-word vocabulary | **61.95%** | **53.06%** |

**Augmentation recalibration (`525cal`) was a regression.** The original
augmentation inherited FSL's scale range of 0.8–3.0×, tuned for a corpus with
median hand span 0.102; applied to ASL, whose median is 0.218, it generated
hands with spans up to 1.76 — wider than the frame. Correcting it to a plausible
0.7–1.4× *cost* 4.6 points. A later controlled comparison showed why: with pose
framing augmented explicitly, wide and narrow scale ranges tie (59.86% vs
59.13%). The wide range had been partially compensating for the framing gap, not
providing scale robustness. Measuring one augmentation parameter while another
is miscalibrated attributes the effect to the wrong variable.

**Pose-framing augmentation (`525frame`) was the substantive fix**, worth 11.8
points. Scaling the pose block about the canvas centre across 0.50–1.05 during
training covers the framing range the device produces. The resulting model is
nearly framing-invariant: 67.18% at corpus framing against 66.68% at device
framing, where the original lost 12.4 points across the same gap.

**Vocabulary reduction (`core200`)** adds 2.1 points top-1 and 4.9
user-visible, and lifts top-5 to 87.29%. The 200 words are selected for
communicative need rather than corpus frequency, and retain the full grammatical
core — pronouns, negation, conjunctions, modals — so signs can still be chained.

## Rejected: the lite pose model

Swapping `pose_landmarker_full` for `pose_landmarker_lite` was evaluated as a
route to a higher capture rate and rejected on measurement. On 147 paired clips
extracted with both models:

| pose model | top-1 |
|---|---|
| `full` (matches training) | 74.15% |
| `lite` | 68.71% |

The 5.4-point cost arises because both extractors built the training features
with `full`; the pose block is unnormalised, so a different pose model shifts
those 100 dimensions out of the trained distribution. Landmark agreement is
bimodal — median absolute difference 0.0151 but p90 of 0.3595 — so most points
agree while a minority diverge substantially. Detection reliability was not the
issue: `lite` found a pose in every frame `full` did (0 of 4,410 missing).

Recovering 5.4 points through frame rate would require moving from 5 to about 12
distinct frames, a 2.4× speedup. Pose cannot deliver it: pose already runs
concurrently with hand detection, so the per-frame budget is `max(hand, pose)`
rather than their sum, and accelerating a component that is not the bottleneck
yields approximately nothing.

`RunningMode.VIDEO` was adopted for pose instead — it tracks between frames
rather than re-detecting, at identical model weights, so the feature
distribution is unchanged. It was deliberately not applied to the hand
landmarker, which alternates between the full canvas and a hand crop; VIDEO mode
would carry tracking state across two different coordinate spaces.

## Threat to validity: the residual gap

Simulating every identified device condition predicts a user-visible rate near
35% for the original model. Field observation was substantially worse — signs
frequently not recognised at all. Roughly 35 points therefore remain unaccounted
for by any factor reproducible from the corpus.

The following were each tested and eliminated as explanations:

- **Zero-hand frames.** Training keeps only hand-present frames; the app
  likewise discards frames with no detected hand, so the two agree.
- **Canonicalisation mismatch.** The two-handed ordering compares
  `lh[0] > rh[0]`, which is translation-invariant and therefore returns the same
  result on raw device coordinates as on stored normalised vectors. The Dart and
  Python implementations agree.
- **Pose detection failure.** Training data contains a pose in 99.99% of frames;
  the lite comparison confirms pose detection is not fragile.
- **Confidence calibration.** The gate costs about 9 points, not 35.

Resolving the remainder requires comparing a tensor built on the device against
the corpus distribution directly. A debug-only path writing each classified
`[30][226]` tensor to storage has been implemented for this purpose
(`compare_device_tensor.py`), and dumps are captured *before* the confidence
gate, since attempts producing no output are the ones of interest. **The
measurements in this chapter are offline; on-device validation of the improved
models is outstanding.**

## Reproducing

```
python ml_pipeline/eval/frame_rate_curve.py   --model ml_pipeline/models/asl_core200_best.keras
python ml_pipeline/eval/segmentation_probe.py --model ml_pipeline/models/asl_core200_best.keras
python ml_pipeline/eval/pose_domain_probe.py  --model ml_pipeline/models/asl_core200_best.keras
python ml_pipeline/eval/device_condition.py   ml_pipeline/models/asl_core200_best.keras
python ml_pipeline/eval/pose_model_swap.py --n 150 --model ml_pipeline/models/asl_core200_best.keras
```

All figures use the ASL Citizen signer-disjoint test split. FSL-105 has a single
signer, so its held-out accuracy measures recognition of one person and is not
comparable to any ASL figure in this chapter.
