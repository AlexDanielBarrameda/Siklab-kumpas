# Prompt 3 — approve the pilot, then one more pilot before the batch

Send with the two new files: `avatar/mocap/007_THANK_YOU.json` and
`avatar/mocap/054_SON.json`, plus the updated `MOCAP_FORMAT.md`.

Copy everything below the line.

---

Approved — render the five MP4s. The acceptance test passes and the work is
excellent. Three responses, then one more pilot I want before the batch of 100.

## Your correction to my scaling prescription was right, and mine was wrong

I told you to scale by "~0.75 shoulder width, ~0.80 hip-to-shoulder". Your
measurement shows limb proportions match within 3% and the mismatch is entirely
the head — nose-above-shoulders at 0.457 shoulder-widths on the performer versus
1.053 on the rig. My instruction would have put FATHER 15 cm below the nose and
failed my own acceptance criterion.

The piecewise-linear map through four anatomical anchors is a better solution
than the one I specified, and treating vertical and horizontal differently
because only one carries meaning is exactly right. Keep it.

## Your finding 1 — MOTHER: checked, and it is faithful. Do not "fix" it.

You were right to flag it and right not to assume. I measured the peak thumb
height against the performer's own mouth across six MOTHER takes, in
shoulder-widths, positive meaning above the mouth:

| take | MOTHER | FATHER |
|---|---|---|
| the clip we sent | -0.51 | +0.17 |
| second | -0.47 | +0.18 |
| four more | -0.58 to -0.63 | +0.02 to +0.07 |
| **median** | **-0.59** | **+0.07** |

Every take places the hand below the mouth, and the one we sent is the *highest*
of the six. This is how this signer performs the sign, not a bad exemplar.

**Reproduce the dataset faithfully rather than correcting toward canonical FSL.**
The recogniser in the app is trained on these same clips, so avatar and
recogniser must agree — an avatar that touches the chin would be teaching users
a gesture the recogniser would then reject. The divergence from canonical FSL is
a dataset property, and we will document it as a limitation.

Your instinct to check before it became ground truth for 100 signs was the right
call and it is the reason this got resolved.

## Your findings 2, 3, 4 — all accepted

**2. Wrist orientation from the pose block, not the hand block.** Correct, and it
was a real gap in my spec — `MOCAP_FORMAT.md` now documents that the hand blocks
live in their own canonical frame and must only be used for finger bending.
Carried into this prompt as you asked.

**3. Reach clamping.** Understood. See the two-handed pilot below — I measured
the reach demands ahead of time so you know what is coming.

**4. Left arm held at side rather than T-pose.** Agreed, and better than what I
specified. For the two-handed signs it becomes driven anyway.

## One more pilot before the 100: two-handed signs

The FATHER/MOTHER round taught us that a pilot which misses a whole class of
signs can pass while something important is broken. The five pilot signs are all
one-handed, and **about a third of the 105-word vocabulary is genuinely
two-handed**. Batching now would repeat that mistake at 100x the cost.

Two files attached:

| sign | frames | both hands | max wrist separation | max right-arm reach |
|---|---|---|---|---|
| `007_THANK_YOU` | 38 | **100%** | 0.86 SW | 1.14 SW |
| `054_SON` | 46 | 45% | 1.52 SW | 1.22 SW |

THANK YOU is two-handed throughout — the clean test. SON alternates between one
and two hands mid-sign, which is the harder case: the left arm has to transition
between driven and idle without snapping.

Reach figures are in shoulder-widths for comparison against the rig's 1.515
total arm. Given your clamping already engages on one-handed face-height signs,
these will likely push it further.

What I would like:

1. **Extend the retarget to drive the left arm** when `left_hand_world` is
   present, using the same vertical anchor map.
2. **Handle the transition** in SON — when a hand's data starts or stops
   mid-clip, blend between driven and idle rather than cutting.
3. **Stills for both**, then MP4 if they look right.
4. Tell me if reach clamping distorts either sign. If two-handed signs need the
   hands to meet in front of the body and the arms cannot reach, I would rather
   know now and consider whether the vertical anchoring should relax slightly
   for signs with no face contact.

## After that

Say the word and I will send all 105 mocap files. The full extraction is running
now and will be ready before you need it. Batch is `build()` + `render_all()`
with the filenames added to `CLIPS`, per your notes.

As before: push back on anything above you think is wrong. Two rounds, two
corrections from you that mattered — that is working.
