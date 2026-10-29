# Prompt 2 — retarget and validate (send after the inspection)

Send the 5 files in `avatar/mocap/` with this, plus the corrected
`MOCAP_FORMAT.md`. Two new pilot signs were added: `052_FATHER.json` and
`053_MOTHER.json`. They exist specifically to catch a failure the other three
cannot — see §Validation.

Copy everything below the line.

---

Thank you for the inspection — both findings documents were exactly what was
needed, and your axis correction was right. I verified it independently against
the data before propagating it. Here are answers to your two open questions,
the scope, and what I want built.

## Your question B — resolved: the footage is NOT mirrored, and the signer is right-handed

Two independent lines of evidence:

1. **From a working system.** This mocap comes from the same clips that train
   the app's live recogniser. That recogniser matches this data with the phone
   camera feed **unmirrored** while the user signs **right-handed**. A mirrored
   dataset would not work that way. Handedness convention was the single
   largest bug in this project, so this path is very well tested.

2. **From the data.** Pose landmark 16 (anatomically right wrist) is the active
   one — visibility 0.98, 0.74 m of travel — against 0.42 / 0.68 m of jitter on
   landmark 15. Consistent with your §4.

**Drive the rig's RIGHT arm from the tracked hand.** Your single-still check
will confirm it before anything is baked.

## Your question A — hybrid, and here is the measurement that decides it

Rotation-based is right for handshape and I agree it makes your §5 scale
mismatch irrelevant there. But a rotation-only retarget will destroy part of
this vocabulary, and I can show it.

FATHER and MOTHER use the **same handshape** and differ only in height relative
to the face. From the classifier: zeroing the pose block drops FATHER to **0%**
and MOTHER to **15%** accuracy. Hand placement relative to the head is
semantically load-bearing, not decorative.

Measured wrist height relative to the nose, in shoulder-widths:

| sign | peak | mean |
|---|---|---|
| FATHER | **+0.03** (at nose level) | -0.21 |
| MOTHER | **-0.65** (well below) | -0.78 |

Rotation-only reproduces joint *angles*. On a rig at ~0.75-0.80 of the
performer's proportions, identical angles place the hand at a different height
relative to the head, and these two signs collapse into each other.

**So:**

- **Fingers and wrist orientation — rotation-based.** Derive from the
  `*_hand_world` landmark vectors. Scale-invariant, and the 21-point hand plane
  recovers palm roll, so your scale mismatch does not matter here.
- **Arm (Shoulder / Arm / ForeArm) — position or IK.** Express the hand target
  **relative to a body landmark** (nose, or the shoulder midpoint) rather than
  in absolute space, and scale by the ratios you measured: ~0.75 on shoulder
  width, ~0.80 on hip-to-shoulder. That preserves semantic placement without
  overshooting the rig's reach.

If you see a cleaner way to preserve face-relative hand height, take it — that
property is the requirement, not the specific method.

## Scope — please allocate no effort to these

Per your §4, which matches our own measurements:

- **Legs**: leave `UpLeg`, `Leg`, `Foot`, `ToeBase` at rest, both sides. Knee
  and ankle visibility is 0.00-0.03; those coordinates are pure extrapolation.
- **Hips**: treat as static, pinned to the floor. Visibility 0.50-0.66 is not
  enough to track translation.
- **Left arm**: leave at rest or heavily smoothed. Your jitter diagnosis is
  correct — longer travel path than the right within an 0.08 m bounding box is
  noise, not motion. These are one-handed signs.
- **Two-handed logic**: not needed for this pilot. About a third of the full
  105-word vocabulary is genuinely two-handed, so we will revisit it later, but
  none of these five are.

## Things that will bite, from your own findings

- **The unapplied object transform** (90° X rotation, 0.01 scale) — your §2.
  You flagged this as the most likely way a generated script fails while
  looking almost-right. Please convert through `armature.matrix_world` and the
  bone rest matrix rather than applying the transform.
- **Bone naming** — index by string only, `arm.pose.bones["mixamorig:LeftArm"]`.
  Colon is not a valid identifier character, and Left/Right are prefixes, so
  Blender's symmetry helpers will not pair them.
- **Null hand frames** — right-hand tracking has small gaps (2 / 1 / 2 frames
  in the original three). Interpolate across nulls rather than dropping frames;
  the frame count is the sign's real duration and should be preserved.
- **Corrected axis mapping**, now verified in `MOCAP_FORMAT.md`:

      blender_x =  mp_x
      blender_y =  mp_z
      blender_z = -mp_y

## Validation — do this before baking anything

Your suggestion of a still frame first is right, and I want one specific
comparison made the acceptance test:

1. Render **one still at peak sign pose for FATHER, and one for MOTHER**, side
   by side.
2. **They must be visibly different**, with FATHER's hand near forehead/nose
   height and MOTHER's clearly lower, near the chin. If they look the same, the
   arm retarget is not preserving face-relative placement and no amount of
   finger detail will fix it.
3. In the same stills, confirm the character is upright (axis mapping), and
   signing with its **right** hand (handedness).

Only once those pass, do the remaining three signs, then bake.

## Output format — MP4 per sign, not a .glb

One MP4 per sign, named by class id and label to match the JSON, e.g.
`000_GOOD_MORNING.mp4`.

Rationale: fixing one word must not require re-exporting and re-verifying all
the others. A per-file output means a bad sign is a 30-second re-render with
zero risk to signs already approved, and the app plays a video rather than
carrying a 3D runtime on a mid-range phone.

Suggested settings — adjust if you have better judgement:

- 480x480 or 512x512, square, character framed head to waist
- 30 fps, matching the mocap `fps` field
- H.264 MP4, no audio
- Plain flat background, strong contrast against the character
- Front-facing camera, fixed for every sign so they are comparable
- Eevee rather than Cycles — these need to be quick to re-render

## Deliverables for this round

1. FATHER and MOTHER stills, side by side (the acceptance test above)
2. Once approved: the 5 pilot signs as MP4
3. The retarget script itself, so the remaining 100 signs can be batched

Please flag anything in the above you think is wrong. The previous round's
corrections were valuable and I would rather be corrected than have you work
around a bad instruction.
