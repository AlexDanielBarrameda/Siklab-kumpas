# mocap/*.json format

Contract between the Python extractor (this repo) and the Blender retarget
script. One file per sign.

```json
{
  "label": "GOOD MORNING",
  "class_id": 0,
  "source_clip": "clips/0/xxx.MOV",
  "fps": 30,
  "frame_count": 60,
  "hand_coverage": 1.0,
  "frames": [
    {
      "pose_world":  [[x, y, z, visibility], ...  33 entries],
      "left_hand_world":  [[x, y, z], ... 21 entries]  or null,
      "right_hand_world": [[x, y, z], ... 21 entries]  or null,
      "pose_image": [[x, y, z, visibility], ... 33 entries]
    }
  ]
}
```

## Which coordinates, and why it matters

MediaPipe returns two sets, and for animation the choice is decisive:

- **`*_world` landmarks are in METRES, origin at the hip midpoint** (hands:
  origin at the wrist). These are real 3D and are what a rig should be driven
  from.
- **image landmarks are normalised 0..1 to the frame**, with z a weak relative
  depth. These are what the CLASSIFIER uses. They are the wrong thing to
  animate from — a rig driven by them looks correct only head-on.

`pose_image` is included only for reference/debugging. Drive the rig from
`pose_world` and `*_hand_world`.

## hand_coverage

Fraction of frames with **at least one** hand detected — not both. These signs
are one-handed, so 1.0 here means every frame has the dominant hand, and says
nothing about the other. Individual `left_hand_world` / `right_hand_world`
entries are still null on frames where that hand was not found, and need
interpolating.

## The hand blocks are NOT in body space

`left_hand_world` / `right_hand_world` are expressed in the hand landmarker's
own canonical frame, centred on the hand. They are **not** aligned to the body
frame that `pose_world` lives in.

Consequence: do not derive absolute wrist orientation from them. It produces a
plausible-looking but wrong palm direction — an error that survives visual
inspection.

- **Wrist orientation** -> from POSE landmarks 16 (wrist), 18 (pinky) and 20
  (index), which are in body space and carry >=0.95 visibility on every clip
  measured.
- **Finger bending** -> from the hand block, which is fine because bending is
  frame-independent.

## Axis convention

MediaPipe world space is: +x to the subject's left, **+y DOWN**, **+z AWAY
from the camera**. Verified against 020_ONE.json frame 0 with the hip midpoint
at the origin:

    nose   y = -0.594    foot  y = +0.684     -> +y is down
    nose   z = -0.282    (subject faces camera) -> +z points away

Blender is +z up, and this rig faces -Y. The conversion is therefore:

    blender_x =  mp_x
    blender_y =  mp_z
    blender_z = -mp_y

Determinant +1, so no mirroring is introduced. Subject's left maps to +X,
matching the rig; up maps to +Z; toward-camera maps to -Y, which the rig
already faces, so no extra yaw is needed.

An earlier version of this file claimed +y up and +z toward camera. Both were
wrong, and `blender_z = mp_y` renders the character upside down.

## Landmark indices

Pose (33): 0 nose, 11/12 shoulders, 13/14 elbows, 15/16 wrists,
23/24 hips, 25/26 knees.

Hand (21): 0 wrist; 1-4 thumb (cmc, mcp, ip, tip); 5-8 index
(mcp, pip, dip, tip); 9-12 middle; 13-16 ring; 17-20 pinky.
