# Avatar — what belongs in THIS workspace

The 3D work (rigging, retargeting, rendering) happens in a separate Blender
agent. No .fbx, .blend or model file lives here.

    avatar/
      mocap/        landmark data extracted from the FSL-105 clips.
                    PRODUCED HERE, carried out to the Blender agent.
      MOCAP_FORMAT.md   the contract both sides follow.
      prompts/      text to paste into the Blender agent's chat.

    flutter_app/assets/avatar/
                    finished animations coming BACK from the Blender agent,
                    for the app to display.

## Why mocap is produced here and not there

The FSL-105 videos and the MediaPipe pipeline live in this repo
(ml_pipeline/). Extracting landmarks needs both. The Blender agent should not
need the video dataset at all — it receives JSON.

## The handoff

    1. here      extract mocap/*.json from the clips
    2. -> out    carry those files to the Blender agent
    3. there     inspect model, rig, retarget from the JSON, render/export
    4. <- back   drop the result into flutter_app/assets/avatar/
    5. here      wire it into the app
