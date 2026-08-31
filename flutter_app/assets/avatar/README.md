# Avatar assets

`clips/` holds the app-ready ASL avatar clips — **MP4 only** (H.264 / yuv420p /
30fps CFR / no audio track). These are build output, not source. The editing
masters live in `avatar_pipeline/masters/` and are never bundled.

Do not hand-drop `.mov`, `.webm`, or `.gif` files in here. Android's ExoPlayer
will not reliably decode them and the failure shows up as a black card on the
device rather than as a build error.

See `docs/ASL_AVATAR.md` for the transcode recipe and the naming convention.
