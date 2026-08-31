# ASL avatar clips — source pipeline

Raw avatar footage comes in here; app-ready clips go out to
`flutter_app/assets/avatar/clips/`. Nothing in `masters/` is ever bundled into
the app — those files are the editing masters and are far too large to ship.

```
avatar_pipeline/
  masters/          <- DROP YOUR QUICKTIME EXPORTS HERE (.mov, gitignored)
  build_clips.py    <- (not written yet) transcodes masters -> app clips
```

## What goes in `masters/`

One file per sign, named for the label it represents, spaces and punctuation
replaced by underscores:

```
masters/
  AFTERNOON.mov
  THANK_YOU.mp4
  BATHROOM.mov
  ...
```

`.mov` and `.mp4` are both fine **here** — masters are for editing, and the
build step normalises either one. If your source is one long recording covering
several signs, split it into per-sign files before dropping them in; the build
script takes each file's name as its label and has no way to find cut points.

The label text comes from `flutter_app/assets/labels/asl_core200_labels.txt`.
Case is ignored by the build script; anything that cannot be matched to a label
is reported and skipped rather than silently shipped under the wrong word.

An MP4 master is **not** already app-ready. Capture tools record variable frame
rate, an audio track, and sometimes pixel formats Android cannot decode. See
`docs/ASL_AVATAR.md` for what to check and for the transcode recipe that turns
these into the clips the app actually plays.
