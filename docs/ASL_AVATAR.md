# ASL avatar clips in the Inclusive Communicator

## Is QuickTime the right format?

**As an editing master, yes. As the file the app ships and plays, no.**

Keep both. They are different jobs and the same file cannot do them well.

### Why `.mov` fails in the app

Flutter's `video_player` is a thin wrapper over **ExoPlayer** on Android and
**AVPlayer** on iOS. AVPlayer handles `.mov` natively — it is Apple's own
container — so a `.mov` avatar will look perfectly fine if you ever test on a
Mac or an iPhone. That is the trap. On the Android device you will actually
demo on, two separate things break:

1. **The container.** QuickTime is not on Android's guaranteed-support list.
   MOV and MP4 share an ISO-BMFF lineage, so some devices' extractors parse it
   anyway — and others don't. You get a clip that plays on one phone and shows
   a black card on the next, with no error to catch in code.
2. **The codec inside it, which is the bigger problem.** QuickTime exports out
   of Final Cut, Premiere, After Effects, or a 3D renderer default to **ProRes**
   or **Animation**, usually at `yuv422p10le` or with an alpha channel. Android
   has no hardware decoder for any of those. Even where the container parses,
   the video track will not decode.

A black avatar card during the defense demo, on a format issue, is not a risk
worth carrying.

### What to export instead

| | |
|---|---|
| Container | **MP4** (`.mp4`) |
| Video codec | **H.264 / AVC**, Main profile, Level 4.0 |
| Pixel format | **`yuv420p`** — the single most important setting |
| Frame rate | **30 fps constant** (CFR, not variable) |
| Resolution | **720 × 1024** (see *Framing* below) |
| Audio | **none** — strip it entirely |
| Keyframes | every 15 frames |
| Layout | `faststart` (moov atom at the front) |

Three of those deserve a word:

- **`yuv420p`** is what makes a ProRes master playable. Without it, ffmpeg
  happily preserves the 4:2:2 chroma of the source and Android silently refuses
  the track.
- **Constant frame rate.** Screen recordings and some renderer exports are VFR.
  VFR clips seek badly and stutter when queued back-to-back, which is exactly
  what sentence playback does.
- **No audio.** The Text tab speaks TTS *while* the avatar signs. An avatar
  clip with its own audio track will talk over it. `-an` also cuts file size.

### Framing — settled: crop 1440 × 2048, deliver 720 × 1024

The masters are 1440 × 2560 (9:16) full-body renders. That is more frame than
the sign needs: the bottom fifth is legs and a tool watermark, and keeping it
shrinks the part that carries meaning on an already small card.

The crop keeps the **full width** — arms swing wide on WHERE and BATHROOM — and
the **top 2048 rows**. That bound is set by the hands at rest, which reach about
row 1870 in IDLE; 2048 clears them with margin. Cropping to a square 1440 × 1440
would cut the resting hands off, which is why the obvious "just make it square"
answer is wrong here.

The delivered 720 × 1024 is a 0.703 aspect. The avatar card is built to exactly
that ratio, so nothing letterboxes and nothing is cropped a second time at
display.

**The crop also removes the `MASCOZ / @shinybox68` watermark.** That was not the
goal, but it is the effect, so it needs a decision rather than silence: check
what the avatar tool's licence requires, and credit it in the thesis regardless.
Cropping for framing and citing the source is defensible; quietly shipping the
asset with the attribution removed is not.

---

## The source is already MP4 — does it still need transcoding?

Almost certainly yes. **MP4 is a container, not a guarantee**, and a VTuber
capture is one of the likelier ways to end up with an MP4 that Android chokes
on or that sequences badly:

| Risk | Where it comes from | Symptom in the app |
|---|---|---|
| **Variable frame rate** | OBS and most capture tools record VFR by default | Stutter and mistimed seeks when clips play back-to-back |
| **Audio track** | Capture tools record desktop/mic audio unless told not to | Avatar talks over the Text tab's TTS |
| **`yuv444p` / `NV12` / 10-bit** | Some renderer and capture presets | Track fails to decode on real hardware |
| **`moov` atom at the end** | Anything not exported with `faststart` | Visible delay before the first frame of every clip |
| **Long single recording** | One take covering many signs | Needs splitting into one file per sign before anything else |

So the transcode step does not go away — it changes from *conversion* to
*normalisation*, and the ffmpeg recipe below is the same either way. Run it even
on files that are already MP4.

**Do this before cutting the rest:** drop one representative file into
`avatar_pipeline/masters/` and probe it. That turns the table above from a list
of things to worry about into a short list of things actually wrong:

```bash
ffprobe -v error -show_entries stream=codec_type,codec_name,profile,pix_fmt,width,height,r_frame_rate,avg_frame_rate \
  -show_entries format=duration,size -of default=nw=1 avatar_pipeline/masters/YOUR_FILE.mp4
```

If `r_frame_rate` and `avg_frame_rate` disagree, it is VFR.

### Background and alpha

The one thing specific to a VTuber-style avatar: these are usually captured
over a **green screen or transparent background**, and **H.264 in MP4 cannot
carry an alpha channel**. There is no format you can export to that gives you a
transparent video and still plays reliably on Android.

Bake the background in at edit time, on a flat colour that suits the card.
Chroma-keying live in Flutter is possible with a fragment shader but is a
disproportionate amount of work and a real performance cost on mid-range
phones — not worth it here.

If the background is baked flat, the current gradient on the avatar card
(`c.avatarCard`) will fight it. Simplest fix is to let the video fill the card
and drop the gradient behind it; that is a one-line change in step 5.

---

## Folder layout

Created and ready for you:

```
avatar_pipeline/
  masters/                        <- your QuickTime .mov exports go HERE
  README.md

flutter_app/assets/avatar/
  clips/                          <- build output: the .mp4s the app ships
  README.md
```

`masters/` is the drop zone. `clips/` is generated — treat anything in it as
disposable and rebuildable from the masters.

### Naming

One file per sign, label text with spaces and punctuation collapsed to
underscores. Source of truth is
`flutter_app/assets/labels/asl_core200_labels.txt`.

```
AFTERNOON.mov      ->  AFTERNOON.mp4
THANK YOU          ->  THANK_YOU.mov      ->  THANK_YOU.mp4
DON'T FEEL LIKE    ->  DONT_FEEL_LIKE.mov ->  DONT_FEEL_LIKE.mp4
CALL (TTY)         ->  CALL_TTY.mov       ->  CALL_TTY.mp4
```

This matches the convention already documented in `FSL_RECORDING.md`. Note it
differs from `core200_videos/`, which uses `THANKYOU.mp4` with no separator —
those are ASL Citizen reference clips for *you* to study, not app assets, and
they are not affected.

### Git

`masters/` must be gitignored — ProRes runs into gigabytes. Add to `.gitignore`:

```
avatar_pipeline/masters/
```

The built `clips/` are ~200–400 KB each. For 200 words that is 40–80 MB, which
is committable but heavy for a normal repo. Either commit them (fine for a
thesis; keeps the build reproducible from a fresh clone) or add Git LFS. Decide
once, early — moving 200 binaries into LFS later rewrites history.

---

## Transcoding

ffmpeg 8.0 is already on your machine, so no new tooling is needed.

Single clip, to check the settings look right before committing to a batch:

```bash
ffmpeg -y -i avatar_pipeline/masters/AFTERNOON.mov \
  -an \
  -c:v libx264 -profile:v main -level 4.0 -pix_fmt yuv420p \
  -vf "fps=30,scale=720:720:force_original_aspect_ratio=increase,crop=720:720" \
  -crf 23 -preset slow \
  -g 15 -keyint_min 15 -sc_threshold 0 \
  -movflags +faststart \
  flutter_app/assets/avatar/clips/AFTERNOON.mp4
```

Whole folder (Git Bash). Handles `.mov` and `.mp4` masters alike — remember the
normalisation applies to both:

```bash
for f in avatar_pipeline/masters/*.mov avatar_pipeline/masters/*.mp4; do
  [ -e "$f" ] || continue
  n=$(basename "${f%.*}")
  ffmpeg -y -i "$f" -an \
    -c:v libx264 -profile:v main -level 4.0 -pix_fmt yuv420p \
    -vf "fps=30,scale=720:720:force_original_aspect_ratio=increase,crop=720:720" \
    -crf 23 -preset slow -g 15 -keyint_min 15 -sc_threshold 0 \
    -movflags +faststart \
    "flutter_app/assets/avatar/clips/$n.mp4"
done
```

Verify one before trusting the batch — `pix_fmt` and the absence of an audio
stream are the two things worth eyeballing:

```bash
ffprobe -v error -show_entries stream=codec_type,codec_name,profile,pix_fmt,width,height,r_frame_rate \
  -of default=nw=1 flutter_app/assets/avatar/clips/AFTERNOON.mp4
```

Expect exactly one stream, `codec_type=video`, `pix_fmt=yuv420p`,
`r_frame_rate=30/1`. If a second `codec_type=audio` stream appears, `-an` was
dropped somewhere.

Note the output filename is derived from the master's basename, so masters must
already be named for their sign before this runs. If your source is one long
recording covering several signs, split it first — `ffmpeg -ss <start> -to <end>
-i long.mp4 -c copy masters/WORD.mp4` is enough for a rough pass, and the
transcode above cleans up the cut points.

This loop is worth promoting to `avatar_pipeline/build_clips.py` once the
settings are locked, so it can also validate names against the label file and
report which words have no clip yet. That script is not written — it is step 2
of the plan below.

---

## Editing guidance

Same discipline as `FSL_RECORDING.md`, for the same reason — the avatar is a
teaching artefact and its readability is what gets graded.

- **Rest-to-rest.** Start hands at rest, sign once, return to rest. Trim to
  about 3 frames of rest at each end — enough to read as a boundary, short
  enough that a five-word sentence does not take fifteen seconds.
- **Trim tight.** Aim for 1.0–2.0 s per clip. The ASL Citizen references
  average ~1.5 s; that is a good target.
- **Uniform framing across all clips.** The avatar should not jump position
  between words in a sentence. This is the most visible quality problem in
  clip-sequenced sign output, and it is fixed in editing, not in code.
- **Consistent background and lighting**, for the same reason.
- **Natural signing speed.** Playback rate can be slowed in-app later if
  testers ask for it; a slow master cannot be sped back up cleanly.

---

## What is built

Working on device (Redmi Turbo 4 Pro, `e4720831`), verified end to end.

| Piece | Where |
|---|---|
| Transcoder + manifest | `avatar_pipeline/build_clips.py` |
| Shipped clips | `flutter_app/assets/avatar/clips/` (13 files, 1.4 MB) |
| Lexicon + glosser | `lib/services/avatar_service.dart` |
| Queue player | `lib/widgets/avatar_player.dart` |
| Wiring | `lib/screens/hearing_user_screen.dart`, Voice and Text tabs |

**Vocabulary** — 112 signs, 26 fingerspelling letters, plus IDLE. 10.5 MB
built. 112 of the 200 CORE200 words are covered, so the avatar and the
recogniser share one vocabulary.

Letters live in `masters/FSL LETTERS/` and are built into a `LETTER_*`
namespace. That prefix is not cosmetic: a clip called `A.mp4` and the article
"a" are the same key, and the article is a function word the glosser drops, so
without the namespace the letter is unreachable and the article plays a
handshape.

**Playback.** IDLE bookends every utterance and loops at rest, so "idle at the
end" and "idle while waiting" are the same state rather than two behaviours.
Two `VideoPlayerController`s are kept in flight — beat *n+1* initialises while
beat *n* plays. A `blackdetect` pass over a screen recording of a four-sign
sentence found no black frames, which is the thing that check exists to catch.

**Glossing** is rule-based and offline. Function words are dropped, everyday
spellings are mapped onto the clip set (`hi`→HELLO, `restroom`→BATHROOM), and
matching is greedy longest-first so multi-word signs beat their first word.
`"hello what is your name"` → `HELLO · WHAT · YOUR · NAME`.

**Missing words are fingerspelled.** A word with no sign clip is spelled letter
by letter, captioned `DAN . D` so the viewer can see both the word and the
progress through it. Spelling is all-or-nothing: a word containing anything
without a letter clip (a digit, say) falls back to the orange `?` chip instead,
because a word spelled with holes in it is worse than one honestly named — the
viewer cannot tell a missing letter from a letter that was not there.

The gloss line shows spelled words letter-joined (`HELLO . MY . NAME . D-A-N`)
and still marks a genuinely unconveyable word with a trailing `*`.

**Two rules keep the glosser honest as the clip set grows.** A function word is
dropped only when nothing can sign it, so adding a clip for TO or WILL makes it
signable with no code change; and a synonym only redirects a spelling that has
no clip of its own, which is why MY and MINE stopped mapping to ME once they
were rendered. Both were live bugs: without them the newly added clips would
have sat unused in the bundle.

### Rebuilding after adding clips

```bash
python avatar_pipeline/build_clips.py     # masters/ -> clips/ + manifest.json
cd flutter_app && flutter build apk --debug
adb -s e4720831 install -r build/app/outputs/flutter-apk/app-debug.apk
```

No Dart change is needed to add a word. The manifest drives the lexicon, so a
new master in `masters/` becomes a signable word on the next build.

## Known problems

1. ~~`GOOD.mp4` is a different avatar.~~ **Fixed.** `GOOD(new).mp4` is the
   re-render and matches the set. `build_clips.py` maps it over the original
   via `ALIASES` rather than deleting anything, so `masters/` stays intact.
   `THANKYOU.mp4` is aliased the same way, to `THANK_YOU` — the label is
   "THANK YOU", so without it the clip would be bundled under a key nothing
   ever asks for.

2. **The watermark question above is unresolved.** See *Framing*.

3. **No non-manual markers.** The avatar's face stays neutral across all signs.
   ASL marks questions, negation, and degree on the face, so "what is your
   name" is missing the brow raise that makes it a question. Worth naming as a
   scoped limitation in the write-up rather than leaving for a panel member to
   raise.

4. **Two seconds of IDLE lead-in** before the first sign on every utterance. It
   is masked in the Text tab because TTS speaks over it, but it is dead time in
   the Voice tab. Trimming IDLE to ~0.6s for the lead-in would tighten this.

## Decisions

Settled:

| | |
|---|---|
| Master format | MP4 or MOV, 1440 × 2560, whatever the tool exports |
| Delivered format | MP4 / H.264 Main / yuv420p / 30fps CFR / no audio / faststart |
| Delivered size | 720 × 1024, cropped from the top 2048 rows |
| Vocabulary | 12 signs + IDLE, all inside CORE200 |
| Duration | Not normalised — clips run 1.02–2.04s as recorded |
| UI language label | "ASL Avatar" (was "3D FSL Avatar", which was wrong twice over) |

Still open:

1. **Fingerspelling.** 26 handshape clips would turn the missing-word chip into
   an actual sign. With a 12-word vocabulary the fallback fires constantly, so
   this buys more here than it would with a large clip set.
2. **Whether to grow the vocabulary** toward CORE200, or keep it demo-sized and
   lean on the fallback.
