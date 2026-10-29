"""Normalise avatar masters into the clips the app ships.

The masters come out of the avatar tool as 1440x2560 HEVC with an audio track.
None of that is wrong for editing, but three parts of it are wrong for the app:

  HEVC        -> H.264. HEVC decodes on modern hardware but H.264 is the format
                 every Android device is required to support, and at this size
                 the quality difference is invisible.
  audio track -> stripped. The Text tab speaks TTS while the avatar signs; an
                 audio track on the clip would talk over it.
  1440x2560   -> cropped and scaled. See CROP below.

CROP: the bottom fifth of the frame is legs and a tool watermark, neither of
which carries any linguistic information, and keeping them shrinks the part
that does. The crop keeps the full width (arms swing wide on WHERE and
BATHROOM) and the top 2048 rows, which clears the fingertips at rest -- they
reach about row 1870 -- with margin.

    python avatar_pipeline/build_clips.py
    python avatar_pipeline/build_clips.py --crf 20      # higher quality
"""

import os
import re
import sys
import json
import shutil
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MASTERS = os.path.join(HERE, "masters")
CLIPS = os.path.join(ROOT, "flutter_app", "assets", "avatar", "clips")
LABELS = os.path.join(ROOT, "flutter_app", "assets", "labels",
                      "asl_core200_labels.txt")

# Source geometry, and the window of it worth keeping.
SRC_W, SRC_H = 1440, 2560
CROP_W, CROP_H, CROP_X, CROP_Y = 1440, 2048, 0, 0
OUT_W, OUT_H = 720, 1024

# The resting-pose clip. Bookends every utterance and loops while idle, so it
# is required rather than just another word.
IDLE = "IDLE"

SOURCE_EXTS = (".mp4", ".mov", ".m4v", ".webm")

# Filenames whose slug is not the gloss they represent.
#
# THANKYOU: the label is "THANK YOU", so the app looks up THANK_YOU. Without
# this the clip is bundled under a key nothing ever asks for -- a silent miss
# that shows up as an unsigned word rather than an error.
#
# GOODNEW: "GOOD(new).mp4" is a re-render replacing the original GOOD, whose
# outfit and framing did not match the rest of the set. Mapping it rather than
# deleting the original keeps masters/ intact; the plain name is skipped below.
ALIASES = {
    "THANKYOU": "THANK_YOU",
    "GOODNEW": "GOOD",
}

# Fingerspelling clips live in their own folder and get their own namespace.
# A letter clip called A.mp4 would otherwise collide with the article "a",
# which the glosser drops as a function word -- the letter would be
# unreachable and the article would play a handshape.
LETTERS_DIR = "FSL LETTERS"
LETTER_PREFIX = "LETTER_"


def slug(label):
    """THANK YOU -> THANK_YOU, DON'T FEEL LIKE -> DONT_FEEL_LIKE."""
    s = re.sub(r"[^A-Za-z0-9 ]", "", label.upper())
    return re.sub(r"\s+", "_", s.strip())


def read_labels():
    if not os.path.exists(LABELS):
        return set()
    out = set()
    with open(LABELS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "," in line:
                out.add(slug(line.split(",", 1)[1]))
    return out


def probe(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-show_entries", "format=duration",
         "-of", "json", path],
        capture_output=True, text=True)
    if r.returncode != 0:
        return None
    d = json.loads(r.stdout)
    st = (d.get("streams") or [{}])[0]
    return {
        "w": st.get("width"),
        "h": st.get("height"),
        "duration": float(d.get("format", {}).get("duration", 0) or 0),
    }


def build(src, dst, crf):
    """Transcode one master. Returns (ok, message)."""
    info = probe(src)
    if not info:
        return False, "unreadable"

    # Only crop when the master matches the geometry the crop was measured
    # against. A differently sized master gets letterboxed into the output
    # frame instead, which looks wrong but is obvious -- better than silently
    # cropping someone's head off.
    if (info["w"], info["h"]) == (SRC_W, SRC_H):
        vf = (f"crop={CROP_W}:{CROP_H}:{CROP_X}:{CROP_Y},"
              f"scale={OUT_W}:{OUT_H},fps=30")
        note = ""
    else:
        vf = (f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
              f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2:color=0x8FA3BC,fps=30")
        note = f"  [!] {info['w']}x{info['h']}, not {SRC_W}x{SRC_H} - padded, not cropped"

    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", src,
        "-an",
        "-c:v", "libx264", "-profile:v", "main", "-level", "4.0",
        "-pix_fmt", "yuv420p",
        "-vf", vf,
        "-crf", str(crf), "-preset", "slow",
        "-g", "15", "-keyint_min", "15", "-sc_threshold", "0",
        "-movflags", "+faststart",
        dst,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return False, (r.stderr or "ffmpeg failed").strip().splitlines()[-1]
    return True, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crf", type=int, default=23,
                    help="quality; lower is better and bigger (default 23)")
    ap.add_argument("--masters", default=MASTERS)
    ap.add_argument("--out", default=CLIPS)
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not on PATH")
    if not os.path.isdir(args.masters):
        raise SystemExit(f"no masters directory at {args.masters}")

    # Names an alias supersedes: the plainly-named master is reference only.
    superseded = set(ALIASES.values())

    sources, letters = {}, {}
    for fn in sorted(os.listdir(args.masters)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in SOURCE_EXTS:
            continue
        key = slug(stem)
        path = os.path.join(args.masters, fn)
        if key in ALIASES:
            sources[ALIASES[key]] = path
        elif key not in superseded:
            sources.setdefault(key, path)

    letters_dir = os.path.join(args.masters, LETTERS_DIR)
    if os.path.isdir(letters_dir):
        for fn in sorted(os.listdir(letters_dir)):
            stem, ext = os.path.splitext(fn)
            if ext.lower() not in SOURCE_EXTS:
                continue
            key = slug(stem)
            if len(key) == 1 and key.isalpha():
                letters[key] = os.path.join(letters_dir, fn)
                sources[LETTER_PREFIX + key] = os.path.join(letters_dir, fn)

    if not sources:
        raise SystemExit(f"no video files in {args.masters}")

    os.makedirs(args.out, exist_ok=True)
    known = read_labels()
    manifest, failed, unknown = {}, [], []
    total_bytes = 0

    print(f"building {len(sources)} clip(s) -> {args.out}\n")
    for word, src in sources.items():
        dst = os.path.join(args.out, f"{word}.mp4")
        ok, note = build(src, dst, args.crf)
        if not ok:
            failed.append((word, note))
            print(f"  FAIL  {word}: {note}")
            continue

        size = os.path.getsize(dst)
        total_bytes += size
        info = probe(dst)
        manifest[word] = {
            "file": f"{word}.mp4",
            "duration": round(info["duration"], 3) if info else None,
        }
        if (known and word not in known and word != IDLE
                and not word.startswith(LETTER_PREFIX)):
            unknown.append(word)
        print(f"  ok    {word:<12} {info['duration']:>5.2f}s  "
              f"{size/1024:>6.0f} KB{note}")

    # The app reads this instead of guessing filenames or scanning the bundle.
    with open(os.path.join(args.out, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "idle": IDLE if IDLE in manifest else None,
            "width": OUT_W,
            "height": OUT_H,
            # letter -> key in "clips", so the app can fingerspell a word it
            # has no sign for without guessing at filenames.
            "letters": {
                k: LETTER_PREFIX + k
                for k in sorted(letters)
                if LETTER_PREFIX + k in manifest
            },
            "clips": manifest,
        }, f, indent=2, sort_keys=True)

    words = sorted(w for w in manifest
                   if w != IDLE and not w.startswith(LETTER_PREFIX))
    built_letters = sorted(w for w in manifest if w.startswith(LETTER_PREFIX))
    print(f"\n{len(words)} sign(s), {len(built_letters)} letter(s), "
          f"{'IDLE' if IDLE in manifest else 'NO IDLE'}"
          f" -- {total_bytes/1024/1024:.1f} MB total")
    print(f"vocabulary: {', '.join(words)}")

    if IDLE not in manifest:
        print(f"\n  [!] no {IDLE} clip. The player bookends every utterance with"
              f"\n      it and loops it at rest, so add {IDLE}.mp4 to masters/.")
    if unknown:
        print(f"\n  [!] not in asl_core200_labels.txt: {', '.join(unknown)}"
              f"\n      the avatar can sign words the recogniser does not know,"
              f"\n      but keeping the two vocabularies aligned is cleaner.")
    if failed:
        print(f"\n  [!] {len(failed)} failed: "
              f"{', '.join(w for w, _ in failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
