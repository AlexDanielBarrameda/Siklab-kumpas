"""Pull one reference video per CORE200 word out of the ASL Citizen zip.

For learning the 200 words the app ships, and for filming yourself signing them.
Streams members straight out of the 42.8 GB archive -- nothing is unpacked.

Variant handling matches extract_asl_citizen.py: ASL-LEX stores regional
variants as WHAT1 / WHAT2, and the corpus keeps one variant per concept under
the plain name. The same base_concept() collapse is applied here, so the video
you watch for WHAT is the variant the model was actually trained on. Getting
this wrong would have you learning a sign the app cannot recognise.

    python export_core200_videos.py
    python export_core200_videos.py --per-word 3   # more examples, more signers
"""

import os
import re
import csv
import sys
import zipfile
import argparse
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../models"))

RAW = os.path.join(HERE, "../data/raw/asl_citizen")
ZIP = os.path.join(RAW, "ASL_Citizen.zip")
SPLITS = os.path.join(RAW, "ASL_Citizen/splits")
OUT = os.path.join(RAW, "core200_videos")

# Display spellings, matching flutter_app/assets/labels/asl_core200_labels.txt.
DISPLAY = {"THANKYOU": "THANK YOU"}


def base_concept(gloss):
    """WHAT1 -> WHAT. Identical to extract_asl_citizen.py."""
    return re.sub(r"\d+$", "", gloss) or gloss


def read_splits():
    rows = []
    for split in ("train", "val", "test"):
        p = os.path.join(SPLITS, f"{split}.csv")
        if not os.path.exists(p):
            continue
        with open(p, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                rows.append({
                    "signer": r.get("Participant ID", "").strip(),
                    "video": r["Video file"].strip(),
                    "gloss": r["Gloss"].strip().upper(),
                    "split": split,
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-word", type=int, default=1,
                    help="videos per word; >1 picks different signers")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    from asl_core200 import CORE200, GROUPS

    rows = read_splits()
    if not rows:
        raise SystemExit(f"no split CSVs under {SPLITS}")

    # gloss -> rows, and concept -> the variant with the most recordings, so the
    # exported clip comes from the same variant the model was trained on.
    by_gloss = defaultdict(list)
    for r in rows:
        by_gloss[r["gloss"]].append(r)

    variants = defaultdict(list)
    for g in by_gloss:
        variants[base_concept(g)].append(g)
    chosen = {b: max(vs, key=lambda g: len(by_gloss[g]))
              for b, vs in variants.items()}

    os.makedirs(args.out, exist_ok=True)
    zf = zipfile.ZipFile(ZIP)
    names = set(zf.namelist())

    # word -> list of exported filenames, for the index page
    exported, missing = {}, []

    for word in CORE200:
        g = chosen.get(word)
        if not g:
            missing.append(word)
            continue

        # Spread across signers so multiple examples are actually different
        # people rather than the same person twice.
        cands = sorted(by_gloss[g], key=lambda r: (r["signer"], r["video"]))
        seen_signers, picks = set(), []
        for r in cands:
            if r["signer"] in seen_signers:
                continue
            seen_signers.add(r["signer"])
            picks.append(r)
            if len(picks) >= args.per_word:
                break
        if not picks:
            missing.append(word)
            continue

        files = []
        for i, r in enumerate(picks):
            member = f"ASL_Citizen/videos/{r['video']}"
            if member not in names:
                continue
            suffix = "" if args.per_word == 1 else f"_{i+1}"
            fname = f"{word}{suffix}.mp4"
            dest = os.path.join(args.out, fname)
            if not os.path.exists(dest):
                with zf.open(member) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
            files.append(fname)

        if files:
            exported[word] = files
        else:
            missing.append(word)

    zf.close()
    write_index(args.out, exported, GROUPS)

    total = sum(len(v) for v in exported.values())
    print(f"exported {total} video(s) for {len(exported)}/{len(CORE200)} words")
    print(f"  -> {os.path.abspath(args.out)}")
    print(f"  -> open index.html in that folder to browse them by category")
    if missing:
        print(f"\nno video found for {len(missing)}: {', '.join(missing)}")


def write_index(out, exported, groups):
    """A local page that plays every clip, grouped the way the vocabulary is
    organised, so the 200 can be worked through category by category."""
    word_group = {}
    for gname, block in groups.items():
        for w in block.split():
            word_group.setdefault(w, gname)

    ordered = defaultdict(list)
    for w in sorted(exported):
        ordered[word_group.get(w, "other")].append(w)

    parts = ["""<!doctype html>
<meta charset="utf-8">
<title>ASL CORE200 reference</title>
<style>
 :root{--bg:#101318;--card:#181d25;--ink:#e7eaef;--mut:#98a1b0;--line:#262c36;--acc:#dfa162}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);
      font:15px/1.5 "Segoe UI",system-ui,sans-serif;padding:28px}
 h1{font-size:22px;margin:0 0 4px}
 .sub{color:var(--mut);margin:0 0 26px;font-size:13.5px}
 h2{font-size:13px;text-transform:uppercase;letter-spacing:.09em;color:var(--acc);
    margin:34px 0 12px;padding-bottom:7px;border-bottom:1px solid var(--line)}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:14px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:6px;
       overflow:hidden}
 video{width:100%;display:block;background:#000;aspect-ratio:4/3;object-fit:cover}
 .cap{padding:8px 10px;font-weight:600;font-size:13.5px;letter-spacing:.02em}
 .n{color:var(--mut);font-weight:400;font-size:12px}
 input{width:100%;max-width:340px;padding:9px 12px;border-radius:5px;
       border:1px solid var(--line);background:var(--card);color:var(--ink);
       font-size:14px;margin-bottom:8px}
</style>
<h1>ASL CORE200 &mdash; reference clips</h1>
<p class="sub">The 200 words the app recognises. Hover or tap a clip to play it.
Type to filter.</p>
<input id="q" placeholder="filter words&hellip;" autocomplete="off">
<div id="all">"""]

    for gname in groups:
        words = ordered.get(gname, [])
        if not words:
            continue
        parts.append(f'<section><h2>{gname} &middot; {len(words)}</h2>'
                     f'<div class="grid">')
        for w in words:
            files = exported[w]
            label = DISPLAY.get(w, w)
            extra = (f' <span class="n">&times;{len(files)}</span>'
                     if len(files) > 1 else "")
            for f in files:
                parts.append(
                    f'<div class="card" data-w="{w}">'
                    f'<video src="{f}" muted loop playsinline preload="none" '
                    f'onmouseover="this.play()" onmouseout="this.pause()" '
                    f'onclick="this.paused?this.play():this.pause()"></video>'
                    f'<div class="cap">{label}{extra}</div></div>')
        parts.append("</div></section>")

    parts.append("""</div>
<script>
 const q=document.getElementById('q');
 q.addEventListener('input',()=>{
   const v=q.value.trim().toUpperCase();
   document.querySelectorAll('.card').forEach(c=>{
     c.style.display = !v || c.dataset.w.includes(v) ? '' : 'none';
   });
   document.querySelectorAll('section').forEach(s=>{
     const any=[...s.querySelectorAll('.card')].some(c=>c.style.display!=='none');
     s.style.display = any ? '' : 'none';
   });
 });
</script>""")

    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
