"""Can a multimodal LLM recognise ASL signs? Measure before integrating.

No free ASL-to-text API exists. Signapse and AWS GenASL run the opposite
direction (text -> sign); the ASLLRP Sign Bank does accept sign video and return
top-5 matches but is a web form with no API and is an academic service. The only
genuinely available online route is a general vision model, which is NOT a sign
recogniser and has never been shown to do this reliably.

So this scores it on clips we already have, before any app changes, API keys in
production, or the privacy tradeoff of uploading video of a deaf signer.

Two conditions, because they measure different things:

  CONSTRAINED  the candidate vocabulary is supplied and the model picks from it.
               This is how the app would actually use it -- a fixed 525-word
               vocabulary -- so it is the fair comparison against the on-device
               model's 67.4%.

  OPEN         "what sign is this?" with no list. Much harder, and indicates
               whether the model has any real grasp of ASL or is just
               pattern-matching against the options.

Usage:
    set GEMINI_API_KEY=...
    python gemini_asl_probe.py --n 20
"""

import os
import csv
import time
import json
import random
import zipfile
import argparse
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP = os.path.join(HERE, "../data/raw/asl_citizen/ASL_Citizen.zip")
SPLITS = os.path.join(HERE, "../data/raw/asl_citizen/ASL_Citizen/splits")
APP_LABELS = os.path.join(HERE, "../../flutter_app/assets/labels/asl_525_labels.txt")

# 2.5-flash is retired for new API keys ("no longer available to new users").
MODEL = "gemini-3.6-flash"
# The first 20-clip run died with 429 RESOURCE_EXHAUSTED after ~6 clips. The
# binding free-tier limit is tokens-per-minute, not requests-per-day, and the
# constrained prompt carries all 523 vocabulary words -- so each call is
# expensive regardless of how few calls are made. Hence a slow base interval
# plus real backoff.
SLEEP = 8.0
MAX_RETRIES = 4

CONSTRAINED = (
    "This video shows one isolated American Sign Language sign.\n"
    "Identify it. Choose the single best match from this list:\n\n{options}\n\n"
    "Reply with ONLY the chosen word, nothing else."
)
OPEN = (
    "This video shows one isolated American Sign Language sign.\n"
    "What English word does it mean? Reply with ONLY that word, nothing else."
)


ENV_FILE = os.path.join(HERE, "../../.env")


def load_key():
    """Key from the gitignored .env, or the environment as a fallback.

    Deliberately read from a file rather than pasted anywhere it could be
    committed or end up in a chat log.
    """
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                if name.strip() in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                    value = value.strip().strip('"').strip("'")
                    if value and value != "paste_your_key_here":
                        return value
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def app_vocabulary():
    """The words the app can actually output, so the test matches deployment."""
    words = []
    with open(APP_LABELS, encoding="utf-8") as f:
        for line in f:
            if "," in line:
                w = line.strip().split(",", 1)[1]
                if w:
                    words.append(w)
    return words


def test_rows():
    rows = []
    with open(os.path.join(SPLITS, "test.csv"), newline="",
              encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append({"gloss": r["Gloss"].strip().upper(),
                         "video": r["Video file"].strip()})
    return rows


def normalise(s):
    return "".join(c for c in s.upper() if c.isalnum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="clips to test")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    key = load_key()
    if not key:
        raise SystemExit(
            "No API key found.\n"
            f"  Put it in: {os.path.abspath(ENV_FILE)}\n"
            "  as a single line:  GEMINI_API_KEY=your_key_here\n"
            "  Free key (no card): https://aistudio.google.com"
        )

    from google import genai
    client = genai.Client(api_key=key)

    vocab = app_vocabulary()
    vocab_set = {normalise(w) for w in vocab}

    # Only test signs the app claims to know, drawn from the held-out test split
    # so the signers are ones the on-device model never saw either.
    rows = [r for r in test_rows() if normalise(r["gloss"]) in vocab_set]
    random.seed(11)
    random.shuffle(rows)

    picked, seen = [], set()
    for r in rows:
        if r["gloss"] in seen:
            continue
        seen.add(r["gloss"])
        picked.append(r)
        if len(picked) >= args.n:
            break

    options = ", ".join(sorted(vocab))
    zf = zipfile.ZipFile(ZIP)
    results = []

    print(f"{args.model}   {len(picked)} clips   vocabulary {len(vocab)} words\n")
    print(f"{'truth':<16} {'constrained':<16} {'open':<16}")
    print("-" * 50)

    for r in picked:
        tmp = None
        try:
            data = zf.read(f"ASL_Citizen/videos/{r['video']}")
            fd, tmp = tempfile.mkstemp(suffix=".mp4")
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)

            uploaded = client.files.upload(file=tmp)
            # Uploaded files need a moment to become ACTIVE.
            for _ in range(30):
                info = client.files.get(name=uploaded.name)
                if str(info.state).endswith("ACTIVE"):
                    break
                time.sleep(1)

            answers = {}
            for tag, prompt in (("constrained", CONSTRAINED.format(options=options)),
                                ("open", OPEN)):
                answers[tag] = "<no answer>"
                for attempt in range(MAX_RETRIES):
                    try:
                        resp = client.models.generate_content(
                            model=args.model, contents=[uploaded, prompt])
                        answers[tag] = (resp.text or "").strip()
                        answers[tag] = (answers[tag].splitlines() or [""])[0][:24]
                        break
                    except Exception as e:
                        msg = str(e)
                        transient = "429" in msg or "503" in msg or "RESOURCE" in msg
                        if transient and attempt < MAX_RETRIES - 1:
                            wait = SLEEP * (2 ** attempt) + 5
                            print(f"    rate limited, waiting {wait:.0f}s "
                                  f"(attempt {attempt + 1})", flush=True)
                            time.sleep(wait)
                            continue
                        answers[tag] = f"<err: {msg[:50]}>"
                        break
                time.sleep(SLEEP)

            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass

            ok_c = normalise(answers["constrained"]) == normalise(r["gloss"])
            ok_o = normalise(answers["open"]) == normalise(r["gloss"])
            results.append({"gloss": r["gloss"], **answers,
                            "constrained_ok": ok_c, "open_ok": ok_o})
            print(f"{r['gloss']:<16} {answers['constrained'][:15]:<16}"
                  f" {answers['open'][:15]:<16}"
                  f" {'C' if ok_c else '.'}{'O' if ok_o else '.'}")
        except Exception as e:
            print(f"{r['gloss']:<16} failed: {type(e).__name__}: {e}")
        finally:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    if results:
        c = sum(r["constrained_ok"] for r in results) / len(results) * 100
        o = sum(r["open_ok"] for r in results) / len(results) * 100
        print("\n" + "=" * 50)
        print(f"clips scored          : {len(results)}")
        print(f"CONSTRAINED accuracy  : {c:.0f}%   <- compare to on-device 67.4%")
        print(f"OPEN accuracy         : {o:.0f}%")
        out = os.path.join(HERE, "gemini_asl_probe_results.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
        print(f"detail -> {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()
