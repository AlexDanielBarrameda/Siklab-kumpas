# Prompt for the UI v2 conversation

Copy everything below the line into a new Claude Code conversation.

---

I need you to implement a UI redesign for SIKLAB-KUMPAS, a Flutter Android app
that does bi-directional Filipino Sign Language translation, fully offline.

## Read these first

The new designs are in `E:\Thesis\Siklab-kumpas\design\ui_v2\`:

- `UI.1.png` through `UI.9.png` — the new screens (dark mode)
- `UI.10.png` — light mode version
- `App-Icon.svg` — new app launcher icon
- `FSL-User-Icon.svg` — icon for the FSL user profile card
- `Hearing-Speaking-Icon.svg` — icon for the hearing/speaking profile card
- `Wordmark.svg` — the "SIKLAB KUMPAS" wordmark

**Look at every one of these before changing any code.** Map each mockup to the
screen it replaces and tell me your reading of them before you start, so we can
correct any misunderstanding early.

## What changes

**Home screen**
- Replace the "SIKLAB-KUMPAS" text + icon with `Wordmark.svg`
- Hearing/speaking profile icon becomes `Hearing-Speaking-Icon.svg`
- FSL user profile icon becomes `FSL-User-Icon.svg`
- Both profile cards move inside a container
- Settings modal stays as it is

**Translation Hub (hearing user screen)**
- Add a Camera option alongside Voice and Text. Rationale: if a hearing user
  cannot understand an FSL user, they should be able to switch to camera mode
  in place rather than backing out and switching profiles. It should reuse the
  existing recognition pipeline from the Deaf/Mute screen.
- The voice speed box becomes collapsible: it starts expanded showing the
  slider, and collapses when tapped.

**Initiator mode (deaf/mute screen)**
- "Signal for Conversation" becomes a button that starts full width, then after
  a few seconds animates down to just the bell icon.
- The Signal modal itself is unchanged.

**Light mode**
- Implement to match `UI.10.png`.

**App icon**
- Replace the launcher icon with `App-Icon.svg`.

## Design tokens already extracted from the assets

| token | value | current code has |
|---|---|---|
| brand orange | `#FF6B01` | `#FF6B00` — needs correcting |
| dark background | `#000000` | `#0D0D0D` |
| dark card/surface | `#181818` | `#1A1A2E` |
| light background | `#F0F0F0` with lavender/pink tints (`#F0D8F0`, `#F0D8D8`) | `#F5F0FF` |

App-Icon also uses a light lavender gradient: `#F7EEFF`, `#F1E6FF`, `#E0DEFF`.

## Where things are

```
E:\Thesis\Siklab-kumpas\
  flutter_app\lib\
    main.dart                       theme + language state (SiklabKumpasApp.of)
    screens\home_screen.dart
    screens\hearing_user_screen.dart   Voice / Text tabs, TTS speed slider
    screens\deaf_mute_screen.dart      camera, recognition, Signal modal
    services\camera_service.dart
    services\inference_service.dart
    services\speech_service.dart       TTS + STT
    widgets\skeleton_overlay.dart      debug landmark overlay
  flutter_app\assets\models\           .tflite models
  flutter_app\assets\labels\
```

## Do not disturb

The recognition pipeline took a long debugging effort and is finally working.
Restrict changes to presentation. In particular:

- `inference_service.dart` — the canonicalisation and normalisation logic must
  stay byte-equivalent to `ml_pipeline/preprocessing/canonicalize.py`. The
  models expect canonicalised input and mispredict badly without it.
- `MainActivity.kt` — the MediaPipe/image pipeline.
- Do not change which `.tflite` files are loaded.

Recent behaviour that is intentional and should survive the redesign:

- Letters: majority voting, 3 of the last 5 frames must agree before a letter
  is announced.
- Words: segmented on hand presence — buffering starts when hands enter frame,
  the sign is classified when they leave. Not a fixed time window.
- TTS speed is a multiplier where 1.0 = natural pace (Android's engine rate is
  0.5 at normal, which is why the raw value is halved internally).
- The camera overlay intentionally has only FRONT/BACK and SKELETON chips.
  Earlier MIRROR and hand-block chips were removed because they no longer
  affect the result.

## Environment

- Test device: Xiaomi M2004J19C, Android 12, id `005a098d0410`, already connected
- Build: `cd flutter_app && flutter build apk --debug`
- Install: `adb -s 005a098d0410 install -r build\app\outputs\flutter-apk\app-debug.apk`
- MIUI quirk: if install fails with `INSTALL_FAILED_USER_RESTRICTED`, run
  `adb -s 005a098d0410 shell am force-stop com.siklabkumpas.siklab_kumpas`
  first and retry — that has worked every time.
- Flutter analyze currently reports 3 pre-existing cosmetic lints; that is the
  clean baseline.

## How I would like you to work

Ask before making assumptions — if a mockup is ambiguous, ask rather than guess.
Show me your plan for each screen before implementing it. I will test on the
device between changes.

## Still open, for context (not part of this task)

STT (the microphone path) is unfinished: never tested on device, and recognised
text is not yet wired into the hearing screen's output box.
