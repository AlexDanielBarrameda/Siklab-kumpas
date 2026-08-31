# Device setup for speech

Speech is the one part of the app that depends on something the phone provides
rather than something we ship. Two separate things have to be true, and the
failure looks identical either way: you press a button and nothing is spoken.

## 1. The manifest has to declare what it wants to see

Fixed in `flutter_app/android/app/src/main/AndroidManifest.xml` — recorded here
because the symptom is so misleading.

Android 11 (API 30) filters which packages an app is allowed to see. Without a
`<queries>` entry, the app cannot bind to a text-to-speech engine or a speech
recogniser **even when one is installed and working**:

```
I AppsFilter: interaction: com.siklabkumpas.siklab_kumpas -> com.xiaomi.mibrain.speech BLOCKED
E TTS     : Failed to initialize TextToSpeech with status: -1
W TextToSpeech: getAvailableLanguages failed: not bound to TTS engine
I flutter : STT init error: PlatformException(recognizerNotAvailable, ...)
```

The `AppsFilter ... BLOCKED` line is the only real clue, and it is easy to miss
because it is logged by the system, not the app.

This is invisible on Android 10 and below, where no filtering happens. That is
why it reproduced on a HyperOS 3 device while a Redmi 9 and a Realme 9i were
fine — those are old enough not to filter. **A device that works proves nothing
about a newer one.**

Neither `flutter_tts` nor `speech_to_text` declares these, so the app must:

```xml
<queries>
    <intent><action android:name="android.intent.action.TTS_SERVICE"/></intent>
    <intent><action android:name="android.speech.RecognitionService"/></intent>
</queries>
```

## 2. The device needs a TTS engine, set as default

Even with the manifest right, `TextToSpeech` binds to whatever engine is set as
the system default. If none is set, it can fail with `status: -1` while an
engine sits installed and unused.

Check:

```bash
adb -s <device> shell settings get secure tts_default_synth
adb -s <device> shell cmd package query-services -a android.intent.action.TTS_SERVICE
```

`null` from the first command with a service listed by the second is the
broken-but-fixable case: pick that engine as the default.

Through the UI on Xiaomi/HyperOS:
**Settings → Additional settings → Accessibility → Text-to-speech output →
Preferred engine**. On stock Android it is under **Settings → System →
Languages & input → Text-to-speech output**.

Or over adb:

```bash
adb -s <device> shell settings put secure tts_default_synth <package.name>
```

This persists across reboots. To undo it, set it back to the engine you had, or
clear it with `settings delete secure tts_default_synth`.

### The Redmi Turbo 4 Pro specifically

It is a **China-region ROM** (`ro.miui.region=CN`, build `OS3.0.305.0.WOLCNXM`).
Google Text-to-Speech is not installed; the only engine present is Xiaomi's
`com.xiaomi.mibrain.speech`, and no default was configured.

Setting that engine as the default was enough — it turns out to support English:

```
I TextToSpeech: Sucessfully bound to com.xiaomi.mibrain.speech
I flutter : TTS ready: en  (asked for en-US)  rate 0.5
E TtsService: onSynthesizeText Start: text=where is the bathroom please
```

Two consequences worth knowing before the demo:

- **Filipino will not be available on this device.** `SpeechService`
  `_resolveLanguage` falls back to English rather than failing, so the app keeps
  working, but the status row will say so. If the demo needs Filipino TTS, use
  a phone with Google TTS installed.
- **Installing Google Text-to-Speech from the Play Store** (which is present on
  this device) and setting it as the preferred engine would give better voices
  and more languages. The Xiaomi engine is sufficient, not ideal.

## Speech that works, then stops until the app is restarted

Fixed in `SpeechService`. Recorded because the cause is three separate traps
stacked on each other, and the symptom — the Convert button doing nothing, with
an empty log — points at none of them.

**What happens.** Xiaomi runs the speech engine on demand and freezes
background services aggressively, so the binding the app got at startup does
not last. Once the engine goes away:

1. `speak()` never returns. `awaitSpeakCompletion(true)` waits for a completion
   callback the dead engine will never send.
2. Nothing is logged, because every `debugPrint` in that path sits after the
   await. The button simply stops responding.
3. `flutter_tts` cannot recover on its own. It constructs its Android
   `TextToSpeech` **once**, when the plugin attaches, and never retries — on
   failure it logs `Failed to initialize TextToSpeech with status: -1` and
   leaves a dead object behind whose `getLanguages` returns empty forever.
   That empty list is what surfaced as *"No text-to-speech engine on this
   device"* on a phone that had a working engine.

**What was added:**

- **Deadlines on every flutter_tts call.** Any of them can hang: the plugin
  queues calls while its TextToSpeech is initialising and answers them only
  once that finishes.
- **A liveness check before speaking.** A dead binding still looks healthy to
  the plugin, so the app asks for the language list first — milliseconds, and
  empty means dead. Without it, every failure cost the full speak timeout.
- **Rebinding via `setEngine`.** This is the only way back: it drops the dead
  `TextToSpeech` and constructs a new one. Note its Dart wrapper awaits the
  channel call *without returning the result*, so it resolves to `null` even on
  success — completion is the only usable success signal, and testing the
  value reads every success as a failure.
- **Polling after a rebind.** A rebind cold-starts the engine process, which is
  not answerable for around 700 ms.

Measured, killing the engine mid-session with
`adb shell am force-stop com.xiaomi.mibrain.speech`:

```
19:47:46.249  TTS: binding is dead before speaking; recovering first
19:47:46.260  TTS: rebinding to com.xiaomi.mibrain.speech
19:47:46.952  TTS ready: en  (asked for en)
19:47:49.694  TTS spoke 2339ms  result=1
```

One press, about a second of extra latency, and the utterance is spoken.
Previously this state was permanent until the app was restarted.

## The Voice Speed slider does nothing on this device

Not a bug in the slider or in `SpeechService`. **The Xiaomi engine ignores
`setSpeechRate`.** Android documents rate support as engine-dependent, and this
one does not implement it.

Measured on device, same sentence, using the `TTS spoke ...ms` line that
`SpeechService.speak` now logs:

| Slider | Rate sent to engine | Actual spoken duration |
|---|---|---|
| 0.50x | 0.25 | 2861 ms |
| 1.60x | 0.80 | 2616 ms |

A 3.2× spread in requested rate produced a 9% difference in output — noise. If
the rate were honoured, 0.50x would take roughly three times as long as 1.60x.

The mapping itself is correct and worth not re-deriving: `flutter_tts` on
Android multiplies by 2 before calling `TextToSpeech.setSpeechRate`, so flutter
0.5 is Android 1.0 — normal pace. That is why `SpeechService._platformNormal`
is 0.5 and not 1.0.

**To confirm the diagnosis rather than take it on trust**, move the slider on
the Redmi 9 or Realme 9i. Those have Google TTS, which does honour the rate. If
speed changes there and not here, the engine is the whole story and no code
change is warranted.

**To fix it on this device**, install Google Text-to-Speech from the Play Store
and set it as the preferred engine. Untested here — Google TTS is not installed
on this ROM.

## Confirming it works

```bash
adb -s <device> logcat -c
# press Convert to Speech + ASL in the app
adb -s <device> logcat -d | grep -iE "flutter :|TextToSpeech|onSynthesizeText"
```

Healthy output names the engine, reports the resolved language, and shows the
text being synthesised. `onSynthesizeText Start: text=...` followed by an
`AudioTrack` line is proof that audio really was produced — useful when you
cannot hear the device.

## The app tells you now

The Text tab's status row used to read a hardcoded green "Audio ready"
regardless of whether speech worked. It now reflects `SpeechService.ttsAvailable`
and `ttsStatus`: green when speech works, orange when it fell back to a
different language than requested, red with the reason when it cannot speak at
all. A communication aid that silently stops talking is worse than one that
says it cannot.
