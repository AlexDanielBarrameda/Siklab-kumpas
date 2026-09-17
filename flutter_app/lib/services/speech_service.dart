import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter_tts/flutter_tts.dart';

class SpeechService {
  static final FlutterTts _tts = FlutterTts();

  // Speech INPUT does not live here any more. Every system recognition path
  // on the target device failed (see OfflineSpeechService), and the one that
  // partly worked sent audio to Google — the single component of this app that
  // was not on-device. Voice input is Vosk, bundled, and the INTERNET
  // permission is gone from the release manifest, so "fully offline" is now a
  // property a reviewer can verify rather than a claim they have to take on
  // trust.

  /// Platform speech rate that sounds like normal speaking pace.
  ///
  /// flutter_tts takes 0.0-1.0 on Android, where **0.5 is normal** and 1.0 is
  /// roughly double speed. The previous default of 1.0 was therefore reading
  /// everything at ~2x, which is why output was hard to follow.
  static const double _platformNormal = 0.5;

  /// User-facing speed, where 1.0x means natural speaking pace. Callers work in
  /// these units; [_toPlatformRate] converts to what the engine expects.
  static double speedMultiplier = 1.0;

  static double _toPlatformRate(double multiplier) =>
      (multiplier * _platformNormal).clamp(0.05, 1.0);

  /// Whether TTS actually works on this device. False means every speak()
  /// call will be silent, which the UI needs to surface — a communication aid
  /// that quietly stops talking is worse than one that says it cannot.
  static bool ttsAvailable = false;

  /// Human-readable reason, shown in the UI and the logs when TTS is degraded.
  static String ttsStatus = 'not initialised';

  /// The language actually in use, which may not be the one requested.
  static String ttsLanguage = '';

  static List<String> _languages = const [];

  /// Every language the engine reports. Surfaced so a diagnostic run can show
  /// whether a Filipino voice is actually installed, rather than inferring it
  /// from how the output sounds.
  static List<String> get ttsLanguages => _languages;
  static bool _ttsInitialised = false;

  /// Whether a Filipino voice is actually downloaded. Null until checked.
  static bool? filipinoInstalled;

  static final Map<String, bool> _installedCache = {};

  /// Whether the engine can actually SPEAK this locale, not merely list it.
  static Future<bool> _installed(String code) async {
    final hit = _installedCache[code];
    if (hit != null) return hit;
    try {
      final r = await _tts
          .isLanguageInstalled(code)
          .timeout(const Duration(seconds: 3));
      final ok = r == true || r == 1;
      _installedCache[code] = ok;
      return ok;
    } catch (e) {
      debugPrint('TTS: isLanguageInstalled($code) failed: $e');
      return false;
    }
  }

  /// Resolves a requested locale against what the device really has.
  ///
  /// getLanguages reports what the ENGINE SUPPORTS, not what is downloaded, and
  /// those are different sets. Google TTS lists fil-PH on a phone carrying no
  /// Filipino voice data, so matching against that list alone resolved to a
  /// locale the engine cannot speak: setLanguage returns LANG_MISSING_DATA, the
  /// utterance is rendered by the fallback voice or not at all, and
  /// [ttsLanguage] is left claiming 'fil-PH'.
  ///
  /// Measured on device, that one mistake produced every reported symptom at
  /// once — Filipino quick messages read by an English voice, "usap" spelled
  /// out, "Pipe" said as /paip/, and the phrase surfacing minutes later glued
  /// to the front of an unrelated sentence. It also defeated the respelling
  /// meant to rescue an English voice, because the app believed it had a
  /// Filipino one.
  ///
  /// isLanguageInstalled is the question that actually matters.
  static Future<String?> _resolveLanguage(String requested) async {
    if (_languages.isEmpty) return null;

    bool listed(String code) =>
        _languages.any((l) => l.toLowerCase() == code.toLowerCase());

    if (listed(requested) && await _installed(requested)) return requested;

    // Same language, different region: fil-PH -> fil, en-GB for en-US.
    final prefix = requested.split(RegExp('[-_]')).first.toLowerCase();
    for (final l in _languages) {
      if (l.toLowerCase().startsWith(prefix) && await _installed(l)) return l;
    }
    // Any English that is genuinely present.
    for (final l in _languages) {
      if (l.toLowerCase().startsWith('en') && await _installed(l)) return l;
    }
    // Nothing could be verified — the check itself may be unavailable. Prefer
    // being wrong and audible over being right and silent.
    for (final l in _languages) {
      if (l.toLowerCase().startsWith('en')) return l;
    }
    return _languages.first;
  }

  /// Runs a flutter_tts call with a deadline, returning null if it does not
  /// finish in time.
  ///
  /// Every method on the plugin can hang forever. It queues incoming calls
  /// while its TextToSpeech is initialising and only answers them once that
  /// finishes, so an engine that never comes back leaves those Futures pending
  /// indefinitely. `awaitSpeakCompletion(true)` adds a second route to the same
  /// place: speak() waits for a completion callback a dead engine will never
  /// send.
  ///
  /// Without a deadline this is invisible — the button stops responding and
  /// nothing reaches the log, because every debugPrint sits after the await.
  /// That is exactly how "the speech suddenly stopped working" presented.
  static Future<T?> _withDeadline<T>(
    Future<T> call,
    Duration limit,
    String what,
  ) async {
    try {
      return await call.timeout(limit);
    } on TimeoutException {
      debugPrint('TTS: $what timed out after ${limit.inSeconds}s');
      return null;
    } catch (e) {
      debugPrint('TTS: $what failed: $e');
      return null;
    }
  }

  /// Generous upper bound on how long an utterance may legitimately take, so a
  /// slow engine is never mistaken for a dead one.
  static Duration _speakBudget(String text) => Duration(
      milliseconds: (6000 + text.length * 250).clamp(6000, 45000));

  /// Reads the engine's language list.
  ///
  /// Returns null when the engine did not ANSWER — a timeout or a platform
  /// error — and an empty list only when it answered and had nothing to give.
  ///
  /// Those two were previously collapsed into "empty", and that is what made a
  /// busy engine indistinguishable from an absent one. Under load the 5s
  /// deadline expires, the app concluded the phone has no text-to-speech at
  /// all, and the Initiator showed "No speech voice on this device" on a
  /// handset whose engine was merely congested. Measured: the frame rate falls
  /// to 5.6 fps as the device heats to 41C, the UI isolate saturates, and
  /// every deadline in this file starts expiring together.
  static Future<List<String>?> _readLanguages() async {
    try {
      final langs =
          await _tts.getLanguages.timeout(const Duration(seconds: 5));
      return (langs is List)
          ? langs.map((e) => e.toString()).toList()
          : <String>[];
    } on TimeoutException {
      debugPrint('TTS: getLanguages timed out — engine busy, not absent');
      return null;
    } catch (e) {
      debugPrint('TTS: getLanguages failed: $e');
      return null;
    }
  }

  /// Forces a fresh binding to a TTS engine, and reports whether one exists.
  ///
  /// flutter_tts constructs its Android TextToSpeech once, when the plugin
  /// attaches, and never retries if that construction fails. It logs
  /// "Failed to initialize TextToSpeech with status: -1" and leaves a dead
  /// object behind whose getLanguages returns empty forever — which this class
  /// would otherwise report as "no engine on this device" for the rest of the
  /// session, on a phone that has a perfectly good engine.
  ///
  /// That race is real on phones where the engine process runs on demand
  /// rather than always; Xiaomi's is one. setEngine is the way back: it drops
  /// the dead object and constructs a new one, which rebinds.
  static Future<bool> _rebindEngine() async {
    const limit = Duration(seconds: 5);

    var engine = (await _withDeadline(
            _tts.getDefaultEngine, limit, 'getDefaultEngine'))
        ?.toString();

    if (engine == null || engine.isEmpty) {
      final engines = await _withDeadline(_tts.getEngines, limit, 'getEngines');
      if (engines is List && engines.isNotEmpty) {
        engine = engines.first.toString();
      }
    }
    if (engine == null || engine.isEmpty) return false;

    debugPrint('TTS: rebinding to $engine');
    // setEngine sets the plugin's status back to "initialising", which is what
    // makes every later call queue. If it does not come back, say so rather
    // than leaving the app wedged behind it.
    //
    // Completion is the only success signal here: flutter_tts's setEngine
    // wrapper awaits the channel call without returning its result, so it
    // resolves to null even when the platform side reported success. Testing
    // the value instead of the completion reads every success as a failure.
    try {
      await _tts.setEngine(engine).timeout(limit);
      return true;
    } on TimeoutException {
      debugPrint('TTS: setEngine timed out after ${limit.inSeconds}s');
      return false;
    } catch (e) {
      debugPrint('TTS: setEngine failed: $e');
      return false;
    }
  }

  static Future<void> initTTS({
    String language = 'en-US',
    double? speed,
  }) async {
    if (speed != null) speedMultiplier = speed;
    try {
      // await speak() completing makes behaviour consistent across engines;
      // some return immediately and drop the utterance otherwise.
      await _withDeadline(_tts.awaitSpeakCompletion(true),
          const Duration(seconds: 5), 'awaitSpeakCompletion');

      var langs = await _readLanguages();

      // No answer at all: the engine is busy, not missing.
      //
      // Returning here rather than falling through to the rebind is
      // deliberate. setEngine resets the plugin to "initialising" and makes
      // every queued call wait on it, so rebinding a congested engine adds to
      // the congestion. Leave the door open and let the next utterance try.
      if (langs == null) {
        ttsAvailable = false;
        ttsStatus = 'Text-to-speech is busy — will retry';
        debugPrint('TTS: engine did not answer; treating as busy, not absent');
        return;
      }
      _languages = langs;

      // Answered, but with nothing. Rebind once, then wait for it.
      //
      // On phones where the engine runs on demand, a rebind starts its process
      // from cold and it is not answerable for a second or two. Poll rather
      // than sleeping a fixed time, so the common case — engine already warm —
      // still returns straight away. Rebinding only once matters: each
      // setEngine resets the plugin to "initialising" and would restart the
      // wait it is supposed to be ending.
      if (_languages.isEmpty && await _rebindEngine()) {
        for (var wait = 0; _languages.isEmpty && wait < 6; wait++) {
          await Future<void>.delayed(const Duration(milliseconds: 500));
          langs = await _readLanguages();
          if (langs == null) {
            ttsAvailable = false;
            ttsStatus = 'Text-to-speech is busy — will retry';
            return;
          }
          _languages = langs;
        }
      }

      if (_languages.isEmpty) {
        ttsAvailable = false;
        _ttsInitialised = false;
        ttsStatus = 'No text-to-speech engine on this device';
        debugPrint('TTS: no engine / no languages reported');
        return;
      }

      final resolved = await _resolveLanguage(language);
      if (resolved == null) {
        ttsAvailable = false;
        ttsStatus = 'No usable voice installed';
        return;
      }

      const limit = Duration(seconds: 5);
      await _withDeadline(_tts.setLanguage(resolved), limit, 'setLanguage');
      await _withDeadline(_tts.setSpeechRate(_toPlatformRate(speedMultiplier)),
          limit, 'setSpeechRate');
      await _withDeadline(_tts.setVolume(1.0), limit, 'setVolume');
      await _withDeadline(_tts.setPitch(1.0), limit, 'setPitch');

      // Recorded once so the trace can state it as fact rather than leaving it
      // to be inferred from how the output sounded.
      filipinoInstalled = await _installed('fil-PH') || await _installed('fil');

      ttsLanguage = resolved;
      ttsAvailable = true;
      _ttsInitialised = true;
      ttsStatus = resolved.toLowerCase().startsWith(
              language.split(RegExp('[-_]')).first.toLowerCase())
          ? 'ok'
          : 'Requested $language unavailable — speaking $resolved';
      debugPrint('TTS ready: $resolved  (asked for $language)  '
          'rate ${_toPlatformRate(speedMultiplier)}');
    } catch (e) {
      ttsAvailable = false;
      ttsStatus = 'Text-to-speech failed to start';
      debugPrint('TTS init error: $e');
    }
  }

  /// [speed] is a multiplier where 1.0 is natural pace, not the raw engine
  /// value. Returns false when nothing was spoken.
  static bool _recovering = false;

  /// True when TTS can speak, re-binding first if the engine has gone away.
  ///
  /// The binding does not survive forever on every phone — Xiaomi's process
  /// manager freezes background services aggressively — so "it worked ten
  /// minutes ago" is not evidence it works now.
  static Future<bool> _ensureReady() async {
    if (ttsAvailable && _ttsInitialised) return true;
    if (_recovering) return false;
    _recovering = true;
    try {
      await initTTS(language: ttsLanguage.isEmpty ? 'en-US' : ttsLanguage);
    } finally {
      _recovering = false;
    }
    return ttsAvailable;
  }

  static Future<bool> _speakOnce(String text, double requested) async {
    // Flush whatever Android's TextToSpeech is still holding.
    //
    // flutter_tts hands the utterance to the platform engine, which keeps its
    // OWN queue that no Dart-side guard can see. An utterance the engine could
    // not render — a locale whose voice data is missing — stays in that queue
    // and is emitted the moment the engine becomes able to speak again, which
    // is how a Filipino quick message arrived later attached to the front of an
    // unrelated English sentence. Dropping stale entries on this side did not
    // help, because the copy that eventually spoke was never on this side.
    await _withDeadline(_tts.stop(), const Duration(seconds: 2), 'stop');

    final rate = _toPlatformRate(requested);
    await _withDeadline(_tts.setSpeechRate(rate),
        const Duration(seconds: 5), 'setSpeechRate');

    // awaitSpeakCompletion(true) makes this return when playback ends, so the
    // elapsed time is the real spoken duration. Logging it alongside the
    // requested rate is how you tell "the slider is broken" apart from "this
    // engine ignores setSpeechRate" — the two look identical from the couch.
    final sw = Stopwatch()..start();
    final result =
        await _withDeadline(_tts.speak(text), _speakBudget(text), 'speak');
    sw.stop();
    debugPrint('TTS spoke ${sw.elapsedMilliseconds}ms  '
        'requested ${requested.toStringAsFixed(2)}x  '
        'engine rate $rate  engine=$ttsLanguage  result=$result');

    // flutter_tts returns 1 on success on Android; null means we gave up.
    return result == 1 || result == true;
  }

  /// Whether an utterance is in flight, and the one waiting behind it.
  static bool _speaking = false;
  static String? _pendingText;
  static double? _pendingSpeed;
  static int _pendingAtMs = 0;

  /// How stale a queued utterance may be before it is dropped.
  ///
  /// A failing engine can hold a speak() call for its whole 6-45s budget, and
  /// whatever queued behind it would then be spoken long after the moment it
  /// belonged to — reported on device as a quick message that stayed silent
  /// when tapped and then arrived glued to the front of the NEXT sentence. For
  /// a communication aid, saying nothing is better than saying something the
  /// user asked for twenty seconds ago.
  static const int _pendingMaxAgeMs = 4000;

  /// When a failed attempt allows the next one.
  static int _busyUntilMs = 0;
  static const int _busyBackoffMs = 8000;

  static void _markBusy() {
    _busyUntilMs = DateTime.now().millisecondsSinceEpoch + _busyBackoffMs;
  }

  static Future<bool> speak(String text, {double? speed}) async {
    if (text.trim().isEmpty) return false;

    // ONE utterance at a time.
    //
    // speak() is driven by the phrase timer and the frame handler, so a new
    // phrase routinely begins while the previous one is still being spoken.
    // Android's TextToSpeech flushes its queue on a second speak(), so the
    // first utterance's completion callback never arrives — and with
    // awaitSpeakCompletion(true) that Future then sits pending until the
    // 6-45s deadline, after which a perfectly healthy engine is marked dead.
    // That is the path by which the Initiator lost its voice mid-session while
    // the Hub, whose speak() is user-triggered and awaited one at a time,
    // carried on working with the same static engine.
    //
    // A single pending slot rather than a queue: during a burst the signer has
    // already moved on, so the newest phrase is the one worth speaking and a
    // backlog would only fall further behind.
    if (_speaking) {
      _pendingText = text;
      _pendingSpeed = speed;
      _pendingAtMs = DateTime.now().millisecondsSinceEpoch;
      return true;
    }

    // Back off after a failure instead of retrying on the next phrase.
    //
    // Each failed attempt spends several seconds of platform calls on the UI
    // isolate — the same isolate running interpreter.run() and marshalling
    // every camera frame — so retrying on every phrase while the engine is
    // congested slows the pipeline that congested it. The backoff breaks that
    // loop.
    if (DateTime.now().millisecondsSinceEpoch < _busyUntilMs) {
      debugPrint('TTS: within backoff window, not speaking');
      return false;
    }

    _speaking = true;
    try {
      return await _speakSerialised(text, speed);
    } finally {
      _speaking = false;
      final next = _pendingText;
      final nextSpeed = _pendingSpeed;
      final age = DateTime.now().millisecondsSinceEpoch - _pendingAtMs;
      _pendingText = null;
      _pendingSpeed = null;
      if (next != null && age <= _pendingMaxAgeMs) {
        unawaited(speak(next, speed: nextSpeed));
      } else if (next != null) {
        debugPrint('TTS: dropped a queued utterance ${age}ms old');
      }
    }
  }

  static Future<bool> _speakSerialised(String text, double? speed) async {
    if (!await _ensureReady()) {
      debugPrint('TTS unavailable, not speaking: $ttsStatus');
      _markBusy();
      return false;
    }

    // Cheap liveness check before committing to an utterance. A binding whose
    // engine has gone away still looks healthy to the plugin, and speaking
    // into it hangs until the deadline — ten seconds of a dead button. Asking
    // for the language list costs milliseconds and a dead binding answers with
    // an empty one, so the recovery starts almost immediately instead.
    final live = await _readLanguages();
    if (live == null) {
      // No answer: congested. Spending a 6-45s speak deadline on it would
      // only hold the UI isolate longer, so report and back off instead.
      ttsAvailable = false;
      ttsStatus = 'Text-to-speech is busy — will retry';
      _markBusy();
      return false;
    }
    if (live.isEmpty) {
      debugPrint('TTS: binding is dead before speaking; recovering first');
      ttsAvailable = false;
      _ttsInitialised = false;
      if (!await _ensureReady()) {
        _markBusy();
        return false;
      }
      await Future<void>.delayed(const Duration(milliseconds: 400));
    }

    final requested = speed ?? speedMultiplier;
    if (await _speakOnce(text, requested)) return true;

    // One retry against a freshly bound engine. Beyond that, report honestly
    // rather than looping.
    debugPrint('TTS: speak did not succeed; rebinding and retrying');
    ttsAvailable = false;
    _ttsInitialised = false;
    if (!await _ensureReady()) {
      _markBusy();
      return false;
    }

    // A freshly rebound engine answers getLanguages before it is willing to
    // speak: handed an utterance immediately, it returns 0 without producing
    // audio. Give it a moment, and allow a second go before giving up.
    for (var attempt = 0; attempt < 2; attempt++) {
      await Future<void>.delayed(const Duration(milliseconds: 400));
      if (await _speakOnce(text, requested)) return true;
    }

    ttsAvailable = false;
    _ttsInitialised = false;
    ttsStatus = 'Text-to-speech stopped responding';
    _markBusy();
    return false;
  }

  /// Dump of what the device actually offers. Call this when diagnosing a
  /// phone where speech does not work; the answer is almost always a missing
  /// engine or missing voice data rather than anything in this app.
  static Future<String> ttsDiagnostics() async {
    final out = StringBuffer('=== TTS diagnostics ===');
    out.writeln();
    try {
      final engines = await _tts.getEngines;
      final def = await _tts.getDefaultEngine;
      final langs = await _tts.getLanguages;
      final voices = await _tts.getVoices;
      out.writeln('engines: $engines');
      out.writeln('default engine: $def');
      out.writeln('languages: $langs');
      out.writeln('voice count: ${voices is List ? voices.length : "?"}');
      out.writeln('en-US available: ${await _tts.isLanguageAvailable("en-US")}');
      out.writeln('fil-PH available: ${await _tts.isLanguageAvailable("fil-PH")}');
    } catch (e) {
      out.writeln('diagnostics failed: $e');
    }
    out.writeln('available=$ttsAvailable  using=$ttsLanguage  status=$ttsStatus');
    final s = out.toString();
    debugPrint(s);
    return s;
  }

  static Future<void> stopTTS() async {
    await _tts.stop();
  }

  static Future<void> setTTSLanguage(String lang) async {
    final wanted = lang == 'fil' ? 'fil-PH' : 'en-US';
    if (!_ttsInitialised) {
      await initTTS(language: wanted);
      return;
    }
    final resolved = await _resolveLanguage(wanted);
    if (resolved == null) return;
    await _tts.setLanguage(resolved);
    ttsLanguage = resolved;
    if (resolved.toLowerCase().startsWith(
        wanted.split(RegExp('[-_]')).first.toLowerCase())) {
      ttsStatus = 'ok';
    } else {
      ttsStatus = 'Requested $wanted unavailable — speaking $resolved';
    }
  }


  static void dispose() {
    _tts.stop();
  }
}
