import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:vosk_flutter_service/vosk_flutter_service.dart';

/// On-device speech recognition, using a bundled Vosk model.
///
/// WHY THIS EXISTS
///
/// Every system recognition path on the target device failed, measured in
/// order:
///
///   Google RecognitionService (direct bind)   error_too_many_requests
///   Google TTS RecognitionService             error_language_unavailable
///   MIUI AsrActivity (system dialog)          errors and closes
///   Google voice dialog (explicit package)    "voice search isn't available"
///
/// The first is Google throttling the client and does not recover with
/// spacing; the rest are device configuration we cannot change. Speech input
/// is a core feature of this app, so it cannot depend on any of them.
///
/// Running the model in-process removes the dependency entirely, and restores
/// something the system paths had taken away: the app no longer sends audio
/// anywhere. Every other component — camera, sign models, avatar, TTS — was
/// already on-device, and speech-to-text was the one exception.
///
/// Costs ~40MB of APK for the small English model.
class OfflineSpeechService {
  static const _modelAsset = 'assets/models/vosk-model-small-en-us-0.15.zip';
  static const _sampleRate = 16000;

  static VoskFlutterPlugin? _vosk;
  static Model? _model;
  static Recognizer? _recognizer;
  static SpeechService? _service;

  static bool _ready = false;
  static bool _loading = false;
  static String? _loadError;

  static bool get isReady => _ready;
  static String? get loadError => _loadError;

  /// Unpacks and loads the model. Safe to call repeatedly.
  ///
  /// The first call costs a second or two — the model is unzipped out of the
  /// bundle — so it is triggered when the Voice tab opens rather than on the
  /// first tap, or the microphone appears to hang on its first use.
  static Future<bool> initialize() async {
    if (_ready) return true;
    if (_loading) {
      while (_loading) {
        await Future<void>.delayed(const Duration(milliseconds: 100));
      }
      return _ready;
    }
    _loading = true;
    try {
      _vosk ??= VoskFlutterPlugin.instance();
      final path = await ModelLoader().loadFromAssets(_modelAsset);
      _model = await _vosk!.createModel(path);
      _recognizer = await _vosk!.createRecognizer(
        model: _model!,
        sampleRate: _sampleRate,
      );
      _ready = true;
      _loadError = null;
      debugPrint('Vosk model ready');
      return true;
    } catch (e) {
      _loadError = '$e';
      _ready = false;
      debugPrint('Vosk load failed: $e');
      return false;
    } finally {
      _loading = false;
    }
  }

  /// Listens until [onResult] fires or [timeout] elapses.
  ///
  /// Partial results are surfaced through [onPartial] so the UI can show words
  /// appearing as they are spoken, which is what makes it feel responsive —
  /// the final result only arrives once the speaker stops.
  static Future<void> listen({
    required void Function(String text) onResult,
    void Function(String partial)? onPartial,
    void Function(String reason)? onFailure,
    Duration timeout = const Duration(seconds: 15),
  }) async {
    if (!await initialize()) {
      onFailure?.call('Offline speech model failed to load');
      return;
    }

    try {
      await stop();
      _service = await _vosk!.initSpeechService(_recognizer!);

      StreamSubscription<String>? partialSub;
      StreamSubscription<String>? resultSub;
      Timer? deadline;
      var finished = false;

      Future<void> done(String? text, String? failure) async {
        if (finished) return;
        finished = true;
        deadline?.cancel();
        await partialSub?.cancel();
        await resultSub?.cancel();
        await stop();
        if (text != null && text.isNotEmpty) {
          onResult(text);
        } else {
          onFailure?.call(failure ?? '');
        }
      }

      partialSub = _service!.onPartial().listen((p) {
        final t = _extract(p, 'partial');
        if (t.isNotEmpty) onPartial?.call(t);
      });

      resultSub = _service!.onResult().listen((r) {
        final t = _extract(r, 'text');
        if (t.isNotEmpty) done(t, null);
      });

      // Vosk emits a final result only when it detects a pause, so a speaker
      // who trails off without one would otherwise leave this listening
      // forever.
      deadline = Timer(timeout, () => done(null, 'No speech heard'));

      await _service!.start(
        onRecognitionError: (dynamic e) {
          debugPrint('Vosk recognition error: $e');
          done(null, 'Speech recognition failed');
        },
      );
    } catch (e) {
      debugPrint('Vosk listen failed: $e');
      onFailure?.call('Speech recognition failed');
      await stop();
    }
  }

  /// Stops AND releases the session.
  ///
  /// stop() alone is not enough: the native SpeechService keeps the microphone
  /// and the recogniser keeps the previous utterance, so the next
  /// initSpeechService fails and every attempt after the first reported
  /// "speech recognition failed". Disposing the service and resetting the
  /// recogniser returns both to the state they were in before the first use.
  static Future<void> stop() async {
    final svc = _service;
    _service = null;
    if (svc != null) {
      try {
        await svc.stop();
      } catch (_) {
        // Was not running.
      }
      try {
        await svc.dispose();
      } catch (_) {
        // Already released.
      }
    }
    try {
      // Clears accumulated audio, so the next utterance does not inherit the
      // tail of the last one.
      await _recognizer?.reset();
    } catch (_) {
      // Recogniser not yet built.
    }
  }

  /// Vosk returns JSON: {"text": "..."} or {"partial": "..."}.
  static String _extract(String raw, String key) {
    try {
      final j = jsonDecode(raw);
      if (j is Map && j[key] is String) return (j[key] as String).trim();
    } catch (_) {
      // Not JSON: some builds emit the bare string.
      return raw.trim();
    }
    return '';
  }
}
