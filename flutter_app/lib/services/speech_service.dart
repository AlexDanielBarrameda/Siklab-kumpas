import 'package:flutter/foundation.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;

class SpeechService {
  static final FlutterTts _tts = FlutterTts();
  static final stt.SpeechToText _stt = stt.SpeechToText();
  static bool _sttInitialized = false;

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

  static Future<void> initTTS({
    String language = 'en-US',
    double? speed,
  }) async {
    try {
      if (speed != null) speedMultiplier = speed;
      await _tts.setLanguage(language);
      await _tts.setSpeechRate(_toPlatformRate(speedMultiplier));
      await _tts.setVolume(1.0);
      await _tts.setPitch(1.0);
      debugPrint('TTS initialized (speed ${speedMultiplier}x -> '
          '${_toPlatformRate(speedMultiplier)})');
    } catch (e) {
      debugPrint('TTS init error: $e');
    }
  }

  /// [speed] is a multiplier where 1.0 is natural pace, not the raw engine
  /// value. Omit it to use [speedMultiplier].
  static Future<void> speak(String text, {double? speed}) async {
    try {
      final rate = _toPlatformRate(speed ?? speedMultiplier);
      debugPrint('Speaking (${speed ?? speedMultiplier}x -> $rate): $text');
      await _tts.setSpeechRate(rate);
      final result = await _tts.speak(text);
      debugPrint('TTS speak result: $result');
    } catch (e) {
      debugPrint('TTS speak error: $e');
    }
  }

  static Future<void> stopTTS() async {
    await _tts.stop();
  }

  static void setTTSLanguage(String lang) {
    final locale = lang == 'fil' ? 'fil-PH' : 'en-US';
    _tts.setLanguage(locale);
  }

  static Future<bool> initSTT() async {
    if (_sttInitialized) return true;
    try {
      debugPrint('Initializing STT...');
      _sttInitialized = await _stt.initialize();
      debugPrint('STT initialized: $_sttInitialized');
      debugPrint('STT available: ${_stt.isAvailable}');
    } catch (e) {
      debugPrint('STT init error: $e');
    }
    return _sttInitialized;
  }

  static Future<void> startListening({
    required Function(String text) onResult,
    String lang = 'en',
  }) async {
    if (!_sttInitialized) await initSTT();
    debugPrint('STT available: ${_stt.isAvailable}');
    debugPrint('STT listening: ${_stt.isListening}');
    if (!_stt.isAvailable) {
      debugPrint('STT not available on this device');
      return;
    }

    final locale = lang == 'fil' ? 'fil-PH' : 'en-US';
    debugPrint('Starting STT with locale: $locale');

    await _stt.listen(
      onResult: (result) {
        debugPrint('STT result: ${result.recognizedWords} final: ${result.finalResult}');
        if (result.finalResult) {
          onResult(result.recognizedWords);
        }
      },
      listenOptions: stt.SpeechListenOptions(
        localeId: locale,
        listenMode: stt.ListenMode.confirmation,
        cancelOnError: true,
      ),
    );
  }

  static Future<void> stopListening() async {
    await _stt.stop();
  }

  static bool get isListening => _stt.isListening;

  static void dispose() {
    _tts.stop();
    _stt.stop();
  }
}