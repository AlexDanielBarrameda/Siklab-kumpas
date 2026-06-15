import 'package:flutter/foundation.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;

class SpeechService {
  static final FlutterTts _tts = FlutterTts();
  static final stt.SpeechToText _stt = stt.SpeechToText();
  static bool _sttInitialized = false;

  static Future<void> initTTS({
    String language = 'en-US',
    double rate = 1.0,
  }) async {
    try {
      await _tts.setLanguage(language);
      await _tts.setSpeechRate(rate);
      await _tts.setVolume(1.0);
      await _tts.setPitch(1.0);
      debugPrint('TTS initialized successfully');
    } catch (e) {
      debugPrint('TTS init error: $e');
    }
  }

  static Future<void> speak(String text, {double rate = 1.0}) async {
    try {
      debugPrint('Speaking: $text');
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