import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import '../models/translation_result.dart';

class InferenceService {
  static const _channel = MethodChannel('com.siklabkumpas/inference');

  static List<String> _fsl105Labels = [];
  static List<String> _lettersLabels = [];
  static bool _isInitialized = false;
  static bool _isProcessing = false;

  static const int sequenceLength = 30;
  static const int landmarkDim = 226;
  static const double confidenceThreshold = 0.6;

  // Rolling buffer for FSL-105
  static final List<List<double>> _frameBuffer = [];

  // Cooldown to avoid repeating same result
  static String _lastLabel = '';
  static DateTime _lastResultTime = DateTime(0);

  static Future<void> initialize() async {
    if (_isInitialized) return;
    try {
      _fsl105Labels = await _loadLabels('assets/labels/fsl105_labels.txt');
      _lettersLabels =
          await _loadLabels('assets/labels/fsl_letters_labels.txt');
      _isInitialized = true;
      debugPrint('InferenceService initialized: '
          '${_fsl105Labels.length} word labels, '
          '${_lettersLabels.length} letter labels');
    } catch (e) {
      debugPrint('InferenceService init error: $e');
    }
  }

  static Future<List<String>> _loadLabels(String assetPath) async {
    final raw = await rootBundle.loadString(assetPath);
    final lines = raw.trim().split('\n');
    final labels = <String>[];
    for (final line in lines) {
      final parts = line.split(',');
      if (parts.length >= 2) {
        labels.add(parts.sublist(1).join(',').trim());
      }
    }
    return labels;
  }

  static Future<TranslationResult?> processFrame(
      CameraImage image) async {
    if (_isProcessing) return null;
    _isProcessing = true;

    try {
      // Get the Y plane bytes (luminance only — faster to transfer)
      final plane = image.planes[0];
      final bytes = plane.bytes;

      final landmarks = await _channel.invokeMethod<List<dynamic>>(
        'processFrame',
        {
          'bytes': bytes,
          'width': image.width,
          'height': image.height,
          'rotation': 0,
        },
      );

      if (landmarks == null || landmarks.isEmpty) return null;

      final lmList =
          landmarks.map((e) => (e as num).toDouble()).toList();

      // Add to rolling buffer
      _frameBuffer.add(lmList);
      if (_frameBuffer.length > sequenceLength) {
        _frameBuffer.removeAt(0);
      }

      // Only run inference when buffer is full
      if (_frameBuffer.length < sequenceLength) return null;

      // Check if any landmarks were detected (non-zero values)
      final hasLandmarks = lmList.any((v) => v != 0.0);
      if (!hasLandmarks) return null;

      return null; // TFLite inference will be added next

    } catch (e) {
      debugPrint('processFrame error: $e');
      return null;
    } finally {
      _isProcessing = false;
    }
  }

  static void clearBuffer() => _frameBuffer.clear();

  static void dispose() {
    _frameBuffer.clear();
    _isInitialized = false;
  }
}