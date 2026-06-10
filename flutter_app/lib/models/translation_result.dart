class TranslationResult {
  final String label;
  final double confidence;
  final DateTime timestamp;

  TranslationResult({
    required this.label,
    required this.confidence,
    required this.timestamp,
  });

  bool get isHighConfidence => confidence >= 0.7;

  String get confidencePercent =>
      '${(confidence * 100).toStringAsFixed(1)}%';
}