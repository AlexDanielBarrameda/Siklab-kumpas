import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:siklab_kumpas/services/features_v3.dart';

/// Golden-vector test against the Python implementation.
///
/// The model is trained on whatever features_v3.py produces. If this port
/// diverges, inference runs in a different feature space than training, and the
/// symptom is not a crash but quietly degraded accuracy -- the same class of
/// failure that once took the word model from 99.0% to 13.5%.
///
/// Regenerate the fixture with:
///   python -c "import sys,json,numpy as np; \
///     sys.path.insert(0,'ml_pipeline/preprocessing'); \
///     from features_v3 import build, trim_idle; ..."
void main() {
  late List<List<List<double>>> input;
  late List<List<List<double>>> trimmed;
  late List<List<List<double>>> expected;

  setUpAll(() {
    final f = File('test/features_v3_golden.json');
    expect(f.existsSync(), isTrue,
        reason: 'missing features_v3_golden.json — regenerate from Python');
    final j = jsonDecode(f.readAsStringSync()) as Map<String, dynamic>;

    List<List<List<double>>> parse(String key) => (j[key] as List)
        .map((clip) => (clip as List)
            .map((frame) =>
                (frame as List).map((x) => (x as num).toDouble()).toList())
            .toList())
        .toList();

    input = parse('input');
    trimmed = parse('trimmed');
    expected = parse('expected');
  });

  test('trimIdle matches the Python implementation', () {
    for (int c = 0; c < input.length; c++) {
      final got = FeaturesV3.trimIdle(input[c]);
      expect(got.length, trimmed[c].length);
      double worst = 0.0;
      for (int t = 0; t < got.length; t++) {
        for (int i = 0; i < got[t].length; i++) {
          final d = (got[t][i] - trimmed[c][t][i]).abs();
          if (d > worst) worst = d;
        }
      }
      expect(worst, lessThan(1e-5),
          reason: 'clip $c: trimIdle diverged by $worst');
    }
  });

  test('build matches the Python implementation', () {
    double worst = 0.0;
    var worstDim = -1;
    for (int c = 0; c < trimmed.length; c++) {
      final got = FeaturesV3.build(trimmed[c]);
      expect(got.length, expected[c].length);
      expect(got[0].length, FeaturesV3.dim);
      for (int t = 0; t < got.length; t++) {
        for (int i = 0; i < FeaturesV3.dim; i++) {
          final d = (got[t][i] - expected[c][t][i]).abs();
          if (d > worst) {
            worst = d;
            worstDim = i;
          }
        }
      }
    }
    // Float32 in numpy vs float64 in Dart, so exact equality is not the bar;
    // anything above 1e-4 means a real logic difference, not rounding.
    expect(worst, lessThan(1e-4),
        reason: 'max divergence $worst at feature index $worstDim');
  });

  test('block boundaries are where the model expects them', () {
    final got = FeaturesV3.build(trimmed[0]);
    // flags are the only block with a known exact value, so they pin the
    // layout: if any earlier block changed width, these land elsewhere.
    for (final row in got) {
      expect(row[332] == 0.0 || row[332] == 1.0, isTrue);
      expect(row[333] == 0.0 || row[333] == 1.0, isTrue);
    }
  });
}
