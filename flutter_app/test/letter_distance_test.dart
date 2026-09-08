import 'package:flutter_test/flutter_test.dart';
import 'package:siklab_kumpas/services/inference_service.dart';

/// The threshold was set from two measured runs on the same device minutes
/// apart. These pin the separation it was drawn from, so a later change to
/// minLetterExtentPx has to face the data that justified this one.
void main() {
  tearDown(() => InferenceService.lastHandExtentPx = 0);

  // Every letter committed while the app was emitting P/Q/O and spelling
  // nothing.
  const garbageRun = [22.2, 55.2, 60.7];

  // Every letter committed while SIKLAB and similar spelled correctly.
  const workingRun = [
    63.6, 117.5, 72.3, 90.5, 99.5, 97.8, 88.8, 96.9, 103.1, 113.3, 73.3, 118.6
  ];

  test('every garbage-run reading is rejected', () {
    for (final px in garbageRun) {
      InferenceService.lastHandExtentPx = px;
      expect(InferenceService.handTooFarForLetters, isTrue,
          reason: '${px}px produced nonsense on device');
    }
  });

  test('every working-run reading is accepted', () {
    for (final px in workingRun) {
      InferenceService.lastHandExtentPx = px;
      expect(InferenceService.handTooFarForLetters, isFalse,
          reason: '${px}px spelled correctly on device');
    }
  });

  test('the threshold sits inside the gap between the two runs', () {
    final worstGood = workingRun.reduce((a, b) => a < b ? a : b);
    final bestBad = garbageRun.reduce((a, b) => a > b ? a : b);
    expect(bestBad, lessThan(InferenceService.minLetterExtentPx));
    expect(worstGood, greaterThan(InferenceService.minLetterExtentPx));
  });

  test('no measurement means no gate — the crop never ran', () {
    // extent 0 is "no hand crop happened", not "a tiny hand". Gating on it
    // would suppress the full-frame path that words and the first letter
    // acquisition both use.
    InferenceService.lastHandExtentPx = 0;
    expect(InferenceService.handTooFarForLetters, isFalse);
  });
}
