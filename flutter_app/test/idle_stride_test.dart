import 'package:flutter_test/flutter_test.dart';
import 'package:siklab_kumpas/services/camera_service.dart';
import 'package:siklab_kumpas/services/inference_service.dart';
import 'package:siklab_kumpas/services/phrase_service.dart';

/// Idle frame-skipping trades sign onset for heat, and the two invariants that
/// make that trade safe are pinned here rather than left to the comments.
void main() {
  tearDown(() {
    CameraService.setIdle(false);
    InferenceService.clearBuffer();
  });

  test('idle can never begin inside a phrase', () {
    // A phrase is signs with hands DOWN between them, and the phrase timer
    // waits phraseGapMs for the next sign. Idling before that elapses would
    // put the pipeline to sleep exactly when the next sign is due.
    expect(InferenceService.idleAfterMs,
        greaterThan(PhraseService.phraseGapMs));
  });

  test('the idle stride only ever halves the active rate', () {
    // Onset cost is idleStride / camera fps; doubling the stride is the
    // measured trade that was accepted. Anything more aggressive is a new
    // decision, not a tuning.
    expect(CameraService.idleStride, 2 * CameraService.activeStride);
    expect(CameraService.activeStride, 2,
        reason: 'must equal the previous constant _frameStride exactly, '
            'or ASL behaviour with hands present would change');
  });

  test('CameraService reports the stride it is actually on', () {
    CameraService.setIdle(true);
    expect(CameraService.isIdle, isTrue);
    CameraService.setIdle(false);
    expect(CameraService.isIdle, isFalse);
  });

  test('every reset wakes both sides together', () {
    // clearBuffer runs on stop/start, mode switch, camera switch and the
    // watchdog. Each must leave InferenceService and CameraService agreeing,
    // or the badge would say one thing and the stride do another.
    CameraService.setIdle(true);
    InferenceService.isIdle = true;
    InferenceService.clearBuffer();
    expect(InferenceService.isIdle, isFalse);
    expect(CameraService.isIdle, isFalse);
  });
}
