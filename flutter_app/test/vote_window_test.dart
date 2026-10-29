import 'package:flutter_test/flutter_test.dart';
import 'package:siklab_kumpas/services/inference_service.dart';

/// The letter-vote window is the only behavioural change in the thermal work,
/// so the guarantee it was accepted under is pinned here: at any frame rate the
/// app reaches while healthy, it must behave EXACTLY as before.
void main() {
  tearDown(() => InferenceService.processingFps = 0);

  group('effectiveVoteMaxAgeMs', () {
    test('is unchanged at healthy frame rates', () {
      // 13.2 fps was measured on device at launch, 5.6 fps after five minutes
      // of heating. Both must keep the original fixed window.
      for (final fps in [13.2, 8.0, 5.6, 4.0, 3.4]) {
        InferenceService.processingFps = fps;
        expect(InferenceService.effectiveVoteMaxAgeMs,
            InferenceService.letterVoteMaxAgeMs,
            reason: '$fps fps must not change the window');
      }
    });

    test('is unchanged before any frame has been timed', () {
      InferenceService.processingFps = 0;
      expect(InferenceService.effectiveVoteMaxAgeMs,
          InferenceService.letterVoteMaxAgeMs);
    });

    test('widens only below the rate that can supply the required votes', () {
      // At 2.5 fps the fixed 1500ms window holds 3 votes, one short of the 4
      // required, so no letter could ever be committed. Widening to 2000ms
      // restores exactly enough room and no more.
      InferenceService.processingFps = 2.5;
      expect(InferenceService.effectiveVoteMaxAgeMs, 2000);

      InferenceService.processingFps = 2.0;
      expect(InferenceService.effectiveVoteMaxAgeMs, 2500);
    });

    test('always holds at least the required number of votes', () {
      // The property that matters: whatever the frame rate, the window must be
      // long enough for letterVotesRequired frames to fit inside it. Below
      // ~2.7 fps the old fixed window failed this, which is why fingerspelling
      // went silent rather than merely slow.
      for (final fps in [1.5, 2.0, 2.5, 3.0, 3.5, 5.0, 10.0, 15.0]) {
        InferenceService.processingFps = fps;
        final votesThatFit =
            InferenceService.effectiveVoteMaxAgeMs / 1000 * fps;
        expect(votesThatFit,
            greaterThanOrEqualTo(InferenceService.letterVotesRequired.toDouble()),
            reason: 'at $fps fps only $votesThatFit votes fit in the window');
      }
    });
  });
}
