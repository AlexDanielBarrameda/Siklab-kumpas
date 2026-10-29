import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import '../models/translation_result.dart';
import 'camera_service.dart';
import 'trace_service.dart';
import 'features_v3.dart';

/// Recognition modes, in the order the badge cycles them.
///
/// ASL first and by default: it is the primary language of the app, has the
/// adapted 56-word model behind it, and is what a demonstration opens on.
/// FSL words are last — that model is trained on a single signer and does not
/// generalise, so it should never be what a user lands on by accident.
///
/// Order matters only here: every model is selected by NAME below, so this
/// list defines the cycle and nothing else.
enum SignMode { aslWords, letters, fslWords }

/// One extracted frame plus the moment it was captured.
class _TimedFrame {
  final int t;
  final List<double> data;
  const _TimedFrame(this.t, this.data);
}

/// One frame's letter prediction, for temporal voting.
class _Vote {
  final int t;
  final String label;
  final double confidence;
  const _Vote(this.t, this.label, this.confidence);
}

class InferenceService {
  static const _channel = MethodChannel('com.siklabkumpas/inference');

  static Interpreter? _fsl105Interpreter;
  static Interpreter? _lettersInterpreter;
  static List<String> _fsl105Labels = [];
  static List<String> _lettersLabels = [];
  static Interpreter? _aslInterpreter;
  static List<String> _aslLabels = [];
  static bool _isInitialized = false;
  static bool _isProcessing = false;

  static const int sequenceLength = 30;
  static const int landmarkDim = 226;
  static const double confidenceThreshold = 0.6;

  /// The ASL v4 model is trained with label smoothing 0.1, which caps the
  /// achievable softmax near 0.9 and shifts the whole confidence distribution
  /// down. Reusing 0.6 there rejects predictions that are correct: measured on
  /// the signer-disjoint test split at the device capture rate,
  ///
  ///     0.60   shown 47.10%   coverage 52.24%   precision 90.16%
  ///     0.35   shown 61.13%   coverage 73.55%   precision 83.11%
  ///
  /// At 0.6 the app stays silent on nearly half of all signs, which is the
  /// complaint that started this work. 0.35 trades 7 points of precision for
  /// 21 points of coverage.
  static const double aslConfidenceThreshold = 0.35;

  static double get activeConfidenceThreshold =>
      mode == SignMode.aslWords ? aslConfidenceThreshold : confidenceThreshold;

  /// Temporal voting for fingerspelling. A letter is announced only when
  /// [letterVotesRequired] of the last [letterVoteWindow] frames agree.
  static const int letterVoteWindow = 5;
  static const int letterVotesRequired = 4;
  static const int letterVoteMaxAgeMs = 1500;

  /// Age limit actually applied to votes, widened only when the frame rate is
  /// too low to supply [letterVotesRequired] of them inside the fixed window.
  ///
  /// The fixed 1500ms silently caps how many votes can ever be held at once:
  /// the buffer holds `fps * 1.5` of them, so below about 2.7 fps it holds
  /// three, the four-vote bar can never be met, and fingerspelling produces
  /// NOTHING however well it is signed. That is not degradation, it is an
  /// unsatisfiable rule — and it presents exactly like a dead camera.
  ///
  /// Measured on device, the frame rate falls from 13.2 fps to 5.6 fps as the
  /// phone heats from 36.9C to 41.3C over five minutes of continuous capture,
  /// so this is a regime the app reaches on its own during a demonstration.
  ///
  /// Widened by the MINIMUM needed to hold the required votes plus one frame
  /// of slack. Above ~3.3 fps the computed value is below 1500ms and the fixed
  /// window still wins, so behaviour there is unchanged — this only ever
  /// applies where the alternative is silence.
  static int get effectiveVoteMaxAgeMs {
    final fps = processingFps;
    if (fps <= 0) return letterVoteMaxAgeMs;
    final needed = ((letterVotesRequired + 1) / fps * 1000).round();
    return needed > letterVoteMaxAgeMs ? needed : letterVoteMaxAgeMs;
  }

  /// Why a buffer was too short to classify.
  ///
  /// "Sign was too quick" is wrong when the app is the slow one: at 2.5 fps a
  /// normal 1.2s sign yields 3 frames and is rejected, and telling the signer
  /// to slow down blames them for the phone throttling. [wordMinFrames] is
  /// deliberately NOT lowered to compensate — measured accuracy is 55.3% at
  /// four frames and 41.3% at three, so classifying from three produces wrong
  /// answers rather than silence, which is worse.
  static String _tooFewFramesMessage(int n) {
    final slow = processingFps > 0 && processingFps < 5.0;
    final base = slow
        ? "> Running slowly — hold each sign a little longer"
        : "> Sign was too quick — hold it a moment longer";
    TraceService.log('rejected  too few frames  n=$n  '
        'fps=${processingFps.toStringAsFixed(1)}  slow=$slow');
    return kDebugMode
        ? "$base  ($n frames, ${processingFps.toStringAsFixed(1)} fps)"
        : base;
  }

  /// Confidence a letter must reach to be committed.
  ///
  /// Much higher than the word threshold, and deliberately so. Fingerspelling
  /// has no posture gate — it is a continuous stream with no rest position —
  /// so the ONLY thing standing between a spurious hand detection and a
  /// committed letter is this number. At 0.6 the back camera spelled
  /// "qqapqqc" from whatever hand-like shapes it found in the background,
  /// with the user not signing at all.
  ///
  /// A 26-class softmax clears 0.6 on noise readily; 0.88 across four of the
  /// last five frames does not. The cost is that a genuine letter must be held
  /// slightly longer, which is the right trade when the alternative is the
  /// textbox filling itself during a demo.
  static const double letterConfidenceThreshold = 0.82;

  /// Mean confidence of the leading letter on the last frame, shown on screen.
  ///
  /// Without it the only visible signal was the vote count, so a letter that
  /// had 5 of 5 votes and simply missed the confidence bar looked identical to
  /// one that was not being seen at all — which is exactly what happened on
  /// the back camera at 0.88.
  static double lastLetterMean = 0;

  static final List<_Vote> _letterVotes = [];

  /// Whether the scene has shown NO hand at least once since letters began.
  ///
  /// A static background object detected as a hand produces PERFECTLY
  /// consistent votes, which is exactly what the [letterVotesRequired]-of-
  /// [letterVoteWindow] rule rewards, and a clean shape clears the 0.82 gate
  /// without difficulty. Voting defends against flicker; it has no defence
  /// against something that simply never moves.
  ///
  /// Observed on device: switching to letters on the BACK camera with the
  /// signer motionless spelled "QOQOQQ" out of the room behind them, and
  /// spelling then worked normally on the very next attempt.
  ///
  /// Requiring an empty frame before any letter is accepted discards whatever
  /// happened to be in view at the moment of the switch. It cannot reject a
  /// real letter: a signer always has their hands down before they start
  /// spelling, so the gate is open by the time the first handshape forms.
  static bool _sawEmptyFrame = false;

  /// Fingerspelling buffer: letters accumulated into a word.
  ///
  /// This is the only path that can produce names and proper nouns — no word
  /// model contains them, whatever its vocabulary — so it matters more than its
  /// size suggests.
  static String spelledWord = '';

  /// A word is finished when the hands LEAVE FRAME for this long — not when no
  /// new letter has arrived.
  ///
  /// Counting from the last committed letter cut words short: at 4-5 fps,
  /// voting alone costs ~0.7s before the signer has even finished forming the
  /// next handshape, so "SIKLAB" ended at "SIK". Hands-down is also the same
  /// gesture that ends a word in WORDS mode, so the two modes behave alike.
  static const int spellIdleMs = 2500;
  static int _lastHandAt = 0;

  /// The last letter committed, and whether the handshape has changed since.
  ///
  /// Double letters are the awkward case: holding one handshape must not spam
  /// L-L-L-L, but "HELLO" genuinely needs two Ls. So the same letter can only
  /// repeat after the recogniser has seen something else in between — which is
  /// what lowering and re-forming the hand produces, and roughly what signers
  /// do anyway.
  static String _lastCommitted = '';
  static int _gapFrames = spellRepeatGapFrames;

  /// Frames without a confident reading of the SAME letter before it may repeat.
  ///
  /// A single gap frame was far too lax: moving between handshapes produces
  /// brief drops constantly, so holding K through a transition committed it
  /// twice. Requiring a sustained gap is the same fix the posture exit needed —
  /// a momentary dip is not the same as actually stopping.
  static const int spellRepeatGapFrames = 4;

  /// Set when a word is completed, for the UI to display.
  static String? completedWord;

  /// How the completed word should be SPOKEN.
  ///
  /// Kept separate from [completedWord] because the two want different casing.
  /// The strip shows S I K L A B in caps, which reads correctly as
  /// fingerspelling — but Android's TTS treats an unfamiliar all-caps token as
  /// an initialism and reads it out letter by letter. "DAN" and "BRUCE" are in
  /// its lexicon so they survive; "SIKLAB" is not, and came out spelled.
  /// Title case makes the engine pronounce it phonetically instead.
  static String? completedWordSpoken;

  /// Title-cases a label so the TTS engine pronounces it instead of spelling it.
  ///
  /// Android reads an unfamiliar ALL-CAPS token as an initialism, so the word
  /// labels — which are stored capitalised — came out letter by letter.
  /// Reported for BREAKFAST and YOUR; the same bug that once spelled SIKLAB.
  /// Words already in the engine lexicon (DAN, BRUCE) happen to survive, which
  /// is why it looks intermittent rather than systematic.
  ///
  /// Multi-word labels are handled per word, so THANK YOU becomes "Thank you"
  /// rather than "Thank YOU".
  static String speakableForm(String word) {
    if (word.isEmpty) return word;
    return word
        .split(' ')
        .map((w) => w.isEmpty
            ? w
            : w[0].toUpperCase() + w.substring(1).toLowerCase())
        .join(' ');
  }

  /// Spoken form for a FINGERSPELLED word.
  ///
  /// Title case, same as [speakableForm]. Two alternatives were tried on the
  /// Redmi and both were worse:
  ///
  ///   lowercase "siklab"    spelled out, 1600ms (letter-by-letter timing)
  ///   syllables "sik lab"   spelled "sik", pronounced "lab"
  ///
  /// That second result is the diagnostic one: the engine pronounces
  /// dictionary words and spells everything else. Fingerspelling exists to
  /// produce names no dictionary contains, so no transformation of the string
  /// can fix it — the fix, if one is wanted, is a better voice on the device
  /// (Google Text-to-Speech with an en-US voice; this device resolved to a
  /// bare "en"). Keeping the simplest form rather than a clever one that
  /// makes the failure sound stranger.
  static String spellableForm(String word) => speakableForm(word);

  /// Sign segmentation for WORDS. A sign is the span between the hands
  /// appearing and leaving frame, so its length adapts to the sign instead of
  /// being forced into a fixed window.
  /// End-of-sign and minimum-length thresholds, in TIME rather than frames.
  ///
  /// These were frame counts, which meant something different at every frame
  /// rate: at the ~4 fps this device manages, "3 frames of silence" is 0.75s
  /// and "6 frames minimum" demands 1.5s of signing. A one-second sign produced
  /// 4 frames, fell under the floor, and was discarded without ever being
  /// classified — silence rather than a wrong answer, which is what made ASL
  /// look completely broken.
  static const int wordEndSilenceMs = 800;
  static const int wordMinDurationMs = 500;

  /// Absolute floor on distinct frames. Measured on the ASL test set:
  /// 8 frames 66.3%, 6 frames 63.8%, 5 frames 62.3%, 4 frames 55.3%,
  /// 3 frames 41.3%, 2 frames 17.2%. Accuracy holds to about 5 and collapses
  /// below 4, so 4 is the point where a degraded answer still beats silence.
  static const int wordMinFrames = 4;
  static const int wordMaxDurationMs = 5000;

  /// Posture gating: distinguishes resting from signing by how high the wrists
  /// sit, so a signer whose hands stay visible at rest still gets a clean
  /// start and end.
  ///
  /// Hands-leaving-frame alone fails when the whole body is in view: the hands
  /// never disappear, so the sign never ends and nothing is ever classified.
  ///
  /// Height is measured as (shoulder_y - wrist_y) / shoulder_width, using the
  /// higher wrist. Shoulders carry visibility 1.00 on every frame, unlike hips
  /// (0.57), and shoulder width gives a scale that does not depend on how far
  /// away the signer stands. Measured on the source clips:
  ///   resting  median -1.8   (p90 -0.9)
  ///   signing  median -1.1   (p90 -0.1)
  ///
  /// The two overlap at the low end, because plenty of signs sit at chest
  /// height, so a single instantaneous threshold would cut a sign in half.
  /// Entry is therefore a high bar crossed once, and exit must be *sustained*
  /// — rest is continuous, whereas a mid-sign dip is brief.
  static const double postureEnter = -0.6;

  /// Bar for "this buffer contains real signing, not rest". Sits between the
  /// measured rest median (-1.83) and signing median (-1.14).
  static const double postureRaised = -1.45;
  static const double postureExit = -1.5;
  /// Time, not frames, for the same reason as [wordEndSilenceMs].
  static const int postureExitMs = 1000;

  /// Last computed wrist height, and whether a sign is currently in progress.
  static double postureHeight = 0;
  static bool postureValid = false;
  /// Whether the current buffer ever rose to signing height.
  static bool _sawRaised = false;
  static int _belowExitSince = 0;

  /// When the hands were last seen, for time-based end-of-sign.
  static int _lastHandSeenMs = 0;

  static String get postureLabel {
    if (!postureValid) return 'posture n/a';
    return '${_sawRaised ? "signing" : "low"} '
        '${postureHeight.toStringAsFixed(1)}';
  }

  /// Wrist height above the shoulder line, in shoulder-widths. Returns null
  /// when the pose landmarks needed are missing, in which case the caller falls
  /// back to hands-in-frame segmentation.
  static double? _wristHeight(List<double> v) {
    double px(int j, int c) => v[126 + j * 4 + c];
    final lsX = px(11, 0), lsY = px(11, 1);
    final rsX = px(12, 0), rsY = px(12, 1);
    if (lsX == 0 && lsY == 0) return null;
    if (rsX == 0 && rsY == 0) return null;

    final width = math.sqrt(
        (lsX - rsX) * (lsX - rsX) + (lsY - rsY) * (lsY - rsY));
    if (width < 1e-4) return null;

    final lwY = px(15, 1), rwY = px(16, 1);
    if (lwY == 0 && rwY == 0) return null;
    // y grows downward, so the higher wrist is the smaller value.
    final wrist = (lwY == 0)
        ? rwY
        : (rwY == 0 ? lwY : math.min(lwY, rwY));

    return ((lsY + rsY) / 2 - wrist) / width;
  }

  /// Duration of signing the 30-frame buffer represents.
  ///
  /// Measured: the FSL-105 clips are 60 fps / ~244 frames / ~4.07 s. The original
  /// extractor kept 30 *consecutive* centre frames — 0.50 s, only 12% of each
  /// sign — so the LSTM never saw a complete sign. extract_fsl105_v2.py instead
  /// samples across the whole clip and keeps the hand-present frames, so a
  /// training sequence now spans the full gesture.
  ///
  /// NO LONGER USED for inference — kept only as the documented median.
  ///
  /// A fixed window was the wrong idea. Per-class measurement shows signs run
  /// from 1.02s (TWO) to 2.67s (NO); 1400ms is merely the median, so it padded
  /// short signs with idle frames and truncated long ones — GOOD MORNING (1.88s,
  /// two movements) was cut mid-sign, which is why it had to be signed in two
  /// halves to register. Words are now segmented on hand presence instead; see
  /// [_decideWord].
  static const int windowMedianMs = 1400;

  static final List<_TimedFrame> _frameBuffer = [];
  static String _lastLabel = '';
  static DateTime _lastResultTime = DateTime(0);
  static String debugString = "> Waiting for sign input...";
  /// What the recogniser is currently doing.
  ///
  /// ASL words behave exactly like FSL words everywhere else in this file —
  /// unmirrored, positional hand ordering, posture-gated segmentation — so
  /// [isLettersMode] stays the discriminator for those paths and only the
  /// interpreter and label list differ.
  static SignMode mode = SignMode.aslWords;

  static bool get isLettersMode => mode == SignMode.letters;

  static void cycleMode() {
    mode = SignMode.values[(mode.index + 1) % SignMode.values.length];
    fullReset();
  }

  /// Resets EVERY piece of cross-frame state, not just the frame buffer.
  ///
  /// clearBuffer deliberately keeps some continuity, which is right between
  /// signs and wrong between modes: LETTERS and WORDS drive the posture gate,
  /// the hand-seen clocks and the repeat suppression completely differently,
  /// and switching modes left a mixture of the two. The symptom is that
  /// recognition goes silent after a mode switch and stays silent even after
  /// switching back — frames keep flowing, so the stream watchdog sees nothing
  /// wrong, and only killing the app recovered it.
  ///
  /// Anything here that persists across a mode switch is a bug waiting to
  /// happen, so this list must stay exhaustive.
  static void fullReset() {
    clearBuffer();
    _lastHandSeenMs = 0;
    _lastHandAt = 0;
    _lastResultTime = DateTime(0);
    _lastCommitted = '';
    _gapFrames = spellRepeatGapFrames;
    completedWord = null;
    completedWordSpoken = null;
    postureValid = false;
    postureHeight = 0;
    lastRawLandmarks = null;
    debugString = '';
  }

  static String get modeLabel {
    switch (mode) {
      case SignMode.fslWords:
        return 'FSL WORDS';
      case SignMode.letters:
        return 'LETTERS';
      case SignMode.aslWords:
        return 'ASL WORDS';
    }
  }

  /// Reproduces `cv2.flip(frame, 1)` from test_webcam.py. MediaPipe assigns its
  /// "Left"/"Right" handedness label from the image as presented, and the training
  /// pipeline normalises against the right-hand block with no fallback — so this
  /// flag decides which feature space the landmarks land in. Exposed so it can be
  /// A/B tested on-device without a rebuild.
  static bool mirrorInput = true;

  /// Whether the frame handed to MediaPipe is mirrored, per mode.
  ///
  /// LETTERS keeps the mirror: fingerspelling is accurate with it, and the
  /// hand-block pin makes the handedness label irrelevant there.
  ///
  /// WORDS must not mirror. The FSL-105 clips are unmirrored, and mirroring
  /// flips the geometry as well as the label — a mirrored right hand has
  /// left-hand shape — so a mirrored frame cannot match unmirrored training no
  /// matter which block it lands in. Extraction was measured both ways;
  /// see extract_fsl105_v2.py.
  /// WORDS never mirrors, on either lens.
  ///
  /// Mirroring the back camera to "match" the front was tried and is a clear
  /// regression, measured on device:
  ///
  ///     word        back, unmirrored   back, mirrored
  ///     HELLO       8/8                4/8
  ///     THANK YOU   8/8                2/8
  ///     EAT         8/8                1/8
  ///     DRINK       8/8                6/8
  ///     NO          4/8                0/8
  ///
  /// The reasoning behind it was wrong: an Android front sensor outputs the
  /// scene unmirrored and the flip is applied only when the PREVIEW is drawn,
  /// so both lenses already agree and the correction introduced a flip that
  /// was never there. Whatever still differs between the lenses for GOOD, NO
  /// and WE is not handedness.
  static bool get effectiveMirror => isLettersMode && mirrorInput;

  /// Which hand block fingerspelling lands in: 'left', 'right', or 'auto'.
  ///
  /// The blocks are not interchangeable — a hand in the right block is
  /// wrist-normalised below, while one in the left block keeps raw image
  /// coordinates, and the letters model learned those as separate regimes.
  /// MediaPipe's own handedness label flips between frames, so pinning the block
  /// is what makes letter output repeatable. Measured on-device: a held 'A' read
  /// as S/N through the right block and A at 100% through the left.
  ///
  /// Words mode uses [wordHandBlock] instead — two-handed signs need both blocks.
  /// Now 'auto': geometric canonicalisation makes the block irrelevant, so
  /// there is nothing left to pin. Kept as a toggle only for A/B checks.
  static String letterHandBlock = 'auto';

  static const List<String> handBlockOptions = ['left', 'right', 'auto'];

  /// Two-handed signs need both blocks, so words cannot pin to one.
  ///
  /// Back to 'auto' now that words no longer mirrors the frame.
  ///
  /// 'swap' was correct only as a *compensation* for the mirror: with a mirrored
  /// frame MediaPipe reported the signing hand as "Right" where unmirrored
  /// training had it on the left, and inverting the label recovered TWO and
  /// THREE where 'auto' gave BREAD. Removing the mirror removes the reason for
  /// the inversion — the label now matches training directly, and the geometry
  /// does too, which swapping alone could never fix.
  ///
  /// Kept as a toggle so this can be re-checked on device against the v2 model.
  static String wordHandBlock = 'auto';

  static const List<String> wordBlockOptions = ['auto', 'swap'];

  static String get activeHandBlock =>
      isLettersMode ? letterHandBlock : wordHandBlock;

  /// Emits periodic landmark diagnostics from the Kotlin side (logcat tag
  /// "MediaPipe"), useful for confirming handedness on a real device.
  static bool debugLogging = kDebugMode;

  /// Raw landmark vector from the last processed frame, before canonicalisation
  /// or normalisation — i.e. still in the coordinates MediaPipe reported. Used
  /// only by the debug skeleton overlay.
  static List<double>? lastRawLandmarks;

  /// Source frame dimensions and rotation for the last processed frame, needed
  /// to map canvas coordinates back onto the preview.
  static int lastSrcWidth = 0;
  static int lastSrcHeight = 0;
  static int lastRotation = 270;

  /// Draw the detected landmarks over the camera preview.
  static bool showSkeleton = false;

  /// Write every classified sequence to disk for offline comparison against the
  /// training corpus. Debug only — off by default, and never enabled in release.
  ///
  /// Offline evaluation predicts ~48% of ASL signs should be recognised and
  /// shown under simulated device conditions (device framing + 5 distinct
  /// frames + the confidence gate), against almost none observed live. Frame
  /// rate, segmentation, canonicalisation and zero-hand frames have each been
  /// measured and ruled out as the cause of the remainder, so the only way to
  /// find it is to compare the tensor the device actually builds against the
  /// distribution the model was trained on.
  ///
  /// Dumps land in the app's external files dir; pull them with:
  ///   adb pull /sdcard/Android/data/com.siklabkumpas.siklab_kumpas/files/tensor_dumps
  /// then run ml_pipeline/eval/compare_device_tensor.py over the directory.
  static bool dumpTensors = false;

  static String? _dumpDir;
  static int _dumpCount = 0;

  /// Resolved once, lazily, so a disabled dump costs nothing.
  static Future<String?> _ensureDumpDir() async {
    if (_dumpDir != null) return _dumpDir;
    try {
      _dumpDir = await _channel.invokeMethod<String>('dumpDir');
    } catch (e) {
      debugPrint('dumpDir unavailable: $e');
    }
    return _dumpDir;
  }

  /// One JSON file per classified sign: the exact [30][226] tensor handed to
  /// the interpreter, plus the context needed to interpret it.
  ///
  /// Fire-and-forget: the sign has already been classified and spoken by the
  /// time this runs, so it must never delay the recognition path.
  static Future<void> _dumpSequence(
    List<List<double>> sequence,
    String? label,
    double? confidence,
    int rawFrames,
    int spanMs,
  ) async {
    final dir = await _ensureDumpDir();
    if (dir == null) return;
    try {
      final n = _dumpCount++;
      final stamp = DateTime.now().toIso8601String().replaceAll(':', '-');
      final f = File('$dir/${mode.name}_${n.toString().padLeft(3, '0')}'
          '_$stamp.json');
      // The sequence is written as raw float32, NOT inside the JSON.
      //
      // jsonEncode of a 30x754 tensor is 22,620 doubles formatted as text on
      // the UI isolate. That blocked the camera stream callback long enough
      // for Android's ImageReader queue to fill, after which the plugin stops
      // delivering frames permanently — the app previews normally and detects
      // nothing until it is killed. Every stall observed on device had the
      // dump enabled.
      //
      // A Float32List copy is ~90KB and costs microseconds, so the callback
      // returns immediately and the queue never backs up.
      final flat = Float32List(sequence.length * sequence[0].length);
      var k = 0;
      for (final row in sequence) {
        for (final x in row) {
          flat[k++] = x;
        }
      }
      final bin = File('${f.path}.f32');
      unawaited(bin.writeAsBytes(flat.buffer.asUint8List(), flush: false));

      final payload = <String, dynamic>{
        'rows': sequence.length,
        'cols': sequence[0].length,
        'bin': bin.path.split('/').last,
        'mode': mode.name,
        'label': label,
        'confidence': confidence,
        'raw_frames': rawFrames,
        'span_ms': spanMs,
        'fps': processingFps,
        'mirror': effectiveMirror,
        'hand_block': activeHandBlock,
        'canvas_w': canvasW,
        'canvas_h': canvasH,
        'posture_height': postureValid ? postureHeight : null,
      };
      // Metadata only — small, so this stays cheap too.
      unawaited(f.writeAsString(jsonEncode(payload), flush: false));
    } catch (e) {
      debugPrint('tensor dump failed: $e');
    }
  }

  /// Canvas size used on the Kotlin side. MUST match CANVAS_W / CANVAS_H in
  /// MainActivity.kt — the overlay mapping is derived from these.
  static const double canvasW = 480;
  static const double canvasH = 360;

  /// Maps a landmark from canvas-normalised space onto the preview widget,
  /// returning null when the point is empty (no detection).
  ///
  /// Kotlin letterboxes the upright frame into a fixed 4:3 canvas, so the
  /// signer occupies only part of it. This undoes that boxing, then accounts
  /// for the preview being mirrored for the front camera while the analysis
  /// frame may not be (words runs unmirrored).
  static Offset? overlayPoint(double cx, double cy) {
    if (cx == 0.0 && cy == 0.0) return null;
    if (lastSrcWidth == 0 || lastSrcHeight == 0) return null;

    // Rotation by 90 or 270 swaps the frame's dimensions.
    final rotated = lastRotation == 90 || lastRotation == 270;
    final upW = (rotated ? lastSrcHeight : lastSrcWidth).toDouble();
    final upH = (rotated ? lastSrcWidth : lastSrcHeight).toDouble();

    final scale = math.min(canvasW / upW, canvasH / upH);
    final dw = upW * scale;
    final dh = upH * scale;
    final left = (canvasW - dw) / 2;
    final top = (canvasH - dh) / 2;

    var x = (cx * canvasW - left) / dw;
    var y = (cy * canvasH - top) / dh;

    final previewMirrored = CameraService.isFrontCamera;
    if (previewMirrored != effectiveMirror) x = 1.0 - x;

    if (x < -0.1 || x > 1.1 || y < -0.1 || y > 1.1) return null;
    return Offset(x, y);
  }

  static bool lastHadHands = false;

  /// Bounding extent of the detected hand, in upright-frame pixels, and which
  /// path found it. Diagnostics only — reported past the 226 dimensions the
  /// model consumes and trimmed off before the tensor is built.
  ///
  /// The letters crop is stretched from `extent / 0.67` up to a 480-wide
  /// canvas, so a small hand is upscaled enormously and the landmarker returns
  /// 21 points on the interpolation regardless. These two numbers are what the
  /// minimum-size gate will be set from, measured rather than assumed.
  static double lastHandExtentPx = 0;
  static bool lastHandFromTracker = false;

  /// Smallest hand the letters model may be asked to read, in upright pixels.
  ///
  /// SET FROM MEASUREMENT, not from reasoning. Two runs were captured on the
  /// same device minutes apart — one at a distance that emitted a stream of
  /// P/Q/O with nothing being spelled, one close enough to spell correctly:
  ///
  ///     garbage   Q 22.2px (14.5x)   O 55.2px (5.8x)   O 60.7px (5.3x)
  ///     real      A 63.6px (5.1x)    S 72.3px (4.4x)   B 118.6px (2.7x)
  ///                                  ... 12 letters, 63.6-118.6px
  ///
  /// The two do not overlap. The crop is sized extent / LETTER_TARGET_SPAN and
  /// stretched to a 480-wide canvas, so the upscale is 321.6 / extent: every
  /// garbage letter came from above 5.3x and every real one from below 5.2x.
  /// Past that the landmarker is reading interpolation, returns 21 points
  /// anyway, and the letters model — which cannot know its input is mush —
  /// settles on whichever classes own that degenerate region. Voting does not
  /// help, because a stationary blur votes consistently.
  ///
  /// 62px sits between the two, and the margin is only ~3px on 15 samples, so
  /// treat it as tuned rather than settled. It errs toward rejecting: a
  /// rejected letter is recoverable in a second by moving closer, where an
  /// accepted one has to be backspaced out of the word.
  static const double minLetterExtentPx = 62.0;

  /// Whether the hand is currently too small for fingerspelling to be read.
  static bool get handTooFarForLetters =>
      lastHandExtentPx > 0 && lastHandExtentPx < minLetterExtentPx;
  static double processingFps = 0;

  /// When a frame last completed processing, for the stream watchdog.
  ///
  /// The pipeline has twice gone silent on device mid-session — camera still
  /// previewing, no crash in logcat, nothing detected until the app was
  /// restarted. Both times it followed a decided word. Rather than keep
  /// guessing at the cause under a deadline, the view watches this timestamp
  /// and restarts the image stream when it goes stale, which recovers whatever
  /// the underlying stall turns out to be.
  static int lastProcessedAtMs = 0;
  static int _lastFrameTime = 0;

  /// How long without hands before the camera drops to the idle stride.
  ///
  /// Must exceed PhraseService.phraseGapMs (5000): a phrase is a sequence of
  /// signs with hands DOWN between them, and idling inside one would put the
  /// pipeline to sleep exactly when the next sign is about to start. 6s sits
  /// just past that gap, so idle only ever begins after a phrase has already
  /// been spoken.
  static const int idleAfterMs = 6000;

  /// Whether the camera is currently on the idle stride.
  static bool isIdle = false;

  /// When hands were last seen, or when the session began — whichever is
  /// later. Separate from _lastHandSeenMs, whose zero value means "never" and
  /// drives the end-of-sign logic; this one must start counting from stream
  /// start so a session that opens with nobody in frame still idles.
  static int _idleAnchorMs = 0;

  static void _enterIdle(int sinceMs) {
    if (isIdle) return;
    isIdle = true;
    CameraService.setIdle(true);
    TraceService.log('idle  entered after ${sinceMs}ms without hands');
  }

  /// Instant, on the first frame with a hand. A false wake costs a little heat;
  /// a slow wake costs sign onset every time.
  static void _wake(String why) {
    if (!isIdle) return;
    isIdle = false;
    CameraService.setIdle(false);
    TraceService.log('idle  woke ($why)');
  }

  /// How long the hands may be absent before the buffer is abandoned.
  ///
  /// This was the last frame-count threshold left, and it tightened as the
  /// pipeline got faster: 5 frames was 2s at 2.5 fps but only 1s at 4.8 fps, so
  /// improving capture made the app *less* tolerant of a hand briefly leaving
  /// frame — which is exactly the over-sensitivity reported on device. In time,
  /// it behaves the same at any frame rate.
  static const int abandonAfterMs = 2000;

  static Future<void> initialize() async {
    if (_isInitialized) return;
    try {
      await _channel.invokeMethod('initMediaPipe');

      // v2canon: full-sign sequences (extract_fsl105_v2.py) + landmark
      // augmentation + canonicalised hand blocks (canonicalize.py).
      // Held-out accuracy 97.19%, vs 96.25% for fsl105_v2.tflite and 72.50% for
      // the original fsl105.tflite. Both earlier models are kept for rollback.
      //
      // REQUIRES _canonicaliseHands() below — the model expects canonicalised
      // input and will mispredict badly without it.
      _fsl105Interpreter = await Interpreter.fromAsset(
        'assets/models/fsl105_v2canon.tflite',
        options: InterpreterOptions()..threads = 2,
      );

      // Canonicalised by geometric chirality (train_fsl_letters_canon.py).
      // 98.62% held-out, vs 97.80% for the original fsl_letters.tflite which is
      // kept for rollback. REQUIRES _canonicaliseLetterHand() below.
      _lettersInterpreter = await Interpreter.fromAsset(
        'assets/models/fsl_letters_canon.tflite',
        options: InterpreterOptions()..threads = 2,
      );

      // ASL core-200: trained on ASL Citizen with signer-disjoint splits, so
      // its accuracy is a cross-signer figure and not comparable to the FSL
      // model's single-signer number. Same 226-d canonical input, so every
      // preprocessing step above is shared.
      //
      // Two changes from the 525-class model this replaces, both measured
      // under the DEVICE condition (device framing + 5 distinct frames, which
      // is what the camera actually delivers) rather than at the corpus
      // framing and 30 frames a published figure assumes:
      //
      //   asl_525       48.06% top-1, 39.35% above the confidence gate
      //   + framing aug 59.86% / 48.12%   pose is never wrist-normalised, so
      //                                   the phone's letterbox put 100 of the
      //                                   226 dims out of distribution
      //   + 200 words   61.95% / 53.06%   fewer confusable competitors, and
      //                                   the vocabulary is chosen for what a
      //                                   user needs to say
      //
      // The 200 keep every grammatical word (pronouns, negation, conjunctions,
      // modals) so signs can still be chained into a sentence.
      _aslInterpreter = await Interpreter.fromAsset(
        'assets/models/asl_demo50.tflite',
        options: InterpreterOptions()..threads = 2,
      );

      _fsl105Labels = await _loadLabels('assets/labels/fsl105_labels.txt');
      _aslLabels = await _loadLabels('assets/labels/asl_demo50_labels.txt');
      _lettersLabels =
          await _loadLabels('assets/labels/fsl_letters_labels.txt');

      _isInitialized = true;
      debugPrint('InferenceService initialized successfully');
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

  static Future<TranslationResult?> processFrame(CameraImage image) async {
    if (_isProcessing || !_isInitialized) return null;
    _isProcessing = true;

    try {
      final planes = image.planes;
      if (planes.isEmpty) return null;

      final yPlane = planes[0];
      final uPlane = planes.length > 1 ? planes[1] : null;
      final vPlane = planes.length > 2 ? planes[2] : null;

      final raw = await _channel.invokeMethod<Float64List>(
        'processFrame',
        {
          'yBytes': yPlane.bytes,
          'uBytes': uPlane?.bytes,
          'vBytes': vPlane?.bytes,
          'width': image.width,
          'height': image.height,
          'yRowStride': yPlane.bytesPerRow,
          'uvRowStride': uPlane?.bytesPerRow ?? yPlane.bytesPerRow,
          'uvPixelStride': uPlane?.bytesPerPixel ?? 1,
          'rotation': CameraService.sensorOrientation,
          'mirror': effectiveMirror,
          'lettersMode': isLettersMode,
          'handBlock': activeHandBlock,
          'debug': debugLogging,
        },
      ).timeout(
        const Duration(seconds: 3),
        // Without this the whole pipeline can wedge silently. processFrame
        // guards re-entry with _isProcessing and clears it in a finally — but
        // a finally never runs if the await never completes, so one hung
        // platform call leaves the flag set and EVERY later frame returns at
        // the guard. The camera keeps previewing and nothing is ever detected,
        // which is exactly the "it stopped detecting" failure seen on device.
        //
        // 3s is far longer than the ~80ms a frame actually takes, so this only
        // fires on a genuine hang, and recovery costs one dropped frame.
        onTimeout: () {
          debugPrint('processFrame: platform call timed out, dropping frame');
          return null;
        },
      );

      if (raw == null || raw.length < landmarkDim) return null;

      // Two diagnostic slots ride past the 226 the model uses; take them and
      // trim, so nothing downstream ever sees a vector of the wrong width.
      if (raw.length > landmarkDim) {
        lastHandExtentPx = raw[landmarkDim];
        lastHandFromTracker = raw[landmarkDim + 1] > 0.5;
      }
      final lmList = List<double>.from(raw.sublist(0, landmarkDim));

      // Snapshot before canonicalisation/normalisation mutate it — the overlay
      // needs the coordinates as MediaPipe actually reported them.
      if (showSkeleton) {
        lastRawLandmarks = List<double>.from(raw);
        lastSrcWidth = image.width;
        lastSrcHeight = image.height;
        lastRotation = CameraService.sensorOrientation;
      }

      _updateFps();

      // Posture, from the raw pose block before any normalisation. Words only:
      // fingerspelling is a continuous stream and has no rest position.
      if (!isLettersMode) {
        final h = _wristHeight(lmList);
        postureValid = h != null;
        if (h != null) postureHeight = h;
      } else {
        postureValid = false;
      }

      // Hands present? Checked before normalisation, while the blocks still hold
      // raw detector output.
      final hasHands = _anyNonZero(lmList, 0, 126);
      lastHadHands = hasHands;

      // WORDS: collapse a single detected hand into the right-hand block before
      // normalising, matching canonicalize.py.
      //
      // Without this the two hand blocks are separate feature spaces: a hand in
      // rh gets wrist-normalised below, one in lh never does (the normalisation
      // has no fallback). Measured: feeding clean training clips through the
      // lh-only path drops the word model from 99.0% to 13.5%, collapsing onto
      // whichever classes own that rare regime. On device MediaPipe labels the
      // signing hand "Left" essentially always, so 100% of live frames hit it
      // while only 11.9% of training frames did — that, not domain shift, is
      // why words decoded as WHEELCHAIR PERSON at 99%.
      //
      // LETTERS uses geometric chirality canonicalisation instead, so the
      // mirror setting, hand block and which hand the user signs with all stop
      // mattering. Measured: 98.62% accuracy, and 99.7% of predictions
      // unchanged when the input is mirrored.
      if (isLettersMode) {
        _canonicaliseLetterHand(lmList);
      } else {
        _canonicaliseHands(lmList);
      }

      // Normalise relative to the RIGHT wrist, with no fallback — mirrors
      // extract_fsl105.py / extract_fsl_letters.py. (utils.py's
      // normalize_landmarks, which prefers the left wrist, was never used to
      // build either dataset.)
      final rx = lmList[63];
      final ry = lmList[64];
      final rz = lmList[65];

      if (rx != 0.0 || ry != 0.0 || rz != 0.0) {
        for (int i = 0; i < 126; i += 3) {
          lmList[i] -= rx;
          lmList[i + 1] -= ry;
          lmList[i + 2] -= rz;
        }
      }

      if (!hasHands) {
        // The scene has proved it can report "no hand", so anything detected
        // from here on entered the frame rather than being there already.
        _sawEmptyFrame = true;
        final nowMs = DateTime.now().millisecondsSinceEpoch;
        final goneMs = _lastHandSeenMs == 0 ? 0 : nowMs - _lastHandSeenMs;

        // Idle is decided here, before the end-of-sign logic below, and that
        // is safe: a word is decided within wordEndSilenceMs (800ms) of the
        // hands leaving, so by 6s the buffer has long since been emptied.
        if (_idleAnchorMs == 0) _idleAnchorMs = nowMs;
        final sinceAnyHandMs = nowMs - _idleAnchorMs;
        if (sinceAnyHandMs >= idleAfterMs) _enterIdle(sinceAnyHandMs);

        // LETTERS: hands down ends the word.
        if (isLettersMode && spelledWord.isNotEmpty && _lastHandAt != 0 &&
            DateTime.now().millisecondsSinceEpoch - _lastHandAt > spellIdleMs) {
          finishSpelling();
          return null;
        }

        // WORDS: hands leaving frame ends the sign. Decide once, on everything
        // that was actually signed, instead of guessing continuously.
        if (!isLettersMode && goneMs >= wordEndSilenceMs &&
            _frameBuffer.isNotEmpty) {
          if (_frameBuffer.length >= wordMinFrames) {
            final decided = _decideWord(DateTime.now());
            _frameBuffer.clear();
            if (decided != null) return decided;
          } else {
            // Too few frames to classify. Say so rather than going silent —
            // silence reads as "the app is broken", where this tells the signer
            // exactly what to change.
            final n = _frameBuffer.length;
            _frameBuffer.clear();
            debugString = _tooFewFramesMessage(n);
            return null;
          }
        }

        final abandon = goneMs >= abandonAfterMs;
        if (abandon) {
          _frameBuffer.clear();
          _sawRaised = false;
        }
        debugString = isIdle
            ? "> Idle — raise your hands to begin"
            : abandon
                ? "> Ready — sign, then lower your hands"
                : kDebugMode
                    ? "> hands out of frame (${goneMs}ms) — still holding "
                        "${_frameBuffer.length} frames"
                    : "> Waiting...";
        return null;
      }

      final now = DateTime.now();
      _lastHandSeenMs = now.millisecondsSinceEpoch;
      _idleAnchorMs = now.millisecondsSinceEpoch;
      _wake('hands');

      // Per-frame trace. Two frames per sign has been reported even when
      // filming the source dataset videos, which cannot be a signing-style
      // problem — so the pipeline itself needs measuring rather than guessing.
      if (debugLogging) {
        debugPrint('FRAME|${mode.name}|hands=${hasHands ? 1 : 0}'
            '|buf=${_frameBuffer.length}'
            '|posture=${postureValid ? postureHeight.toStringAsFixed(2) : "na"}'
            '|raised=${_sawRaised ? 1 : 0}'
            '|fps=${processingFps.toStringAsFixed(1)}');
      }
      _frameBuffer.add(_TimedFrame(now.millisecondsSinceEpoch, lmList));
      _pruneBuffer();

      TranslationResult? result;

      if (isLettersMode) {
        final tooFar = handTooFarForLetters;
        final armed = _sawEmptyFrame && !tooFar;
        final raw = armed ? runLettersInference(lmList) : null;
        if (!_sawEmptyFrame) {
          debugString = "> Lower your hands to begin spelling";
        } else if (tooFar) {
          // Say what to change. Silence here is indistinguishable from the app
          // being broken, which is exactly how this was reported.
          debugString = kDebugMode
              ? "> Move your hand closer  "
                  "(${lastHandExtentPx.toStringAsFixed(0)}px, "
                  "need ${minLetterExtentPx.toStringAsFixed(0)})"
              : "> Move your hand closer to the camera";
          // Votes gathered at a readable distance must not combine with frames
          // captured too far away.
          _letterVotes.clear();
        }
        if (raw != null) {
          // Every frame used to be published straight to the screen, so one
          // noisy frame where O narrowly beat C produced the "O, C, O, C"
          // flicker. Require agreement across several frames instead.
          final stable = _stableLetter(raw, now);

          if (stable == null) {
            // Not a confident reading of anything — counts toward the gap a
            // repeated letter must clear.
            _gapFrames++;
          } else if (stable.label != _lastCommitted) {
            // A different letter always commits, and resets the gap.
            spelledWord += stable.label;
            _lastCommitted = stable.label;
            _gapFrames = 0;
            _letterVotes.clear();
            result = stable;
          } else if (_gapFrames >= spellRepeatGapFrames) {
            // Same letter again, but only after a sustained gap — a genuine
            // double letter, not a held handshape.
            spelledWord += stable.label;
            _gapFrames = 0;
            _letterVotes.clear();
            result = stable;
          }

          if (result != null) {
            // Every committed letter, with the hand size that produced it.
            // Comparing a run that spells correctly against one that emits
            // P/Q/O is what sets the minimum extent.
            TraceService.log('letter  ${result.label}  '
                'conf=${result.confidence.toStringAsFixed(3)}  '
                'extent=${lastHandExtentPx.toStringAsFixed(1)}px  '
                'upscale=${lastHandExtentPx > 0 ? (321.6 / lastHandExtentPx).toStringAsFixed(1) : "-"}x  '
                'tracked=${lastHandFromTracker ? 1 : 0}  '
                'fps=${processingFps.toStringAsFixed(1)}');
          }

          final votes = _letterVotes.where((v) => v.label == raw.label).length;
          debugString = result != null
              ? "> $spelledWord"
              // Vote counts and confidences are diagnostics, not guidance —
              // they were added to tune the threshold from data and mean
              // nothing to a user. Debug builds keep them; release shows the
              // word being spelled and nothing else.
              : kDebugMode
                  ? "> $spelledWord   (checking ${raw.label} "
                      "$votes/$letterVotesRequired "
                      "conf ${lastLetterMean.toStringAsFixed(2)}"
                      "/${letterConfidenceThreshold.toStringAsFixed(2)})"
                  : "> $spelledWord";
        }

        // Hands are up, so the signer is still spelling.
        _lastHandAt = now.millisecondsSinceEpoch;
      } else {
        if (postureValid) {
          // Posture ENDS a sign; it never decides whether to record one.
          //
          // This previously gated collection: frames were discarded whenever
          // the wrist sat below postureEnter. Measured signing height is a
          // median of -1.14 shoulder-widths, well below the -0.6 entry bar, so
          // any sign that did not peak high had its buffer emptied every frame
          // and was never classified at all. Silence, not a wrong answer.
          //
          // Now every frame with a hand is kept, and the sign is closed when
          // the hands settle low for postureExitMs. _sawRaised only decides
          // whether the buffer is worth classifying, so resting never produces
          // spurious output but a real sign is never thrown away.
          //
          // NOTE the separate threshold: reusing postureEnter here reintroduced
          // the same fault one level down, because a sign whose peak never
          // reached -0.6 left _sawRaised false and produced no output. This bar
          // sits between the measured rest median (-1.83) and signing median
          // (-1.14), so real signing clears it and resting does not.
          if (postureHeight > postureRaised) _sawRaised = true;

          final nowMs = now.millisecondsSinceEpoch;
          if (postureHeight >= postureExit) {
            _belowExitSince = 0;
          } else if (_belowExitSince == 0) {
            _belowExitSince = nowMs;
          }

          if (_belowExitSince != 0 &&
              nowMs - _belowExitSince >= postureExitMs &&
              _frameBuffer.isNotEmpty) {
            _belowExitSince = 0;
            final n = _frameBuffer.length;
            final worth = _sawRaised && n >= wordMinFrames;
            final decided = worth ? _decideWord(now) : null;
            _frameBuffer.clear();
            _sawRaised = false;
            if (decided != null) return decided;
            if (_sawRaised || n >= wordMinFrames) {
              debugString = _tooFewFramesMessage(n);
            } else {
              // The recovery counter exists to tell US the stall still happens
              // and is being papered over. A user has nothing to do with it.
              debugString = (kDebugMode && watchdogRestarts > 0)
                  ? "> Ready — sign, then rest your hands  "
                      "[recovered ${watchdogRestarts}x]"
                  : "> Ready — sign, then rest your hands";
            }
            return null;
          }
        }


        // Still signing (or posture unavailable, in which case hands leaving
        // frame is the only end signal). Classify anyway past the time cap.
        final spanMs = _frameBuffer.last.t - _frameBuffer.first.t;
        if (spanMs >= wordMaxDurationMs && _frameBuffer.length >= wordMinFrames) {
              _belowExitSince = 0;
          final decided = _decideWord(now);
          _frameBuffer.clear();
          if (decided != null) return decided;
        } else {
          debugString = "> Signing... (${(spanMs / 1000).toStringAsFixed(1)}s, "
              "${_frameBuffer.length} frames) — "
              "${postureValid ? 'lower your hands when done' : 'move hands out of frame when done'}";
        }
      }

      if (result != null) {
        final isNewResult = result.label != _lastLabel ||
            now.difference(_lastResultTime).inSeconds >= 2;

        if (isNewResult && result.confidence >= confidenceThreshold) {
          _lastLabel = result.label;
          _lastResultTime = now;
          // Machine-readable so an accuracy run can be tallied from logcat
          // instead of transcribed by hand.
          debugPrint('RESULT|${mode.name}|${result.label}|'
              '${result.confidence.toStringAsFixed(3)}');
          return result;
        }
      }

      return null;
    } catch (e) {
      debugPrint('processFrame error: $e');
      return null;
    } finally {
      _isProcessing = false;
    }
  }

  /// Port of `canonicalise_frame` in ml_pipeline/preprocessing/canonicalize.py.
  /// Must stay byte-for-byte equivalent to it, or live input lands in a feature
  /// space the model never saw. Operates on the raw vector, before wrist
  /// normalisation. Two-handed frames are left untouched.
  /// Apparent handedness of a hand block, from geometry rather than MediaPipe's
  /// label. Signed area of the palm triangle wrist / index-knuckle /
  /// pinky-knuckle: mirroring flips the sign, in-plane rotation does not.
  /// Returns 0 when the palm is too edge-on to call (~3.5% of frames).
  ///
  /// Port of `hand_chirality` in canonicalize.py — keep the two identical.
  static int _handChirality(List<double> v, int base) {
    final wx = v[base], wy = v[base + 1];
    final ix = v[base + 5 * 3], iy = v[base + 5 * 3 + 1];
    final kx = v[base + 17 * 3], ky = v[base + 17 * 3 + 1];
    final cross = (ix - wx) * (ky - wy) - (iy - wy) * (kx - wx);

    double span = 0;
    for (int j = 0; j < 21; j++) {
      final dx = v[base + j * 3] - wx;
      final dy = v[base + j * 3 + 1] - wy;
      final d = dx * dx + dy * dy;
      if (d > span) span = d;
    }
    span = span <= 0 ? 0 : math.sqrt(span);
    if (span < 1e-6 || cross.abs() < 0.02 * span * span) return 0;
    return cross > 0 ? 1 : -1;
  }

  /// Fingerspelling canonicalisation: whichever block holds the hand, mirror it
  /// to one chirality and put it in the right-hand block.
  ///
  /// This is what removes the constraint that letters only worked for specific
  /// combinations of mirror setting, hand block and which hand the user signs
  /// with. Those combinations were exactly the cases where the three chirality
  /// flips happened to compose correctly. Geometry decides it instead of
  /// MediaPipe's label, which matches the true chirality only 28.6% of the time.
  ///
  /// Port of `canonicalise_letter_frame` in canonicalize.py.
  static void _canonicaliseLetterHand(List<double> v) {
    final hasRight = _anyNonZero(v, 63, 126);
    final hasLeft = _anyNonZero(v, 0, 63);
    if (!hasRight && !hasLeft) return;

    final base = hasRight ? 63 : 0;
    final flip = _handChirality(v, base) == -1;
    final wristX = v[base];

    for (int j = 0; j < 21; j++) {
      final i = j * 3;
      final x = v[base + i];
      v[63 + i] = flip ? (2.0 * wristX - x) : x;
      v[63 + i + 1] = v[base + i + 1];
      v[63 + i + 2] = v[base + i + 2];
    }
    for (int i = 0; i < 63; i++) {
      v[i] = 0.0;
    }
  }

  static void _canonicaliseHands(List<double> v) {
    final hasLeft = _anyNonZero(v, 0, 63);
    final hasRight = _anyNonZero(v, 63, 126);

    if (hasLeft && hasRight) {
      // Two hands: order by position in the image, leftmost wrist into the left
      // block, instead of trusting MediaPipe's label. Swapping the blocks on
      // two-handed clips drops accuracy from 98.9% to 2.3%, and the label flips
      // between consecutive frames — so two-handed signs were a coin flip
      // (measured on device: SON decoded as CRAB, both two-handed signs).
      if (v[0] > v[63]) {
        for (int i = 0; i < 63; i++) {
          final tmp = v[i];
          v[i] = v[63 + i];
          v[63 + i] = tmp;
        }
      }
      return;
    }

    if (!hasLeft) return;

    // One hand, in the left block: move it to the right block, mirroring x
    // about the wrist so a single hand looks the same whichever label
    // MediaPipe assigned it.
    final wristX = v[0];
    for (int j = 0; j < 21; j++) {
      final i = j * 3;
      v[63 + i] = 2.0 * wristX - v[i];
      v[63 + i + 1] = v[i + 1];
      v[63 + i + 2] = v[i + 2];
      v[i] = 0.0;
      v[i + 1] = 0.0;
      v[i + 2] = 0.0;
    }
  }

  static bool _anyNonZero(List<double> v, int start, int end) {
    for (int i = start; i < end; i++) {
      if (v[i] != 0.0) return true;
    }
    return false;
  }

  static void _updateFps() {
    final nowMs = DateTime.now().millisecondsSinceEpoch;
    lastProcessedAtMs = nowMs;
    if (_lastFrameTime != 0) {
      final dt = nowMs - _lastFrameTime;
      if (dt > 0) {
        final instant = 1000.0 / dt;
        processingFps =
            processingFps == 0 ? instant : (processingFps * 0.8 + instant * 0.2);
      }
    }
    _lastFrameTime = nowMs;
  }

  /// Caps the buffer at [wordMaxDurationMs] so a signer who leaves their hands
  /// up indefinitely does not grow it without bound.
  ///
  /// This used to trim to 1.9s, which would have silently
  /// truncated any sign longer than that — NO runs 2.67s — discarding its
  /// opening before classification.
  static void _pruneBuffer() {
    if (_frameBuffer.isEmpty) return;
    final cutoff = _frameBuffer.last.t - wordMaxDurationMs;
    while (_frameBuffer.length > 1 && _frameBuffer.first.t < cutoff) {
      _frameBuffer.removeAt(0);
    }
  }

  /// WHY THERE IS NO HANDSHAPE TIE-BREAK HERE
  ///
  /// One was built and reverted, and the reason is worth keeping.
  ///
  /// ME, PLEASE and SORRY were measured as a confusable cluster, and they are
  /// handshape-distinct — index point, open hand, closed fist. Nearest-centroid
  /// on five numbers (fingertip distance / hand span) separated them at 95.2%
  /// leave-one-out over 62 captures, against ~20% for the model on back-camera
  /// ME. Reassigning within the cluster was a verified no-op across 377
  /// held-out captures and lifted untrained back-camera ME from 3/11 to 8/11.
  ///
  /// On device it made things WORSE: six firings, five of which destroyed a
  /// CORRECT ME (model ME@0.73-0.80 -> SORRY) and broke ME HELP PLEASE and
  /// ME SORRY entirely.
  ///
  /// Two reasons, both worth remembering:
  ///
  ///   1. It was validated on words signed in ISOLATION, with ten-second rests.
  ///      Deployed, the same signs occur inside PHRASES, where they are made
  ///      faster and less deliberately — the measured index extension fell from
  ///      0.98 to 0.63-0.77 and the descriptor stopped recognising a point.
  ///      Validation context has to match deployment context.
  ///
  ///   2. The cluster was wrong. The confusions that actually dominate on
  ///      device are ME<->YOU and SORRY<->YES, and BOTH are same-handshape
  ///      pairs: ME and YOU are the same index point aimed in opposite
  ///      directions, SORRY and YES are both fists differing in location and
  ///      motion. A handshape descriptor is structurally incapable of
  ///      separating either, so no amount of tuning would have helped.
  ///
  /// Runs whichever word model the current mode selects. Both take the same
  /// canonicalised [1, 30, 226] input; only the output width and labels differ.
  static TranslationResult? _runWordInference(List<List<double>> sequence) {
    final asl = mode == SignMode.aslWords;
    final interpreter = asl ? _aslInterpreter : _fsl105Interpreter;
    final labels = asl ? _aslLabels : _fsl105Labels;
    if (interpreter == null || labels.isEmpty) return null;

    try {
      // ASL runs the v4 model on the 736-d v3 representation; FSL still runs
      // the v2 model on the raw 226-d vector. FeaturesV3 is verified
      // byte-equivalent to features_v3.py by test/features_v3_test.dart --
      // a divergence there would not crash, it would quietly put inference in
      // a different feature space than training.
      final prepared = asl
          ? FeaturesV3.build(FeaturesV3.trimIdle(sequence))
          : sequence;
      final input = [prepared];
      final output = [List.filled(labels.length, 0.0)];
      interpreter.run(input, output);

      final scores = output[0];
      final maxScore = scores.reduce((a, b) => a > b ? a : b);
      final maxIndex = scores.indexOf(maxScore);

      if (maxIndex >= labels.length) return null;
      final label = labels[maxIndex];
      final confidence = maxScore;
      return TranslationResult(
        label: label,
        confidence: confidence,
        timestamp: DateTime.now(),
      );
    } catch (e) {
      debugPrint('word inference error: $e');
    }
    return null;
  }

  /// Classifies one complete sign attempt: everything buffered between the
  /// hands entering and leaving frame, resampled to [sequenceLength].
  ///
  /// This replaces the old fixed fixed window. Measured sign lengths in
  /// FSL-105 range from 1.02s (TWO) to 2.67s (NO), so no single window fits:
  /// a 1400ms window truncated GOOD MORNING (1.88s, two movements) mid-sign
  /// while padding TWO with idle frames. Segmenting on hand presence also
  /// matches how the training clips were built — hand-present frames only,
  /// resampled to 30 — and yields one answer per sign instead of a continuous
  /// stream of guesses.
  static TranslationResult? _decideWord(DateTime now) {
    if (_frameBuffer.length < wordMinFrames) return null;

    final idx = List<int>.generate(
      sequenceLength,
      (i) => ((_frameBuffer.length - 1) * i / (sequenceLength - 1)).round(),
    );
    final sequence = [for (final i in idx) _frameBuffer[i].data];

    final result = _runWordInference(sequence);
    final spanMs = _frameBuffer.last.t - _frameBuffer.first.t;

    // Dumped before the confidence gate and before the null check, because the
    // reported symptom is that nothing appears at all — so the attempts that
    // produce no output are exactly the ones worth capturing.
    if (dumpTensors) {
      unawaited(_dumpSequence(sequence, result?.label, result?.confidence,
          _frameBuffer.length, spanMs));
    }

    if (result == null) {
      TraceService.log('decided  NO-OUTPUT  frames=${_frameBuffer.length}  '
          'span=${spanMs}ms  fps=${processingFps.toStringAsFixed(1)}');
      debugString = "> Could not read that sign";
      return null;
    }

    if (result.confidence < activeConfidenceThreshold) {
      TraceService.log('decided  BELOW-GATE  ${result.label}  '
          'conf=${result.confidence.toStringAsFixed(3)}  '
          'gate=$activeConfidenceThreshold  '
          'frames=${_frameBuffer.length}  span=${spanMs}ms  '
          'fps=${processingFps.toStringAsFixed(1)}');
      // Percentages and durations are for tuning, not for the signer. What a
      // user can act on is that the sign was not clear enough.
      debugString = kDebugMode
          ? "> Unsure — ${result.label} "
              "(${result.confidencePercent}, "
              "${(spanMs / 1000).toStringAsFixed(1)}s)"
          : "> Not clear — try that sign again";
      return null;
    }
    debugString = kDebugMode
        ? "> WORDS: ${result.label} (${result.confidencePercent})"
        : "> ${result.label}";

    // Bypass the repeat-suppression used for the continuous letter stream: each
    // sign attempt here is a deliberate act, so signing the same word twice
    // should report twice.
    _lastLabel = result.label;
    _lastResultTime = now;
    debugPrint('RESULT|${mode.name}|${result.label}|'
        '${result.confidence.toStringAsFixed(3)}|${spanMs}ms');
    TraceService.log('decided  ${result.label}  '
        'conf=${result.confidence.toStringAsFixed(3)}  '
        'frames=${_frameBuffer.length}  span=${spanMs}ms  '
        'fps=${processingFps.toStringAsFixed(1)}');
    return result;
  }

  /// Majority vote over recent frames, so a letter is only announced once
  /// several consecutive frames agree.
  ///
  /// Single-frame output flickered between neighbouring handshapes (C/O, D/O)
  /// because the letters model runs independently on every frame and a marginal
  /// frame can flip the winner. Requiring [letterVotesRequired] of the last
  /// [letterVoteWindow] frames to agree costs roughly
  /// letterVotesRequired / fps seconds of latency — about 0.6s at 5 fps — and
  /// removes the flicker.
  ///
  /// Votes older than [letterVoteMaxAgeMs] are discarded so a pause between
  /// letters cannot carry stale evidence into the next one.
  static TranslationResult? _stableLetter(TranslationResult raw, DateTime now) {
    final nowMs = now.millisecondsSinceEpoch;
    _letterVotes.add(_Vote(nowMs, raw.label, raw.confidence));
    _letterVotes.removeWhere((v) => nowMs - v.t > effectiveVoteMaxAgeMs);
    while (_letterVotes.length > letterVoteWindow) {
      _letterVotes.removeAt(0);
    }

    final counts = <String, int>{};
    final sums = <String, double>{};
    for (final v in _letterVotes) {
      counts[v.label] = (counts[v.label] ?? 0) + 1;
      sums[v.label] = (sums[v.label] ?? 0) + v.confidence;
    }

    String? best;
    int bestCount = 0;
    counts.forEach((label, n) {
      if (n > bestCount) {
        bestCount = n;
        best = label;
      }
    });

    if (best == null || bestCount < letterVotesRequired) return null;

    // Average confidence across the agreeing frames, not the single best one —
    // a lucky spike should not carry a letter on its own.
    final meanConfidence = sums[best]! / bestCount;
    lastLetterMean = meanConfidence;
    if (meanConfidence < letterConfidenceThreshold) return null;

    return TranslationResult(
      label: best!,
      confidence: meanConfidence,
      timestamp: now,
    );
  }

  static TranslationResult? runLettersInference(List<double> landmarks) {
    if (_lettersInterpreter == null) return null;

    try {
      final input = [landmarks];
      final output = [List.filled(26, 0.0)];

      _lettersInterpreter!.run(input, output);

      final scores = output[0];
      final maxScore = scores.reduce((a, b) => a > b ? a : b);
      final maxIndex = scores.indexOf(maxScore);

      if (maxIndex < _lettersLabels.length) {
        return TranslationResult(
          label: _lettersLabels[maxIndex],
          confidence: maxScore,
          timestamp: DateTime.now(),
        );
      }
    } catch (e) {
      debugPrint('Letters inference error: $e');
    }
    return null;
  }

  /// Finalises the spelled word so the UI can display and speak it.
  static void finishSpelling() {
    if (spelledWord.isEmpty) return;
    completedWord = spelledWord;
    completedWordSpoken = spellableForm(spelledWord);
    debugPrint('RESULT|spell|$spelledWord|1.000');
    spelledWord = '';
    _lastCommitted = '';
    _gapFrames = spellRepeatGapFrames;
    _lastHandAt = 0;
    _letterVotes.clear();
    debugString = "> spelled: $completedWord";
  }

  static void backspaceLetter() {
    if (spelledWord.isEmpty) return;
    spelledWord = spelledWord.substring(0, spelledWord.length - 1);
    _lastCommitted = '';
    _gapFrames = spellRepeatGapFrames;
    _letterVotes.clear();
  }

  static void clearSpelling() {
    spelledWord = '';
    completedWord = null;
    completedWordSpoken = null;
    _lastCommitted = '';
    _gapFrames = spellRepeatGapFrames;
    _lastHandAt = 0;
    _letterVotes.clear();
  }

  static void clearBuffer() {
    _frameBuffer.clear();
    _letterVotes.clear();
    clearSpelling();
    // Re-arm: a mode switch, a camera switch and a watchdog restart all change
    // what is in view, so whatever was detected before must not carry over.
    _sawEmptyFrame = false;
    _sawRaised = false;
    _belowExitSince = 0;
    _lastLabel = '';
    _lastFrameTime = 0;
    processingFps = 0;
    // _isProcessing gates EVERY frame, and clearing everything except this
    // flag is why the pipeline could not be recovered from inside the app:
    // switching mode and toggling the dump both call clearBuffer, so neither
    // helped, and only killing the process did. If a platform call ever fails
    // to complete, this is the state that has to be released.
    _isProcessing = false;
    // Restart the watchdog's clock rather than zeroing it. Zero reads as
    // "never processed", which the watchdog treats as not-yet-started and
    // skips — and since mode switching calls clearBuffer, that silently
    // disabled recovery for the rest of the session.
    lastProcessedAtMs = DateTime.now().millisecondsSinceEpoch;
    // Every reset — stop/start, mode switch, camera switch, watchdog — begins
    // awake, and the idle clock counts from now.
    _idleAnchorMs = lastProcessedAtMs;
    _wake('reset');
  }

  /// How many times the watchdog has had to restart the image stream.
  ///
  /// Surfaced in the debug string rather than only logged: debugPrint output
  /// is not visible when the app is installed rather than run from the IDE,
  /// which is how it gets onto the demo device.
  static int watchdogRestarts = 0;

  static void dispose() {
    _fsl105Interpreter?.close();
    _lettersInterpreter?.close();
    _aslInterpreter?.close();
    _fsl105Interpreter = null;
    _lettersInterpreter = null;
    _aslInterpreter = null;
    _frameBuffer.clear();
    _isInitialized = false;
  }
}
