import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import '../models/translation_result.dart';
import 'camera_service.dart';

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
  static bool _isInitialized = false;
  static bool _isProcessing = false;

  static const int sequenceLength = 30;
  static const int landmarkDim = 226;
  static const double confidenceThreshold = 0.6;

  /// Temporal voting for fingerspelling. A letter is announced only when
  /// [letterVotesRequired] of the last [letterVoteWindow] frames agree.
  static const int letterVoteWindow = 5;
  static const int letterVotesRequired = 3;
  static const int letterVoteMaxAgeMs = 1500;

  static final List<_Vote> _letterVotes = [];

  /// Sign segmentation for WORDS. A sign is the span between the hands
  /// appearing and leaving frame, so its length adapts to the sign instead of
  /// being forced into a fixed window.
  static const int wordEndSilenceFrames = 3;
  static const int wordMinFrames = 6;
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
  static const double postureExit = -1.5;
  static const int postureExitFrames = 4;

  /// Last computed wrist height, and whether a sign is currently in progress.
  static double postureHeight = 0;
  static bool postureValid = false;
  static bool _signing = false;
  static int _belowExitStreak = 0;

  static String get postureLabel {
    if (!postureValid) return 'posture n/a';
    return '${_signing ? "signing" : "resting"} '
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
  static bool isLettersMode = false;

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
  static double processingFps = 0;
  static int _lastFrameTime = 0;

  /// Consecutive frames with no hand. At ~2.5 fps this is roughly 2 seconds.
  static int _missStreak = 0;
  static const int _missesBeforeClear = 5;

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

      _fsl105Labels = await _loadLabels('assets/labels/fsl105_labels.txt');
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
      );

      if (raw == null || raw.length < landmarkDim) return null;

      final lmList = List<double>.from(raw);

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
        _missStreak++;

        // WORDS: hands leaving frame ends the sign. Decide once, on everything
        // that was actually signed, instead of guessing continuously.
        if (!isLettersMode &&
            _missStreak >= wordEndSilenceFrames &&
            _frameBuffer.length >= wordMinFrames) {
          final decided = _decideWord(DateTime.now());
          _frameBuffer.clear();
          if (decided != null) return decided;
        }

        if (_missStreak >= _missesBeforeClear) {
          _frameBuffer.clear();
        }
        debugString = _missStreak >= _missesBeforeClear
            ? "> Ready — sign, then lower your hands"
            : "> No hands detected ($_missStreak)";
        return null;
      }
      _missStreak = 0;

      final now = DateTime.now();
      _frameBuffer.add(_TimedFrame(now.millisecondsSinceEpoch, lmList));
      _pruneBuffer();

      TranslationResult? result;

      if (isLettersMode) {
        final raw = runLettersInference(lmList);
        if (raw != null) {
          // Every frame used to be published straight to the screen, so one
          // noisy frame where O narrowly beat C produced the "O, C, O, C"
          // flicker. Require agreement across several frames instead.
          result = _stableLetter(raw, now);
          final votes = _letterVotes.where((v) => v.label == raw.label).length;
          debugString = result != null
              ? "> LETTERS: ${result.label} (${result.confidencePercent})"
              : "> LETTERS: checking ${raw.label}... "
                  "($votes/$letterVotesRequired)";
        }
      } else {
        if (postureValid) {
          // Posture gating. Entry is a single crossing; exit must persist, so a
          // sign that dips to waist height mid-gesture is not cut in two.
          if (!_signing) {
            if (postureHeight > postureEnter) {
              _signing = true;
              _belowExitStreak = 0;
              // Drop anything buffered while at rest.
              final last = _frameBuffer.last;
              _frameBuffer
                ..clear()
                ..add(last);
            } else {
              _frameBuffer.clear();
              debugString = "> Ready — raise your hands to sign "
                  "(${postureHeight.toStringAsFixed(1)})";
              return null;
            }
          } else {
            _belowExitStreak =
                postureHeight < postureExit ? _belowExitStreak + 1 : 0;
            if (_belowExitStreak >= postureExitFrames) {
              _signing = false;
              _belowExitStreak = 0;
              if (_frameBuffer.length >= wordMinFrames) {
                final decided = _decideWord(now);
                _frameBuffer.clear();
                if (decided != null) return decided;
              }
              _frameBuffer.clear();
              debugString = "> Ready — raise your hands to sign";
              return null;
            }
          }
        }

        // Still signing (or posture unavailable, in which case hands leaving
        // frame is the only end signal). Classify anyway past the time cap.
        final spanMs = _frameBuffer.last.t - _frameBuffer.first.t;
        if (spanMs >= wordMaxDurationMs && _frameBuffer.length >= wordMinFrames) {
          _signing = false;
          _belowExitStreak = 0;
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
          debugPrint('RESULT|${isLettersMode ? "L" : "W"}|${result.label}|'
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

  static TranslationResult? _runFSL105Inference(List<List<double>> sequence) {
    if (_fsl105Interpreter == null) return null;

    try {
      // Input: [1, 30, 226]
      final input = [sequence];
      final output = [List.filled(105, 0.0)];

      _fsl105Interpreter!.run(input, output);

      final scores = output[0];
      final maxScore = scores.reduce((a, b) => a > b ? a : b);
      final maxIndex = scores.indexOf(maxScore);

      if (maxIndex < _fsl105Labels.length) {
        return TranslationResult(
          label: _fsl105Labels[maxIndex],
          confidence: maxScore,
          timestamp: DateTime.now(),
        );
      }
    } catch (e) {
      debugPrint('FSL-105 inference error: $e');
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

    final result = _runFSL105Inference(sequence);
    if (result == null) {
      debugString = "> Could not read that sign";
      return null;
    }

    final spanMs = _frameBuffer.last.t - _frameBuffer.first.t;
    if (result.confidence < confidenceThreshold) {
      debugString = "> Unsure — ${result.label} "
          "(${result.confidencePercent}, ${(spanMs / 1000).toStringAsFixed(1)}s)";
      return null;
    }
    debugString = "> WORDS: ${result.label} (${result.confidencePercent})";

    // Bypass the repeat-suppression used for the continuous letter stream: each
    // sign attempt here is a deliberate act, so signing the same word twice
    // should report twice.
    _lastLabel = result.label;
    _lastResultTime = now;
    debugPrint('RESULT|W|${result.label}|'
        '${result.confidence.toStringAsFixed(3)}|${spanMs}ms');
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
    _letterVotes.removeWhere((v) => nowMs - v.t > letterVoteMaxAgeMs);
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
    if (meanConfidence < confidenceThreshold) return null;

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

  static void clearBuffer() {
    _frameBuffer.clear();
    _letterVotes.clear();
    _signing = false;
    _belowExitStreak = 0;
    _lastLabel = '';
    _lastFrameTime = 0;
    _missStreak = 0;
    processingFps = 0;
  }

  static void dispose() {
    _fsl105Interpreter?.close();
    _lettersInterpreter?.close();
    _fsl105Interpreter = null;
    _lettersInterpreter = null;
    _frameBuffer.clear();
    _isInitialized = false;
  }
}
