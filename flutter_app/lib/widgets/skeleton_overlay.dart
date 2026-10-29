import 'package:flutter/material.dart';

import '../services/inference_service.dart';

/// Draws MediaPipe's detected hand and pose landmarks over the camera preview.
///
/// Diagnostic, not decorative: it shows directly whether the hands are being
/// tracked, which hand is which, and whether the skeleton is aligned with the
/// signer — questions that otherwise have to be inferred from wrong
/// translations.
class SkeletonOverlay extends StatelessWidget {
  final bool enabled;

  /// Bumped by the parent on every processed frame. Without a changing field
  /// the widget compares equal between builds and the painter never re-runs.
  final int revision;

  const SkeletonOverlay({
    super.key,
    required this.enabled,
    required this.revision,
  });

  @override
  Widget build(BuildContext context) {
    if (!enabled) return const SizedBox.shrink();
    return Positioned.fill(
      child: IgnorePointer(
        child: CustomPaint(painter: _SkeletonPainter(revision)),
      ),
    );
  }
}

class _SkeletonPainter extends CustomPainter {
  final int revision;
  _SkeletonPainter(this.revision);

  // MediaPipe hand topology: five fingers plus the palm arch.
  static const List<List<int>> _handBones = [
    [0, 1], [1, 2], [2, 3], [3, 4],           // thumb
    [0, 5], [5, 6], [6, 7], [7, 8],           // index
    [9, 10], [10, 11], [11, 12],              // middle
    [13, 14], [14, 15], [15, 16],             // ring
    [0, 17], [17, 18], [18, 19], [19, 20],    // pinky
    [5, 9], [9, 13], [13, 17],                // knuckle line
  ];

  // Upper body only — the lower pose landmarks are not used by the models.
  static const List<List<int>> _poseBones = [
    [11, 12],                                  // shoulders
    [11, 13], [13, 15],                        // left arm
    [12, 14], [14, 16],                        // right arm
    [11, 23], [12, 24], [23, 24],              // torso
  ];

  @override
  void paint(Canvas canvas, Size size) {
    final v = InferenceService.lastRawLandmarks;
    if (v == null || v.length < 226) return;

    Offset? at(double nx, double ny) {
      final p = InferenceService.overlayPoint(nx, ny);
      return p == null ? null : Offset(p.dx * size.width, p.dy * size.height);
    }

    // ── pose: 25 landmarks of 4 values (x, y, z, visibility) from index 126
    final posePaint = Paint()
      ..color = const Color(0xFF4FC3F7).withValues(alpha: 0.75)
      ..strokeWidth = 2.5
      ..strokeCap = StrokeCap.round;

    Offset? posePoint(int j) {
      if (j >= 25) return null;
      final b = 126 + j * 4;
      if (v[b + 3] < 0.4) return null;   // low visibility
      return at(v[b], v[b + 1]);
    }

    for (final bone in _poseBones) {
      final a = posePoint(bone[0]);
      final c = posePoint(bone[1]);
      if (a != null && c != null) canvas.drawLine(a, c, posePaint);
    }

    // ── hands: 21 landmarks of 3 values. Left block is orange, right is green,
    // so which block a hand landed in is visible at a glance.
    for (final hand in [
      (base: 0, color: const Color(0xFFFF6B00), tag: 'L'),
      (base: 63, color: const Color(0xFF4CAF50), tag: 'R'),
    ]) {
      final base = hand.base;
      bool any = false;
      for (int i = base; i < base + 63; i++) {
        if (v[i] != 0.0) {
          any = true;
          break;
        }
      }
      if (!any) continue;

      final bonePaint = Paint()
        ..color = hand.color
        ..strokeWidth = 3
        ..strokeCap = StrokeCap.round;
      final jointPaint = Paint()..color = hand.color;

      Offset? joint(int j) => at(v[base + j * 3], v[base + j * 3 + 1]);

      for (final bone in _handBones) {
        final a = joint(bone[0]);
        final c = joint(bone[1]);
        if (a != null && c != null) canvas.drawLine(a, c, bonePaint);
      }
      for (int j = 0; j < 21; j++) {
        final p = joint(j);
        if (p != null) canvas.drawCircle(p, j == 0 ? 5 : 3, jointPaint);
      }

      // Label the wrist so the block assignment is unambiguous on screen.
      final wrist = joint(0);
      if (wrist != null) {
        final tp = TextPainter(
          text: TextSpan(
            text: hand.tag,
            style: TextStyle(
              color: hand.color,
              fontSize: 13,
              fontWeight: FontWeight.w700,
              shadows: const [Shadow(color: Colors.black, blurRadius: 3)],
            ),
          ),
          textDirection: TextDirection.ltr,
        )..layout();
        tp.paint(canvas, wrist + const Offset(8, -6));
      }
    }
  }

  @override
  bool shouldRepaint(covariant _SkeletonPainter oldDelegate) =>
      oldDelegate.revision != revision;
}
