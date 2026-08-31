import 'dart:math' as math;

import 'package:flutter/material.dart';

/// Head-and-shoulders guide drawn over the camera preview while setting up.
///
/// WHY FRAMING IS A FEATURE, NOT A PREFERENCE
///
/// The posture gate measures (shoulder_y - wrist_y) / shoulder_width, so it is
/// scale-invariant: how far the signer stands from the phone genuinely does not
/// matter. What does matter is TILT. A phone held low and angled upward
/// compresses the wrist-to-shoulder geometry until hands at rest read as
/// raised, and resting starts being classified as signing. Both shoulders must
/// also be in view — they carry visibility 1.00 on every frame, they anchor the
/// crop that frames everything else, and losing them is what used to freeze the
/// signer-tracking box.
///
/// None of that is discoverable by looking at the screen, and coaching each
/// person verbally does not scale to a panel passing the phone between them. A
/// shape to line yourself up with does.
///
/// Hidden once recognition starts: it has done its job by then, and a signer
/// watching their own hands does not need furniture drawn over them.
class FramingGuide extends StatelessWidget {
  final bool visible;
  final bool isFilipino;
  final Color color;

  const FramingGuide({
    super.key,
    required this.visible,
    required this.isFilipino,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    if (!visible) return const SizedBox.shrink();
    return Positioned.fill(
      // Never intercept taps: the chips and the preview underneath stay live.
      child: IgnorePointer(
        child: Stack(
          children: [
            Positioned.fill(
              child: CustomPaint(painter: _GuidePainter(color)),
            ),
            Positioned(
              left: 12,
              right: 12,
              bottom: 14,
              child: Center(
                child: Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 12, vertical: 7),
                  decoration: BoxDecoration(
                    color: Colors.black.withValues(alpha: 0.55),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    isFilipino
                        ? 'Ihanay ang ulo at balikat · ipantay ang telepono'
                        : 'Line up head and shoulders · hold the phone level',
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                      color: Colors.white70,
                      fontSize: 11.5,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _GuidePainter extends CustomPainter {
  final Color color;

  _GuidePainter(this.color);

  /// Where the shoulders meet the outer edge, as a fraction of preview height.
  ///
  /// Deliberately low. Signing space sits ABOVE the shoulders far more than
  /// below, so leaving the upper half clear is what stops a raised sign being
  /// cropped off the top — the same reasoning as the crop's upward bias in
  /// MainActivity.
  static const double _shoulderY = 0.53;

  @override
  void paint(Canvas canvas, Size size) {
    final w = size.width;
    final h = size.height;

    final stroke = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2
      ..strokeCap = StrokeCap.round
      ..color = color.withValues(alpha: 0.55);

    // Head.
    final headCy = h * 0.26;
    final headRx = w * 0.145;
    final headRy = h * 0.125;
    _dashOval(
      canvas,
      Rect.fromCenter(
        center: Offset(w / 2, headCy),
        width: headRx * 2,
        height: headRy * 2,
      ),
      stroke,
    );

    // Neck and shoulders, drawn as two mirrored curves rather than one arch.
    // A single smooth curve across the top reads as a dome, not a body; the
    // shape people recognise is the drop from the neck out to each shoulder.
    final neckY = headCy + headRy * 0.92;
    final neckHalf = w * 0.055;
    final sy = h * _shoulderY;
    final left = w * 0.215;
    final right = w * 0.785;
    final bottom = h * 0.99;

    for (final side in [-1.0, 1.0]) {
      final neckX = w / 2 + neckHalf * side;
      final outerX = side < 0 ? left : right;
      final path = Path()
        ..moveTo(neckX, neckY)
        // Down the neck, then out along the shoulder line.
        ..cubicTo(
          neckX, neckY + h * 0.045,
          w / 2 + (outerX - w / 2) * 0.72, sy - h * 0.018,
          outerX, sy,
        )
        // Upper arm, straight down out of frame.
        ..lineTo(outerX, bottom);
      _dashPath(canvas, path, stroke);
    }

    // Rest line: where hands should hang when NOT signing. This is the part
    // people get wrong — resting hands high in frame is what makes rest look
    // like signing to the posture gate, and no amount of scale invariance
    // fixes a phone that is tilted upward.
    final restY = h * 0.82;
    final hint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.2
      ..color = color.withValues(alpha: 0.3);
    _dashLine(canvas, Offset(w * 0.28, restY), Offset(w * 0.72, restY), hint);
  }

  void _dashLine(Canvas canvas, Offset a, Offset b, Paint p) {
    const dash = 9.0, gap = 7.0;
    final total = (b - a).distance;
    if (total <= 0) return;
    final dir = (b - a) / total;
    var t = 0.0;
    while (t < total) {
      final end = math.min(t + dash, total);
      canvas.drawLine(a + dir * t, a + dir * end, p);
      t = end + gap;
    }
  }

  void _dashPath(Canvas canvas, Path path, Paint p) {
    const dash = 9.0, gap = 7.0;
    for (final metric in path.computeMetrics()) {
      var t = 0.0;
      while (t < metric.length) {
        final end = math.min(t + dash, metric.length);
        canvas.drawPath(metric.extractPath(t, end), p);
        t = end + gap;
      }
    }
  }

  void _dashOval(Canvas canvas, Rect rect, Paint p) {
    _dashPath(canvas, Path()..addOval(rect), p);
  }

  @override
  bool shouldRepaint(covariant _GuidePainter old) => old.color != color;
}
