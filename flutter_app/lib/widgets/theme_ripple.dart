import 'dart:math' as math;

import 'package:flutter/material.dart';

/// Circular reveal for the light/dark toggle.
///
/// Paints the *incoming* theme's background gradient into a circle that grows
/// from the toggle button until it covers the screen, flips the theme at full
/// coverage, then fades the circle out to reveal the newly themed content.
/// Because the circle is painted with the same gradient the new theme uses, the
/// hand-off at the switch point is invisible — only the content fades in.
class ThemeRipple {
  const ThemeRipple._();

  /// Guards against a second ripple starting while one is mid-flight.
  static bool _running = false;

  static void run({
    required BuildContext context,
    required GlobalKey originKey,
    required Gradient targetGradient,
    required VoidCallback onSwitch,
  }) {
    final overlay = Overlay.maybeOf(context);
    final box = originKey.currentContext?.findRenderObject() as RenderBox?;
    if (overlay == null || box == null || !box.hasSize) {
      // No overlay, or the button has not been laid out — just switch.
      onSwitch();
      return;
    }

    if (_running) {
      // A ripple is already playing; don't stack a second one.
      return;
    }

    final center = box.localToGlobal(box.size.center(Offset.zero));
    final maxRadius = _radiusToCover(center, MediaQuery.sizeOf(context));

    _running = true;
    late final OverlayEntry entry;
    entry = OverlayEntry(
      builder: (_) => _RippleLayer(
        center: center,
        maxRadius: maxRadius,
        gradient: targetGradient,
        onSwitch: onSwitch,
        onDone: () {
          entry.remove();
          _running = false;
        },
      ),
    );

    overlay.insert(entry);
  }

  /// Distance from [center] to the furthest corner, so the circle fully covers
  /// the screen no matter where the button sits.
  static double _radiusToCover(Offset center, Size size) {
    final dx = math.max(center.dx, size.width - center.dx);
    final dy = math.max(center.dy, size.height - center.dy);
    return math.sqrt(dx * dx + dy * dy);
  }
}

/// The animating layer.
///
/// This owns its own controller rather than borrowing a vsync from the button
/// that triggered it. SingleTickerProviderStateMixin never clears its `_ticker`
/// field once a ticker has been created, so reusing the button's State would
/// throw "multiple tickers were created" on the second ripple. A fresh State
/// per ripple gets a fresh ticker, and the controller's lifetime is tied to
/// this widget instead of spanning theme rebuilds.
class _RippleLayer extends StatefulWidget {
  final Offset center;
  final double maxRadius;
  final Gradient gradient;
  final VoidCallback onSwitch;
  final VoidCallback onDone;

  const _RippleLayer({
    required this.center,
    required this.maxRadius,
    required this.gradient,
    required this.onSwitch,
    required this.onDone,
  });

  @override
  State<_RippleLayer> createState() => _RippleLayerState();
}

class _RippleLayerState extends State<_RippleLayer>
    with SingleTickerProviderStateMixin {
  /// Point in the timeline where the circle has covered the screen and the
  /// theme actually flips.
  static const _switchAt = 0.62;

  late final AnimationController _controller;
  bool _switched = false;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 720),
    )..addListener(_maybeSwitch);

    _controller.forward().whenCompleteOrCancel(() {
      if (!_switched) {
        _switched = true;
        widget.onSwitch();
      }
      // Removing the entry tears down this State, so it must not happen while
      // the ticker is still dispatching. Defer to after the frame.
      WidgetsBinding.instance.addPostFrameCallback((_) => widget.onDone());
    });
  }

  void _maybeSwitch() {
    if (!_switched && _controller.value >= _switchAt) {
      _switched = true;
      widget.onSwitch();
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Positioned.fill(
      child: IgnorePointer(
        child: AnimatedBuilder(
          animation: _controller,
          builder: (_, __) {
            final t = _controller.value;
            final grow =
                Curves.easeOutCubic.transform((t / _switchAt).clamp(0.0, 1.0));
            final fade = t <= _switchAt
                ? 1.0
                : 1.0 - ((t - _switchAt) / (1.0 - _switchAt));

            return Opacity(
              opacity: fade.clamp(0.0, 1.0),
              child: ClipPath(
                clipper: _CircleReveal(widget.center, grow * widget.maxRadius),
                child: DecoratedBox(
                  decoration: BoxDecoration(gradient: widget.gradient),
                  child: const SizedBox.expand(),
                ),
              ),
            );
          },
        ),
      ),
    );
  }
}

class _CircleReveal extends CustomClipper<Path> {
  final Offset center;
  final double radius;

  const _CircleReveal(this.center, this.radius);

  @override
  Path getClip(Size size) =>
      Path()..addOval(Rect.fromCircle(center: center, radius: radius));

  @override
  bool shouldReclip(_CircleReveal oldClipper) =>
      oldClipper.radius != radius || oldClipper.center != center;
}
