import 'package:flutter/material.dart';

/// Route transition used when opening a profile from Home.
///
/// Slide carries the motion and fade softens the entry. The scale that used to
/// sit under these was dropped: three stacked transitions meant three layers to
/// composite per frame, and the fade alone already forces a saveLayer over the
/// whole incoming screen.
Route<T> appPageRoute<T>(Widget page) {
  return PageRouteBuilder<T>(
    transitionDuration: const Duration(milliseconds: 380),
    reverseTransitionDuration: const Duration(milliseconds: 280),
    pageBuilder: (_, __, ___) => page,
    transitionsBuilder: (_, animation, __, child) {
      final curved = CurvedAnimation(
        parent: animation,
        curve: Curves.easeOutCubic,
        reverseCurve: Curves.easeInCubic,
      );
      return FadeTransition(
        opacity: curved,
        child: SlideTransition(
          position: Tween<Offset>(
            begin: const Offset(0, 0.04),
            end: Offset.zero,
          ).animate(curved),
          child: child,
        ),
      );
    },
  );
}

/// Runs [action] once this route's entry transition has settled.
///
/// Plugin bring-up (camera, speech) is expensive enough to drop frames, and
/// doing it in initState meant that cost landed in the middle of the push
/// animation — which is why opening a profile looked like it skipped its
/// transition entirely. Holding the work until the route is at rest keeps the
/// animation clean.
void afterRouteTransition(BuildContext context, VoidCallback action) {
  final animation = ModalRoute.of(context)?.animation;
  if (animation == null || animation.isCompleted) {
    action();
    return;
  }

  void listener(AnimationStatus status) {
    if (status == AnimationStatus.completed) {
      animation.removeStatusListener(listener);
      action();
    } else if (status == AnimationStatus.dismissed) {
      // Route was popped before it ever settled; drop the work.
      animation.removeStatusListener(listener);
    }
  }

  animation.addStatusListener(listener);
}

/// Plays an entry animation whenever [tabKey] changes.
///
/// Unlike AnimatedSwitcher this does not keep the outgoing child mounted for
/// the length of the transition — it is discarded on the same frame the new one
/// appears. That matters for the camera tab: RecognitionView owns the global
/// CameraService, so an outgoing copy lingering through a cross-fade could tear
/// down the camera a newly built copy had just initialised.
class TabSwitcher extends StatefulWidget {
  final Object tabKey;
  final Widget child;

  const TabSwitcher({super.key, required this.tabKey, required this.child});

  @override
  State<TabSwitcher> createState() => _TabSwitcherState();
}

class _TabSwitcherState extends State<TabSwitcher>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 300),
    value: 1.0,
  );

  late final Animation<double> _fade =
      CurvedAnimation(parent: _controller, curve: Curves.easeOut);

  late final Animation<Offset> _slide = Tween<Offset>(
    begin: const Offset(0.06, 0),
    end: Offset.zero,
  ).animate(CurvedAnimation(parent: _controller, curve: Curves.easeOutCubic));

  @override
  void didUpdateWidget(TabSwitcher oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.tabKey != oldWidget.tabKey) {
      _controller.forward(from: 0);
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return FadeTransition(
      opacity: _fade,
      child: SlideTransition(position: _slide, child: widget.child),
    );
  }
}

/// Scale-on-press wrapper. Dips while held — including through a long press —
/// and springs back past 1.0 on release for the pop.
class PressablePop extends StatefulWidget {
  final Widget child;
  final VoidCallback onTap;

  /// How far the card dips while held.
  final double pressedScale;

  const PressablePop({
    super.key,
    required this.child,
    required this.onTap,
    this.pressedScale = 0.955,
  });

  @override
  State<PressablePop> createState() => _PressablePopState();
}

class _PressablePopState extends State<PressablePop> {
  bool _pressed = false;

  void _setPressed(bool value) {
    if (_pressed != value) setState(() => _pressed = value);
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      behavior: HitTestBehavior.opaque,
      onTapDown: (_) => _setPressed(true),
      onTapUp: (_) => _setPressed(false),
      onTapCancel: () => _setPressed(false),
      onTap: widget.onTap,
      // Holding keeps the card dipped; releasing a long press pops it back.
      onLongPress: () {},
      onLongPressStart: (_) => _setPressed(true),
      onLongPressEnd: (_) => _setPressed(false),
      onLongPressCancel: () => _setPressed(false),
      child: AnimatedScale(
        scale: _pressed ? widget.pressedScale : 1.0,
        // easeOutBack overshoots on the way back to 1.0, which is the pop.
        duration: Duration(milliseconds: _pressed ? 120 : 340),
        curve: _pressed ? Curves.easeOut : Curves.easeOutBack,
        child: widget.child,
      ),
    );
  }
}
