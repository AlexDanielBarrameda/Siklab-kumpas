import 'dart:async';

import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

import '../main.dart';
import '../services/avatar_service.dart';
import '../theme/tokens.dart';

/// One utterance for the avatar to sign.
///
/// [id] is what makes the same sentence playable twice: the player restarts
/// whenever the id changes, so pressing the button again replays rather than
/// being swallowed as "no change".
class AvatarRequest {
  final int id;
  final List<GlossStep> steps;

  const AvatarRequest(this.id, this.steps);
}

/// A single step of playback: either a clip, or a word the avatar has no clip
/// for and can only name.
class _Beat {
  final String word;
  final String? asset;
  final bool loop;

  const _Beat({required this.word, this.asset, this.loop = false});

  bool get isGap => asset == null;
}

/// Plays a queue of avatar clips back to back.
///
/// Two controllers are kept in flight: while beat *n* plays, beat *n+1* is
/// already initialising. A single controller re-initialising between words
/// leaves a black frame on every word boundary, which on a four-word sentence
/// is four visible flashes.
class AvatarPlayer extends StatefulWidget {
  final AvatarRequest? request;
  final bool isFilipino;

  /// Called when the queue finishes and the avatar returns to rest.
  final VoidCallback? onFinished;

  const AvatarPlayer({
    super.key,
    this.request,
    this.isFilipino = false,
    this.onFinished,
  });

  @override
  State<AvatarPlayer> createState() => _AvatarPlayerState();
}

class _AvatarPlayerState extends State<AvatarPlayer> {
  /// Matches the clips produced by avatar_pipeline/build_clips.py.
  static const _aspect = 720 / 1024;

  /// How long a word with no clip is held on screen before moving on. Long
  /// enough to read, short enough not to stall a sentence.
  static const _gapHold = Duration(milliseconds: 750);

  VideoPlayerController? _cur;
  VideoPlayerController? _pre;
  String? _preAsset;

  List<_Beat> _plan = const [];
  int _beat = -1;
  Timer? _gapTimer;

  /// Backstop that advances the sequence if clip completion is never detected.
  ///
  /// video_player exposes no completion callback, so [_tick] infers it from
  /// "stopped, and within 80ms of the end". That window is easy to miss on a
  /// short clip — LETTER_I is 0.77s where every other letter is 1.0s or more,
  /// and it was the one letter that stalled fingerspelling. Rather than widen
  /// the window and risk advancing early, this fires slightly after the clip
  /// should have ended and only if nothing else has moved on.
  Timer? _endTimer;

  /// Bumped whenever playback restarts. Every async step carries the run it
  /// belongs to and abandons itself if that run is no longer current, which is
  /// what makes interrupting a half-played sentence safe. A boolean "busy"
  /// guard cannot do this job: it drops the interrupting request instead of
  /// the interrupted one.
  int _run = 0;

  String? _caption;

  @override
  void initState() {
    super.initState();
    AvatarService.load().then((_) {
      if (!mounted) return;
      // A request may already exist when this player is built -- switching to
      // the Text tab after signing on the Voice tab, for instance. Start from
      // it rather than from idle, so the tab does not come up blank.
      if (widget.request != null) {
        _start(widget.request!);
      } else {
        _enterIdle();
      }
    });
  }

  @override
  void didUpdateWidget(covariant AvatarPlayer old) {
    super.didUpdateWidget(old);
    final r = widget.request;
    if (r != null && r.id != old.request?.id) _start(r);
  }

  @override
  void dispose() {
    _run++;
    _cancelTimers();
    _cur?.removeListener(_tick);
    _cur?.dispose();
    _pre?.dispose();
    super.dispose();
  }

  // --------------------------------------------------------------- playback

  /// IDLE bookends the utterance: once at the front, then looping at the end.
  /// The trailing loop doubles as the resting state, so "idle at the end" and
  /// "idle while waiting" are the same thing rather than two behaviours.
  void _start(AvatarRequest request) {
    final idle = AvatarService.idleAsset;
    final plan = <_Beat>[
      if (idle != null) _Beat(word: '', asset: idle),
      for (final s in request.steps)
        _Beat(
          // While fingerspelling, name the word as well as the letter: the
          // letter alone tells the viewer nothing about how far through it is.
          word: s.isSpelled ? '${s.spelledFrom} · ${s.word}' : s.word,
          asset: s.asset,
        ),
    ];
    if (plan.isEmpty) return;
    _restart(plan);
  }

  void _cancelTimers() {
    _gapTimer?.cancel();
    _endTimer?.cancel();
  }

  void _enterIdle() {
    final idle = AvatarService.idleAsset;
    if (idle == null) return;
    _restart([_Beat(word: '', asset: idle, loop: true)]);
  }

  /// Abandons whatever was playing and begins [plan] from its first beat.
  void _restart(List<_Beat> plan) {
    _run++;
    _gapTimer?.cancel();
    _cur?.removeListener(_tick);
    _plan = plan;
    _beat = -1;
    _step(_run);
  }

  /// Plays the next beat of [run]. Every suspension point re-checks the run,
  /// so a request that arrives mid-step simply wins.
  Future<void> _step(int run) async {
    if (run != _run || !mounted) return;

    _beat++;
    if (_beat >= _plan.length) {
      // Finished a real utterance rather than the idle loop.
      if (_plan.length > 1) {
        widget.onFinished?.call();
        _enterIdle();
      }
      return;
    }

    final beat = _plan[_beat];
    setState(() => _caption = beat.word.isEmpty ? null : beat.word);

    if (beat.isGap) {
      // No clip: hold the last frame and name the word.
      _cur?.pause();
      _gapTimer?.cancel();
      _gapTimer = Timer(_gapHold, () => _step(run));
      return;
    }

    final next = await _acquire(beat.asset!);
    if (run != _run || !mounted) {
      next?.dispose();
      return;
    }
    if (next == null) {
      // Could not open this clip; treat it as a gap rather than stalling.
      _step(run);
      return;
    }

    final old = _cur;
    old?.removeListener(_tick);

    await next.setLooping(beat.loop);
    await next.seekTo(Duration.zero);
    if (run != _run || !mounted) {
      next.dispose();
      return;
    }

    _cur = next;
    next.addListener(_tick);
    unawaited(next.play());
    setState(() {});

    // Arm the backstop for this beat. Cancelled by _tick on normal completion
    // and by the next _step, so it only ever fires when detection failed.
    _endTimer?.cancel();
    if (!beat.loop) {
      final total = next.value.duration;
      if (total > Duration.zero) {
        final beatAtArm = _beat;
        _endTimer = Timer(total + const Duration(milliseconds: 250), () {
          if (!mounted || run != _run || _beat != beatAtArm) return;
          _cur?.removeListener(_tick);
          _step(run);
        });
      }
    }

    // Dispose only after the new controller is on screen, so the card never
    // shows an empty frame during the swap.
    old?.dispose();

    unawaited(_preload(_beat + 1, run));
  }

  /// Uses the preloaded controller when it matches, otherwise builds one.
  Future<VideoPlayerController?> _acquire(String asset) async {
    if (_preAsset == asset && _pre != null) {
      final c = _pre;
      _pre = null;
      _preAsset = null;
      return c;
    }
    return _make(asset);
  }

  Future<VideoPlayerController?> _make(String asset) async {
    try {
      final c = VideoPlayerController.asset(asset);
      await c.initialize();
      await c.setVolume(0);
      return c;
    } catch (_) {
      return null;
    }
  }

  Future<void> _preload(int index, int run) async {
    if (index >= _plan.length) return;
    final beat = _plan[index];
    if (beat.isGap) return;
    if (_preAsset == beat.asset) return;

    final c = await _make(beat.asset!);
    if (run != _run || !mounted) {
      c?.dispose();
      return;
    }
    _pre?.dispose();
    _pre = c;
    _preAsset = c == null ? null : beat.asset;
  }

  /// video_player has no completion callback, so completion is inferred:
  /// stopped, and within a frame or two of the end.
  void _tick() {
    final c = _cur;
    if (c == null || !c.value.isInitialized) return;
    if (_plan.isEmpty || _beat < 0 || _beat >= _plan.length) return;
    if (_plan[_beat].loop) return;

    final v = c.value;
    if (v.isPlaying) return;
    final dur = v.duration;
    if (dur <= Duration.zero) return;
    if (v.position >= dur - const Duration(milliseconds: 80)) {
      c.removeListener(_tick);
      _endTimer?.cancel();
      _step(_run);
    }
  }

  // ------------------------------------------------------------------- view

  @override
  Widget build(BuildContext context) {
    final c = AppColors.of(SiklabKumpasApp.of(context)?.isDark ?? true);
    final controller = _cur;
    final ready = controller != null && controller.value.isInitialized;

    return AspectRatio(
      aspectRatio: _aspect,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(20),
        child: Container(
          decoration: BoxDecoration(
            gradient: ready ? null : c.avatarCard,
            border: Border.all(color: c.border),
            borderRadius: BorderRadius.circular(20),
          ),
          child: Stack(
            fit: StackFit.expand,
            children: [
              if (ready)
                FittedBox(
                  fit: BoxFit.cover,
                  clipBehavior: Clip.hardEdge,
                  child: SizedBox(
                    width: controller.value.size.width,
                    height: controller.value.size.height,
                    child: VideoPlayer(controller),
                  ),
                )
              else
                _placeholder(c),
              if (_caption != null) _captionChip(_caption!),
            ],
          ),
        ),
      ),
    );
  }

  Widget _placeholder(AppColors c) {
    final noClips = AvatarService.isReady == false && AvatarService.idleAsset == null;
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            width: 92,
            height: 92,
            decoration: const BoxDecoration(
              shape: BoxShape.circle,
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [Color(0xFF7C5CFF), Color(0xFF5B4BFF)],
              ),
            ),
            child: const Icon(Icons.back_hand_rounded,
                color: Colors.white, size: 44),
          ),
          const SizedBox(height: 14),
          Text(
            noClips
                ? (widget.isFilipino ? 'Walang clip' : 'No clips bundled')
                : (widget.isFilipino ? 'ASL Avatar' : 'ASL Avatar'),
            style: TextStyle(color: c.subtitle, fontSize: 14),
          ),
        ],
      ),
    );
  }

  /// Names the sign currently playing. Doubles as the fallback for words with
  /// no clip, where it is the only thing telling the user what was meant.
  Widget _captionChip(String word) {
    final gap = _beat >= 0 && _beat < _plan.length && _plan[_beat].isGap;
    return Positioned(
      left: 0,
      right: 0,
      bottom: 10,
      child: Center(
        child: AnimatedSwitcher(
          duration: const Duration(milliseconds: 160),
          child: Container(
            key: ValueKey('$word$gap'),
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: Colors.black.withValues(alpha: 0.55),
              borderRadius: BorderRadius.circular(12),
              border: gap
                  ? Border.all(color: AppTokens.orange.withValues(alpha: 0.8))
                  : null,
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                if (gap) ...[
                  const Icon(Icons.help_outline_rounded,
                      color: AppTokens.orange, size: 14),
                  const SizedBox(width: 5),
                ],
                Text(
                  word,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                    letterSpacing: 0.4,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
