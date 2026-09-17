import 'package:flutter/material.dart';
import '../main.dart';
import '../services/avatar_service.dart';
import '../services/speech_service.dart';
import '../services/offline_speech_service.dart';
import '../theme/tokens.dart';
import '../widgets/app_scaffold.dart';
import '../widgets/onboarding_tour.dart';
import '../widgets/avatar_player.dart';
import '../widgets/recognition_view.dart';
import '../widgets/transitions.dart';

/// Tabs in the order they appear in the bottom bar.
enum _HubTab { camera, voice, text }

class HearingUserScreen extends StatefulWidget {
  const HearingUserScreen({super.key});

  @override
  State<HearingUserScreen> createState() => _HearingUserScreenState();
}

class _HearingUserScreenState extends State<HearingUserScreen> {
  _HubTab _tab = _HubTab.voice;
  bool _isListening = false;
  bool _speedExpanded = true;
  String _textOutput = '';
  final TextEditingController _textController = TextEditingController();

  /// Multiplier where 1.0 = natural speaking pace (see SpeechService).
  double _voiceSpeed = SpeechService.speedMultiplier;

  /// Range capped at 1.6x: the engine tops out at its own maximum, so a higher
  /// ceiling would be a slider that visibly moves without changing anything
  /// (the old 0.5-2.0 range did exactly that above 1.0). The tick labels below
  /// the slider therefore read 0.5/0.9/1.2/1.6 rather than the mockup's
  /// 0.5/1.0/1.5/2.0.
  static const _speedMin = 0.5;
  static const _speedMax = 1.6;

  /// Matches TabSwitcher's duration, so the camera does not start spinning up
  /// while the tab transition is still playing.
  static const _tabAnimation = Duration(milliseconds: 300);

  /// Current utterance for the avatar, and the counter that makes replaying
  /// the same sentence a new request rather than a no-op.
  AvatarRequest? _avatarRequest;
  int _requestId = 0;

  /// The gloss actually being signed, shown to the user because ASL gloss is
  /// not the English that was spoken — dropping "is"/"the" and reordering is
  /// something they should be able to see rather than guess at.
  String _glossLine = '';

  /// Glosses [text] and hands the result to the avatar.
  void _sign(String text) {
    final steps = AvatarService.gloss(text);
    if (steps.isEmpty) return;
    setState(() {
      _glossLine = AvatarService.describe(steps);
      _avatarRequest = AvatarRequest(++_requestId, steps);
    });
  }

  @override
  void initState() {
    super.initState();
    // Speech plugin bring-up is slow enough to swallow the push animation from
    // Home, which made this screen look like it appeared with no transition at
    // all. Hold it until the route has settled.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      afterRouteTransition(context, () async {
        if (!mounted) return;
        await SpeechService.initTTS();
        if (!mounted) return;
        // After speech bring-up, so the dialog is not competing with it.
        final app = SiklabKumpasApp.of(context);
        await OnboardingTour.maybeShow(
          context,
          TourId.hub,
          (app?.language ?? 'en') == 'fil',
        );
        // Both report availability through SpeechService rather than throwing,
        // so the status row only becomes accurate once this has run.
        if (mounted) setState(() {});
      });
    });
  }

  @override
  void dispose() {
    _textController.dispose();
    SpeechService.dispose();
    super.dispose();
  }

  String _modeLine(bool isFilipino) {
    switch (_tab) {
      case _HubTab.camera:
        return isFilipino
            ? 'Sign Language → Teksto + Pagsasalita'
            : 'Sign Language → Text + Speech';
      case _HubTab.voice:
        return isFilipino
            ? 'Boses → Teksto + ASL Avatar'
            : 'Voice → Text + ASL Avatar';
      case _HubTab.text:
        return isFilipino
            ? 'Teksto → Pagsasalita + ASL Avatar'
            : 'Text → Speech + ASL Avatar';
    }
  }

  @override
  Widget build(BuildContext context) {
    final app = SiklabKumpasApp.of(context);
    final c = AppColors.of(app?.isDark ?? true);
    final lang = app?.language ?? 'en';
    final isFilipino = lang == 'fil';

    return AppScaffold(
      helpTour: TourId.hub,
      showBack: true,
      title: isFilipino ? 'Hub ng Pagsasalin' : 'Translation Hub',
      subtitle: isFilipino
          ? 'Real-time na Tulay sa 3-Daang Komunikasyon'
          : 'Real-time 3-Way Communication Bridge',
      modeLine: _modeLine(isFilipino),
      child: Column(
        children: [
          Expanded(child: _buildBody(c, lang, isFilipino)),
          _buildTabBar(c, isFilipino),
        ],
      ),
    );
  }

  Widget _buildBody(AppColors c, String lang, bool isFilipino) {
    // One transition covers all three tabs. TabSwitcher animates the incoming
    // child only and drops the outgoing one immediately, which is what lets the
    // camera tab take part: an AnimatedSwitcher would keep a disposing
    // RecognitionView alive through the cross-fade, and that copy tearing down
    // the global CameraService could kill a camera the new copy had just
    // started.
    const pad = EdgeInsets.fromLTRB(16, 0, 16, 8);

    // The voice tab does not scroll: the avatar takes whatever height is left
    // over, which keeps it as large as the device allows. The text tab has
    // more stacked controls than fit on a short screen, so that one still does.
    final Widget body;
    switch (_tab) {
      case _HubTab.camera:
        body = Padding(
          key: const ValueKey(_HubTab.camera),
          padding: pad,
          child: RecognitionView(
            isFilipino: isFilipino,
            initDelay: _tabAnimation,
          ),
        );
      case _HubTab.voice:
        body = Padding(
          key: const ValueKey(_HubTab.voice),
          padding: pad,
          child: _buildVoiceTab(c, lang, isFilipino),
        );
      case _HubTab.text:
        body = SingleChildScrollView(
          key: const ValueKey(_HubTab.text),
          padding: pad,
          child: _buildTextTab(c, isFilipino),
        );
    }

    return TabSwitcher(tabKey: _tab, child: body);
  }

  // ---------------------------------------------------------------- voice tab

  Widget _buildVoiceTab(AppColors c, String lang, bool isFilipino) {
    return Column(
      children: [
        // Output first, matching UI.3 — the hearing user reads this, so it sits
        // above the avatar rather than below it.
        AppCard(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                isFilipino ? 'Output ng Teksto:' : 'Text Output:',
                style: TextStyle(color: c.subtitle, fontSize: 12),
              ),
              const SizedBox(height: 6),
              Text(
                _textOutput.isEmpty
                    ? (isFilipino
                        ? 'Pindutin ang mikropono para magsimula'
                        : 'Press microphone to start')
                    : _textOutput,
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
                style: TextStyle(
                  color: _textOutput.isEmpty ? c.subtitle : c.text,
                  fontSize: 16,
                  fontWeight: FontWeight.w500,
                ),
              ),
              if (_glossLine.isNotEmpty) ...[
                const SizedBox(height: 6),
                _glossText(c),
              ],
            ],
          ),
        ),
        const SizedBox(height: 12),
        Expanded(
          child: Center(
            child: AvatarPlayer(
              request: _avatarRequest,
              isFilipino: isFilipino,
            ),
          ),
        ),
        const SizedBox(height: 14),
        _buildMicButton(c, lang, isFilipino),
        const SizedBox(height: 8),
      ],
    );
  }

  /// Real TTS state, not a decoration.
  ///
  /// This row used to be a hardcoded green "Audio ready", which said the same
  /// thing on a device with no speech engine at all. SpeechService already
  /// works out whether it can speak and why not; the only missing piece was
  /// showing it.
  Widget _speechStatusCard(AppColors c, bool isFilipino) {
    final ok = SpeechService.ttsAvailable;
    final degraded = ok && SpeechService.ttsStatus != 'ok';

    const green = Color(0xFF28C840);
    const red = Color(0xFFFF453A);
    final tint = !ok ? red : (degraded ? c.orange : green);

    final String label;
    if (ok && !degraded) {
      label = isFilipino ? 'Handa na ang audio' : 'Audio ready';
    } else if (!ok) {
      label = isFilipino
          ? 'Walang text-to-speech sa device na ito'
          : SpeechService.ttsStatus;
    } else {
      label = SpeechService.ttsStatus;
    }

    return AppCard(
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            isFilipino ? 'Output ng Pagsasalita:' : 'Speech Output:',
            style: TextStyle(color: c.subtitle, fontSize: 12),
          ),
          const SizedBox(width: 10),
          Padding(
            padding: const EdgeInsets.only(top: 3),
            child: Icon(
              ok
                  ? Icons.circle
                  : Icons.error_outline_rounded,
              color: tint,
              size: ok ? 9 : 14,
            ),
          ),
          const SizedBox(width: 5),
          Expanded(
            child: Text(
              label,
              style: TextStyle(
                color: tint,
                fontSize: 13,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      ),
    );
  }

  /// The gloss line. Letter-joined runs are fingerspelled; a trailing `*`
  /// marks a word the avatar could neither sign nor spell, which the player
  /// names on screen instead. Hiding either would overstate what the avatar
  /// actually conveyed.
  Widget _glossText(AppColors c) {
    return Text(
      _glossLine,
      maxLines: 2,
      overflow: TextOverflow.ellipsis,
      style: TextStyle(
        color: c.orange,
        fontSize: 12,
        fontWeight: FontWeight.w600,
        letterSpacing: 0.3,
      ),
    );
  }

  /// Switches tab and clears everything the previous tab left behind.
  ///
  /// _textOutput and _avatarRequest are single fields shared by the Voice and
  /// Text tabs, so without this a sentence typed in Text was still sitting in
  /// the Voice tab, the avatar replayed it there, and a stale "Listening..."
  /// stayed on screen because it is just _textOutput and nothing had
  /// overwritten it. Any live microphone session is stopped too — leaving one
  /// running behind a hidden tab holds the mic and blocks the next attempt.
  Future<void> _switchTab(_HubTab next) async {
    if (_tab == next) return;
    if (_isListening) {
      await OfflineSpeechService.stop();
    }
    if (!mounted) return;
    setState(() {
      _tab = next;
      _isListening = false;
      _textOutput = '';
      _glossLine = '';
      _avatarRequest = null;
    });
  }

  /// Guards against a second listen() starting while one is being set up.
  ///
  /// Google throttles rapid start/stop cycles with error_too_many_requests,
  /// and a double tap or an impatient second press was enough to trip it —
  /// after which every attempt fails for a while, which reads as the mic
  /// being broken rather than busy.
  bool _micBusy = false;

  /// Set once every speech path has failed, so the button stops pretending.
  ///
  /// Measured on this device: the Google recogniser is throttled
  /// (error_too_many_requests), the Google TTS recogniser has no matching
  /// language (error_language_unavailable), and MIUI's own dialog errors and
  /// closes. There is no working speech-to-text here, and a button that keeps
  /// offering one is worse than a button that says so.
  bool _micUnavailable = false;


  Widget _buildMicButton(AppColors c, String lang, bool isFilipino) {
    return GestureDetector(
      onTap: () async {
        if (_micUnavailable) {
          setState(() => _textOutput =
              'Voice input is not available on this device — '
              'use the Text tab instead');
          return;
        }
        if (_micBusy) return;
        _micBusy = true;
        try {
          await _handleMicTap(lang);
        } finally {
          _micBusy = false;
        }
      },
      child: _micVisual(c),
    );
  }

  Future<void> _handleMicTap(String lang) async {
    if (_isListening) {
      await OfflineSpeechService.stop();
      setState(() => _isListening = false);
      return;
    }

    setState(() {
      _isListening = true;
      _textOutput = 'Loading...';
    });

    // Offline first, and in practice only. Every system recogniser on this
    // device failed — Google's is throttled, Google TTS has no matching
    // language, MIUI's dialog errors out — so the bundled model is not a
    // fallback, it is the implementation.
    final ok = await OfflineSpeechService.initialize();
    if (!mounted) return;
    if (!ok) {
      setState(() {
        _isListening = false;
        _micUnavailable = true;
        _textOutput = 'Speech model failed to load — use the Text tab';
      });
      return;
    }

    setState(() => _textOutput = 'Listening...');
    await OfflineSpeechService.listen(
      onPartial: (partial) {
        // Words appear as they are spoken, so the pause before the final
        // result does not read as the app having missed you.
        if (!mounted || !_isListening) return;
        setState(() => _textOutput = partial);
      },
      onResult: (text) {
        if (!mounted) return;
        setState(() {
          _isListening = false;
          _textOutput = text;
        });
        // Sign it, but do NOT speak it back. The hearing user just said this
        // out loud; repeating it serves nobody, and it talks over them while
        // the avatar — the output the deaf user actually needs — is playing.
        _sign(text);
      },
      onFailure: (reason) {
        if (!mounted) return;
        setState(() {
          _isListening = false;
          _textOutput = reason.isEmpty ? '' : reason;
        });
      },
    );
  }

  Widget _micVisual(AppColors c) {
    return AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        width: 76,
        height: 76,
        decoration: BoxDecoration(
          color: _micUnavailable
              ? c.surface.withValues(alpha: 0.5)
              : (_isListening ? c.orange.withValues(alpha: 0.2) : c.surface),
          shape: BoxShape.circle,
          border: Border.all(
            color: _isListening ? c.orange : c.border,
            width: _isListening ? 2 : 1,
          ),
        ),
        child: Icon(
          _micUnavailable
              ? Icons.mic_off_rounded
              : (_isListening ? Icons.mic_rounded : Icons.mic_none_rounded),
          color: _micUnavailable
              ? c.subtitle
              : (_isListening ? c.orange : c.text),
          size: 34,
        ),
    );
  }

  // ----------------------------------------------------------------- text tab

  Widget _buildTextTab(AppColors c, bool isFilipino) {
    return Column(
      children: [
        AppCard(
          padding: EdgeInsets.zero,
          child: TextField(
            controller: _textController,
            maxLines: 4,
            style: TextStyle(color: c.text, fontSize: 15),
            decoration: InputDecoration(
              hintText: isFilipino
                  ? 'I-type ang iyong mensahe dito...'
                  : 'Type your message here...',
              hintStyle: TextStyle(color: c.text.withValues(alpha: 0.35)),
              border: InputBorder.none,
              contentPadding: const EdgeInsets.all(16),
            ),
          ),
        ),
        const SizedBox(height: 14),
        _buildVoiceSpeed(c, isFilipino),
        const SizedBox(height: 14),
        Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxHeight: 320),
            child: AvatarPlayer(
              request: _avatarRequest,
              isFilipino: isFilipino,
            ),
          ),
        ),
        if (_glossLine.isNotEmpty) ...[
          const SizedBox(height: 10),
          Center(child: _glossText(c)),
        ],
        const SizedBox(height: 14),
        _speechStatusCard(c, isFilipino),
        const SizedBox(height: 14),
        SizedBox(
          width: double.infinity,
          height: 54,
          child: ElevatedButton(
            onPressed: () async {
              final text = _textController.text.trim();
              if (text.isEmpty) return;
              // Sign first: speak() waits for playback to finish, so awaiting
              // it before signing would hold the avatar back for the whole
              // utterance instead of running the two together.
              _sign(text);
              await SpeechService.speak(text, speed: _voiceSpeed);
              // speak() may have rebound a dead engine, or given up on one.
              // Either way the status row above is now out of date.
              if (mounted) setState(() {});
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: c.orange,
              foregroundColor: Colors.white,
              elevation: 0,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16),
              ),
            ),
            child: Text(
              isFilipino
                  ? 'I-convert sa Pagsasalita + ASL'
                  : 'Convert to Speech + ASL',
              style:
                  const TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
            ),
          ),
        ),
        const SizedBox(height: 8),
      ],
    );
  }

  /// Collapsible speed control: starts expanded showing the slider, collapses
  /// to just the header row when the header is tapped.
  Widget _buildVoiceSpeed(AppColors c, bool isFilipino) {
    return AppCard(
      padding: EdgeInsets.zero,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          GestureDetector(
            behavior: HitTestBehavior.opaque,
            onTap: () => setState(() => _speedExpanded = !_speedExpanded),
            child: Padding(
              padding:
                  const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
              child: Row(
                children: [
                  Icon(Icons.volume_up_rounded, color: c.orange, size: 20),
                  const SizedBox(width: 10),
                  Text(
                    isFilipino ? 'Bilis ng Boses' : 'Voice Speed',
                    style: TextStyle(
                      color: c.text,
                      fontSize: 15,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const Spacer(),
                  Text(
                    '${_voiceSpeed.toStringAsFixed(2)}x',
                    style: TextStyle(
                      color: c.orange,
                      fontSize: 15,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(width: 8),
                  AnimatedRotation(
                    turns: _speedExpanded ? 0.5 : 0,
                    duration: const Duration(milliseconds: 200),
                    child: Icon(Icons.keyboard_arrow_down_rounded,
                        color: c.subtitle, size: 22),
                  ),
                ],
              ),
            ),
          ),
          AnimatedCrossFade(
            duration: const Duration(milliseconds: 220),
            sizeCurve: Curves.easeOutCubic,
            crossFadeState: _speedExpanded
                ? CrossFadeState.showFirst
                : CrossFadeState.showSecond,
            firstChild: Column(
              children: [
                Divider(height: 1, color: c.border),
                Padding(
                  padding: const EdgeInsets.fromLTRB(12, 6, 12, 12),
                  child: Column(
                    children: [
                      SliderTheme(
                        data: SliderTheme.of(context).copyWith(
                          trackHeight: 5,
                          overlayShape: const RoundSliderOverlayShape(
                              overlayRadius: 16),
                        ),
                        child: Slider(
                          value: _voiceSpeed,
                          min: _speedMin,
                          max: _speedMax,
                          divisions: 11,
                          onChanged: (v) => setState(() {
                            _voiceSpeed = v;
                            SpeechService.speedMultiplier = v;
                          }),
                        ),
                      ),
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 10),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: ['0.5x', '0.9x', '1.2x', '1.6x']
                              .map((t) => Text(
                                    t,
                                    style: TextStyle(
                                        color: c.subtitle, fontSize: 11),
                                  ))
                              .toList(),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            secondChild: const SizedBox(width: double.infinity),
          ),
        ],
      ),
    );
  }

  // ------------------------------------------------------------------ shared

  Widget _buildTabBar(AppColors c, bool isFilipino) {
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 8, 16, 12),
      padding: const EdgeInsets.all(6),
      decoration: BoxDecoration(
        color: c.surface,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: c.border),
      ),
      child: Row(
        children: [
          _TabButton(
            icon: Icons.photo_camera_outlined,
            label: isFilipino ? 'Kamera' : 'Camera',
            isSelected: _tab == _HubTab.camera,
            colors: c,
            onTap: () => _switchTab(_HubTab.camera),
          ),
          _TabButton(
            icon: Icons.mic_none_rounded,
            label: isFilipino ? 'Boses' : 'Voice',
            isSelected: _tab == _HubTab.voice,
            colors: c,
            onTap: () => _switchTab(_HubTab.voice),
          ),
          _TabButton(
            icon: Icons.title_rounded,
            label: isFilipino ? 'Teksto' : 'Text',
            isSelected: _tab == _HubTab.text,
            colors: c,
            onTap: () => _switchTab(_HubTab.text),
          ),
        ],
      ),
    );
  }
}

class _TabButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool isSelected;
  final AppColors colors;
  final VoidCallback onTap;

  const _TabButton({
    required this.icon,
    required this.label,
    required this.isSelected,
    required this.colors,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final c = colors;
    final fg = isSelected ? c.orange : c.subtitle;
    return Expanded(
      child: GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTap: onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          padding: const EdgeInsets.symmetric(vertical: 10),
          decoration: BoxDecoration(
            color: isSelected
                ? c.orange.withValues(alpha: 0.18)
                : Colors.transparent,
            borderRadius: BorderRadius.circular(14),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon, color: fg, size: 22),
              const SizedBox(height: 5),
              Text(
                label,
                style: TextStyle(
                  color: fg,
                  fontSize: 12,
                  fontWeight:
                      isSelected ? FontWeight.w600 : FontWeight.w400,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
