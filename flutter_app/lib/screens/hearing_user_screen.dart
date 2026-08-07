import 'package:flutter/material.dart';
import '../main.dart';
import '../services/speech_service.dart';
import '../theme/tokens.dart';
import '../widgets/app_scaffold.dart';
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

  @override
  void initState() {
    super.initState();
    // Speech plugin bring-up is slow enough to swallow the push animation from
    // Home, which made this screen look like it appeared with no transition at
    // all. Hold it until the route has settled.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      afterRouteTransition(context, () {
        if (!mounted) return;
        SpeechService.initSTT();
        SpeechService.initTTS();
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
            ? 'FSL → Teksto + Pagsasalita'
            : 'FSL → Text + Speech';
      case _HubTab.voice:
        return isFilipino
            ? 'Boses → Teksto + FSL Avatar'
            : 'Voice → Text + FSL Avatar';
      case _HubTab.text:
        return isFilipino
            ? 'Teksto → Pagsasalita + FSL Avatar'
            : 'Text → Speech + FSL Avatar';
    }
  }

  @override
  Widget build(BuildContext context) {
    final app = SiklabKumpasApp.of(context);
    final c = AppColors.of(app?.isDark ?? true);
    final lang = app?.language ?? 'en';
    final isFilipino = lang == 'fil';

    return AppScaffold(
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
    return TabSwitcher(
      tabKey: _tab,
      child: _tab == _HubTab.camera
          ? Padding(
              key: const ValueKey(_HubTab.camera),
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
              child: RecognitionView(
                isFilipino: isFilipino,
                initDelay: _tabAnimation,
              ),
            )
          : SingleChildScrollView(
              key: ValueKey(_tab),
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
              child: _tab == _HubTab.voice
                  ? _buildVoiceTab(c, lang, isFilipino)
                  : _buildTextTab(c, isFilipino),
            ),
    );
  }

  // ---------------------------------------------------------------- voice tab

  Widget _buildVoiceTab(AppColors c, String lang, bool isFilipino) {
    return Column(
      children: [
        // Output first, matching UI.3 — the hearing user reads this, so it sits
        // above the avatar rather than below it.
        AppCard(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                isFilipino ? 'Output ng Teksto:' : 'Text Output:',
                style: TextStyle(color: c.subtitle, fontSize: 12),
              ),
              const SizedBox(height: 8),
              Text(
                _textOutput.isEmpty
                    ? (isFilipino
                        ? 'Pindutin ang mikropono para magsimula'
                        : 'Press microphone to start')
                    : _textOutput,
                style: TextStyle(
                  color: _textOutput.isEmpty ? c.subtitle : c.text,
                  fontSize: 16,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 16),
        _avatarCard(
          c,
          height: 210,
          label: isFilipino ? '3D FSL Avatar' : '3D FSL Avatar',
          puck: 100,
        ),
        const SizedBox(height: 20),
        _buildMicButton(c, lang, isFilipino),
        const SizedBox(height: 8),
      ],
    );
  }

  Widget _buildMicButton(AppColors c, String lang, bool isFilipino) {
    return GestureDetector(
      onTap: () async {
        final available = await SpeechService.initSTT();
        if (!available) {
          setState(() => _textOutput =
              'Speech recognition not available on this device');
          return;
        }

        if (_isListening) {
          await SpeechService.stopListening();
          setState(() => _isListening = false);
        } else {
          setState(() {
            _isListening = true;
            _textOutput = 'Listening...';
          });
          await SpeechService.startListening(
            lang: lang,
            onResult: (text) {
              setState(() {
                _textOutput = text;
                _isListening = false;
              });
              SpeechService.speak(text);
            },
          );
        }
      },
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        width: 76,
        height: 76,
        decoration: BoxDecoration(
          color: _isListening ? c.orange.withValues(alpha: 0.2) : c.surface,
          shape: BoxShape.circle,
          border: Border.all(
            color: _isListening ? c.orange : c.border,
            width: _isListening ? 2 : 1,
          ),
        ),
        child: Icon(
          _isListening ? Icons.mic_rounded : Icons.mic_none_rounded,
          color: _isListening ? c.orange : c.text,
          size: 34,
        ),
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
        _avatarCard(
          c,
          height: 190,
          label: isFilipino ? 'Output ng FSL Avatar' : 'FSL Avatar Output',
          puck: 90,
        ),
        const SizedBox(height: 14),
        AppCard(
          child: Row(
            children: [
              Text(
                isFilipino ? 'Output ng Pagsasalita:' : 'Speech Output:',
                style: TextStyle(color: c.subtitle, fontSize: 12),
              ),
              const SizedBox(width: 10),
              const Icon(Icons.circle, color: Color(0xFF28C840), size: 9),
              const SizedBox(width: 5),
              Text(
                isFilipino ? 'Handa na ang audio' : 'Audio ready',
                style: const TextStyle(
                  color: Color(0xFF28C840),
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 14),
        SizedBox(
          width: double.infinity,
          height: 54,
          child: ElevatedButton(
            onPressed: () {
              final text = _textController.text.trim();
              if (text.isNotEmpty) {
                SpeechService.speak(text, speed: _voiceSpeed);
              }
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
                  ? 'I-convert sa Pagsasalita + FSL'
                  : 'Convert to Speech + FSL',
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

  Widget _avatarCard(
    AppColors c, {
    required double height,
    required String label,
    required double puck,
  }) {
    return AppCard(
      gradient: c.avatarCard,
      padding: EdgeInsets.zero,
      radius: 20,
      child: SizedBox(
        height: height,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              width: puck,
              height: puck,
              decoration: const BoxDecoration(
                shape: BoxShape.circle,
                gradient: LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [Color(0xFF7C5CFF), Color(0xFF5B4BFF)],
                ),
              ),
              child: Icon(Icons.back_hand_rounded,
                  color: Colors.white, size: puck * 0.48),
            ),
            const SizedBox(height: 14),
            Text(label, style: TextStyle(color: c.subtitle, fontSize: 14)),
          ],
        ),
      ),
    );
  }

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
            onTap: () => setState(() => _tab = _HubTab.camera),
          ),
          _TabButton(
            icon: Icons.mic_none_rounded,
            label: isFilipino ? 'Boses' : 'Voice',
            isSelected: _tab == _HubTab.voice,
            colors: c,
            onTap: () => setState(() => _tab = _HubTab.voice),
          ),
          _TabButton(
            icon: Icons.title_rounded,
            label: isFilipino ? 'Teksto' : 'Text',
            isSelected: _tab == _HubTab.text,
            colors: c,
            onTap: () => setState(() => _tab = _HubTab.text),
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
