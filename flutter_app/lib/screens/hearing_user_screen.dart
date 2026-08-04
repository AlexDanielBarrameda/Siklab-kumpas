import 'package:flutter/material.dart';
import '../main.dart';
import '../services/speech_service.dart';

class HearingUserScreen extends StatefulWidget {
  const HearingUserScreen({super.key});

  @override
  State<HearingUserScreen> createState() => _HearingUserScreenState();
}

class _HearingUserScreenState extends State<HearingUserScreen> {
  static const _orange = Color(0xFFFF6B00);

  int _selectedTab = 0;
  bool _isListening = false;
  String _textOutput = '';
  final TextEditingController _textController = TextEditingController();
  /// Multiplier where 1.0 = natural speaking pace (see SpeechService).
  double _voiceSpeed = SpeechService.speedMultiplier;

  @override
  void initState() {
    super.initState();
    SpeechService.initSTT();
    SpeechService.initTTS();
  }

  @override
  void dispose() {
    _textController.dispose();
    SpeechService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final app = SiklabKumpasApp.of(context);
    final isDark = app?.isDark ?? true;
    final lang = app?.language ?? 'en';
    final isFilipino = lang == 'fil';

    final bgColor =
        isDark ? const Color(0xFF0D0D0D) : const Color(0xFFF5F0FF);
    final cardColor = isDark ? const Color(0xFF1A1A2E) : Colors.white;
    final textColor = isDark ? Colors.white : const Color(0xFF1A1A2E);
    final subtitleColor = isDark ? Colors.white54 : Colors.black45;
    final inputBg =
        isDark ? const Color(0xFF0D0D0D) : const Color(0xFFF5F5F5);

    return Scaffold(
      backgroundColor: bgColor,
      body: SafeArea(
        child: Column(
          children: [
            // Header
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 0),
              child: Row(
                children: [
                  GestureDetector(
                    onTap: () => Navigator.pop(context),
                    child: Icon(Icons.arrow_back_ios_new_rounded,
                        color: textColor, size: 20),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          isFilipino
                              ? 'Hub ng Pagsasalin'
                              : 'Translation Hub',
                          style: TextStyle(
                            color: textColor,
                            fontSize: 18,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                        Text(
                          'Real-time 3-Way Communication Bridge',
                          style: TextStyle(
                              color: subtitleColor, fontSize: 12),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),

            // Mode indicator
            Padding(
              padding:
                  const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              child: AnimatedSwitcher(
                duration: const Duration(milliseconds: 300),
                child: Align(
                  key: ValueKey(_selectedTab),
                  alignment: Alignment.centerLeft,
                  child: Text(
                    _selectedTab == 0
                        ? (isFilipino
                            ? 'Boses → Teksto + FSL Avatar'
                            : 'Voice → Text + FSL Avatar')
                        : (isFilipino
                            ? 'Teksto → Pagsasalita + FSL Avatar'
                            : 'Text → Speech + FSL Avatar'),
                    style: const TextStyle(
                      color: _orange,
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              ),
            ),

            // Content area with fade+slide transition
            Expanded(
              child: AnimatedSwitcher(
                duration: const Duration(milliseconds: 350),
                transitionBuilder: (child, animation) {
                  final offsetAnimation = Tween<Offset>(
                    begin: const Offset(0.08, 0),
                    end: Offset.zero,
                  ).animate(CurvedAnimation(
                    parent: animation,
                    curve: Curves.easeOutCubic,
                  ));
                  return FadeTransition(
                    opacity: animation,
                    child: SlideTransition(
                      position: offsetAnimation,
                      child: child,
                    ),
                  );
                },
                child: SingleChildScrollView(
                  key: ValueKey(_selectedTab),
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  child: _selectedTab == 0
                      ? _buildVoiceTab(cardColor, textColor,
                          subtitleColor, lang, isFilipino)
                      : _buildTextTab(cardColor, textColor,
                          subtitleColor, inputBg, isFilipino),
                ),
              ),
            ),

            // Bottom tab bar
            Container(
              margin: const EdgeInsets.all(16),
              padding: const EdgeInsets.all(6),
              decoration: BoxDecoration(
                color: cardColor,
                borderRadius: BorderRadius.circular(16),
              ),
              child: Row(
                children: [
                  _TabButton(
                    icon: Icons.mic_rounded,
                    label: isFilipino ? 'Boses' : 'Voice',
                    isSelected: _selectedTab == 0,
                    onTap: () => setState(() => _selectedTab = 0),
                  ),
                  _TabButton(
                    icon: Icons.text_fields_rounded,
                    label: isFilipino ? 'Teksto' : 'Text',
                    isSelected: _selectedTab == 1,
                    onTap: () => setState(() => _selectedTab = 1),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildVoiceTab(
    Color cardColor,
    Color textColor,
    Color subtitleColor,
    String lang,
    bool isFilipino,
  ) {
    return Column(
      children: [
        // Avatar placeholder
        Container(
          width: double.infinity,
          height: 220,
          decoration: BoxDecoration(
            color: cardColor,
            borderRadius: BorderRadius.circular(20),
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                width: 100,
                height: 100,
                decoration: const BoxDecoration(
                  color: Color(0xFF6C63FF),
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.back_hand_rounded,
                    color: Colors.white, size: 48),
              ),
              const SizedBox(height: 12),
              Text(
                '3D FSL Avatar',
                style: TextStyle(color: subtitleColor, fontSize: 14),
              ),
            ],
          ),
        ),

        const SizedBox(height: 20),

        // Mic button
        GestureDetector(
          onTap: () async {
            final available = await SpeechService.initSTT();
            if (!available) {
              setState(() => _textOutput = 'Speech recognition not available on this device');
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
            width: 72,
            height: 72,
            decoration: BoxDecoration(
              color: _isListening
                  ? _orange.withValues(alpha: 0.2)
                  : cardColor,
              shape: BoxShape.circle,
              border: Border.all(
                color: _isListening ? _orange : Colors.transparent,
                width: 2,
              ),
            ),
            child: Icon(
              _isListening
                  ? Icons.mic_rounded
                  : Icons.mic_none_rounded,
              color: _isListening ? _orange : textColor,
              size: 32,
            ),
          ),
        ),

        const SizedBox(height: 20),

        // Text output box
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: cardColor,
            borderRadius: BorderRadius.circular(16),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                isFilipino ? 'Output ng Teksto:' : 'Text Output:',
                style: TextStyle(color: subtitleColor, fontSize: 12),
              ),
              const SizedBox(height: 8),
              Text(
                _textOutput.isEmpty
                    ? (isFilipino
                        ? 'Pindutin ang mikropono para magsimula'
                        : 'Press microphone to start')
                    : _textOutput,
                style: TextStyle(
                  color: _textOutput.isEmpty ? subtitleColor : textColor,
                  fontSize: 15,
                ),
              ),
            ],
          ),
        ),

        const SizedBox(height: 16),
      ],
    );
  }

  Widget _buildTextTab(
    Color cardColor,
    Color textColor,
    Color subtitleColor,
    Color inputBg,
    bool isFilipino,
  ) {
    return Column(
      children: [
        // Voice speed
        Container(
          padding:
              const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          decoration: BoxDecoration(
            color: cardColor,
            borderRadius: BorderRadius.circular(16),
          ),
          child: Row(
            children: [
              const Icon(Icons.volume_up_rounded,
                  color: _orange, size: 20),
              const SizedBox(width: 8),
              Text(
                isFilipino ? 'Bilis ng Boses' : 'Voice Speed',
                style: TextStyle(color: textColor, fontSize: 14),
              ),
              const Spacer(),
              Text(
                '${_voiceSpeed.toStringAsFixed(2)}x',
                style: const TextStyle(
                  color: _orange,
                  fontSize: 14,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
        ),

        const SizedBox(height: 8),

        // Speed slider
        // Range capped at 1.6x: the engine tops out at its own maximum, so a
        // higher ceiling would be a slider that visibly moves without changing
        // anything (the old 0.5-2.0 range did exactly that above 1.0).
        Slider(
          value: _voiceSpeed,
          min: 0.5,
          max: 1.6,
          divisions: 11,
          activeColor: _orange,
          inactiveColor: _orange.withValues(alpha: 0.2),
          onChanged: (v) => setState(() {
            _voiceSpeed = v;
            SpeechService.speedMultiplier = v;
          }),
        ),

        const SizedBox(height: 8),

        // Text input
        Container(
          decoration: BoxDecoration(
            color: cardColor,
            borderRadius: BorderRadius.circular(16),
          ),
          child: TextField(
            controller: _textController,
            maxLines: 4,
            style: TextStyle(color: textColor, fontSize: 15),
            decoration: InputDecoration(
              hintText: isFilipino
                  ? 'I-type ang iyong mensahe dito...'
                  : 'Type your message here...',
              hintStyle:
                  TextStyle(color: textColor.withValues(alpha: 0.3)),
              border: InputBorder.none,
              contentPadding: const EdgeInsets.all(16),
            ),
          ),
        ),

        const SizedBox(height: 16),

        // Avatar placeholder
        Container(
          width: double.infinity,
          height: 200,
          decoration: BoxDecoration(
            color: cardColor,
            borderRadius: BorderRadius.circular(20),
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                width: 90,
                height: 90,
                decoration: const BoxDecoration(
                  color: Color(0xFF6C63FF),
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.back_hand_rounded,
                    color: Colors.white, size: 44),
              ),
              const SizedBox(height: 12),
              Text(
                isFilipino
                    ? 'Output ng FSL Avatar'
                    : 'FSL Avatar Output',
                style: TextStyle(color: subtitleColor, fontSize: 14),
              ),
            ],
          ),
        ),

        const SizedBox(height: 16),

        // Speech output status
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: cardColor,
            borderRadius: BorderRadius.circular(16),
          ),
          child: Row(
            children: [
              Text(
                isFilipino
                    ? 'Output ng Pagsasalita:'
                    : 'Speech Output:',
                style: TextStyle(color: subtitleColor, fontSize: 12),
              ),
              const SizedBox(width: 8),
              const Icon(Icons.circle, color: Colors.green, size: 10),
              const SizedBox(width: 4),
              Text(
                isFilipino ? 'Handa na ang audio' : 'Audio ready',
                style: const TextStyle(
                    color: Colors.green, fontSize: 13),
              ),
            ],
          ),
        ),

        const SizedBox(height: 16),

        // Convert button
        SizedBox(
          width: double.infinity,
          height: 56,
          child: ElevatedButton(
            onPressed: () {
              final text = _textController.text.trim();
              if (text.isNotEmpty) {
                SpeechService.speak(text, speed: _voiceSpeed);
              }
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: _orange,
              foregroundColor: Colors.white,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(32),
              ),
            ),
            child: Text(
              isFilipino
                  ? 'I-convert sa Pagsasalita + FSL'
                  : 'Convert to Speech + FSL',
              style: const TextStyle(
                  fontSize: 16, fontWeight: FontWeight.w700),
            ),
          ),
        ),

        const SizedBox(height: 16),
      ],
    );
  }
}

class _TabButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool isSelected;
  final VoidCallback onTap;

  const _TabButton({
    required this.icon,
    required this.label,
    required this.isSelected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: GestureDetector(
        onTap: onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          padding: const EdgeInsets.symmetric(vertical: 10),
          decoration: BoxDecoration(
            color: isSelected
                ? const Color(0xFFFF6B00).withValues(alpha: 0.15)
                : Colors.transparent,
            borderRadius: BorderRadius.circular(12),
          ),
          child: Column(
            children: [
              Icon(
                icon,
                color: isSelected
                    ? const Color(0xFFFF6B00)
                    : Colors.grey,
                size: 22,
              ),
              const SizedBox(height: 4),
              Text(
                label,
                style: TextStyle(
                  color: isSelected
                      ? const Color(0xFFFF6B00)
                      : Colors.grey,
                  fontSize: 12,
                  fontWeight: isSelected
                      ? FontWeight.w600
                      : FontWeight.w400,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}