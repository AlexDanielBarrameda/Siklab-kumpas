import 'package:flutter/material.dart';
import '../main.dart';

class DeafMuteScreen extends StatefulWidget {
  const DeafMuteScreen({super.key});

  @override
  State<DeafMuteScreen> createState() => _DeafMuteScreenState();
}

class _DeafMuteScreenState extends State<DeafMuteScreen> {
  static const _orange = Color(0xFFFF6B00);

  bool _isDetecting = false;
  List<String> _translationLog = [];

  final List<String> _preRecordedEn = [
    'None',
    'Hello, can I talk to you?',
    'Excuse me, may I have a moment?',
    'Hi, can you help me with something?',
    'Good day! Can we have a conversation?',
  ];

  final List<String> _preRecordedFil = [
    'Wala',
    'Kumusta, maari ka bang makausap?',
    'Pasensya na, maari po bang makahingi ng sandali?',
    'Hi, maari mo ba akong tulungan?',
    'Magandang araw! Maari ba tayong mag-usap?',
  ];

  void _showSignalModal(BuildContext context, bool isFilipino) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => _SignalModal(
        isFilipino: isFilipino,
        preRecorded: isFilipino ? _preRecordedFil : _preRecordedEn,
      ),
    );
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
                          isFilipino ? 'Initiator Mode' : 'Initiator Mode',
                          style: TextStyle(
                            color: textColor,
                            fontSize: 18,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                        Text(
                          isFilipino ? 'Bingi / Pipi POV' : 'Deaf / Mute POV',
                          style: TextStyle(
                              color: subtitleColor, fontSize: 12),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),

            // Mode label
            Padding(
              padding:
                  const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
              child: Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  isFilipino
                      ? 'FSL → Teksto + Pagsasalita'
                      : 'FSL → Text + Speech Conversion',
                  style: const TextStyle(
                    color: _orange,
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ),

            // Signal button
            Padding(
              padding:
                  const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              child: GestureDetector(
                onTap: () => _showSignalModal(context, isFilipino),
                child: Container(
                  width: double.infinity,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  decoration: BoxDecoration(
                    color: _isDetecting
                        ? _orange
                        : cardColor,
                    borderRadius: BorderRadius.circular(32),
                    border: Border.all(
                      color: _orange.withValues(alpha: 0.3),
                      width: 1,
                    ),
                  ),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(
                        Icons.notifications_outlined,
                        color: _isDetecting ? Colors.white : textColor,
                        size: 20,
                      ),
                      const SizedBox(width: 8),
                      Text(
                        isFilipino
                            ? 'Senyas para sa Pag-uusap'
                            : 'Signal for Conversation',
                        style: TextStyle(
                          color:
                              _isDetecting ? Colors.white : textColor,
                          fontSize: 15,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),

            // Camera view
            Expanded(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: Container(
                  width: double.infinity,
                  decoration: BoxDecoration(
                    color: const Color(0xFF1A2332),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                      color: _orange.withValues(alpha: 0.4),
                      width: 1.5,
                    ),
                  ),
                  child: Stack(
                    children: [
                      // Camera placeholder
                      Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.camera_alt_outlined,
                                color: Colors.white24, size: 48),
                            const SizedBox(height: 8),
                            Text(
                              isFilipino
                                  ? 'Kamera'
                                  : 'Camera',
                              style: const TextStyle(
                                  color: Colors.white24, fontSize: 13),
                            ),
                          ],
                        ),
                      ),

                      // LIVE badge
                      Positioned(
                        top: 12,
                        left: 12,
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 10, vertical: 4),
                          decoration: BoxDecoration(
                            color: Colors.red,
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: const [
                              Icon(Icons.circle,
                                  color: Colors.white, size: 8),
                              SizedBox(width: 4),
                              Text(
                                'LIVE',
                                style: TextStyle(
                                  color: Colors.white,
                                  fontSize: 11,
                                  fontWeight: FontWeight.w700,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),

                      // Start/Stop detecting button
                      Positioned(
                        bottom: 12,
                        right: 12,
                        child: GestureDetector(
                          onTap: () =>
                              setState(() => _isDetecting = !_isDetecting),
                          child: Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 14, vertical: 8),
                            decoration: BoxDecoration(
                              color: _isDetecting
                                  ? Colors.red
                                  : _orange,
                              borderRadius: BorderRadius.circular(20),
                            ),
                            child: Text(
                              _isDetecting
                                  ? (isFilipino ? 'Ihinto' : 'Stop')
                                  : (isFilipino ? 'Simulan' : 'Start'),
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 12,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),

            // Translation output log
            Container(
              margin: const EdgeInsets.all(16),
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: cardColor,
                borderRadius: BorderRadius.circular(16),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Icon(Icons.circle,
                          color: Colors.red, size: 8),
                      const SizedBox(width: 4),
                      const Icon(Icons.circle,
                          color: Colors.orange, size: 8),
                      const SizedBox(width: 4),
                      const Icon(Icons.circle,
                          color: Colors.green, size: 8),
                      const SizedBox(width: 8),
                      Text(
                        isFilipino
                            ? 'Output ng Pagsalin (Teksto + Pagsasalita)'
                            : 'Translation Output (Text + Speech)',
                        style: TextStyle(
                            color: subtitleColor, fontSize: 11),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    _isDetecting
                        ? (isFilipino
                            ? '> Naghihintay ng sign input...'
                            : '> Waiting for sign input...')
                        : (isFilipino
                            ? '> Pindutin ang Simulan para magsimula'
                            : '> Press Start to begin detecting'),
                    style: TextStyle(
                      color: _isDetecting ? _orange : subtitleColor,
                      fontSize: 13,
                      fontFamily: 'monospace',
                    ),
                  ),
                  if (_translationLog.isNotEmpty) ...[
                    const SizedBox(height: 4),
                    ..._translationLog.map((log) => Text(
                          '> $log',
                          style: TextStyle(
                              color: textColor, fontSize: 13),
                        )),
                  ],
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _SignalModal extends StatefulWidget {
  final bool isFilipino;
  final List<String> preRecorded;

  const _SignalModal({
    required this.isFilipino,
    required this.preRecorded,
  });

  @override
  State<_SignalModal> createState() => _SignalModalState();
}

class _SignalModalState extends State<_SignalModal> {
  bool _flickerEnabled = false;
  int _selectedMessage = 0;

  static const _orange = Color(0xFFFF6B00);

  @override
  Widget build(BuildContext context) {
    final app = SiklabKumpasApp.of(context);
    final isDark = app?.isDark ?? true;
    final isFilipino = widget.isFilipino;

    final bgColor = isDark ? const Color(0xFF1A1A2E) : Colors.white;
    final cardColor =
        isDark ? const Color(0xFF0D0D0D) : const Color(0xFFF5F5F5);
    final textColor = isDark ? Colors.white : const Color(0xFF1A1A2E);
    final subtitleColor = isDark ? Colors.white54 : Colors.black45;

    return Container(
      margin: const EdgeInsets.all(16),
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(24),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Title row
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    isFilipino
                        ? 'Senyas para sa Pag-uusap'
                        : 'Signal for Conversation',
                    style: TextStyle(
                      color: textColor,
                      fontSize: 18,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  Text(
                    isFilipino
                        ? 'Pumili kung paano kunin ang kanilang atensyon'
                        : 'Choose how to get their attention',
                    style:
                        TextStyle(color: subtitleColor, fontSize: 12),
                  ),
                ],
              ),
              GestureDetector(
                onTap: () => Navigator.pop(context),
                child: Icon(Icons.close, color: subtitleColor),
              ),
            ],
          ),

          const SizedBox(height: 20),

          // Flicker lights section
          Row(
            children: [
              const Icon(Icons.lightbulb_outline,
                  color: _orange, size: 20),
              const SizedBox(width: 8),
              Text(
                isFilipino ? 'Kumikislap na Ilaw' : 'Flicker Lights',
                style: TextStyle(
                  color: textColor,
                  fontSize: 15,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            isFilipino
                ? 'I-flash ang screen para makuha ang atensyon'
                : 'Flash the screen to get attention',
            style: TextStyle(color: subtitleColor, fontSize: 12),
          ),
          const SizedBox(height: 10),

          // Flicker toggle
          Container(
            padding:
                const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            decoration: BoxDecoration(
              color: _flickerEnabled
                  ? _orange.withValues(alpha: 0.1)
                  : cardColor,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(
                color:
                    _flickerEnabled ? _orange : Colors.transparent,
                width: 1.5,
              ),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  isFilipino ? 'Kumikislap na Ilaw' : 'Flicker Lights',
                  style: TextStyle(
                    color: _flickerEnabled ? _orange : textColor,
                    fontSize: 14,
                    fontWeight: FontWeight.w500,
                  ),
                ),
                Switch(
                  value: _flickerEnabled,
                  onChanged: (v) =>
                      setState(() => _flickerEnabled = v),
                  activeColor: _orange,
                ),
              ],
            ),
          ),

          const SizedBox(height: 20),

          // Audio spiel section
          Row(
            children: [
              const Icon(Icons.volume_up_rounded,
                  color: _orange, size: 20),
              const SizedBox(width: 8),
              Text(
                isFilipino ? 'Mag-play ng Audio Spiel' : 'Play Audio Spiel',
                style: TextStyle(
                  color: textColor,
                  fontSize: 15,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            isFilipino
                ? 'Pumili ng pre-recorded na mensahe'
                : 'Select a pre-recorded message',
            style: TextStyle(color: subtitleColor, fontSize: 12),
          ),
          const SizedBox(height: 10),

          // Message list
          ...List.generate(widget.preRecorded.length, (i) {
            final isSelected = _selectedMessage == i;
            return Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: GestureDetector(
                onTap: () => setState(() => _selectedMessage = i),
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 200),
                  padding: const EdgeInsets.symmetric(
                      horizontal: 16, vertical: 12),
                  decoration: BoxDecoration(
                    color: isSelected
                        ? _orange.withValues(alpha: 0.1)
                        : cardColor,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                      color: isSelected ? _orange : Colors.transparent,
                      width: 1.5,
                    ),
                  ),
                  child: Row(
                    children: [
                      Expanded(
                        child: Text(
                          widget.preRecorded[i],
                          style: TextStyle(
                            color: isSelected ? _orange : textColor,
                            fontSize: 14,
                            fontWeight: isSelected
                                ? FontWeight.w600
                                : FontWeight.w400,
                          ),
                        ),
                      ),
                      if (isSelected)
                        const Icon(Icons.circle,
                            color: _orange, size: 10),
                    ],
                  ),
                ),
              ),
            );
          }),

          const SizedBox(height: 16),

          // Start conversation button
          SizedBox(
            width: double.infinity,
            height: 52,
            child: ElevatedButton(
              onPressed: () {
                Navigator.pop(context);
                // Signal logic will be wired here
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
                    ? 'Simulan ang Pag-uusap'
                    : 'Start Conversation',
                style: const TextStyle(
                    fontSize: 16, fontWeight: FontWeight.w700),
              ),
            ),
          ),

          const SizedBox(height: 10),

          // Cancel button
          SizedBox(
            width: double.infinity,
            height: 48,
            child: TextButton(
              onPressed: () => Navigator.pop(context),
              child: Text(
                isFilipino ? 'Kanselahin' : 'Cancel',
                style: TextStyle(color: subtitleColor, fontSize: 15),
              ),
            ),
          ),
        ],
      ),
    );
  }
}