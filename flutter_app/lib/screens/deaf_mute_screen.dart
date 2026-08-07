import 'dart:async';

import 'package:flutter/material.dart';
import 'package:torch_light/torch_light.dart';
import '../main.dart';
import '../services/speech_service.dart';
import '../theme/tokens.dart';
import '../widgets/app_scaffold.dart';
import '../widgets/recognition_view.dart';

class DeafMuteScreen extends StatefulWidget {
  const DeafMuteScreen({super.key});

  @override
  State<DeafMuteScreen> createState() => _DeafMuteScreenState();
}

class _DeafMuteScreenState extends State<DeafMuteScreen> {
  /// The signal button starts as a labelled pill and shrinks to the bare bell
  /// after this long, freeing the screen once the affordance has been read.
  static const _collapseAfter = Duration(seconds: 3);

  /// Clearance kept at the bottom of the scroll area so content can be scrolled
  /// clear of the floating signal button.
  static const _signalReserve = 70.0;

  bool _signalCollapsed = false;
  Timer? _collapseTimer;

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

  @override
  void initState() {
    super.initState();
    _collapseTimer = Timer(_collapseAfter, () {
      if (mounted) setState(() => _signalCollapsed = true);
    });
  }

  @override
  void dispose() {
    _collapseTimer?.cancel();
    super.dispose();
  }

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
    final isFilipino = (app?.language ?? 'en') == 'fil';

    return AppScaffold(
      showBack: true,
      title: isFilipino ? 'Modong Tagapagsimula' : 'Initiator Mode',
      subtitle: isFilipino ? 'Bingi / Pipi POV' : 'Deaf / Mute POV',
      modeLine: isFilipino
          ? 'FSL → Teksto + Pagsasalita'
          : 'FSL → Text + Speech Conversion',
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
        // The signal button floats over the scrolling content instead of
        // sitting under it in a Column, so it stays reachable no matter how
        // far the translation log has grown.
        child: Stack(
          children: [
            Positioned.fill(
              child: RecognitionView(
                isFilipino: isFilipino,
                bottomInset: _signalReserve,
              ),
            ),
            Positioned(
              right: 0,
              bottom: 0,
              child: _SignalButton(
                collapsed: _signalCollapsed,
                label: isFilipino
                    ? 'Senyas para sa Pag-uusap'
                    : 'Signal for Conversation',
                onTap: () => _showSignalModal(context, isFilipino),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Orange pill that shrinks to a circular bell. Tapping it opens the signal
/// sheet in either state.
///
/// Deliberately not AnimatedCrossFade. Its default layout builder pins the
/// outgoing child with `top: 0.0` and no bottom constraint, so the label sat
/// flush against the top of the capsule instead of centred; and because the two
/// children had different intrinsic heights it animated the capsule's *height*
/// as well, re-laying out the column on every frame. Here the height is a
/// constant [_height] and only the label's width collapses, so the bell holds
/// its baseline and the column never reflows.
class _SignalButton extends StatefulWidget {
  final bool collapsed;
  final String label;
  final VoidCallback onTap;

  const _SignalButton({
    required this.collapsed,
    required this.label,
    required this.onTap,
  });

  static const _height = 56.0;

  @override
  State<_SignalButton> createState() => _SignalButtonState();
}

class _SignalButtonState extends State<_SignalButton>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 460),
    // 1.0 = fully expanded, 0.0 = bare bell.
    value: widget.collapsed ? 0.0 : 1.0,
  );

  late final Animation<double> _t = CurvedAnimation(
    parent: _controller,
    curve: Curves.easeInOutCubic,
  );

  @override
  void didUpdateWidget(_SignalButton oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.collapsed != oldWidget.collapsed) {
      widget.collapsed ? _controller.reverse() : _controller.forward();
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    const bell =
        Icon(Icons.notifications_rounded, color: Colors.white, size: 22);

    return GestureDetector(
      onTap: widget.onTap,
      child: AnimatedBuilder(
        animation: _t,
        builder: (context, _) {
          return Container(
            height: _SignalButton._height,
            clipBehavior: Clip.antiAlias,
            decoration: BoxDecoration(
              color: AppTokens.orange,
              borderRadius:
                  BorderRadius.circular(_SignalButton._height / 2),
              boxShadow: [
                BoxShadow(
                  color: AppTokens.orange.withValues(alpha: 0.35),
                  blurRadius: 18,
                  offset: const Offset(0, 6),
                ),
              ],
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                // Fixed-width slot equal to the collapsed diameter, so the bell
                // stays perfectly centred at both ends of the animation.
                const SizedBox(
                  width: _SignalButton._height,
                  child: Center(child: bell),
                ),
                ClipRect(
                  child: Align(
                    alignment: Alignment.centerLeft,
                    widthFactor: _t.value,
                    child: Opacity(
                      opacity: _t.value,
                      child: Padding(
                        padding: const EdgeInsets.only(right: 22),
                        child: Text(
                          widget.label,
                          maxLines: 1,
                          softWrap: false,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 15,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          );
        },
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

  @override
  Widget build(BuildContext context) {
    final c = AppColors.of(SiklabKumpasApp.of(context)?.isDark ?? true);
    final isFilipino = widget.isFilipino;

    return Container(
      margin: const EdgeInsets.all(16),
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: c.panel,
        borderRadius: BorderRadius.circular(24),
      ),
      child: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        isFilipino
                            ? 'Senyas para sa Pag-uusap'
                            : 'Signal for Conversation',
                        style: TextStyle(
                          color: c.text,
                          fontSize: 18,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      Text(
                        isFilipino
                            ? 'Pumili kung paano kunin ang kanilang atensyon'
                            : 'Choose how to get their attention',
                        style: TextStyle(color: c.subtitle, fontSize: 12),
                      ),
                    ],
                  ),
                ),
                GestureDetector(
                  onTap: () => Navigator.pop(context),
                  child: Icon(Icons.close, color: c.subtitle),
                ),
              ],
            ),

            const SizedBox(height: 20),

            // Flicker section
            Row(
              children: [
                Icon(Icons.lightbulb_outline, color: c.orange, size: 20),
                const SizedBox(width: 8),
                Text(
                  isFilipino ? 'Kumikislap na Ilaw' : 'Flicker Lights',
                  style: TextStyle(
                    color: c.text,
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
              style: TextStyle(color: c.subtitle, fontSize: 12),
            ),
            const SizedBox(height: 10),

            Container(
              padding:
                  const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
              decoration: BoxDecoration(
                color: _flickerEnabled
                    ? c.orange.withValues(alpha: 0.1)
                    : c.panelAlt,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(
                  color: _flickerEnabled ? c.orange : Colors.transparent,
                  width: 1.5,
                ),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    isFilipino ? 'Kumikislap na Ilaw' : 'Flicker Lights',
                    style: TextStyle(
                      color: _flickerEnabled ? c.orange : c.text,
                      fontSize: 14,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                  Switch(
                    value: _flickerEnabled,
                    onChanged: (v) => setState(() => _flickerEnabled = v),
                    activeThumbColor: c.orange,
                    activeTrackColor: c.orange.withValues(alpha: 0.3),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 20),

            // Audio spiel section
            Row(
              children: [
                Icon(Icons.volume_up_rounded, color: c.orange, size: 20),
                const SizedBox(width: 8),
                Text(
                  isFilipino ? 'Mag-play ng Audio Spiel' : 'Play Audio Spiel',
                  style: TextStyle(
                    color: c.text,
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
              style: TextStyle(color: c.subtitle, fontSize: 12),
            ),
            const SizedBox(height: 10),

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
                          ? c.orange.withValues(alpha: 0.1)
                          : c.panelAlt,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                        color: isSelected ? c.orange : Colors.transparent,
                        width: 1.5,
                      ),
                    ),
                    child: Row(
                      children: [
                        Expanded(
                          child: Text(
                            widget.preRecorded[i],
                            style: TextStyle(
                              color: isSelected ? c.orange : c.text,
                              fontSize: 14,
                              fontWeight: isSelected
                                  ? FontWeight.w600
                                  : FontWeight.w400,
                            ),
                          ),
                        ),
                        if (isSelected)
                          Icon(Icons.circle, color: c.orange, size: 10),
                      ],
                    ),
                  ),
                ),
              );
            }),

            const SizedBox(height: 16),

            SizedBox(
              width: double.infinity,
              height: 52,
              child: ElevatedButton(
                onPressed: () async {
                  Navigator.pop(context); // Close the modal

                  // Flash the torch 3 times
                  if (_flickerEnabled) {
                    try {
                      final isAvailable = await TorchLight.isTorchAvailable();
                      if (isAvailable) {
                        for (int i = 0; i < 3; i++) {
                          await TorchLight.enableTorch();
                          await Future.delayed(
                              const Duration(milliseconds: 200));
                          await TorchLight.disableTorch();
                          await Future.delayed(
                              const Duration(milliseconds: 200));
                        }
                      }
                    } catch (e) {
                      debugPrint('Torch error: $e');
                    }
                  }

                  // Play the audio spiel if not "None" (index 0)
                  if (_selectedMessage > 0) {
                    SpeechService.setTTSLanguage(isFilipino ? 'fil' : 'en');
                    SpeechService.speak(widget.preRecorded[_selectedMessage]);
                  }
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: c.orange,
                  foregroundColor: Colors.white,
                  elevation: 0,
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

            SizedBox(
              width: double.infinity,
              height: 48,
              child: TextButton(
                onPressed: () => Navigator.pop(context),
                child: Text(
                  isFilipino ? 'Kanselahin' : 'Cancel',
                  style: TextStyle(color: c.subtitle, fontSize: 15),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
