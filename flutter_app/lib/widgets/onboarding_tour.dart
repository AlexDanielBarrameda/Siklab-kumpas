import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../main.dart';
import '../theme/tokens.dart';

/// One step of a tour.
class TourStep {
  final String titleEn;
  final String titleFil;
  final String bodyEn;
  final String bodyFil;
  final IconData icon;

  const TourStep({
    required this.titleEn,
    required this.titleFil,
    required this.bodyEn,
    required this.bodyFil,
    required this.icon,
  });
}

/// Which tour, so each screen remembers its own "seen" state.
enum TourId { initiator, hub }

/// First-run tours, and the (?) button that reopens them.
///
/// Shown once per screen on first use, then only on request. The two screens
/// are explained separately rather than from a single tour on Home: the
/// Initiator is a camera and the Hub is three tabs, and a combined tour would
/// have to describe controls the user cannot see yet.
class OnboardingTour {
  static const _channel = MethodChannel('com.siklabkumpas/inference');

  /// Marker files rather than shared_preferences.
  ///
  /// One boolean per screen does not justify a new dependency days before a
  /// defence — especially after the permission_handler upgrade needed for
  /// offline speech. The platform channel already exists and works.
  static String? _dir;

  static Future<String?> _filesDir() async {
    if (_dir != null) return _dir;
    try {
      _dir = await _channel.invokeMethod<String>('appFilesDir');
    } catch (e) {
      debugPrint('appFilesDir unavailable: $e');
    }
    return _dir;
  }

  static Future<bool> hasSeen(TourId id) async {
    final dir = await _filesDir();
    if (dir == null) return true;   // cannot tell: do not nag
    return File('$dir/tour_${id.name}.seen').existsSync();
  }

  static Future<void> markSeen(TourId id) async {
    final dir = await _filesDir();
    if (dir == null) return;
    try {
      await File('$dir/tour_${id.name}.seen').writeAsString('1', flush: true);
    } catch (e) {
      debugPrint('could not record tour state: $e');
    }
  }

  /// Shows the tour if this screen has not been seen before.
  static Future<void> maybeShow(
      BuildContext context, TourId id, bool isFilipino) async {
    if (await hasSeen(id)) return;
    if (!context.mounted) return;
    await show(context, id, isFilipino);
  }

  /// Shows the tour unconditionally — what the (?) button calls.
  static Future<void> show(
      BuildContext context, TourId id, bool isFilipino) async {
    final steps = id == TourId.initiator ? _initiatorSteps : _hubSteps;
    await showDialog<void>(
      context: context,
      barrierDismissible: false,
      builder: (_) => _TourDialog(steps: steps, isFilipino: isFilipino),
    );
    await markSeen(id);
  }

  static const _initiatorSteps = <TourStep>[
    TourStep(
      icon: Icons.videocam_rounded,
      titleEn: 'Point and start',
      titleFil: 'Itutok at simulan',
      bodyEn: 'Aim the camera at the person signing, then press Start. '
          'Keep the head and both hands in the frame.',
      bodyFil: 'Itutok ang camera sa taong nagse-sign, pagkatapos pindutin '
          'ang Simulan. Panatilihing nasa frame ang ulo at dalawang kamay.',
    ),
    TourStep(
      icon: Icons.back_hand_rounded,
      titleEn: 'Sign, rest, sign',
      titleFil: 'Sign, pahinga, sign',
      // The rhythm is not discoverable and is the most common reason a first
      // attempt fails, so it gets a step of its own.
      bodyEn: 'Make one sign, lower your hands for about a second, then make '
          'the next. The app uses that pause to tell signs apart.',
      bodyFil: 'Gumawa ng isang sign, ibaba ang kamay ng humigit-kumulang '
          'isang segundo, pagkatapos ay ang susunod. Ginagamit ng app ang '
          'paghinto na iyon upang paghiwalayin ang mga sign.',
    ),
    TourStep(
      icon: Icons.record_voice_over_rounded,
      titleEn: 'Signs become sentences',
      titleFil: 'Ang mga sign ay nagiging pangungusap',
      bodyEn: 'Signs are joined into a phrase and spoken aloud. '
          'HELLO GOOD MORNING is read as "Hello, good morning".',
      bodyFil: 'Pinagsasama ang mga sign sa isang parirala at binibigkas. '
          'Ang HELLO GOOD MORNING ay binabasa bilang "Hello, good morning".',
    ),
    TourStep(
      icon: Icons.abc_rounded,
      titleEn: 'Switch languages',
      titleFil: 'Magpalit ng wika',
      bodyEn: 'Tap the mode chip to move between ASL words, letters for '
          'fingerspelling, and FSL words.',
      bodyFil: 'Pindutin ang mode chip upang lumipat sa ASL words, letters '
          'para sa fingerspelling, at FSL words.',
    ),
  ];

  static const _hubSteps = <TourStep>[
    TourStep(
      icon: Icons.record_voice_over_rounded,
      titleEn: 'Speak or type',
      titleFil: 'Magsalita o mag-type',
      bodyEn: 'Say or write something and the avatar signs it back, so a deaf '
          'person can read you without an interpreter.',
      bodyFil: 'Magsalita o sumulat at isa-sign ito ng avatar, upang mabasa '
          'ka ng bingi nang walang interpreter.',
    ),
    TourStep(
      icon: Icons.mic_rounded,
      titleEn: 'Voice',
      titleFil: 'Boses',
      bodyEn: 'Tap the microphone and speak normally. Your words appear as '
          'you say them, then the avatar signs them.',
      bodyFil: 'Pindutin ang mikropono at magsalita nang normal. Lilitaw ang '
          'iyong mga salita habang sinasabi mo, pagkatapos ay isa-sign ito '
          'ng avatar.',
    ),
    TourStep(
      icon: Icons.keyboard_rounded,
      titleEn: 'Text',
      titleFil: 'Teksto',
      bodyEn: 'Type a sentence and send it. Useful in a noisy room, or when '
          'you would rather not speak.',
      bodyFil: 'Mag-type ng pangungusap at ipadala. Kapaki-pakinabang sa '
          'maingay na lugar, o kung ayaw mong magsalita.',
    ),
    TourStep(
      icon: Icons.videocam_rounded,
      titleEn: 'Camera',
      titleFil: 'Kamera',
      bodyEn: 'Point the camera at someone signing to read their signs as '
          'text and speech.',
      bodyFil: 'Itutok ang camera sa taong nagse-sign upang mabasa ang '
          'kanilang mga sign bilang teksto at pagsasalita.',
    ),
  ];
}

class _TourDialog extends StatefulWidget {
  final List<TourStep> steps;
  final bool isFilipino;

  const _TourDialog({required this.steps, required this.isFilipino});

  @override
  State<_TourDialog> createState() => _TourDialogState();
}

class _TourDialogState extends State<_TourDialog> {
  int _i = 0;

  @override
  Widget build(BuildContext context) {
    final c = AppColors.of(SiklabKumpasApp.of(context)?.isDark ?? true);
    final step = widget.steps[_i];
    final fil = widget.isFilipino;
    final last = _i == widget.steps.length - 1;

    return Dialog(
      backgroundColor: c.panel,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
      insetPadding: const EdgeInsets.symmetric(horizontal: 28, vertical: 24),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(22, 24, 22, 16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: 54,
              height: 54,
              decoration: BoxDecoration(
                color: c.orange.withValues(alpha: 0.15),
                shape: BoxShape.circle,
              ),
              child: Icon(step.icon, color: c.orange, size: 28),
            ),
            const SizedBox(height: 16),
            Text(
              fil ? step.titleFil : step.titleEn,
              style: TextStyle(
                color: c.text,
                fontSize: 19,
                fontWeight: FontWeight.w700,
                height: 1.2,
              ),
            ),
            const SizedBox(height: 10),
            Text(
              fil ? step.bodyFil : step.bodyEn,
              style: TextStyle(color: c.subtitle, fontSize: 14.5, height: 1.45),
            ),
            const SizedBox(height: 20),
            Row(
              children: [
                // Progress dots, so the reader knows how much is left.
                for (var k = 0; k < widget.steps.length; k++) ...[
                  Container(
                    width: k == _i ? 18 : 7,
                    height: 7,
                    decoration: BoxDecoration(
                      color: k == _i ? c.orange : c.border,
                      borderRadius: BorderRadius.circular(4),
                    ),
                  ),
                  const SizedBox(width: 5),
                ],
                const Spacer(),
                if (!last)
                  TextButton(
                    onPressed: () => Navigator.of(context).pop(),
                    child: Text(
                      fil ? 'Laktawan' : 'Skip all',
                      style: TextStyle(color: c.subtitle, fontSize: 14),
                    ),
                  ),
                const SizedBox(width: 4),
                FilledButton(
                  style: FilledButton.styleFrom(
                    backgroundColor: c.orange,
                    foregroundColor: Colors.white,
                    padding:
                        const EdgeInsets.symmetric(horizontal: 22, vertical: 12),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                  ),
                  onPressed: () {
                    if (last) {
                      Navigator.of(context).pop();
                    } else {
                      setState(() => _i++);
                    }
                  },
                  child: Text(
                    last
                        ? (fil ? 'Tapos na' : 'Got it')
                        : (fil ? 'Susunod' : 'Next'),
                    style: const TextStyle(
                        fontSize: 14, fontWeight: FontWeight.w600),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
