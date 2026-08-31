import 'package:flutter/material.dart';
import '../main.dart';
import '../theme/tokens.dart';
import '../widgets/app_scaffold.dart';
import '../widgets/brand_icons.dart';
import '../widgets/transitions.dart';
import 'hearing_user_screen.dart';
import 'deaf_mute_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final app = SiklabKumpasApp.of(context);
    final c = AppColors.of(app?.isDark ?? true);
    final isFilipino = (app?.language ?? 'en') == 'fil';

    return AppScaffold(
      // Scrolls only when the viewport cannot fit the stacked cards; on a
      // normal phone the Spacers below distribute the slack instead.
      child: LayoutBuilder(
        builder: (context, constraints) {
          return SingleChildScrollView(
            child: ConstrainedBox(
              constraints: BoxConstraints(minHeight: constraints.maxHeight),
              child: IntrinsicHeight(
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 20),
                  child: Column(
                    children: [
                      BrandIcons.wordmark(width: 210),
                      const SizedBox(height: 8),
                      Text(
                        isFilipino
                            ? 'Tulay sa Pagitan'
                            : "Let's Bridge the Gap",
                        style: TextStyle(
                          color: c.subtitle,
                          fontSize: 15,
                          fontWeight: FontWeight.w400,
                        ),
                      ),
                      // Most of the slack goes below the cards rather than
                      // above them, so the pair sits nearer the middle of the
                      // screen instead of hugging the bottom.
                      const Spacer(flex: 1),
                      Text(
                        isFilipino
                            ? 'Pumili ng Iyong Profile'
                            : 'Select Your Profile',
                        style: TextStyle(
                          color: c.text.withValues(alpha: 0.85),
                          fontSize: 17,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                      const SizedBox(height: 16),
                      _ProfileCard(
                        title: isFilipino
                            ? 'Kasama sa Komunikasyon'
                            : 'Inclusive Communicator',
                        caption: isFilipino
                            ? 'Nakaririnig / Nakakapagsalita'
                            : 'Hearing / Speaking Users',
                        icon: BrandIcons.hearingSpeaking(size: 74),
                        onTap: () => Navigator.push(
                          context,
                          appPageRoute(const HearingUserScreen()),
                        ),
                      ),
                      const SizedBox(height: 14),
                      _ProfileCard(
                        title: isFilipino
                            ? 'Gumagamit ng Sign Language'
                            : 'Sign Language User',
                        caption: isFilipino
                            ? 'Bingi / Pipi'
                            : 'Deaf / Mute Users',
                        icon: BrandIcons.fslUser(size: 74),
                        onTap: () => Navigator.push(
                          context,
                          appPageRoute(const DeafMuteScreen()),
                        ),
                      ),
                      const Spacer(flex: 3),
                    ],
                  ),
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}

class _ProfileCard extends StatelessWidget {
  final String title;
  final String caption;
  final Widget icon;
  final VoidCallback onTap;

  const _ProfileCard({
    required this.title,
    required this.caption,
    required this.icon,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppColors.of(SiklabKumpasApp.of(context)?.isDark ?? true);

    return PressablePop(
      onTap: onTap,
      child: AppCard(
        elevated: true,
        radius: 22,
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 18),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              title,
              textAlign: TextAlign.center,
              style: TextStyle(
                color: c.text,
                fontSize: 17,
                height: 1.25,
                fontWeight: FontWeight.w700,
              ),
            ),
            const SizedBox(height: 14),
            icon,
            const SizedBox(height: 14),
            Text(
              caption,
              textAlign: TextAlign.center,
              style: TextStyle(color: c.subtitle, fontSize: 13),
            ),
          ],
        ),
      ),
    );
  }
}
