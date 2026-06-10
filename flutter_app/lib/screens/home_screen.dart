import 'package:flutter/material.dart';
import '../main.dart';
import 'hearing_user_screen.dart';
import 'deaf_mute_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  static const _orange = Color(0xFFFF6B00);
  static const _purple = Color(0xFF7B5EA7);

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    // Forces rebuild when parent state (theme/language) changes
    setState(() {});
  }

  void _showSettingsModal(BuildContext context) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => const _SettingsModal(),
    );
  }

  @override
  Widget build(BuildContext context) {
    final app = SiklabKumpasApp.of(context);
    final isDark = app?.isDark ?? true;
    final lang = app?.language ?? 'en';
    final isFilipino = lang == 'fil';

    final bgColor = isDark ? const Color(0xFF0D0D0D) : const Color(0xFFF5F0FF);
    final cardColor = isDark ? const Color(0xFF1A1A2E) : Colors.white;
    final textColor = isDark ? Colors.white : const Color(0xFF1A1A2E);
    final subtitleColor = isDark ? Colors.white54 : Colors.black45;

    return Scaffold(
      backgroundColor: bgColor,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
          child: Column(
            children: [
              // Top bar
              Row(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  _IconButton(
                    icon: Icons.settings_outlined,
                    isDark: isDark,
                    onTap: () => _showSettingsModal(context),
                  ),
                  const SizedBox(width: 8),
                  _IconButton(
                    icon: isDark
                        ? Icons.wb_sunny_outlined
                        : Icons.nightlight_round,
                    isDark: isDark,
                    onTap: () {
                      app?.toggleTheme();
                      setState(() {});
                    },
                  ),
                ],
              ),

              // Center content shifted up 30px
              Expanded(
                child: Transform.translate(
                  offset: const Offset(0, -30),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      // Logo + Title
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(Icons.star_outline_rounded,
                              color: _orange, size: 32),
                          const SizedBox(width: 10),
                          Text(
                            'SIKLAB-KUMPAS',
                            style: TextStyle(
                              color: textColor,
                              fontSize: 22,
                              fontWeight: FontWeight.w800,
                              letterSpacing: 1.2,
                            ),
                          ),
                        ],
                      ),

                      const SizedBox(height: 40),

                      // Subtitle
                      Text(
                        isFilipino
                            ? 'Pumili ng Iyong Profile'
                            : 'Select Your Profile',
                        style: TextStyle(
                          color: subtitleColor,
                          fontSize: 16,
                          fontWeight: FontWeight.w400,
                        ),
                      ),

                      const SizedBox(height: 24),

                      // Profile card — Hearing
                      _ProfileCard(
                        icon: Icons.graphic_eq_rounded,
                        iconColor: _orange,
                        iconBg: _orange.withValues(alpha: 0.15),
                        title: isFilipino
                            ? 'Kasama sa Komunikasyon'
                            : 'Inclusive Communicator',
                        subtitle: isFilipino
                            ? 'Nakaririnig / Nakakapagsalita'
                            : 'Hearing / Speaking',
                        isDark: isDark,
                        cardColor: cardColor,
                        textColor: textColor,
                        subtitleColor: subtitleColor,
                        onTap: () => Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (_) => const HearingUserScreen()),
                        ),
                      ),

                      const SizedBox(height: 16),

                      // Profile card — Deaf/Mute
                      _ProfileCard(
                        icon: Icons.favorite_border_rounded,
                        iconColor: _purple,
                        iconBg: _purple.withValues(alpha: 0.15),
                        title: isFilipino ? 'Gumagamit ng FSL' : 'FSL User',
                        subtitle: isFilipino ? 'Bingi / Pipi' : 'Deaf / Mute',
                        isDark: isDark,
                        cardColor: cardColor,
                        textColor: textColor,
                        subtitleColor: subtitleColor,
                        onTap: () => Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (_) => const DeafMuteScreen()),
                        ),
                      ),

                      const SizedBox(height: 48),

                      // Decorative banner
                      Container(
                        width: double.infinity,
                        height: 56,
                        decoration: BoxDecoration(
                          color: _orange,
                          borderRadius: BorderRadius.circular(32),
                        ),
                        child: Center(
                          child: Text(
                            isFilipino
                                ? 'Tulay sa Pagitan'
                                : "Let's Bridge the Gap",
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 16,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ProfileCard extends StatelessWidget {
  final IconData icon;
  final Color iconColor;
  final Color iconBg;
  final String title;
  final String subtitle;
  final bool isDark;
  final Color cardColor;
  final Color textColor;
  final Color subtitleColor;
  final VoidCallback onTap;

  const _ProfileCard({
    required this.icon,
    required this.iconColor,
    required this.iconBg,
    required this.title,
    required this.subtitle,
    required this.isDark,
    required this.cardColor,
    required this.textColor,
    required this.subtitleColor,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: cardColor,
          borderRadius: BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.08),
              blurRadius: 12,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: Row(
          children: [
            Container(
              width: 48,
              height: 48,
              decoration: BoxDecoration(
                color: iconBg,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(icon, color: iconColor, size: 24),
            ),
            const SizedBox(width: 16),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: TextStyle(
                    color: textColor,
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  subtitle,
                  style: TextStyle(
                    color: subtitleColor,
                    fontSize: 13,
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

class _IconButton extends StatelessWidget {
  final IconData icon;
  final bool isDark;
  final VoidCallback onTap;

  const _IconButton({
    required this.icon,
    required this.isDark,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 40,
        height: 40,
        decoration: BoxDecoration(
          color: isDark
              ? Colors.white.withValues(alpha: 0.1)
              : Colors.black.withValues(alpha: 0.07),
          shape: BoxShape.circle,
        ),
        child: Icon(
          icon,
          color: isDark ? Colors.white70 : Colors.black54,
          size: 20,
        ),
      ),
    );
  }
}

class _SettingsModal extends StatelessWidget {
  const _SettingsModal();

  @override
  Widget build(BuildContext context) {
    final app = SiklabKumpasApp.of(context);
    final isDark = app?.isDark ?? true;
    final currentLang = app?.language ?? 'en';

    final bgColor = isDark ? const Color(0xFF1A1A2E) : Colors.white;
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
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                currentLang == 'fil' ? 'Mga Setting' : 'Settings',
                style: TextStyle(
                  color: textColor,
                  fontSize: 20,
                  fontWeight: FontWeight.w700,
                ),
              ),
              GestureDetector(
                onTap: () => Navigator.pop(context),
                child: Icon(Icons.close, color: subtitleColor),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Divider(color: Colors.white.withValues(alpha: 0.1)),
          const SizedBox(height: 16),
          Row(
            children: [
              const Icon(Icons.language, color: Color(0xFFFF6B00), size: 20),
              const SizedBox(width: 8),
              Text(
                currentLang == 'fil' ? 'Wika' : 'Language',
                style: TextStyle(
                  color: textColor,
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            currentLang == 'fil'
                ? 'Pumili ng iyong gustong wika'
                : 'Choose your preferred language',
            style: TextStyle(color: subtitleColor, fontSize: 13),
          ),
          const SizedBox(height: 16),
          _LangOption(
            code: 'EN',
            label: currentLang == 'fil' ? 'Ingles' : 'English',
            isSelected: currentLang == 'en',
            isDark: isDark,
            onTap: () {
              app?.setLanguage('en');
              Navigator.pop(context);
            },
          ),
          const SizedBox(height: 10),
          _LangOption(
            code: 'PH',
            label: 'Filipino',
            isSelected: currentLang == 'fil',
            isDark: isDark,
            onTap: () {
              app?.setLanguage('fil');
              Navigator.pop(context);
            },
          ),
          const SizedBox(height: 24),
          SizedBox(
            width: double.infinity,
            height: 52,
            child: ElevatedButton(
              onPressed: () => Navigator.pop(context),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFFFF6B00),
                foregroundColor: Colors.white,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(32),
                ),
              ),
              child: Text(
                currentLang == 'fil' ? 'Isara' : 'Close',
                style: const TextStyle(
                    fontSize: 16, fontWeight: FontWeight.w700),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _LangOption extends StatelessWidget {
  final String code;
  final String label;
  final bool isSelected;
  final bool isDark;
  final VoidCallback onTap;

  const _LangOption({
    required this.code,
    required this.label,
    required this.isSelected,
    required this.isDark,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final cardColor =
        isDark ? const Color(0xFF0D0D0D) : const Color(0xFFF5F5F5);
    final textColor = isDark ? Colors.white : const Color(0xFF1A1A2E);

    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        decoration: BoxDecoration(
          color: isSelected
              ? const Color(0xFFFF6B00).withValues(alpha: 0.15)
              : cardColor,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: isSelected ? const Color(0xFFFF6B00) : Colors.transparent,
            width: 1.5,
          ),
        ),
        child: Row(
          children: [
            CircleAvatar(
              radius: 16,
              backgroundColor: isSelected
                  ? const Color(0xFFFF6B00)
                  : Colors.grey.shade400,
              child: Text(
                code,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                label,
                style: TextStyle(
                  color: isSelected ? const Color(0xFFFF6B00) : textColor,
                  fontSize: 15,
                  fontWeight:
                      isSelected ? FontWeight.w600 : FontWeight.w400,
                ),
              ),
            ),
            if (isSelected)
              const Icon(Icons.circle, color: Color(0xFFFF6B00), size: 12),
          ],
        ),
      ),
    );
  }
}