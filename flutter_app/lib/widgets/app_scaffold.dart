import 'package:flutter/material.dart';
import '../main.dart';
import '../theme/tokens.dart';
import 'settings_modal.dart';
import 'theme_ripple.dart';

/// Circular chrome button used in every header.
class AppIconButton extends StatelessWidget {
  final IconData icon;
  final VoidCallback onTap;
  final Color? iconColor;
  final double size;

  const AppIconButton({
    super.key,
    required this.icon,
    required this.onTap,
    this.iconColor,
    this.size = 40,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppColors.of(SiklabKumpasApp.of(context)?.isDark ?? true);
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: size,
        height: size,
        decoration: BoxDecoration(
          color: c.iconButtonBg,
          shape: BoxShape.circle,
          border: Border.all(color: c.border),
        ),
        child: Icon(icon, color: iconColor ?? c.iconButtonFg, size: size * 0.5),
      ),
    );
  }
}

/// Settings + theme toggle, top right on every screen.
class AppTopActions extends StatefulWidget {
  const AppTopActions({super.key});

  @override
  State<AppTopActions> createState() => _AppTopActionsState();
}

class _AppTopActionsState extends State<AppTopActions> {
  /// Moon tint in light mode, sampled from UI.10.
  static const _moon = Color(0xFF9810FA);

  /// Ripple originates from the centre of this button.
  final _themeButtonKey = GlobalKey();

  @override
  Widget build(BuildContext context) {
    final app = SiklabKumpasApp.of(context);
    final isDark = app?.isDark ?? true;

    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        AppIconButton(
          icon: Icons.settings_outlined,
          onTap: () => showSettingsModal(context),
        ),
        const SizedBox(width: 10),
        AppIconButton(
          key: _themeButtonKey,
          icon: isDark ? Icons.wb_sunny_outlined : Icons.nightlight_round,
          iconColor: isDark ? AppTokens.orange : _moon,
          onTap: () {
            if (app == null) return;
            ThemeRipple.run(
              context: context,
              originKey: _themeButtonKey,
              // The gradient we are moving *to*.
              targetGradient: isDark
                  ? AppTokens.lightBackground
                  : AppTokens.darkBackground,
              onSwitch: app.toggleTheme,
            );
          },
        ),
      ],
    );
  }
}

/// Gradient background, safe area, and the shared header treatment: a chrome
/// row (optional back button on the left, settings + theme on the right) with
/// a centred title block underneath.
class AppScaffold extends StatelessWidget {
  final Widget child;
  final bool showBack;
  final String? title;
  final String? subtitle;

  /// Orange line under the subtitle, e.g. "Voice -> Text + FSL Avatar".
  final String? modeLine;

  /// Trailing widget on the mode line, e.g. the LETTERS/WORDS toggle.
  final Widget? modeLineTrailing;

  const AppScaffold({
    super.key,
    required this.child,
    this.showBack = false,
    this.title,
    this.subtitle,
    this.modeLine,
    this.modeLineTrailing,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppColors.of(SiklabKumpasApp.of(context)?.isDark ?? true);

    return Scaffold(
      backgroundColor: Colors.transparent,
      body: Container(
        decoration: BoxDecoration(gradient: c.background),
        child: SafeArea(
          child: Column(
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
                child: Row(
                  children: [
                    if (showBack)
                      AppIconButton(
                        icon: Icons.arrow_back_rounded,
                        onTap: () => Navigator.pop(context),
                      ),
                    const Spacer(),
                    const AppTopActions(),
                  ],
                ),
              ),
              if (title != null) ...[
                const SizedBox(height: 12),
                Text(
                  title!,
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    color: c.text,
                    fontSize: 20,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ],
              if (subtitle != null) ...[
                const SizedBox(height: 2),
                Text(
                  subtitle!,
                  textAlign: TextAlign.center,
                  style: TextStyle(color: c.subtitle, fontSize: 13),
                ),
              ],
              if (modeLine != null) ...[
                const SizedBox(height: 8),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  child: Row(
                    mainAxisAlignment: modeLineTrailing == null
                        ? MainAxisAlignment.center
                        : MainAxisAlignment.spaceBetween,
                    children: [
                      Flexible(
                        child: Text(
                          modeLine!,
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            color: c.orange,
                            fontSize: 13,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                      if (modeLineTrailing != null) modeLineTrailing!,
                    ],
                  ),
                ),
              ],
              const SizedBox(height: 12),
              Expanded(child: child),
            ],
          ),
        ),
      ),
    );
  }
}

/// Rounded translucent card used for every content panel on the gradient.
class AppCard extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry padding;
  final double radius;
  final Color? borderColor;
  final Gradient? gradient;

  /// Lifts the card off the background with a stronger fill and shadow.
  final bool elevated;

  const AppCard({
    super.key,
    required this.child,
    this.padding = const EdgeInsets.all(16),
    this.radius = 18,
    this.borderColor,
    this.gradient,
    this.elevated = false,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppColors.of(SiklabKumpasApp.of(context)?.isDark ?? true);
    return Container(
      width: double.infinity,
      padding: padding,
      decoration: BoxDecoration(
        color: gradient == null
            ? (elevated ? c.surfaceElevated : c.surface)
            : null,
        gradient: gradient,
        borderRadius: BorderRadius.circular(radius),
        border: Border.all(color: borderColor ?? c.border),
        boxShadow: elevated ? c.elevatedShadow : c.cardShadow,
      ),
      child: child,
    );
  }
}
