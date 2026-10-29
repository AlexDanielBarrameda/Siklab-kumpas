import 'package:flutter/material.dart';

/// Design tokens for the v2 UI.
///
/// Values are sampled from design/ui_v2. Every screen resolves its palette
/// through [AppColors] rather than declaring local constants, so a token only
/// ever changes in one place.
class AppTokens {
  const AppTokens._();

  /// Brand orange. Note this is FF6B01, not the FF6B00 the first build used.
  static const orange = Color(0xFFFF6B01);

  static const darkBg = Color(0xFF000000);
  static const darkPanel = Color(0xFF181818);
  static const darkPanelAlt = Color(0xFF262626);

  static const lightBg = Color(0xFFF0F0F0);
  static const lightTintLavender = Color(0xFFF0D8F0);
  static const lightTintPink = Color(0xFFF0D8D8);

  static const ink = Color(0xFF1A1A2E);

  /// Camera preview backdrop. Deliberately a cool near-black so the orange
  /// skeleton overlay stays legible against it.
  static const previewBg = Color(0xFF1A2332);

  /// Puck behind the FSL avatar placeholder.
  static const avatarPurple = Color(0xFF6C63FF);

  /// Warm-to-violet wash behind the whole app in dark mode: black at the top,
  /// orange bleeding down into purple around the middle, back to black.
  static const darkBackground = LinearGradient(
    begin: Alignment.topCenter,
    end: Alignment.bottomCenter,
    colors: [
      Color(0xFF000000),
      Color(0xFF3B1C0E),
      Color(0xFF371817),
      Color(0xFF2C1230),
      Color(0xFF280D3A),
      Color(0xFF1E0F28),
      Color(0xFF110D13),
      Color(0xFF000000),
    ],
    stops: [0.0, 0.16, 0.30, 0.45, 0.56, 0.72, 0.86, 1.0],
  );

  /// Light counterpart: peach at the top through pink into lavender.
  static const lightBackground = LinearGradient(
    begin: Alignment.topCenter,
    end: Alignment.bottomCenter,
    colors: [
      Color(0xFFFFF3EA),
      Color(0xFFFBDFD5),
      Color(0xFFFADFDF),
      Color(0xFFF5E3F4),
      Color(0xFFF3E1FB),
      Color(0xFFF4E4F7),
      Color(0xFFFAEFFB),
      Color(0xFFFCF3FB),
    ],
    stops: [0.0, 0.16, 0.30, 0.44, 0.56, 0.72, 0.88, 1.0],
  );

  /// Violet card behind the FSL avatar placeholder.
  static const avatarCardDark = LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [Color(0xFF2D1340), Color(0xFF2E1F38)],
  );

  static const avatarCardLight = LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [Color(0xFFEDE2FB), Color(0xFFF3E6F4)],
  );
}

/// Brightness-resolved palette. Obtain with `AppColors.of(isDark)`.
class AppColors {
  final bool isDark;

  const AppColors._(this.isDark);

  static const dark = AppColors._(true);
  static const light = AppColors._(false);

  factory AppColors.of(bool isDark) => isDark ? dark : light;

  Color get orange => AppTokens.orange;

  Gradient get background =>
      isDark ? AppTokens.darkBackground : AppTokens.lightBackground;

  /// Card sitting directly on the gradient. Translucent rather than opaque so
  /// the wash reads through it, which is what the mockups show.
  Color get surface => isDark
      ? Colors.white.withValues(alpha: 0.065)
      : Colors.white.withValues(alpha: 0.72);

  /// Opaque panel for modals, which sit over a scrim instead of the gradient.
  Color get panel => isDark ? AppTokens.darkPanel : Colors.white;

  /// Inset row inside a [panel] — language options, spiel choices.
  Color get panelAlt =>
      isDark ? AppTokens.darkPanelAlt : const Color(0xFFF1F1F4);

  Color get border => isDark
      ? Colors.white.withValues(alpha: 0.10)
      : Colors.black.withValues(alpha: 0.06);

  Color get text => isDark ? Colors.white : AppTokens.ink;

  Color get subtitle => isDark
      ? Colors.white.withValues(alpha: 0.54)
      : Colors.black.withValues(alpha: 0.45);

  Color get iconButtonBg => isDark
      ? Colors.white.withValues(alpha: 0.10)
      : Colors.white.withValues(alpha: 0.85);

  Color get iconButtonFg =>
      isDark ? Colors.white70 : Colors.black.withValues(alpha: 0.65);

  Gradient get avatarCard =>
      isDark ? AppTokens.avatarCardDark : AppTokens.avatarCardLight;

  List<BoxShadow> get cardShadow => isDark
      ? const []
      : [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.06),
            blurRadius: 16,
            offset: const Offset(0, 6),
          ),
        ];

  /// Surface for cards that should read as lifted off the background rather
  /// than painted onto it — the profile pickers on Home.
  Color get surfaceElevated => isDark
      ? Colors.white.withValues(alpha: 0.105)
      : Colors.white.withValues(alpha: 0.94);

  /// Deeper, wider shadow to match [surfaceElevated]. In dark mode a plain
  /// black shadow still reads, because the backdrop there is the purple/orange
  /// wash rather than true black.
  List<BoxShadow> get elevatedShadow => isDark
      ? [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.60),
            blurRadius: 30,
            spreadRadius: -6,
            offset: const Offset(0, 14),
          ),
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.30),
            blurRadius: 8,
            spreadRadius: -2,
            offset: const Offset(0, 3),
          ),
        ]
      : [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.14),
            blurRadius: 30,
            spreadRadius: -4,
            offset: const Offset(0, 14),
          ),
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.06),
            blurRadius: 6,
            offset: const Offset(0, 2),
          ),
        ];
}
