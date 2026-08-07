import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';

/// The brand artwork from design/ui_v2, wrapped so screens never repeat asset
/// paths or intrinsic aspect ratios.
class BrandIcons {
  const BrandIcons._();

  static const _base = 'assets/icons';

  /// "SIKLAB KUMPAS" lockup. Source viewBox is 1337x703, so height follows
  /// from width at that ratio.
  static const wordmarkAspect = 1337 / 703;

  static Widget wordmark({double width = 240}) => SvgPicture.asset(
        '$_base/Wordmark.svg',
        width: width,
        height: width / wordmarkAspect,
        fit: BoxFit.contain,
        semanticsLabel: 'Siklab Kumpas',
      );

  /// Waving hand — the FSL user profile.
  static Widget fslUser({double size = 72}) => SvgPicture.asset(
        '$_base/FSL-User-Icon.svg',
        width: size,
        height: size,
        fit: BoxFit.contain,
      );

  /// ABC over a waveform — the hearing/speaking profile.
  static Widget hearingSpeaking({double size = 72}) => SvgPicture.asset(
        '$_base/Hearing-Speaking-Icon.svg',
        width: size,
        height: size,
        fit: BoxFit.contain,
      );
}
