import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'screens/home_screen.dart';
import 'theme/tokens.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  runApp(const SiklabKumpasApp());
}

class SiklabKumpasApp extends StatefulWidget {
  const SiklabKumpasApp({super.key});

  /// Theme and language state for any descendant, including pushed routes.
  ///
  /// This resolves through an [_AppScope] inherited widget rather than
  /// `findAncestorStateOfType`, so callers actually register a dependency and
  /// rebuild when the theme or language changes. The scope sits above
  /// [MaterialApp], which puts it above the Navigator too — that is what lets
  /// the sun/moon button work from the Hub and Initiator screens and not just
  /// from Home.
  static _SiklabKumpasAppState? of(BuildContext context) =>
      context.dependOnInheritedWidgetOfExactType<_AppScope>()?.state;

  @override
  State<SiklabKumpasApp> createState() => _SiklabKumpasAppState();
}

class _SiklabKumpasAppState extends State<SiklabKumpasApp> {
  ThemeMode _themeMode = ThemeMode.dark;
  String _language = 'en';

  void toggleTheme() {
    setState(() {
      _themeMode =
          _themeMode == ThemeMode.dark ? ThemeMode.light : ThemeMode.dark;
    });
  }

  void setLanguage(String lang) {
    setState(() => _language = lang);
  }

  String get language => _language;
  bool get isDark => _themeMode == ThemeMode.dark;

  @override
  Widget build(BuildContext context) {
    return _AppScope(
      state: this,
      themeMode: _themeMode,
      language: _language,
      child: MaterialApp(
        title: 'Siklab-Kumpas',
        debugShowCheckedModeBanner: false,
        themeMode: _themeMode,
        theme: _buildLightTheme(),
        darkTheme: _buildDarkTheme(),
        home: const HomeScreen(),
      ),
    );
  }

  ThemeData _buildLightTheme() => _buildTheme(Brightness.light);

  ThemeData _buildDarkTheme() => _buildTheme(Brightness.dark);

  ThemeData _buildTheme(Brightness brightness) {
    final isDark = brightness == Brightness.dark;
    final c = AppColors.of(isDark);
    return ThemeData(
      brightness: brightness,
      // Screens paint their own gradient via AppScaffold; keeping the scaffold
      // colour transparent stops a flat fill showing through it.
      scaffoldBackgroundColor: Colors.transparent,
      canvasColor: isDark ? AppTokens.darkBg : AppTokens.lightBg,
      colorScheme: ColorScheme(
        brightness: brightness,
        primary: AppTokens.orange,
        onPrimary: Colors.white,
        secondary: AppTokens.avatarPurple,
        onSecondary: Colors.white,
        surface: c.panel,
        onSurface: c.text,
        error: const Color(0xFFE5484D),
        onError: Colors.white,
      ),
      sliderTheme: SliderThemeData(
        activeTrackColor: AppTokens.orange,
        inactiveTrackColor: AppTokens.orange.withValues(alpha: 0.2),
        thumbColor: AppTokens.orange,
        overlayColor: AppTokens.orange.withValues(alpha: 0.15),
      ),
      fontFamily: 'Inter',
      useMaterial3: true,
    );
  }
}

class _AppScope extends InheritedWidget {
  final _SiklabKumpasAppState state;
  final ThemeMode themeMode;
  final String language;

  const _AppScope({
    required this.state,
    required this.themeMode,
    required this.language,
    required super.child,
  });

  @override
  bool updateShouldNotify(_AppScope oldWidget) =>
      themeMode != oldWidget.themeMode || language != oldWidget.language;
}
