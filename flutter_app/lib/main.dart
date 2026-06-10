import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'screens/home_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  runApp(const SiklabKumpasApp());
}

class SiklabKumpasApp extends StatefulWidget {
  const SiklabKumpasApp({super.key});

  static _SiklabKumpasAppState? of(BuildContext context) =>
      context.findAncestorStateOfType<_SiklabKumpasAppState>();

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
    return MaterialApp(
      title: 'Siklab-Kumpas',
      debugShowCheckedModeBanner: false,
      themeMode: _themeMode,
      theme: _buildLightTheme(),
      darkTheme: _buildDarkTheme(),
      home: const HomeScreen(),
    );
  }

  ThemeData _buildLightTheme() {
    return ThemeData(
      brightness: Brightness.light,
      scaffoldBackgroundColor: const Color(0xFFF5F0FF),
      colorScheme: const ColorScheme.light(
        primary: Color(0xFFFF6B00),
        secondary: Color(0xFF7B5EA7),
        surface: Colors.white,
      ),
      fontFamily: 'Inter',
      useMaterial3: true,
    );
  }

  ThemeData _buildDarkTheme() {
    return ThemeData(
      brightness: Brightness.dark,
      scaffoldBackgroundColor: const Color(0xFF0D0D0D),
      colorScheme: const ColorScheme.dark(
        primary: Color(0xFFFF6B00),
        secondary: Color(0xFF7B5EA7),
        surface: Color(0xFF1A1A2E),
      ),
      fontFamily: 'Inter',
      useMaterial3: true,
    );
  }
}