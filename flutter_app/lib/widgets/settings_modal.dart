import 'package:flutter/material.dart';
import '../main.dart';
import '../theme/tokens.dart';

/// Opens the language/settings sheet. Lifted out of home_screen so the
/// Translation Hub and Initiator Mode headers can reach it too.
void showSettingsModal(BuildContext context) {
  showModalBottomSheet(
    context: context,
    isScrollControlled: true,
    backgroundColor: Colors.transparent,
    builder: (_) => const SettingsModal(),
  );
}

class SettingsModal extends StatelessWidget {
  const SettingsModal({super.key});

  @override
  Widget build(BuildContext context) {
    final app = SiklabKumpasApp.of(context);
    final c = AppColors.of(app?.isDark ?? true);
    final currentLang = app?.language ?? 'en';
    final isFilipino = currentLang == 'fil';

    return Container(
      margin: const EdgeInsets.all(16),
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: c.panel,
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
                isFilipino ? 'Mga Setting' : 'Settings',
                style: TextStyle(
                  color: c.text,
                  fontSize: 20,
                  fontWeight: FontWeight.w700,
                ),
              ),
              GestureDetector(
                onTap: () => Navigator.pop(context),
                child: Icon(Icons.close, color: c.subtitle),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Divider(color: c.border),
          const SizedBox(height: 16),
          Row(
            children: [
              Icon(Icons.language, color: c.orange, size: 20),
              const SizedBox(width: 8),
              Text(
                isFilipino ? 'Wika' : 'Language',
                style: TextStyle(
                  color: c.text,
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            isFilipino
                ? 'Pumili ng iyong gustong wika'
                : 'Choose your preferred language',
            style: TextStyle(color: c.subtitle, fontSize: 13),
          ),
          const SizedBox(height: 16),
          _LangOption(
            code: 'EN',
            label: isFilipino ? 'Ingles' : 'English',
            isSelected: currentLang == 'en',
            colors: c,
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
            colors: c,
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
                backgroundColor: c.orange,
                foregroundColor: Colors.white,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(32),
                ),
              ),
              child: Text(
                isFilipino ? 'Isara' : 'Close',
                style:
                    const TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
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
  final AppColors colors;
  final VoidCallback onTap;

  const _LangOption({
    required this.code,
    required this.label,
    required this.isSelected,
    required this.colors,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final c = colors;
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        decoration: BoxDecoration(
          color: isSelected ? c.orange.withValues(alpha: 0.15) : c.panelAlt,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: isSelected ? c.orange : Colors.transparent,
            width: 1.5,
          ),
        ),
        child: Row(
          children: [
            CircleAvatar(
              radius: 16,
              backgroundColor:
                  isSelected ? c.orange : Colors.grey.shade600,
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
                  color: isSelected ? c.orange : c.text,
                  fontSize: 15,
                  fontWeight:
                      isSelected ? FontWeight.w600 : FontWeight.w400,
                ),
              ),
            ),
            if (isSelected)
              Icon(Icons.circle, color: c.orange, size: 12),
          ],
        ),
      ),
    );
  }
}
