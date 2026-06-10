import 'package:flutter_test/flutter_test.dart';
import 'package:siklab_kumpas/main.dart';

void main() {
  testWidgets('App smoke test', (WidgetTester tester) async {
    await tester.pumpWidget(const SiklabKumpasApp());
    expect(find.text('SIKLAB-KUMPAS'), findsOneWidget);
  });
}