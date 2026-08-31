import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:siklab_kumpas/main.dart';

void main() {
  // This asserted on the literal text "SIKLAB-KUMPAS", which the UI v2
  // redesign replaced with an SVG lockup (see widgets/brand_icons.dart). Text
  // finders cannot see SVG, so the test failed for a cosmetic reason and hid
  // whatever else the suite might have caught.
  //
  // What a smoke test is actually for here is that the widget tree builds
  // without throwing — plugin channels are unavailable under flutter_test, so
  // anything deeper belongs in an integration test on a real device.
  testWidgets('app builds without throwing', (WidgetTester tester) async {
    await tester.pumpWidget(const SiklabKumpasApp());
    expect(find.byType(MaterialApp), findsOneWidget);
    expect(tester.takeException(), isNull);
  });
}
