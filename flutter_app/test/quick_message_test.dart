import 'package:flutter_test/flutter_test.dart';
import 'package:siklab_kumpas/screens/deaf_mute_screen.dart';

/// The respelled forms exist only to survive an ENGLISH voice. Applying them
/// when a Filipino voice is loaded would make pronunciation worse, so the
/// scoping is the part worth pinning.
void main() {
  const deaf = QuickMessage(
    'Bingi po ako. Sign language po ang paraan ko ng pakikipag-usap.',
    'Bingi po ako. Sign language po ang paraan ko ng pakikipag oosap.',
  );
  const mute = QuickMessage(
    'Pipe po ako. Sign language po ang paraan ko ng pakikipag-usap.',
    'Pee-peh po ako. Sign language po ang paraan ko ng pakikipag oosap.',
  );
  const plain = QuickMessage('Hello, can I talk to you?');

  test('an English voice gets the respelled form', () {
    expect(mute.spokenFor('en-US'), startsWith('Pee-peh'));
    expect(deaf.spokenFor('en-US'), contains('oosap'));
  });

  test('a Filipino voice gets the real spelling', () {
    for (final lang in ['fil-PH', 'fil', 'tl-PH', 'FIL-PH']) {
      expect(mute.spokenFor(lang), startsWith('Pipe'),
          reason: '$lang is a Filipino voice and needs no respelling');
      expect(deaf.spokenFor(lang), contains('pakikipag-usap'));
    }
  });

  test('a message with no override is spoken as displayed', () {
    expect(plain.spokenFor('en-US'), plain.display);
    expect(plain.spokenFor('fil-PH'), plain.display);
  });

  test('an unknown or empty voice is treated as not Filipino', () {
    // Falling back to the respelling is the safer default: the override is
    // only ever wrong on a Filipino voice, and an unnamed voice is not one.
    expect(mute.spokenFor(''), startsWith('Pee-peh'));
  });
}
