import 'package:flutter_test/flutter_test.dart';
import 'package:siklab_kumpas/services/phrase_service.dart';

/// Covers every phrase in the demo script, plus the compositional rules, so a
/// change to one gloss cannot silently break another.
void main() {
  setUp(PhraseService.clear);

  group('gloss to English', () {
    final cases = <List<String>, String>{
      ['HELLO', 'GOOD', 'MORNING']: 'Hello, good morning',
      ['HELLO', 'GOOD', 'AFTERNOON']: 'Hello, good afternoon',
      ['GOOD', 'DAY']: 'Good day',
      ['HOW', 'YOU']: 'How are you?',
      ['ME', 'FINE', 'THANK YOU']: "I'm fine, thank you",
      ['WATER', 'PLEASE']: 'Water, please',
      ['EAT', 'PLEASE']: 'Food, please',
      ['ME', 'HANDSOME']: "I'm handsome",
      ['YOU', 'HANDSOME']: "You're handsome",
    };
    cases.forEach((signs, english) {
      test('${signs.join(" ")} -> $english', () {
        expect(PhraseService.toEnglish(signs), english);
      });
    });
  });

  test('subject + adjective composes without a table entry', () {
    // WE SMART is in no lookup table; the copula rule has to supply "are".
    expect(PhraseService.toEnglish(['WE', 'SMART']), 'We are smart');
    expect(PhraseService.toEnglish(['ME', 'SAD']), "I'm sad");
  });

  test('an unknown gloss still produces readable text', () {
    expect(PhraseService.toEnglish(['CAR', 'BIG']), 'Car big');
  });

  test('a repeated sign is not doubled', () {
    // The segmenter can close and reopen mid-sign, so the same sign arriving
    // twice in a row is nearly always one sign seen twice.
    PhraseService.add('GOOD', 1000);
    PhraseService.add('GOOD', 1500);
    PhraseService.add('MORNING', 2000);
    expect(PhraseService.glossLine, 'GOOD MORNING');
  });

  test('a long silence starts a new phrase', () {
    PhraseService.add('WATER', 1000);
    expect(PhraseService.isComplete(2000), isFalse);
    expect(PhraseService.isComplete(1000 + PhraseService.phraseGapMs + 1),
        isTrue);
    // A sign after the gap begins again rather than extending the old phrase.
    PhraseService.add('HELLO', 1000 + PhraseService.phraseGapMs + 100);
    expect(PhraseService.glossLine, 'HELLO');
  });

  group('immediate completion', () {
    test('a complete phrase fires without waiting', () {
      PhraseService.add('GOOD', 1000);
      expect(PhraseService.isRecognisedComplete(), isFalse);
      PhraseService.add('DAY', 1500);
      expect(PhraseService.isRecognisedComplete(), isTrue);
    });

    test('a phrase that PREFIXES a longer one waits', () {
      // ME FINE is a phrase, but also the start of ME FINE THANK YOU, so
      // firing here would cut the signer off mid-sentence.
      PhraseService.add('ME', 1000);
      PhraseService.add('FINE', 1500);
      expect(PhraseService.isRecognisedComplete(), isFalse);
      PhraseService.add('THANK YOU', 2000);
      expect(PhraseService.isRecognisedComplete(), isTrue);
    });

    test('HELLO GOOD does not fire before the third sign', () {
      PhraseService.add('HELLO', 1000);
      PhraseService.add('GOOD', 1500);
      expect(PhraseService.isRecognisedComplete(), isFalse);
      PhraseService.add('AFTERNOON', 2000);
      expect(PhraseService.isRecognisedComplete(), isTrue);
    });

    test('an unknown sequence never fires immediately', () {
      PhraseService.add('CAR', 1000);
      PhraseService.add('BIG', 1500);
      expect(PhraseService.isRecognisedComplete(), isFalse);
    });
  });

  group('removeLast', () {
    test('drops the last sign and leaves the rest', () {
      PhraseService.add('HELLO', 1000);
      PhraseService.add('GOOD', 1500);
      PhraseService.add('AFTERNOON', 2000);
      expect(PhraseService.removeLast(2100), 'AFTERNOON');
      expect(PhraseService.glossLine, 'HELLO GOOD');
    });

    test('returns null on an empty buffer', () {
      expect(PhraseService.removeLast(1000), isNull);
    });

    test('restarts the phrase clock so a replacement can be signed', () {
      PhraseService.add('HELLO', 1000);
      PhraseService.add('WATER', 1500);
      // Without the clock restart the correction would inherit whatever was
      // left of the old window and the phrase could close mid-correction.
      PhraseService.removeLast(1500 + PhraseService.phraseGapMs - 10);
      expect(PhraseService.isComplete(1500 + PhraseService.phraseGapMs + 10),
          isFalse);
    });

    test('a corrected phrase still completes normally', () {
      PhraseService.add('ME', 1000);
      PhraseService.add('WATER', 1500);
      PhraseService.removeLast(1600);
      PhraseService.add('FINE', 1700);
      PhraseService.add('THANK YOU', 2000);
      expect(PhraseService.glossLine, 'ME FINE THANK YOU');
      expect(PhraseService.isRecognisedComplete(), isTrue);
    });
  });

  group('single-sign entries do not truncate longer phrases', () {
    test('HELP waits, because HELP PLEASE exists', () {
      PhraseService.add('HELP', 1000);
      expect(PhraseService.isRecognisedComplete(), isFalse);
      PhraseService.add('PLEASE', 1500);
      expect(PhraseService.isRecognisedComplete(), isTrue);
      expect(PhraseService.toEnglish(PhraseService.take()), 'Help, please');
    });

    test('HOW waits, because HOW YOU and HOW MANY exist', () {
      PhraseService.add('HOW', 1000);
      expect(PhraseService.isRecognisedComplete(), isFalse);
      PhraseService.add('YOU', 1500);
      expect(PhraseService.isRecognisedComplete(), isTrue);
      expect(PhraseService.toEnglish(PhraseService.take()), 'How are you?');
    });

    test('PLEASE waits, because PLEASE MORE exists', () {
      PhraseService.add('PLEASE', 1000);
      expect(PhraseService.isRecognisedComplete(), isFalse);
    });

    test('NO speaks at once but still builds ME NO KNOW', () {
      PhraseService.add('NO', 1000);
      expect(PhraseService.isRecognisedComplete(), isTrue);
      PhraseService.clear();

      PhraseService.add('ME', 2000);
      PhraseService.add('NO', 2400);
      expect(PhraseService.isRecognisedComplete(), isFalse);
      PhraseService.add('KNOW', 2800);
      expect(PhraseService.isRecognisedComplete(), isTrue);
      expect(PhraseService.toEnglish(PhraseService.take()), "I don't know");
    });

    test('YES and WHY end a sentence, so they speak at once', () {
      PhraseService.add('YES', 1000);
      expect(PhraseService.isRecognisedComplete(), isTrue);
      PhraseService.clear();
      PhraseService.add('WHY', 2000);
      expect(PhraseService.isRecognisedComplete(), isTrue);
    });
  });

  group('new public-space phrases render as English', () {
    test('help and health', () {
      expect(PhraseService.toEnglish(['ME', 'HELP', 'PLEASE']), 'I need help');
      expect(PhraseService.toEnglish(['YOU', 'HELP', 'ME']),
          'Can you help me?');
      expect(PhraseService.toEnglish(['ME', 'HEADACHE']),
          'I have a headache');
      expect(PhraseService.toEnglish(['ME', 'PAIN']), "I'm in pain");
    });

    test('ordering', () {
      expect(PhraseService.toEnglish(['DRINK', 'PLEASE']), 'A drink, please');
      expect(PhraseService.toEnglish(['HOW', 'MANY']), 'How many?');
    });
  });

  group('ME is I as a subject and me as an object', () {
    test('after a modal it is the subject', () {
      expect(PhraseService.toEnglish(['CAN', 'ME', 'DRINK', 'WATER', 'PLEASE']),
          'Can I drink water, please');
    });

    test('opening the sentence it is the subject', () {
      expect(PhraseService.toEnglish(['ME', 'DRINK', 'WATER', 'PLEASE']),
          'I drink water, please');
    });

    test('as an object it stays lowercase me', () {
      expect(PhraseService.toEnglish(['CAN', 'YOU', 'HELP', 'ME']),
          'Can you help me');
    });

    test('other pronouns follow the same rule', () {
      expect(PhraseService.toEnglish(['CAN', 'WE', 'EAT', 'PLEASE']),
          'Can we eat, please');
    });

    test('the phrase table still wins over the general rule', () {
      expect(PhraseService.toEnglish(['ME', 'HELP', 'PLEASE']), 'I need help');
      expect(PhraseService.toEnglish(['YOU', 'HELP', 'ME']),
          'Can you help me?');
    });
  });

  test('take() empties the buffer', () {
    PhraseService.add('GOOD', 1000);
    PhraseService.add('DAY', 1200);
    expect(PhraseService.take(), ['GOOD', 'DAY']);
    expect(PhraseService.isEmpty, isTrue);
  });
}
