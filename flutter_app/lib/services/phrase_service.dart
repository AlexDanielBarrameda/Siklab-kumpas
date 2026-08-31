/// Chains recognised signs into a phrase, and turns ASL gloss into English.
///
/// Each recognised sign was previously spoken on its own and discarded, so
/// "HELLO GOOD MORNING" came out as three unrelated words. ASL is not English
/// word-for-word either: it drops the copula and articles, so the signs a user
/// actually makes are HOW YOU, not "how are you", and ME FINE, not "I am fine".
///
/// This holds recognised signs in a buffer, closes the phrase when signing
/// stops, and renders the gloss as English. The rules are explicit and small
/// rather than a language model: everything runs offline, nothing is uploaded,
/// and the output is inspectable — which matters more here than covering
/// arbitrary sentences.
class PhraseService {
  /// Signs recognised since the phrase began.
  static final List<String> _buffer = [];

  /// When the last sign landed, for deciding the phrase has ended.
  static int _lastSignMs = 0;

  /// Silence that closes a phrase, when nothing else does.
  ///
  /// A single timeout cannot be right for both jobs. Dropping the hands is how
  /// a sign ENDS and also how a phrase ends, so the only thing separating them
  /// is duration — and 2s, which felt responsive, was shorter than the natural
  /// pause between signs, so phrases split into single words.
  ///
  /// Raised 3.5s -> 5s from measurement, not preference. The gap between
  /// consecutive accepted signs was logged across 237 transitions in one
  /// session:
  ///
  ///     p50 2.74s   p75 3.28s   p85 3.88s   p90 4.53s   p95 5.85s
  ///
  /// 3.5s covered only 80.2% of them, so ONE TRANSITION IN FIVE closed the
  /// phrase early — GOOD spoken alone ten times where GOOD DAY was intended,
  /// HELLO six times, and the following sign then opened a fresh phrase. It
  /// reads as "it cannot detect GOOD" when GOOD was recognised perfectly well
  /// (50 decisions, 4 below the gate) and simply arrived alone.
  ///
  /// 5s covers 91.6% and costs nothing when things go RIGHT: a phrase the
  /// table recognises is spoken the instant it completes, so this timeout only
  /// ever governs sequences that are still incomplete.
  ///
  /// This is now only a fallback. A phrase matching a known utterance is
  /// spoken the instant it completes (see [isRecognisedComplete]), so the
  /// scripted phrases do not wait for it at all — the timeout applies to sign
  /// sequences the table does not know.
  static const int phraseGapMs = 5000;

  /// Signs in one phrase. Past this it is almost certainly two utterances that
  /// ran together, and a runaway buffer reads as the app having hung.
  static const int maxSigns = 8;

  static bool get isEmpty => _buffer.isEmpty;

  /// The phrase as signed, for display while it is being built.
  static String get glossLine => _buffer.join(' ');

  /// Adds a recognised sign. Returns true when it started a new phrase.
  static bool add(String sign, int nowMs) {
    final fresh = _buffer.isEmpty || (nowMs - _lastSignMs) > phraseGapMs;
    if (fresh) _buffer.clear();
    // A sign repeated immediately is nearly always the same sign recognised
    // twice, not a genuine repetition — the segmenter can close and reopen
    // mid-sign. Real doubling ("very very") is rare enough to lose.
    if (_buffer.isEmpty || _buffer.last != sign) {
      _buffer.add(sign);
    }
    if (_buffer.length > maxSigns) _buffer.removeAt(0);
    _lastSignMs = nowMs;
    return fresh;
  }

  /// True when nothing has been signed for [phraseGapMs] and a phrase is held.
  static bool isComplete(int nowMs) =>
      _buffer.isNotEmpty && (nowMs - _lastSignMs) > phraseGapMs;

  /// True when the buffer already spells a complete known phrase, so there is
  /// nothing to wait for.
  ///
  /// Deliberately does NOT fire when the gloss is the start of a longer known
  /// phrase: ME FINE is a phrase on its own but also the opening of ME FINE
  /// THANK YOU, and speaking at ME FINE would cut the user off mid-sentence.
  static bool isRecognisedComplete() {
    if (_buffer.isEmpty) return false;
    final gloss = _buffer.join(' ').toUpperCase();
    if (!_phrases.containsKey(gloss)) return false;
    final prefix = '$gloss ';
    for (final known in _phrases.keys) {
      if (known.startsWith(prefix)) return false;   // a longer one may follow
    }
    return true;
  }

  /// Drops the last sign from the pending phrase, returning what was removed.
  ///
  /// Words had no correction affordance at all: a misread could only be fixed
  /// by waiting out [phraseGapMs] and letting the wrong sentence be spoken
  /// aloud. Letters has had a backspace since fingerspelling was added, and the
  /// mistake costs more here — a wrong SIGN changes the sentence, where a wrong
  /// letter only misspells a word.
  static String? removeLast(int nowMs) {
    if (_buffer.isEmpty) return null;
    final gone = _buffer.removeLast();
    // Restart the clock. The signer is mid-correction and should get the full
    // window to sign a replacement, not whatever remained of the old one.
    _lastSignMs = nowMs;
    return gone;
  }

  static void clear() {
    _buffer.clear();
    _lastSignMs = 0;
  }

  /// Takes the finished phrase, clearing the buffer.
  static List<String> take() {
    final out = List<String>.from(_buffer);
    _buffer.clear();
    _lastSignMs = 0;
    return out;
  }

  // ── gloss to English ────────────────────────────────────────────────────

  /// Whole-phrase glosses, matched before any word-level rule.
  ///
  /// Fixed greetings and courtesies are idioms rather than compositions —
  /// "HOW YOU" is not "how you" — and they are also exactly what gets
  /// demonstrated, so they are worth getting exactly right.
  static const Map<String, String> _phrases = {
    'HELLO GOOD MORNING': 'Hello, good morning',
    'HELLO GOOD AFTERNOON': 'Hello, good afternoon',
    'GOOD MORNING': 'Good morning',
    'GOOD AFTERNOON': 'Good afternoon',
    'GOOD DAY': 'Good day',
    'HOW YOU': 'How are you?',
    'YOU HOW': 'How are you?',
    'ME FINE THANK YOU': "I'm fine, thank you",
    'FINE THANK YOU': "I'm fine, thank you",
    'ME FINE': "I'm fine",
    'WATER PLEASE': 'Water, please',
    'EAT PLEASE': 'Food, please',
    'ME HANDSOME': "I'm handsome",
    'YOU HANDSOME': "You're handsome",
    'THANK YOU': 'Thank you',

    // ── asking a stranger for help ──────────────────────────────────────
    //
    // The highest-value set in a public space, and the reason the app exists:
    // these are the sentences a Deaf signer cannot get across to a hearing
    // person without an interpreter.
    'HELP': 'Help',
    'HELP PLEASE': 'Help, please',
    'ME HELP PLEASE': 'I need help',
    'YOU HELP ME': 'Can you help me?',
    'SORRY': 'Sorry',
    'ME SORRY': "I'm sorry",
    'ME NO KNOW': "I don't know",
    'YOU KNOW': 'Do you know?',
    'PLEASE MORE': 'Please, more',

    // ── ordering and shopping ───────────────────────────────────────────
    'PIZZA PLEASE': 'Pizza, please',
    'DRINK PLEASE': 'A drink, please',
    'MORE PLEASE': 'More, please',
    'ME FINISH': "I'm finished",
    'FINISH': 'Finished',
    'HOW MANY': 'How many?',
    'ME CAN': 'I can',

    // ── health and urgency ──────────────────────────────────────────────
    'ME PAIN': "I'm in pain",
    'ME HEADACHE': 'I have a headache',
    'ME SAD': "I'm sad",
    'ME UPSET': "I'm upset",

    // ── small talk ──────────────────────────────────────────────────────
    'YOU SMART': "You're smart",
    'YOU FRIENDLY': "You're friendly",
    'WE FRIENDLY': "We're friends",
    'ME MOTHER': 'My mother',
    'ME FATHER': 'My father',
    'ME FAMILY': 'My family',
    'ME PARENTS': 'My parents',

    // ── single signs worth speaking on their own ────────────────────────
    //
    // Safe to list because isRecognisedComplete() refuses to fire on anything
    // that PREFIXES a longer entry: HELP waits because HELP PLEASE exists,
    // HOW waits for HOW YOU and HOW MANY, PLEASE waits for PLEASE MORE. Only
    // signs that genuinely end a sentence speak immediately.
    'YES': 'Yes',
    'NO': 'No',
    'MAYBE': 'Maybe',
    'PLEASE': 'Please',
    'NEAR': 'Nearby',
    'WHY': 'Why?',
  };

  /// Pronouns, so a subject can be recognised without a lookup table entry.
  static const Map<String, String> _subject = {
    'ME': 'I', 'YOU': 'You', 'WE': 'We', 'HE': 'He', 'THEY': 'They',
  };

  /// Signs that can stand before the subject, so a pronoun following one is
  /// still the subject of the sentence rather than its object.
  static const Set<String> _modals = {
    'CAN', 'MAYBE', 'WILL', 'MUST', 'SHOULD',
  };

  /// The copula ASL omits, chosen to agree with the subject.
  static const Map<String, String> _copula = {
    'I': 'am', 'You': 'are', 'We': 'are', 'He': 'is', 'They': 'are',
  };

  /// Signs that describe rather than act, so SUBJECT + X takes a copula.
  static const Set<String> _adjectives = {
    'HANDSOME', 'FINE', 'GOOD', 'BAD', 'SAD', 'HAPPY', 'SMART', 'FRIENDLY',
    'BORED', 'UPSET', 'BIG', 'SMALL', 'HOT', 'COLD', 'DARK', 'DIRTY',
    'HUNGRY', 'SICK', 'NICE', 'RUDE', 'DEAF', 'HEARING',
  };

  static String _title(String w) => w.isEmpty
      ? w
      : w[0].toUpperCase() + w.substring(1).toLowerCase();

  /// Renders a gloss sequence as English.
  static String toEnglish(List<String> signs) {
    if (signs.isEmpty) return '';
    final gloss = signs.join(' ').toUpperCase();

    final exact = _phrases[gloss];
    if (exact != null) return exact;

    // SUBJECT + ADJECTIVE, the pattern ASL uses constantly and English cannot:
    // ME HANDSOME -> I'm handsome.
    if (signs.length == 2) {
      final subj = _subject[signs[0].toUpperCase()];
      if (subj != null && _adjectives.contains(signs[1].toUpperCase())) {
        final cop = _copula[subj] ?? 'is';
        final adj = signs[1].toLowerCase();
        return subj == 'I' ? "I'm $adj" : '$subj $cop $adj';
      }
      // SUBJECT + VERB/NOUN: no copula, just join.
      if (subj != null) return '$subj ${signs[1].toLowerCase()}';
    }

    // Courtesy words read better as their own clause.
    final out = <String>[];
    for (var i = 0; i < signs.length; i++) {
      final w = signs[i].toUpperCase();
      if (w == 'PLEASE' && i == signs.length - 1 && signs.length > 1) {
        out.add(', please');
        continue;
      }
      // ASL has ONE sign where English has two forms: the same pointing
      // gesture is "I" as a subject and "me" as an object. English will not
      // accept the wrong one, so position has to decide it — subject position
      // being sentence-initial or directly after a modal.
      //
      // CAN ME DRINK WATER PLEASE came out as "Can me drink water, please":
      // the pronoun rule existed but was reachable only from the two-sign
      // SUBJECT + ADJECTIVE path, so it never applied to a longer sentence.
      final subj = _subject[w];
      if (subj != null &&
          (i == 0 || _modals.contains(signs[i - 1].toUpperCase()))) {
        // "I" is capitalised wherever it falls; the others only when they
        // open the sentence.
        out.add(i == 0 ? subj : (subj == 'I' ? 'I' : subj.toLowerCase()));
        continue;
      }
      out.add(i == 0 ? _title(signs[i]) : signs[i].toLowerCase());
    }
    var s = out.join(' ').replaceAll(' , ', ', ');
    if (!s.endsWith('?') && !s.endsWith('.')) s = s;
    return s;
  }
}
