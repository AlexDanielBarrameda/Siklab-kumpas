import 'dart:convert';
import 'package:flutter/services.dart';

/// One step in an utterance the avatar is asked to sign.
///
/// A step either resolves to a clip or it does not. Unresolved steps are kept
/// rather than dropped: silently skipping a word the avatar cannot sign would
/// let the app claim it had signed a sentence it had only signed half of.
class GlossStep {
  /// Display spelling, e.g. `THANK YOU`.
  final String word;

  /// Bundle path, or null when no clip exists for this word.
  final String? asset;

  /// When this step is one letter of a word being fingerspelled, the word it
  /// came from. Null for an ordinary sign.
  final String? spelledFrom;

  const GlossStep(this.word, this.asset, {this.spelledFrom});

  bool get hasClip => asset != null;

  bool get isSpelled => spelledFrom != null;

  @override
  String toString() => hasClip ? word : '$word?';
}

/// Loads the avatar clip set and turns spoken/typed English into an ordered
/// list of clips to play.
///
/// The glosser is deliberately rule-based and offline. A network-backed
/// translator would be a liability in a live demo, and an offline one is
/// something the write-up can actually describe.
class AvatarService {
  AvatarService._();

  static const _dir = 'assets/avatar/clips';
  static const _manifestPath = '$_dir/manifest.json';

  static Map<String, String> _clips = {};

  /// 'A' -> clip. Kept apart from [_clips] because a letter clip and the
  /// article "a" would otherwise be the same key.
  static Map<String, String> _letters = {};
  static String? _idleAsset;
  static bool _loaded = false;

  /// Prefix the build script gives letter clips inside the shared clip table.
  static const _letterPrefix = 'LETTER_';

  /// Function words to drop, minus any that actually have a clip. Computed at
  /// load rather than hardcoded: see [_skip].
  static Set<String> _drop = {};

  /// Longest first, so `THANK YOU` is matched before `THANK`.
  static List<List<String>> _phrases = [];

  static bool get isReady => _loaded && _clips.isNotEmpty;

  /// Whether unknown words can be fingerspelled rather than just named.
  static bool get canFingerspell => _letters.isNotEmpty;
  static String? get idleAsset => _idleAsset;
  static Iterable<String> get vocabulary => _clips.keys;

  /// Function words ASL does not sign. Dropping them is what stops
  /// "where is the bathroom" from spending two clips on "is" and "the".
  ///
  /// A word here is only dropped when the clip set has nothing for it. If a
  /// clip exists, someone rendered it deliberately and it gets signed — that
  /// keeps this list from silently swallowing new work. The trade is
  /// linguistic: TO has a clip, and signing it in "want to go" is not how ASL
  /// works.
  static const _skip = {
    'A', 'AN', 'THE', 'IS', 'ARE', 'AM', 'WAS', 'WERE', 'BE', 'BEEN', 'BEING',
    'TO', 'OF', 'DO', 'DOES', 'DID', 'HAS', 'HAVE', 'HAD', 'WILL', 'SHALL',
    'THERE', 'IT', 'ITS', 'THAT', 'THIS',
  };

  /// Everyday spellings mapped onto the gloss the clip set actually uses.
  ///
  /// Only for spellings with no clip of their own. MY and MINE used to map to
  /// ME as an approximation; both now have their own clips, so mapping them
  /// away would throw that work on the floor.
  static const _synonyms = {
    'HI': 'HELLO',
    'HEY': 'HELLO',
    'I': 'ME',
    'IM': 'ME',
    'YOURE': 'YOU',
    'YOURS': 'YOUR',
    'WHATS': 'WHAT',
    'WHERES': 'WHERE',
    'RESTROOM': 'BATHROOM',
    'TOILET': 'BATHROOM',
    'WASHROOM': 'BATHROOM',
    'CR': 'BATHROOM',
    'MORNINGS': 'MORNING',
  };

  static Future<void> load() async {
    if (_loaded) return;
    try {
      final raw = await rootBundle.loadString(_manifestPath);
      final json = jsonDecode(raw) as Map<String, dynamic>;

      final clips = (json['clips'] as Map<String, dynamic>?) ?? {};
      _clips = {
        for (final e in clips.entries)
          e.key: '$_dir/${(e.value as Map<String, dynamic>)['file']}',
      };

      final idle = json['idle'] as String?;
      _idleAsset = idle == null ? null : _clips[idle];
      // IDLE is the resting state, not a word anybody says.
      if (idle != null) _clips.remove(idle);

      final letters = (json['letters'] as Map<String, dynamic>?) ?? {};
      _letters = {
        for (final e in letters.entries)
          if (_clips[e.value as String] != null)
            e.key.toUpperCase(): _clips[e.value as String]!,
      };
      // Letters are not words: leaving them in the lexicon would let a stray
      // single character match as a sign.
      _clips.removeWhere((k, _) => k.startsWith(_letterPrefix));

      _drop = _skip.where((w) => !_clips.containsKey(w)).toSet();

      _phrases = _clips.keys.map((k) => k.split('_')).toList()
        ..sort((a, b) => b.length.compareTo(a.length));
    } catch (_) {
      // No clips bundled. The player falls back to its static card, so the
      // screen still works -- it just cannot sign.
      _clips = {};
      _letters = {};
      _idleAsset = null;
      _drop = {};
      _phrases = [];
    }
    _loaded = true;
  }

  static String? clipFor(String word) => _clips[_key(word)];

  static String _key(String word) =>
      word.toUpperCase().replaceAll(RegExp(r'[^A-Z0-9 ]'), '').trim()
          .replaceAll(RegExp(r'\s+'), '_');

  /// English sentence -> ordered gloss steps.
  ///
  /// Greedy longest-match over the lexicon, so multi-word signs win over their
  /// first word. No reordering is attempted: English word order is close
  /// enough for the short utterances this vocabulary supports, and a
  /// half-correct topic-comment reordering would be harder to defend than none.
  static List<GlossStep> gloss(String sentence) {
    final words = sentence
        .toUpperCase()
        .replaceAll(RegExp(r"[^A-Z0-9' ]"), ' ')
        .replaceAll("'", '')
        .split(RegExp(r'\s+'))
        .where((w) => w.isNotEmpty)
        .map((w) => _synonyms[w] ?? w)
        .where((w) => !_drop.contains(w))
        .toList();

    final steps = <GlossStep>[];
    var i = 0;
    while (i < words.length) {
      final matched = _matchPhraseAt(words, i);
      if (matched != null) {
        final key = words.sublist(i, i + matched).join('_');
        steps.add(GlossStep(key.replaceAll('_', ' '), _clips[key]));
        i += matched;
        continue;
      }
      // No sign for it: spell it rather than just naming it. With a
      // demo-sized vocabulary this is the common path, not the edge case.
      steps.addAll(_spell(words[i]) ?? [GlossStep(words[i], null)]);
      i += 1;
    }

    // Collapse an immediately repeated sign -- "where where" reads as a
    // stutter, not emphasis. Spelled letters are exempt: dropping the second
    // L of HELLO would spell HELO.
    final out = <GlossStep>[];
    for (final s in steps) {
      final repeat = out.isNotEmpty &&
          out.last.word == s.word &&
          !s.isSpelled &&
          !out.last.isSpelled;
      if (repeat) continue;
      out.add(s);
    }
    return out;
  }

  /// One step per letter, or null when the word cannot be spelled in full.
  ///
  /// All-or-nothing on purpose: a word spelled with holes in it is worse than
  /// one that is honestly named, because the viewer has no way to tell a
  /// missing letter from a letter that was not there.
  static List<GlossStep>? _spell(String word) {
    if (_letters.isEmpty) return null;
    final out = <GlossStep>[];
    for (final ch in word.split('')) {
      final asset = _letters[ch];
      if (asset == null) return null;
      out.add(GlossStep(ch, asset, spelledFrom: word));
    }
    return out.isEmpty ? null : out;
  }

  /// Human-readable gloss line: `HELLO . F-R-I-E-N-D . NAME`.
  ///
  /// Spelled words are shown letter-joined so the reader can tell a
  /// fingerspelled word from a signed one, and a trailing `*` still marks a
  /// word the avatar could neither sign nor spell.
  static String describe(List<GlossStep> steps) {
    final parts = <String>[];
    var i = 0;
    while (i < steps.length) {
      final s = steps[i];
      if (s.isSpelled) {
        final from = s.spelledFrom;
        final letters = <String>[];
        while (i < steps.length && steps[i].spelledFrom == from) {
          letters.add(steps[i].word);
          i++;
        }
        parts.add(letters.join('-'));
        continue;
      }
      parts.add(s.hasClip ? s.word : '${s.word}*');
      i++;
    }
    return parts.join(' · ');
  }

  /// Length in words of the longest lexicon entry starting at [i], or null.
  static int? _matchPhraseAt(List<String> words, int i) {
    for (final p in _phrases) {
      if (i + p.length > words.length) continue;
      var ok = true;
      for (var k = 0; k < p.length; k++) {
        if (words[i + k] != p[k]) {
          ok = false;
          break;
        }
      }
      if (ok) return p.length;
    }
    return null;
  }
}
