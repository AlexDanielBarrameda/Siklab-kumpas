import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// Append-only diagnostic log, written to a file on the device.
///
/// WHY A FILE AND NOT debugPrint
///
/// Dart output does not reach logcat on the target handset. A 167,081-line
/// capture taken while a reported failure was actually happening contained
/// ZERO `I/flutter` lines, so every `RESULT|`, `FRAME|` and `TTS:` line this
/// app emits is invisible once it is installed rather than run from the IDE —
/// which is how it gets onto the demo device. That left screenshots as the
/// only evidence of a failure, and cost several test rounds.
///
/// The system's own logcat output is still useful and still arrives: skin
/// temperature, camera open/close and process lifetime all come from system
/// pids. This file supplies the half that goes missing — what the app itself
/// believed was happening at that moment.
///
/// WHY IT IS BUFFERED
///
/// Writes accumulate in memory and are flushed on a timer, never from the
/// frame callback. That is not tidiness. jsonEncode of a tensor on the UI
/// isolate previously blocked the camera stream long enough for Android's
/// ImageReader queue to fill, after which the plugin stopped delivering frames
/// permanently — the app previewed normally and detected nothing until it was
/// killed. Every stall observed on device had that write enabled. Adding a
/// diagnostic that reintroduces the fault it is meant to diagnose would be a
/// poor trade, so logging costs one string concatenation and a list append.
///
/// Pull it with:
///   adb pull /sdcard/Android/data/com.siklabkumpas.siklab_kumpas/files/trace.log
class TraceService {
  static const _channel = MethodChannel('com.siklabkumpas/inference');

  /// Ceiling on unflushed lines, so a failing flush cannot grow without bound.
  static const _maxBuffered = 500;

  /// Ceiling on the file, so a long session cannot fill the device. Past this
  /// the file is rotated to trace.1.log and started again — one generation is
  /// enough to cover a demo, and keeps the pull to a single small file.
  static const _maxBytes = 512 * 1024;

  static const _flushEvery = Duration(seconds: 5);

  static String? _dir;
  static final List<String> _buffer = [];
  static Timer? _flusher;
  static bool _running = false;
  static bool _flushing = false;

  static bool get isRunning => _running;

  static Future<String?> _filesDir() async {
    if (_dir != null) return _dir;
    try {
      _dir = await _channel.invokeMethod<String>('traceDir');
    } catch (e) {
      debugPrint('trace: traceDir unavailable: $e');
    }
    return _dir;
  }

  static String get _stamp {
    final n = DateTime.now();
    String two(int v) => v.toString().padLeft(2, '0');
    return '${two(n.hour)}:${two(n.minute)}:${two(n.second)}'
        '.${n.millisecond.toString().padLeft(3, '0')}';
  }

  /// Begins a session. Safe to call repeatedly.
  static Future<void> start(String reason) async {
    if (_running) return;
    if (await _filesDir() == null) return;
    _running = true;
    final n = DateTime.now();
    log('=== session start ($reason) '
        '${n.year}-${n.month.toString().padLeft(2, '0')}-'
        '${n.day.toString().padLeft(2, '0')} ===');
    _flusher?.cancel();
    _flusher = Timer.periodic(_flushEvery, (_) => flush());
  }

  /// Records one event. Cheap enough to call from anywhere, including the
  /// frame path — it does no I/O.
  static void log(String event) {
    if (!_running) return;
    if (_buffer.length >= _maxBuffered) {
      // Drop the oldest rather than the newest: the lines nearest a failure
      // are the ones worth keeping.
      _buffer.removeAt(0);
    }
    _buffer.add('$_stamp  $event');
  }

  static Future<void> flush() async {
    if (_flushing || _buffer.isEmpty) return;
    final dir = await _filesDir();
    if (dir == null) return;

    _flushing = true;
    // Taken before the await so lines logged during the write are not lost.
    final pending = List<String>.from(_buffer);
    _buffer.clear();
    try {
      final f = File('$dir/trace.log');
      if (await f.exists() && await f.length() > _maxBytes) {
        try {
          await f.rename('$dir/trace.1.log');
        } catch (_) {
          // Rotation is a convenience; losing it must not lose the session.
        }
      }
      await File('$dir/trace.log')
          .writeAsString('${pending.join('\n')}\n', mode: FileMode.append);
    } catch (e) {
      debugPrint('trace: flush failed: $e');
    } finally {
      _flushing = false;
    }
  }

  /// Ends a session and writes out whatever is left.
  static Future<void> stop(String reason) async {
    if (!_running) return;
    log('=== session end ($reason) ===');
    _flusher?.cancel();
    _flusher = null;
    _running = false;
    await flush();
  }
}
