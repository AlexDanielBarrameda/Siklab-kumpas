import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

class CameraService {
  static CameraController? _controller;
  static List<CameraDescription> _cameras = [];
  static CameraDescription? _active;
  static bool _isInitialized = false;

  static void Function(CameraImage)? _onFrame;
  static int _frameCount = 0;

  /// Only every Nth frame is handed to Dart. The inference pipeline drops frames
  /// on its own while busy, but delivering every frame still costs a plane copy
  /// across the platform boundary, so we halve that traffic up front.
  static const int _frameStride = 2;

  static bool get isInitialized => _isInitialized;
  static CameraController? get controller => _controller;
  static CameraDescription? get activeCamera => _active;
  static bool get hasMultipleCameras => _cameras.length > 1;

  static bool get isFrontCamera =>
      _active?.lensDirection == CameraLensDirection.front;

  /// Clockwise degrees needed to bring the sensor buffer upright while the device
  /// is held in portrait. Passed through to the Kotlin side.
  static int get sensorOrientation => _active?.sensorOrientation ?? 270;

  static Future<void> initialize({CameraLensDirection? lens}) async {
    try {
      if (_cameras.isEmpty) {
        _cameras = await availableCameras();
      }
      if (_cameras.isEmpty) {
        _isInitialized = false;
        return;
      }

      final wanted = lens ?? _active?.lensDirection ?? CameraLensDirection.front;

      // Already running on the requested lens.
      if (_isInitialized &&
          _controller != null &&
          _controller!.value.isInitialized &&
          _active?.lensDirection == wanted) {
        return;
      }

      final description = _cameras.firstWhere(
        (c) => c.lensDirection == wanted,
        orElse: () => _cameras.first,
      );

      await _disposeController();

      final controller = CameraController(
        description,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );

      await controller.initialize();

      _controller = controller;
      _active = description;
      _isInitialized = true;
    } catch (e) {
      debugPrint('Camera init error: $e');
      _isInitialized = false;
    }
  }

  /// Flips between the front and back lens, preserving the streaming state.
  /// Returns true when the active lens actually changed.
  static Future<bool> switchCamera() async {
    if (_cameras.length < 2) return false;

    final wasStreaming = isStreaming;
    final callback = _onFrame;

    if (wasStreaming) {
      await stopImageStream();
    }

    final next = isFrontCamera
        ? CameraLensDirection.back
        : CameraLensDirection.front;

    await initialize(lens: next);

    if (wasStreaming && callback != null && _isInitialized) {
      await startImageStream(callback);
    }
    return _isInitialized;
  }

  /// Whether WE believe a stream is running.
  ///
  /// Deliberately separate from `controller.value.isStreamingImages`. That flag
  /// can be left true after a stop that did not fully take, and both plugin
  /// methods are guarded by it — so startImageStream returns without starting
  /// and stopImageStream returns without stopping. The result is a camera that
  /// previews normally and delivers no frames, recoverable only by killing the
  /// app: stop/start recognition, switching mode and the watchdog all failed
  /// for exactly this reason.
  static bool _streaming = false;

  static bool get isStreaming => _streaming;

  static Future<void> startImageStream(
      void Function(CameraImage) onFrame) async {
    if (!_isInitialized || _controller == null) return;
    if (_streaming) return;

    // Reconcile a stale plugin flag before starting, or the call below is a
    // silent no-op.
    if (_controller!.value.isStreamingImages) {
      try {
        await _controller!.stopImageStream();
      } catch (_) {
        // Already stopped underneath us; the flag was simply wrong.
      }
    }

    _onFrame = onFrame;
    _frameCount = 0;

    await _controller!.startImageStream((image) {
      _frameCount++;
      if (_frameCount % _frameStride == 0) {
        onFrame(image);
      }
    });
    _streaming = true;
  }

  static Future<void> stopImageStream() async {
    // Cleared first so a throw below cannot leave us believing we stream.
    _streaming = false;
    if (_controller == null) return;
    try {
      if (_controller!.value.isStreamingImages) {
        await _controller!.stopImageStream();
      }
    } catch (_) {
      // Nothing to stop.
    }
  }

  /// Tears the camera down and brings it back up, then restarts the stream.
  ///
  /// The escalation when restarting the stream alone is not enough — a fresh
  /// controller is the only thing guaranteed to clear plugin-side state, and
  /// it is what killing the app was achieving by hand.
  static Future<bool> hardRestart(void Function(CameraImage) onFrame) async {
    try {
      await stopImageStream();
      await dispose();
      await initialize();
      if (!_isInitialized) return false;
      await startImageStream(onFrame);
      return _streaming;
    } catch (e) {
      debugPrint('camera hardRestart failed: $e');
      return false;
    }
  }

  static Future<void> _disposeController() async {
    final controller = _controller;
    _controller = null;
    _isInitialized = false;
    // Or a fresh controller inherits the belief that a stream is already
    // running, and startImageStream silently declines to start one.
    _streaming = false;
    if (controller == null) return;
    try {
      if (controller.value.isStreamingImages) {
        await controller.stopImageStream();
      }
    } catch (_) {
      // Controller may already be torn down; nothing useful to do here.
    }
    await controller.dispose();
  }

  static Future<void> dispose() async {
    _onFrame = null;
    await _disposeController();
    _active = null;
  }
}
