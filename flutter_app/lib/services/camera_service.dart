import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

class CameraService {
  static CameraController? _controller;
  static List<CameraDescription> _cameras = [];
  static bool _isInitialized = false;

  static bool get isInitialized => _isInitialized;
  static CameraController? get controller => _controller;

  static Future<void> initialize() async {
    try {
      _cameras = await availableCameras();
      if (_cameras.isEmpty) return;

      // Use front camera for sign language
      final front = _cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => _cameras.first,
      );

      _controller = CameraController(
        front,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );

      await _controller!.initialize();
      _isInitialized = true;
    } catch (e) {
      debugPrint('Camera init error: $e');
      _isInitialized = false;
    }
  }

  static Future<void> startImageStream(
      Function(CameraImage) onFrame) async {
    if (!_isInitialized || _controller == null) return;
    if (_controller!.value.isStreamingImages) return;
    await _controller!.startImageStream(onFrame);
  }

  static Future<void> stopImageStream() async {
    if (_controller == null) return;
    if (!_controller!.value.isStreamingImages) return;
    await _controller!.stopImageStream();
  }

  static Future<void> dispose() async {
    await _controller?.dispose();
    _controller = null;
    _isInitialized = false;
  }
}