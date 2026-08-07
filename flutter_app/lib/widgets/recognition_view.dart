import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import '../main.dart';
import '../theme/tokens.dart';
import '../services/camera_service.dart';
import '../services/inference_service.dart';
import '../services/speech_service.dart';
import 'app_scaffold.dart';
import 'skeleton_overlay.dart';
import 'transitions.dart';

/// Camera preview, recognition controls and translation log.
///
/// Extracted from deaf_mute_screen so the Translation Hub's Camera tab renders
/// the *same* widget rather than a second copy of the pipeline glue: a hearing
/// user who cannot follow an FSL signer switches tab in place instead of
/// backing out to swap profiles.
///
/// This moved widget code only. Frame handling, buffering and the letter/word
/// decision all still live in InferenceService and are called exactly as
/// before.
class RecognitionView extends StatefulWidget {
  final bool isFilipino;

  /// Extra pause before the camera is brought up, on top of waiting for the
  /// route transition. The Hub passes its tab-switch duration here so camera
  /// start-up does not land in the middle of that animation.
  final Duration initDelay;

  /// Space kept clear at the bottom of the scroll area for anything floating
  /// over it — the Initiator screen's signal button.
  final double bottomInset;

  const RecognitionView({
    super.key,
    required this.isFilipino,
    this.initDelay = Duration.zero,
    this.bottomInset = 0,
  });

  @override
  State<RecognitionView> createState() => _RecognitionViewState();
}

class _RecognitionViewState extends State<RecognitionView>
    with WidgetsBindingObserver {
  bool _isDetecting = false;
  bool _cameraReady = false;
  bool _permissionDenied = false;
  bool _isLoadingModels = false;
  bool _isSwitchingCamera = false;

  /// Incremented per processed frame so the skeleton overlay repaints.
  int _skeletonTick = 0;

  /// Height of the translation log. Fixed so the camera above never resizes.
  static const double _logHeight = 74;
  final List<String> _translationLog = [];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    // Camera bring-up costs enough frames to visibly stall an in-flight screen
    // or tab transition, so it waits until the motion has settled.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      afterRouteTransition(context, () {
        if (!mounted) return;
        if (widget.initDelay == Duration.zero) {
          _initCamera();
        } else {
          Future.delayed(widget.initDelay, () {
            if (mounted) _initCamera();
          });
        }
      });
    });
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive) {
      CameraService.stopImageStream();
    } else if (state == AppLifecycleState.resumed && _cameraReady) {
      CameraService.initialize().then((_) {
        if (mounted) setState(() {});
      });
    }
  }

  /// Releases the camera when this view goes away — including when the Hub
  /// switches off the Camera tab, so the sensor is not held while the user is
  /// on Voice or Text.
  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    CameraService.stopImageStream();
    CameraService.dispose();
    super.dispose();
  }

  Future<void> _initCamera() async {
    final status = await Permission.camera.request();
    if (status.isDenied || status.isPermanentlyDenied) {
      if (mounted) setState(() => _permissionDenied = true);
      return;
    }

    await CameraService.initialize();
    if (!mounted) {
      // Tab was switched away mid-initialisation. CameraService is a global
      // singleton, so without this the controller we just built would stay
      // open with no widget owning it.
      await CameraService.dispose();
      return;
    }
    setState(() => _cameraReady = CameraService.isInitialized);
  }

  Future<void> _switchCamera() async {
    if (!_cameraReady || _isSwitchingCamera) return;
    if (!CameraService.hasMultipleCameras) return;

    setState(() => _isSwitchingCamera = true);
    await CameraService.switchCamera();
    InferenceService.clearBuffer();
    if (mounted) {
      setState(() {
        _isSwitchingCamera = false;
        _cameraReady = CameraService.isInitialized;
      });
    }
  }

  Future<void> _toggleDetection() async {
    if (!_cameraReady || _isLoadingModels) return;

    if (_isDetecting) {
      await CameraService.stopImageStream();
      InferenceService.clearBuffer();
      setState(() => _isDetecting = false);
    } else {
      setState(() => _isLoadingModels = true);
      await InferenceService.initialize();
      if (!mounted) return;
      setState(() {
        _isLoadingModels = false;
        _isDetecting = true;
        _translationLog.clear();
      });
      await CameraService.startImageStream((CameraImage image) async {
        final result = await InferenceService.processFrame(image);
        if (mounted) {
          if (result != null) {
            SpeechService.speak(result.label);
            setState(() {
              _skeletonTick++;
              _translationLog.insert(
                  0, '${result.label} (${result.confidencePercent})');
              if (_translationLog.length > 5) {
                _translationLog.removeLast();
              }
            });
          } else {
            // Force UI update to show the debug string and repaint the skeleton
            setState(() => _skeletonTick++);
          }
        }
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = AppColors.of(SiklabKumpasApp.of(context)?.isDark ?? true);
    final isFilipino = widget.isFilipino;

    return LayoutBuilder(
      builder: (context, constraints) {
        // The preview's height comes from the viewport and nothing else.
        //
        // It used to be an Expanded sibling of the output log, which meant
        // every extra line of recognised text stole height from the camera and
        // the preview visibly shrank and grew while signing. Pinning it here
        // means the log can only ever expand *downward*; when it runs past the
        // bottom the page scrolls instead of squeezing the camera.
        //
        // The height is what is left after the fixed chrome below it plus a
        // constant allowance for the translation card. The allowance is
        // deliberately a constant rather than the card's real height — that is
        // the entire point. The card may grow past it and push the page into
        // scrolling, but it can never claw height back from the camera. Sizing
        // it this way reproduces the roomy preview of the pre-scroll layout,
        // where Expanded handed the camera all the slack.
        const gaps = 14.0 + 14.0;
        const actionButton = 54.0;
        const logAllowance = 120.0;
        final previewHeight =
            (constraints.maxHeight - gaps - actionButton - logAllowance)
                .clamp(240.0, 560.0);

        return SingleChildScrollView(
          padding: EdgeInsets.only(bottom: widget.bottomInset),
          child: Column(
            children: [
              SizedBox(
                height: previewHeight,
                child: _buildPreview(c, isFilipino),
              ),
              const SizedBox(height: 14),
              _buildStartButton(isFilipino),
              const SizedBox(height: 14),
              _buildOutputLog(c, isFilipino),
            ],
          ),
        );
      },
    );
  }

  Widget _buildPreview(AppColors c, bool isFilipino) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(20),
      child: Container(
        width: double.infinity,
        decoration: BoxDecoration(
          color: AppTokens.previewBg,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: _isDetecting ? c.orange : c.orange.withValues(alpha: 0.3),
            width: 1.5,
          ),
        ),
        child: Stack(
          children: [
            if (_permissionDenied)
              Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Icon(Icons.videocam_off_outlined,
                        color: Colors.white38, size: 48),
                    const SizedBox(height: 12),
                    Text(
                      isFilipino
                          ? 'Kailangan ng pahintulot sa camera'
                          : 'Camera permission required',
                      style:
                          const TextStyle(color: Colors.white38, fontSize: 13),
                    ),
                    const SizedBox(height: 12),
                    TextButton(
                      onPressed: () => openAppSettings(),
                      child: Text('Open Settings',
                          style: TextStyle(color: c.orange)),
                    ),
                  ],
                ),
              )
            else if (_cameraReady && CameraService.controller != null)
              // Positioned.fill alone stretched the feed to whatever shape the
              // container happened to be, which visibly widened the signer.
              // Size a box to the sensor's own aspect and let BoxFit.cover crop
              // it instead, so proportions are preserved.
              Positioned.fill(
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(20),
                  child: _AspectPreview(
                      controller: CameraService.controller!),
                ),
              )
            else
              Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    CircularProgressIndicator(color: c.orange),
                    const SizedBox(height: 12),
                    Text(
                      isFilipino
                          ? 'Sinisimulan ang camera...'
                          : 'Starting camera...',
                      style:
                          const TextStyle(color: Colors.white38, fontSize: 13),
                    ),
                  ],
                ),
              ),

            // Debug skeleton, drawn over the preview.
            // NOT const: a const widget is canonicalised, so Flutter would skip
            // rebuilding this subtree and the overlay would never react to
            // showSkeleton changing.
            SkeletonOverlay(
              enabled: InferenceService.showSkeleton,
              revision: _skeletonTick,
            ),

            // LIVE badge
            Positioned(
              top: 12,
              left: 12,
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                decoration: BoxDecoration(
                  color: _isDetecting ? Colors.red : Colors.black54,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      Icons.circle,
                      color: _isDetecting ? Colors.white : Colors.white38,
                      size: 8,
                    ),
                    const SizedBox(width: 5),
                    Text(
                      'LIVE',
                      style: TextStyle(
                        color: _isDetecting ? Colors.white : Colors.white38,
                        fontSize: 11,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ],
                ),
              ),
            ),

            // Camera controls.
            //
            // The MIRROR and hand-block chips are gone. Letters canonicalises
            // hand chirality from the palm geometry, so mirroring and the block
            // choice no longer change the result — verified on device across
            // A-Z with either hand and both mirror settings. Words enforces its
            // own settings in InferenceService (unmirrored, positional hand
            // ordering). Both chips were therefore controls that could only
            // produce confusion in a demo.
            //
            // The underlying fields still exist for diagnostics; see
            // InferenceService.mirrorInput and letterHandBlock / wordHandBlock.
            Positioned(
              top: 12,
              right: 12,
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  _CameraChip(
                    label: InferenceService.isLettersMode ? 'LETTERS' : 'WORDS',
                    icon: Icons.abc_rounded,
                    active: true,
                    onTap: () => setState(() {
                      InferenceService.isLettersMode =
                          !InferenceService.isLettersMode;
                      InferenceService.clearBuffer();
                      _translationLog.clear();
                    }),
                  ),
                  const SizedBox(width: 6),
                  _CameraChip(
                    label: 'SKELETON',
                    icon: Icons.share_outlined,
                    active: InferenceService.showSkeleton,
                    onTap: () => setState(() {
                      InferenceService.showSkeleton =
                          !InferenceService.showSkeleton;
                      if (!InferenceService.showSkeleton) {
                        InferenceService.lastRawLandmarks = null;
                      }
                    }),
                  ),
                  if (CameraService.hasMultipleCameras) ...[
                    const SizedBox(width: 6),
                    _CameraChip(
                      label: CameraService.isFrontCamera ? 'FRONT' : 'BACK',
                      icon: Icons.cameraswitch_rounded,
                      active: false,
                      busy: _isSwitchingCamera,
                      onTap: _switchCamera,
                    ),
                  ],
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStartButton(bool isFilipino) {
    return SizedBox(
      width: double.infinity,
      height: 54,
      child: ElevatedButton(
        onPressed: (_cameraReady && !_isLoadingModels) ? _toggleDetection : null,
        style: ElevatedButton.styleFrom(
          backgroundColor: _isDetecting ? Colors.red : AppTokens.orange,
          foregroundColor: Colors.white,
          disabledBackgroundColor: AppTokens.orange.withValues(alpha: 0.35),
          disabledForegroundColor: Colors.white70,
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
        ),
        child: _isLoadingModels
            ? Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(width: 10),
                  Text(
                    isFilipino ? 'Naglo-load...' : 'Loading...',
                    style: const TextStyle(
                        fontSize: 15, fontWeight: FontWeight.w700),
                  ),
                ],
              )
            : Text(
                _isDetecting
                    ? (isFilipino ? 'Ihinto ang Pagkilala' : 'Stop Recognition')
                    : (isFilipino
                        ? 'Simulan ang Pagkilala'
                        : 'Start Recognition'),
                style:
                    const TextStyle(fontSize: 15, fontWeight: FontWeight.w700),
              ),
      ),
    );
  }

  Widget _buildOutputLog(AppColors c, bool isFilipino) {
    return AppCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.circle, color: Color(0xFFFF5F57), size: 9),
              const SizedBox(width: 5),
              const Icon(Icons.circle, color: Color(0xFFFEBC2E), size: 9),
              const SizedBox(width: 5),
              const Icon(Icons.circle, color: Color(0xFF28C840), size: 9),
              const SizedBox(width: 10),
              Flexible(
                child: Text(
                  isFilipino
                      ? 'Output ng Pagsalin (Teksto + Pagsasalita)'
                      : 'Translation Output (Text + Speech)',
                  style: TextStyle(color: c.subtitle, fontSize: 11),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            _isDetecting
                ? InferenceService.debugString
                : (isFilipino
                    ? '> Pindutin ang Simulan para magsimula'
                    : '> Press Start to begin detecting'),
            style: TextStyle(
              color: _isDetecting ? c.orange : c.subtitle,
              fontSize: 13,
              fontFamily: 'monospace',
            ),
          ),
          if (_isDetecting) ...[
            const SizedBox(height: 2),
            Text(
              '> ${InferenceService.processingFps.toStringAsFixed(1)} fps'
              ' | ${CameraService.isFrontCamera ? "front" : "back"}'
              ' | ${InferenceService.effectiveMirror ? "mirrored" : "direct"}'
              ' | hands: ${InferenceService.lastHadHands ? "yes" : "no"}'
              '${InferenceService.isLettersMode ? "" : " | ${InferenceService.postureLabel}"}',
              style: TextStyle(
                color: c.subtitle,
                fontSize: 11,
                fontFamily: 'monospace',
              ),
            ),
          ],
          if (_translationLog.isNotEmpty) ...[
            const SizedBox(height: 4),
            // Fixed height with its own scroll. Previously the log grew with
            // its contents, and because the preview above is Expanded, every
            // long translation shrank the camera and re-laid out the screen.
            SizedBox(
              height: _logHeight,
              child: ListView(
                padding: EdgeInsets.zero,
                children: _translationLog
                    .map((log) => Padding(
                          padding: const EdgeInsets.only(bottom: 2),
                          child: Text(
                            '> $log',
                            style: TextStyle(color: c.text, fontSize: 13),
                          ),
                        ))
                    .toList(),
              ),
            ),
          ],
        ],
      ),
    );
  }
}

/// Small pill overlaid on the camera preview.
class _CameraChip extends StatelessWidget {
  final String label;
  final IconData icon;
  final bool active;
  final bool busy;
  final VoidCallback onTap;

  const _CameraChip({
    required this.label,
    required this.icon,
    required this.active,
    required this.onTap,
    this.busy = false,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: busy ? null : onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 5),
        decoration: BoxDecoration(
          color:
              active ? AppTokens.orange.withValues(alpha: 0.85) : Colors.black54,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (busy)
              const SizedBox(
                width: 12,
                height: 12,
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                  color: Colors.white,
                ),
              )
            else
              Icon(icon, color: Colors.white, size: 12),
            const SizedBox(width: 5),
            Text(
              label,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 10,
                fontWeight: FontWeight.w700,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Camera preview that keeps the sensor's aspect ratio.
///
/// [CameraPreview] fills whatever box it is given, so placing it directly in a
/// Positioned.fill distorts the image whenever the container is not the same
/// shape as the sensor. Sizing to the real preview dimensions and cropping with
/// BoxFit.cover keeps people the shape they actually are.
class _AspectPreview extends StatelessWidget {
  final CameraController controller;

  const _AspectPreview({required this.controller});

  @override
  Widget build(BuildContext context) {
    final preview = controller.value.previewSize;
    if (preview == null) return CameraPreview(controller);

    // previewSize is reported in sensor orientation, which is landscape even
    // when the phone is upright, so the axes are swapped for display.
    return FittedBox(
      fit: BoxFit.cover,
      clipBehavior: Clip.hardEdge,
      child: SizedBox(
        width: preview.height,
        height: preview.width,
        child: CameraPreview(controller),
      ),
    );
  }
}
