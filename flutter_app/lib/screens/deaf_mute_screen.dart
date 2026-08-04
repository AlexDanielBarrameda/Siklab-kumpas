import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import '../main.dart';
import '../services/camera_service.dart';
import '../services/inference_service.dart';
import '../services/speech_service.dart';
import 'package:torch_light/torch_light.dart';

class DeafMuteScreen extends StatefulWidget {
  const DeafMuteScreen({super.key});

  @override
  State<DeafMuteScreen> createState() => _DeafMuteScreenState();
}

class _DeafMuteScreenState extends State<DeafMuteScreen>
    with WidgetsBindingObserver {
  static const _orange = Color(0xFFFF6B00);

  bool _isDetecting = false;
  bool _cameraReady = false;
  bool _permissionDenied = false;
  bool _isLoadingModels = false;
  bool _isSwitchingCamera = false;
  final List<String> _translationLog = [];

  final List<String> _preRecordedEn = [
    'None',
    'Hello, can I talk to you?',
    'Excuse me, may I have a moment?',
    'Hi, can you help me with something?',
    'Good day! Can we have a conversation?',
  ];

  final List<String> _preRecordedFil = [
    'Wala',
    'Kumusta, maari ka bang makausap?',
    'Pasensya na, maari po bang makahingi ng sandali?',
    'Hi, maari mo ba akong tulungan?',
    'Magandang araw! Maari ba tayong mag-usap?',
  ];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initCamera();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive) {
      CameraService.stopImageStream();
    } else if (state == AppLifecycleState.resumed && _cameraReady) {
      CameraService.initialize().then((_) => setState(() {}));
    }
  }

  Future<void> _initCamera() async {
    final status = await Permission.camera.request();
    if (status.isDenied || status.isPermanentlyDenied) {
      setState(() => _permissionDenied = true);
      return;
    }

    await CameraService.initialize();
    if (mounted) {
      setState(() => _cameraReady = CameraService.isInitialized);
    }
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
              _translationLog.insert(0,
                '${result.label} (${result.confidencePercent})');
              if (_translationLog.length > 5) {
                _translationLog.removeLast();
              }
            });
          } else {
            // Force UI update to show the debug string
            setState(() {});
          }
        }
      });
    }
  }

  void _showSignalModal(BuildContext context, bool isFilipino) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => _SignalModal(
        isFilipino: isFilipino,
        preRecorded: isFilipino ? _preRecordedFil : _preRecordedEn,
      ),
    );
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    CameraService.stopImageStream();
    CameraService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final app = SiklabKumpasApp.of(context);
    final isDark = app?.isDark ?? true;
    final lang = app?.language ?? 'en';
    final isFilipino = lang == 'fil';

    final bgColor =
        isDark ? const Color(0xFF0D0D0D) : const Color(0xFFF5F0FF);
    final cardColor = isDark ? const Color(0xFF1A1A2E) : Colors.white;
    final textColor = isDark ? Colors.white : const Color(0xFF1A1A2E);
    final subtitleColor = isDark ? Colors.white54 : Colors.black45;

    return Scaffold(
      backgroundColor: bgColor,
      body: SafeArea(
        child: Column(
          children: [
            // Header
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 0),
              child: Row(
                children: [
                  GestureDetector(
                    onTap: () => Navigator.pop(context),
                    child: Icon(Icons.arrow_back_ios_new_rounded,
                        color: textColor, size: 20),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Initiator Mode',
                          style: TextStyle(
                            color: textColor,
                            fontSize: 18,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                        Text(
                          isFilipino
                              ? 'Bingi / Pipi POV'
                              : 'Deaf / Mute POV',
                          style: TextStyle(
                              color: subtitleColor, fontSize: 12),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),

            // Mode label and Toggle
            Padding(
              padding:
                  const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    isFilipino
                        ? 'FSL → Teksto + Pagsasalita'
                        : 'FSL → Text + Speech Conversion',
                    style: const TextStyle(
                      color: _orange,
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  GestureDetector(
                    onTap: () {
                      setState(() {
                        InferenceService.isLettersMode = !InferenceService.isLettersMode;
                        InferenceService.clearBuffer();
                        _translationLog.clear();
                      });
                    },
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                      decoration: BoxDecoration(
                        color: _orange.withValues(alpha: 0.2),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: _orange, width: 1),
                      ),
                      child: Text(
                        InferenceService.isLettersMode ? 'LETTERS' : 'WORDS',
                        style: const TextStyle(
                          color: _orange,
                          fontSize: 11,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),

            // Signal button
            Padding(
              padding:
                  const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              child: GestureDetector(
                onTap: () => _showSignalModal(context, isFilipino),
                child: Container(
                  width: double.infinity,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  decoration: BoxDecoration(
                    color: cardColor,
                    borderRadius: BorderRadius.circular(32),
                    border: Border.all(
                      color: _orange.withValues(alpha: 0.3),
                      width: 1,
                    ),
                  ),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(Icons.notifications_outlined,
                          color: textColor, size: 20),
                      const SizedBox(width: 8),
                      Text(
                        isFilipino
                            ? 'Senyas para sa Pag-uusap'
                            : 'Signal for Conversation',
                        style: TextStyle(
                          color: textColor,
                          fontSize: 15,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),

            // Camera view
            Expanded(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(20),
                  child: Container(
                    width: double.infinity,
                    decoration: BoxDecoration(
                      color: const Color(0xFF1A2332),
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(
                        color: _isDetecting
                            ? _orange
                            : _orange.withValues(alpha: 0.3),
                        width: 1.5,
                      ),
                    ),
                    child: Stack(
                      children: [
                        // Camera preview or placeholder
                        if (_permissionDenied)
                          Center(
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(Icons.videocam_off_outlined,
                                      color: Colors.white38, size: 48),
                                const SizedBox(height: 12),
                                Text(
                                  isFilipino
                                      ? 'Kailangan ng pahintulot sa camera'
                                      : 'Camera permission required',
                                  style: const TextStyle(
                                      color: Colors.white38,
                                      fontSize: 13),
                                ),
                                const SizedBox(height: 12),
                                TextButton(
                                  onPressed: () =>
                                      openAppSettings(),
                                  child: const Text('Open Settings',
                                      style: TextStyle(
                                          color: _orange)),
                                ),
                              ],
                            ),
                          )
                        else if (_cameraReady &&
                            CameraService.controller != null)
                          Positioned.fill(
                            child: CameraPreview(
                                CameraService.controller!),
                          )
                        else
                          Center(
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                const CircularProgressIndicator(
                                    color: _orange),
                                const SizedBox(height: 12),
                                Text(
                                  isFilipino
                                      ? 'Sinisimulan ang camera...'
                                      : 'Starting camera...',
                                  style: const TextStyle(
                                      color: Colors.white38,
                                      fontSize: 13),
                                ),
                              ],
                            ),
                          ),

                        // LIVE badge
                        Positioned(
                          top: 12,
                          left: 12,
                          child: Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 10, vertical: 4),
                            decoration: BoxDecoration(
                              color: _isDetecting
                                  ? Colors.red
                                  : Colors.black54,
                              borderRadius: BorderRadius.circular(20),
                            ),
                            child: Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Icon(
                                  Icons.circle,
                                  color: _isDetecting
                                      ? Colors.white
                                      : Colors.white38,
                                  size: 8,
                                ),
                                const SizedBox(width: 4),
                                Text(
                                  'LIVE',
                                  style: TextStyle(
                                    color: _isDetecting
                                        ? Colors.white
                                        : Colors.white38,
                                    fontSize: 11,
                                    fontWeight: FontWeight.w700,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),

                        // Camera controls
                        Positioned(
                          top: 12,
                          right: 12,
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              // The MIRROR and hand-block chips are gone.
                              //
                              // Letters canonicalises hand chirality from the
                              // palm geometry, so mirroring and the block choice
                              // no longer change the result — verified on device
                              // across A-Z with either hand and both mirror
                              // settings. Words enforces its own settings in
                              // InferenceService (unmirrored, positional hand
                              // ordering). Both chips were therefore controls
                              // that could only produce confusion in a demo.
                              //
                              // The underlying fields still exist for
                              // diagnostics; see InferenceService.mirrorInput
                              // and letterHandBlock / wordHandBlock.
                              if (CameraService.hasMultipleCameras) ...[
                                _CameraChip(
                                  label: CameraService.isFrontCamera
                                      ? 'FRONT'
                                      : 'BACK',
                                  icon: Icons.cameraswitch_rounded,
                                  active: false,
                                  busy: _isSwitchingCamera,
                                  onTap: _switchCamera,
                                ),
                              ],
                            ],
                          ),
                        ),

                        // Start/Stop button
                        Positioned(
                          bottom: 12,
                          right: 12,
                          child: GestureDetector(
                            onTap: _toggleDetection,
                            child: AnimatedContainer(
                              duration:
                                  const Duration(milliseconds: 200),
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 14, vertical: 8),
                              decoration: BoxDecoration(
                                color: _isDetecting
                                    ? Colors.red
                                    : _orange,
                                borderRadius:
                                    BorderRadius.circular(20),
                              ),
                              child: _isLoadingModels
                                  ? Row(
                                      mainAxisSize: MainAxisSize.min,
                                      children: [
                                        const SizedBox(
                                          width: 12,
                                          height: 12,
                                          child:
                                              CircularProgressIndicator(
                                            strokeWidth: 2,
                                            color: Colors.white,
                                          ),
                                        ),
                                        const SizedBox(width: 8),
                                        Text(
                                          isFilipino
                                              ? 'Naglo-load...'
                                              : 'Loading...',
                                          style: const TextStyle(
                                            color: Colors.white,
                                            fontSize: 12,
                                            fontWeight: FontWeight.w600,
                                          ),
                                        ),
                                      ],
                                    )
                                  : Text(
                                      _isDetecting
                                          ? (isFilipino ? 'Ihinto' : 'Stop')
                                          : (isFilipino
                                              ? 'Simulan'
                                              : 'Start'),
                                      style: const TextStyle(
                                        color: Colors.white,
                                        fontSize: 12,
                                        fontWeight: FontWeight.w600,
                                      ),
                                    ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),

            // Translation output log
            Container(
              margin: const EdgeInsets.all(16),
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: cardColor,
                borderRadius: BorderRadius.circular(16),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Icon(Icons.circle,
                          color: Colors.red, size: 8),
                      const SizedBox(width: 4),
                      const Icon(Icons.circle,
                          color: Colors.orange, size: 8),
                      const SizedBox(width: 4),
                      const Icon(Icons.circle,
                          color: Colors.green, size: 8),
                      const SizedBox(width: 8),
                      Text(
                        isFilipino
                            ? 'Output ng Pagsalin (Teksto + Pagsasalita)'
                            : 'Translation Output (Text + Speech)',
                        style: TextStyle(
                            color: subtitleColor, fontSize: 11),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    _isDetecting
                        ? InferenceService.debugString
                        : (isFilipino
                            ? '> Pindutin ang Simulan para magsimula'
                            : '> Press Start to begin detecting'),
                    style: TextStyle(
                      color:
                          _isDetecting ? _orange : subtitleColor,
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
                      ' | hands: ${InferenceService.lastHadHands ? "yes" : "no"}',
                      style: TextStyle(
                        color: subtitleColor,
                        fontSize: 11,
                        fontFamily: 'monospace',
                      ),
                    ),
                  ],
                  if (_translationLog.isNotEmpty) ...[
                    const SizedBox(height: 4),
                    ..._translationLog.map((log) => Text(
                          '> $log',
                          style: TextStyle(
                              color: textColor, fontSize: 13),
                        )),
                  ],
                ],
              ),
            ),
          ],
        ),
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

  static const _orange = Color(0xFFFF6B00);

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: busy ? null : onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: active ? _orange.withValues(alpha: 0.85) : Colors.black54,
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

class _SignalModal extends StatefulWidget {
  final bool isFilipino;
  final List<String> preRecorded;

  const _SignalModal({
    required this.isFilipino,
    required this.preRecorded,
  });

  @override
  State<_SignalModal> createState() => _SignalModalState();
}

class _SignalModalState extends State<_SignalModal> {
  bool _flickerEnabled = false;
  int _selectedMessage = 0;

  static const _orange = Color(0xFFFF6B00);

  @override
  Widget build(BuildContext context) {
    final app = SiklabKumpasApp.of(context);
    final isDark = app?.isDark ?? true;
    final isFilipino = widget.isFilipino;

    final bgColor = isDark ? const Color(0xFF1A1A2E) : Colors.white;
    final cardColor =
        isDark ? const Color(0xFF0D0D0D) : const Color(0xFFF5F5F5);
    final textColor = isDark ? Colors.white : const Color(0xFF1A1A2E);
    final subtitleColor = isDark ? Colors.white54 : Colors.black45;

    return Container(
      margin: const EdgeInsets.all(16),
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(24),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      isFilipino
                          ? 'Senyas para sa Pag-uusap'
                          : 'Signal for Conversation',
                      style: TextStyle(
                        color: textColor,
                        fontSize: 18,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    Text(
                      isFilipino
                          ? 'Pumili kung paano kunin ang kanilang atensyon'
                          : 'Choose how to get their attention',
                      style: TextStyle(
                          color: subtitleColor, fontSize: 12),
                    ),
                  ],
                ),
              ),
              GestureDetector(
                onTap: () => Navigator.pop(context),
                child: Icon(Icons.close, color: subtitleColor),
              ),
            ],
          ),

          const SizedBox(height: 20),

          // Flicker section
          Row(
            children: [
              const Icon(Icons.lightbulb_outline,
                  color: _orange, size: 20),
              const SizedBox(width: 8),
              Text(
                isFilipino
                    ? 'Kumikislap na Ilaw'
                    : 'Flicker Lights',
                style: TextStyle(
                  color: textColor,
                  fontSize: 15,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            isFilipino
                ? 'I-flash ang screen para makuha ang atensyon'
                : 'Flash the screen to get attention',
            style: TextStyle(color: subtitleColor, fontSize: 12),
          ),
          const SizedBox(height: 10),

          Container(
            padding: const EdgeInsets.symmetric(
                horizontal: 16, vertical: 12),
            decoration: BoxDecoration(
              color: _flickerEnabled
                  ? _orange.withValues(alpha: 0.1)
                  : cardColor,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(
                color: _flickerEnabled
                    ? _orange
                    : Colors.transparent,
                width: 1.5,
              ),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  isFilipino
                      ? 'Kumikislap na Ilaw'
                      : 'Flicker Lights',
                  style: TextStyle(
                    color:
                        _flickerEnabled ? _orange : textColor,
                    fontSize: 14,
                    fontWeight: FontWeight.w500,
                  ),
                ),
                Switch(
                  value: _flickerEnabled,
                  onChanged: (v) =>
                      setState(() => _flickerEnabled = v),
                  activeThumbColor: _orange,
                  activeTrackColor: _orange.withValues(alpha: 0.3),
                ),
              ],
            ),
          ),

          const SizedBox(height: 20),

          // Audio spiel section
          Row(
            children: [
              const Icon(Icons.volume_up_rounded,
                  color: _orange, size: 20),
              const SizedBox(width: 8),
              Text(
                isFilipino
                    ? 'Mag-play ng Audio Spiel'
                    : 'Play Audio Spiel',
                style: TextStyle(
                  color: textColor,
                  fontSize: 15,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            isFilipino
                ? 'Pumili ng pre-recorded na mensahe'
                : 'Select a pre-recorded message',
            style: TextStyle(color: subtitleColor, fontSize: 12),
          ),
          const SizedBox(height: 10),

          ...List.generate(widget.preRecorded.length, (i) {
            final isSelected = _selectedMessage == i;
            return Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: GestureDetector(
                onTap: () => setState(() => _selectedMessage = i),
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 200),
                  padding: const EdgeInsets.symmetric(
                      horizontal: 16, vertical: 12),
                  decoration: BoxDecoration(
                    color: isSelected
                        ? _orange.withValues(alpha: 0.1)
                        : cardColor,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                      color: isSelected
                          ? _orange
                          : Colors.transparent,
                      width: 1.5,
                    ),
                  ),
                  child: Row(
                    children: [
                      Expanded(
                        child: Text(
                          widget.preRecorded[i],
                          style: TextStyle(
                            color: isSelected ? _orange : textColor,
                            fontSize: 14,
                            fontWeight: isSelected
                                ? FontWeight.w600
                                : FontWeight.w400,
                          ),
                        ),
                      ),
                      if (isSelected)
                        const Icon(Icons.circle,
                            color: _orange, size: 10),
                    ],
                  ),
                ),
              ),
            );
          }),

          const SizedBox(height: 16),

          SizedBox(
            width: double.infinity,
            height: 52,
            child: ElevatedButton(
              onPressed: () async {
                Navigator.pop(context); // Close the modal
                
                // Flash the torch 3 times
                if (_flickerEnabled) {
                  try {
                    final isAvailable = await TorchLight.isTorchAvailable();
                    if (isAvailable) {
                      for (int i = 0; i < 3; i++) {
                        await TorchLight.enableTorch();
                        await Future.delayed(const Duration(milliseconds: 200));
                        await TorchLight.disableTorch();
                        await Future.delayed(const Duration(milliseconds: 200));
                      }
                    }
                  } catch (e) {
                    debugPrint('Torch error: $e');
                  }
                }

                // Play the audio spiel if not "None" (index 0)
                if (_selectedMessage > 0) {
                  SpeechService.setTTSLanguage(isFilipino ? 'fil' : 'en');
                  SpeechService.speak(widget.preRecorded[_selectedMessage]);
                }
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: _orange,
                foregroundColor: Colors.white,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(32),
                ),
              ),
              child: Text(
                isFilipino
                    ? 'Simulan ang Pag-uusap'
                    : 'Start Conversation',
                style: const TextStyle(
                    fontSize: 16, fontWeight: FontWeight.w700),
              ),
            ),
          ),

          const SizedBox(height: 10),

          SizedBox(
            width: double.infinity,
            height: 48,
            child: TextButton(
              onPressed: () => Navigator.pop(context),
              child: Text(
                isFilipino ? 'Kanselahin' : 'Cancel',
                style:
                    TextStyle(color: subtitleColor, fontSize: 15),
              ),
            ),
          ),
        ],
      ),
    );
  }
}