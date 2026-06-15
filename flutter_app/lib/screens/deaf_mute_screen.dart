import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import '../main.dart';
import '../services/camera_service.dart';
import '../services/inference_service.dart';

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

  Future<void> _toggleDetection() async {
    if (!_cameraReady) return;

    if (_isDetecting) {
      await CameraService.stopImageStream();
      InferenceService.clearBuffer();
      setState(() => _isDetecting = false);
    } else {
      await InferenceService.initialize();
      setState(() {
        _isDetecting = true;
        _translationLog.clear();
      });
      await CameraService.startImageStream((CameraImage image) async {
        final result = await InferenceService.processFrame(image);
        if (result != null && mounted) {
          setState(() {
            _translationLog.insert(0,
              '${result.label} (${result.confidencePercent})');
            if (_translationLog.length > 5) {
              _translationLog.removeLast();
            }
          });
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

            // Mode label
            Padding(
              padding:
                  const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
              child: Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  isFilipino
                      ? 'FSL → Teksto + Pagsasalita'
                      : 'FSL → Text + Speech Conversion',
                  style: const TextStyle(
                    color: _orange,
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                  ),
                ),
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
                              child: Text(
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
                        ? (isFilipino
                            ? '> Naghihintay ng sign input...'
                            : '> Waiting for sign input...')
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
              onPressed: () => Navigator.pop(context),
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