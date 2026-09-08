import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import '../main.dart';
import '../theme/tokens.dart';
import '../services/camera_service.dart';
import '../services/inference_service.dart';
import '../services/speech_service.dart';
import '../services/phrase_service.dart';
import '../services/trace_service.dart';
import 'app_scaffold.dart';
import 'framing_guide.dart';
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

  /// Watchdog that restarts the image stream if frames stop arriving.
  Timer? _watchdog;
  bool _restarting = false;

  /// Closes a phrase once signing stops, and the gloss built so far.
  Timer? _phraseTimer;
  String _pendingGloss = '';

  /// Incremented per processed frame so the skeleton overlay repaints.
  int _skeletonTick = 0;

  /// Phrase-timer ticks, used only to space out the trace heartbeat.
  int _traceTick = 0;

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

  /// Whether detection was running when the app went to the background.
  bool _resumeDetecting = false;

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Deliberately NOT inactive.
    //
    // Android passes through inactive in BOTH directions — going away and
    // coming back — so the old teardown there also ran on every resume, while
    // the matching branch called only initialize(), which returns early while
    // the controller is still alive and never restarts the stream. Every
    // background round trip therefore left the stream stopped with the button
    // still reading "Stop Recognition" and no frames arriving; the 4s watchdog
    // was the only thing that ever recovered it.
    if (state == AppLifecycleState.paused ||
        state == AppLifecycleState.detached) {
      _resumeDetecting = _isDetecting;
      TraceService.log('lifecycle  backgrounded (detecting=$_isDetecting) '
          '— releasing camera');
      _watchdog?.cancel();
      // Release the sensor, not merely the stream. Logcat shows the camera
      // held OPENED for the full ten minutes the app sat in the background —
      // heating the phone and draining the battery with nothing on screen.
      // Frame rate here is temperature-bound, so that heat is paid back on
      // return as a slower pipeline.
      CameraService.dispose();
      if (mounted) setState(() => _cameraReady = false);
    } else if (state == AppLifecycleState.resumed) {
      if (_resumeDetecting || _cameraReady) _restoreCamera();
    }
  }

  /// Rebuilds the camera after a background trip, restarting the stream if
  /// detection was running when we left.
  Future<void> _restoreCamera() async {
    if (_restarting) return;
    // Held across the rebuild so the watchdog cannot start a second restart
    // on top of this one.
    _restarting = true;
    try {
      await CameraService.initialize();
      if (!mounted) {
        await CameraService.dispose();
        return;
      }
      setState(() => _cameraReady = CameraService.isInitialized);
      TraceService.log('lifecycle  resumed, camera rebuilt '
          '(ok=$_cameraReady, restarting stream=$_resumeDetecting)');
      if (!_cameraReady || !_resumeDetecting) return;
      _resumeDetecting = false;
      await CameraService.startImageStream((CameraImage image) async {
        try {
          await _handleFrame(image);
        } catch (e) {
          debugPrint('frame callback error (after resume): $e');
        }
      });
    } catch (e) {
      debugPrint('camera restore failed: $e');
    } finally {
      _restarting = false;
      if (mounted && _isDetecting) _startWatchdog();
    }
  }

  /// Releases the camera when this view goes away — including when the Hub
  /// switches off the Camera tab, so the sensor is not held while the user is
  /// on Voice or Text.
  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _watchdog?.cancel();
    _phraseTimer?.cancel();
    // Leaving the screen must not lose the session: unflushed lines are the
    // most recent ones, which is exactly the window a failure sits in.
    TraceService.stop('view disposed');
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

  /// Switches recognition mode and rebuilds the pipeline underneath it.
  ///
  /// A mode switch changes what the Kotlin side does per frame — LETTERS skips
  /// pose entirely and runs a second hand pass on a crop — so the landmarker
  /// state carries assumptions from the mode that just ended. Resetting only
  /// the Dart-side buffers left recognition silent after switching away and
  /// back, with frames still flowing so nothing looked wrong. Restarting the
  /// stream rebuilds both sides, which is what killing the app was doing.
  Future<void> _cycleMode() async {
    InferenceService.cycleMode();
    PhraseService.clear();
    setState(() {
      _translationLog.clear();
      _pendingGloss = '';
    });
    if (!_isDetecting || _restarting) return;
    _restarting = true;
    try {
      await CameraService.hardRestart((CameraImage image) async {
        try {
          await _handleFrame(image);
        } catch (e) {
          debugPrint('frame callback error (after mode switch): $e');
        }
      });
    } catch (e) {
      debugPrint('mode switch restart failed: $e');
    } finally {
      _restarting = false;
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
      _watchdog?.cancel();
      _phraseTimer?.cancel();
      PhraseService.clear();
      await CameraService.stopImageStream();
      InferenceService.clearBuffer();
      await TraceService.stop('recognition off');
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
      await TraceService.start('mode=${InferenceService.mode.name}');
      // TTS is brought up AFTER the camera, and not awaited.
      //
      // A failing engine holds initTTS for many seconds — a rebind plus six
      // 500ms polls plus several 5s deadlines — and blocking the camera behind
      // it made pressing Start feel broken while the preview sat empty. Speech
      // is output; nothing about capture depends on it.
      unawaited(SpeechService.initTTS().then((_) {
        TraceService.log('tts available=${SpeechService.ttsAvailable} '
            'status=${SpeechService.ttsStatus} lang=${SpeechService.ttsLanguage}');
        TraceService.log('tts filipino installed='
            '${SpeechService.filipinoInstalled}  '
            'languages: ${SpeechService.ttsLanguages.join(", ")}');
        if (mounted) setState(() {});
      }));
      _startWatchdog();
      _startPhraseTimer();
      await CameraService.startImageStream((CameraImage image) async {
        try {
          await _handleFrame(image);
        } catch (e) {
          // An exception escaping a stream callback kills the subscription
          // silently: the preview keeps running and no frame ever arrives
          // again, which is indistinguishable from the app freezing.
          debugPrint('frame callback error: $e');
        }
      });
    }
  }

  /// One frame, from whichever call site started the stream.
  ///
  /// Extracted because the stream is started in three places — initial start,
  /// watchdog recovery and mode switch — and each previously carried its own
  /// copy of this logic. The copies had already drifted: the watchdog's
  /// version spoke each sign individually and never chained phrases.
  Future<void> _handleFrame(CameraImage image) async {
    final result = await InferenceService.processFrame(image);
    if (!mounted) return;

    // A finished spelled word takes priority: it is a whole name, not a single
    // letter, and is the only output that can carry proper nouns.
    final spelled = InferenceService.completedWord;
    if (spelled != null) {
      final spoken = InferenceService.completedWordSpoken ?? spelled;
      InferenceService.completedWord = null;
      InferenceService.completedWordSpoken = null;
      debugPrint('SPEAK|spelled|display="$spelled"|spoken="$spoken"');
      TraceService.log('spelled  "$spelled"  '
          'fps=${InferenceService.processingFps.toStringAsFixed(1)}');
      SpeechService.speak(spoken);
      setState(() {
        _skeletonTick++;
        _translationLog.insert(0, spelled);
        if (_translationLog.length > 5) _translationLog.removeLast();
      });
      return;
    }

    if (result == null) {
      setState(() => _skeletonTick++);
      return;
    }

    // In LETTERS mode a result is a single letter mid-word. Speaking it would
    // read "S, I, K, L, A, B" instead of the word; the spelling strip already
    // shows progress and only the finished word is spoken.
    if (InferenceService.isLettersMode) {
      setState(() => _skeletonTick++);
      return;
    }

    // Chain signs into a phrase rather than speaking each one. ASL drops the
    // copula and articles, so sign-by-sign speech could never produce
    // "How are you?" from HOW YOU.
    PhraseService.add(result.label, DateTime.now().millisecondsSinceEpoch);
    if (PhraseService.isRecognisedComplete()) {
      _speakPhrase();
    } else {
      setState(() {
        _skeletonTick++;
        _pendingGloss = PhraseService.glossLine;
      });
    }
  }

  /// Speaks a phrase once signing has stopped for [PhraseService.phraseGapMs].
  ///
  /// Driven by a timer rather than by frame arrival: the phrase ends when
  /// NOTHING is recognised, so there is no frame to hang the decision on.
  void _startPhraseTimer() {
    _phraseTimer?.cancel();
    _phraseTimer = Timer.periodic(const Duration(milliseconds: 400), (_) {
      if (!mounted || !_isDetecting) return;
      // Heartbeat every ten seconds. Frame rate is the variable everything
      // else depends on and the one thing a screenshot cannot show over time —
      // the reported failure was a collapse from 13.2 fps to 5.6 fps that only
      // two screenshots five minutes apart happened to capture.
      if (++_traceTick % 25 == 0) {
        TraceService.log(
            'hb  fps=${InferenceService.processingFps.toStringAsFixed(1)}'
            '  mode=${InferenceService.mode.name}'
            '  hands=${InferenceService.lastHadHands ? 1 : 0}'
            '${InferenceService.isLettersMode ? "  extent=${InferenceService.lastHandExtentPx.toStringAsFixed(0)}px" : ""}'
            '  tts=${SpeechService.ttsAvailable ? "ok" : SpeechService.ttsStatus}');
      }
      final now = DateTime.now().millisecondsSinceEpoch;
      if (!PhraseService.isComplete(now)) return;

      _speakPhrase();
    });
  }

  /// Drops the last recognised sign from the pending phrase.
  ///
  /// The correction has to be available BEFORE the phrase is spoken. Without
  /// it a misread could only be fixed by waiting out the 3.5s gap, listening
  /// to the wrong sentence, and signing the whole thing again — which is also
  /// why every test round took longer than it needed to.
  void _backspaceSign() {
    final removed =
        PhraseService.removeLast(DateTime.now().millisecondsSinceEpoch);
    if (removed == null) return;
    setState(() => _pendingGloss = PhraseService.glossLine);
  }

  /// Speaks whatever phrase is buffered and logs it.
  void _speakPhrase() {
    final signs = PhraseService.take();
    if (signs.isEmpty) return;
    final english = PhraseService.toEnglish(signs);
    TraceService.log('phrase  gloss="${signs.join(" ")}"  english="$english"  '
        'fps=${InferenceService.processingFps.toStringAsFixed(1)}  '
        'tts=${SpeechService.ttsAvailable ? "ok" : SpeechService.ttsStatus}');
    SpeechService.speak(english);
    if (!mounted) return;
    setState(() {
      _pendingGloss = '';
      // The gloss is shown next to the English so the panel can see what was
      // recognised as well as what was said — the two differ, and that
      // difference is the feature.
      _translationLog.insert(0, '$english   [${signs.join(" ")}]');
      if (_translationLog.length > 5) _translationLog.removeLast();
    });
  }

  /// Why speech is unavailable, rather than one message for every cause.
  ///
  /// The fixed string claimed the device had no voice whatever the reason, so
  /// a congested engine, a missing voice for the requested language and a
  /// genuinely absent engine all read identically — and the cause that
  /// actually occurs on the test device, congestion once the pipeline slows,
  /// was the one being misreported as missing hardware. SpeechService.ttsStatus
  /// already carries the distinction and the Hub already shows it.
  String _ttsBannerText(bool isFilipino) {
    final status = SpeechService.ttsStatus;
    final unknown = status.isEmpty || status == 'not initialised';
    if (!isFilipino) {
      return unknown
          ? 'No speech voice on this device — text only'
          : '$status — text only';
    }
    switch (status) {
      case 'Text-to-speech is busy — will retry':
        return 'Abala ang text-to-speech — susubukan muli';
      case 'Text-to-speech stopped responding':
        return 'Hindi tumutugon ang text-to-speech';
      case 'No usable voice installed':
        return 'Walang naka-install na boses — teksto lang';
      default:
        return 'Walang boses sa device na ito — teksto lang';
    }
  }

  /// Restarts the image stream if frames stop arriving while detection is on.
  ///
  /// The pipeline has gone silent twice on device — preview alive, no crash in
  /// logcat, nothing recognised until the app was restarted, both times just
  /// after a word was decided. Timeouts were added to the platform call and to
  /// the pose future, and it still happened, so the cause is not yet known.
  ///
  /// This does not diagnose it. It recovers from it, whatever it is, which is
  /// the property that actually matters when the alternative is the app dying
  /// silently in front of a panel.
  void _startWatchdog() {
    _watchdog?.cancel();
    _watchdog = Timer.periodic(const Duration(seconds: 2), (_) async {
      if (!mounted || !_isDetecting || _restarting) return;
      final since = DateTime.now().millisecondsSinceEpoch -
          InferenceService.lastProcessedAtMs;
      // Frames arrive every ~80ms, so 4s of nothing is a stall, not a lull.
      if (InferenceService.lastProcessedAtMs == 0 || since < 4000) return;

      _restarting = true;
      InferenceService.watchdogRestarts++;
      debugPrint('watchdog: no frame for ${since}ms, restarting image stream');
      TraceService.log('WATCHDOG  no frame for ${since}ms, restarting stream  '
          '(restart #${InferenceService.watchdogRestarts})');
      try {
        // clearBuffer releases _isProcessing, which gates every frame.
        InferenceService.clearBuffer();
        // hardRestart, not stop-then-start: the plugin's isStreamingImages flag
        // can be left true after a stop that did not take, and both plugin
        // methods are guarded by it, so stop/start was a pair of silent
        // no-ops. Only a fresh controller reliably clears that state.
        final ok = await CameraService.hardRestart((CameraImage image) async {
          try {
            await _handleFrame(image);
          } catch (e) {
            debugPrint('frame callback error (after restart): $e');
          }
        });
        debugPrint('watchdog: hardRestart ${ok ? "recovered" : "FAILED"}');
      } catch (e) {
        debugPrint('watchdog restart failed: $e');
      } finally {
        _restarting = false;
      }
    });
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

            // Framing guide, while setting up only. Tilt is the framing error
            // that actually breaks the posture gate, and it is invisible to
            // whoever is holding the phone.
            FramingGuide(
              visible: !_isDetecting && _cameraReady && !_permissionDenied,
              isFilipino: isFilipino,
              color: c.orange,
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
                  // Cycles FSL WORDS -> LETTERS -> ASL WORDS. Letters are
                  // shared between the two languages: the fingerspelling
                  // alphabets are the same, so one model serves both.
                  _CameraChip(
                    label: InferenceService.modeLabel,
                    icon: Icons.abc_rounded,
                    active: true,
                    onTap: _cycleMode,
                  ),
                  // SKELETON chip removed for the demo. The overlay recomputes
                  // the old letterbox mapping, but the camera now crops to the
                  // signer, so the joints no longer land on the hands — it
                  // would read as a bug on stage. Recognition never used it.
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
                  // Debug builds only: writes each classified sequence to
                  // external storage so the tensor the device actually builds
                  // can be diffed against the training corpus offline. Never
                  // reachable in a release build.
                  if (kDebugMode) ...[
                    const SizedBox(width: 6),
                    _CameraChip(
                      label: InferenceService.dumpTensors ? 'DUMP ON' : 'DUMP',
                      icon: Icons.save_alt_rounded,
                      active: InferenceService.dumpTensors,
                      onTap: () => setState(() {
                        InferenceService.dumpTensors =
                            !InferenceService.dumpTensors;
                      }),
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
          // A communication aid that quietly stops talking is worse than one
          // that says it cannot. Some devices ship without a TTS engine, or
          // without voice data for the requested language.
          if (_isDetecting && !SpeechService.ttsAvailable) ...[
            const SizedBox(height: 6),
            Row(
              children: [
                const Icon(Icons.volume_off_rounded,
                    color: Color(0xFFFEBC2E), size: 14),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    _ttsBannerText(isFilipino),
                    style: const TextStyle(
                        color: Color(0xFFFEBC2E), fontSize: 11),
                  ),
                ),
              ],
            ),
          ],
          if (InferenceService.isLettersMode) ...[
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 12, vertical: 10),
                    decoration: BoxDecoration(
                      color: c.surface,
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(
                          color: c.orange.withValues(alpha: 0.35)),
                    ),
                    child: Text(
                      InferenceService.spelledWord.isEmpty
                          ? (isFilipino
                              ? 'Baybayin ang salita...'
                              : 'Spell a word...')
                          : InferenceService.spelledWord,
                      style: TextStyle(
                        color: InferenceService.spelledWord.isEmpty
                            ? c.subtitle
                            : c.text,
                        fontSize: 20,
                        letterSpacing: 3,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 6),
                _SpellButton(
                  icon: Icons.backspace_outlined,
                  onTap: () =>
                      setState(InferenceService.backspaceLetter),
                ),
                const SizedBox(width: 6),
                _SpellButton(
                  icon: Icons.volume_up_rounded,
                  filled: true,
                  onTap: () => setState(InferenceService.finishSpelling),
                ),
              ],
            ),
          ],
          // The phrase as it builds, before it is spoken. Without this the
          // app looks unresponsive for the seconds between the first sign and
          // the end of the phrase — the signs ARE being collected, but nothing
          // said so.
          if (_pendingGloss.isNotEmpty) ...[
            const SizedBox(height: 4),
            Row(
              children: [
                Icon(Icons.more_horiz_rounded, size: 16, color: c.orange),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    _pendingGloss,
                    style: TextStyle(
                      color: c.orange,
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                      letterSpacing: 0.4,
                    ),
                  ),
                ),
                const SizedBox(width: 6),
                // Same two controls as the spelling strip, deliberately: the
                // signer should not have to learn a second idiom for the same
                // two actions.
                _SpellButton(
                  icon: Icons.backspace_outlined,
                  onTap: _backspaceSign,
                ),
                const SizedBox(width: 6),
                _SpellButton(
                  icon: Icons.volume_up_rounded,
                  filled: true,
                  onTap: _speakPhrase,
                ),
              ],
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

/// Small square action button beside the spelling strip.
class _SpellButton extends StatelessWidget {
  final IconData icon;
  final bool filled;
  final VoidCallback onTap;

  const _SpellButton({
    required this.icon,
    required this.onTap,
    this.filled = false,
  });

  @override
  Widget build(BuildContext context) {
    final c = AppColors.of(SiklabKumpasApp.of(context)?.isDark ?? true);
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 44,
        height: 44,
        decoration: BoxDecoration(
          color: filled ? c.orange : c.surface,
          borderRadius: BorderRadius.circular(10),
          border: Border.all(
              color: filled ? c.orange : c.orange.withValues(alpha: 0.35)),
        ),
        child: Icon(icon,
            size: 20, color: filled ? Colors.white : c.text),
      ),
    );
  }
}
