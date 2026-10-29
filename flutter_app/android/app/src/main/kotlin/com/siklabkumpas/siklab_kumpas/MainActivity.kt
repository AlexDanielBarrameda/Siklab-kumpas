package com.siklabkumpas.siklab_kumpas

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.os.Handler
import android.os.Looper
import android.util.Log
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import java.io.File
import java.io.FileOutputStream
import java.lang.reflect.Method
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : FlutterActivity() {

    private val CHANNEL = "com.siklabkumpas/inference"

    private var handLandmarker: HandLandmarker? = null
    private var poseLandmarker: PoseLandmarker? = null

    @Volatile
    private var isInitialized = false

    /**
     * MediaPipe IMAGE-mode landmarkers are not thread safe, so everything runs on a
     * single background thread. This also keeps the platform (UI) thread free.
     */
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())

    /**
     * Training geometry. Every sample the models saw was a 640x480 landscape frame
     * (extract_fsl105.py resizes to 640x480, test_webcam.py captures at 640x480).
     *
     * MediaPipe normalises x by image width and y by image height, so feeding it a
     * portrait frame rescales the two axes by different factors and distorts every
     * hand shape. We letterbox the upright frame into the training canvas instead.
     */
    /**
     * Kept at 4:3 to match the training aspect — MediaPipe normalises x by width
     * and y by height, so only the ratio matters, not the absolute size. Dropped
     * from 640x480 to cut the per-frame pixel work and the GPU upload feeding the
     * landmarkers; the hand crop still fills this canvas, and the training stills
     * themselves were only a few hundred pixels across.
     */
    private val CANVAS_W = 480
    private val CANVAS_H = 360

    /**
     * Fraction of the frame the hand spans in the FSL-letters training stills
     * (measured: median 0.670 wrist-to-farthest-landmark over the whole set).
     *
     * These features are wrist-relative but never scale-normalised, so apparent hand
     * size scales every coordinate directly. A signer sitting at arm's length from a
     * phone produces a span near 0.20, where the letters model measures ~30-50%
     * accuracy. Re-running the landmarker on a crop around the hand restores the
     * training framing (measured 80.8% -> 90.7% on held-out stills).
     *
     * FSL-105 does NOT need this: its clips were filmed torso-in-frame and have a
     * median span of 0.130, which already matches phone framing.
     */
    private val LETTER_TARGET_SPAN = 0.67f

    // Scratch buffers, reused across frames to avoid per-frame allocation.
    private var argbBuffer: IntArray? = null
    private var srcBitmap: Bitmap? = null
    private var canvasBitmap: Bitmap? = null
    private var uprightBitmap: Bitmap? = null
    private var cropBitmap: Bitmap? = null
    private val canvasPaint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)

    // Where the upright frame was placed inside the letterbox canvas, so hand
    // coordinates found on the canvas can be mapped back to full-resolution pixels.
    private var boxLeft = 0f
    private var boxTop = 0f
    private var boxScale = 1f

    /**
     * Signer-tracking crop.
     *
     * Letterboxing an upright portrait frame into the 4:3 canvas spends ~44% of
     * the width on black bars and leaves the signer small. Measured on a Redmi
     * Turbo 4 Pro against the training corpus:
     *
     *     shoulder width   corpus 0.417   device 0.19-0.24
     *     hand span        corpus 0.271   device 0.10-0.15
     *
     * The features are scale-invariant, so distance alone is harmless -- but a
     * ~48px hand is below what the landmarker resolves well, and it compresses
     * the detected hand toward an average shape. The measured symptom is that
     * inter-hand offset divided by hand span reads 7.17 on device against 2.88
     * in the corpus, which puts the hand_loc block 21.5% outside the corpus
     * range and drags velocity and acceleration with it.
     *
     * So instead of shrinking the signer to fit, crop to them. The crop is
     * sized from the last pose so their shoulders land near the corpus
     * proportion, and centred on the shoulder midpoint.
     */
    private var signerCx = -1f          // canvas-normalised shoulder midpoint
    private var signerCy = -1f
    private var signerShoulder = -1f    // canvas-normalised shoulder width
    private var cropCx = -1f            // smoothed crop centre, upright pixels
    private var cropCy = -1f
    private var cropW = -1f             // smoothed crop width, upright pixels

    /** Corpus median shoulder width, as a fraction of canvas width. */
    private val TARGET_SHOULDER = 0.40f
    /** How fast the crop follows the signer. Low so framing does not jitter. */
    private val CROP_SMOOTH = 0.15f

    /**
     * Consecutive frames on which pose has failed to locate the signer.
     *
     * Losing the signer used to set signerShoulder = -1f and nothing else,
     * which stopped the crop being UPDATED but left cropW/cropCx/cropCy at
     * their last values -- so the crop went on drawing the same stale region
     * indefinitely. Since pose then runs on that stale crop, a signer who has
     * moved out of it can never be found again, and the failure sustains
     * itself: no pose, so no new crop, so no pose.
     *
     * Measured on device from the diagnostic trace: two dead spells of 28s and
     * 43s in one session and 90s in the next, each ending abruptly when the
     * signer happened to line back up with the frozen box. Frame rate was
     * NORMAL throughout -- in fact higher during the dead spells (15.6 fps
     * against ~10.5 while working), because an empty crop is cheap to process.
     * Letters mode never showed the fault because it clears cropW every frame.
     */
    private var signerLostFrames = 0

    /**
     * Losses tolerated before the crop starts opening up.
     *
     * Pose blinks for a single frame routinely -- hands crossing the face,
     * motion blur -- and widening on those would perturb the framing mid-sign,
     * which is a model input. Below this the behaviour is byte-identical to
     * before; only a SUSTAINED loss triggers recovery.
     */
    private val CROP_GRACE_FRAMES = 3

    /** How fast a lost crop opens back out, per frame. */
    private val CROP_RELAX = 1.18f

    // Cached reflective accessor for pose visibility (MediaPipe 0.10.0 exposes it
    // inconsistently across builds). Looked up once instead of 25x per frame.
    private var visibilityMethod: Method? = null
    private var visibilityResolved = false

    private var frameCounter = 0L


    /**
     * Pose runs on EVERY frame, concurrently with the hand work.
     *
     * It was previously cached and reused across 3 frames, on the reasoning that
     * the body moves slowly. That was wrong: the pose block carries the WRISTS,
     * so caching left hand position stale by up to 0.4s — and FATHER vs MOTHER
     * are distinguished *only* by hand height relative to the face. Zeroing pose
     * drops FATHER to 0% accuracy, so a lagged pose blurred precisely the signal
     * those signs depend on; MOTHER never appeared once in a 34-sign trace.
     *
     * Running it every frame would have cost ~1.3 fps sequentially. Running it
     * on its own thread costs almost nothing instead, since the frame budget
     * becomes max(hand, pose) rather than hand + pose.
     */
    private val poseExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    /**
     * Monotonic timestamp for the pose landmarker's VIDEO mode, which rejects a
     * timestamp that does not strictly increase. Pose runs on a single-thread
     * executor so calls are already serialised; this only has to keep counting.
     */
    private val poseTimestamp = java.util.concurrent.atomic.AtomicLong(0L)

    /** Pose landmarks for one frame, as the flat 25 x (x, y, z, visibility) block. */
    private fun detectPose(frame: Bitmap): FloatArray {
        val out = FloatArray(100)
        // VIDEO mode instead of IMAGE: it tracks the pose between frames rather
        // than re-detecting from scratch, which is cheaper per frame and steadier
        // over time. Same model weights, so the landmark distribution the word
        // models were trained on is unchanged — that matters here, because the
        // pose block is never wrist-normalised and carries roughly 34 points of
        // accuracy, so anything that shifts it is a domain change, not a
        // speedup.
        //
        // Deliberately NOT applied to the hand landmarker: that one alternates
        // between the full canvas and a crop around the hand, two different
        // coordinate spaces, and VIDEO mode would carry tracking state across
        // the two.
        val result = poseLandmarker?.detectForVideo(
            BitmapImageBuilder(frame).build(),
            poseTimestamp.addAndGet(33L)
        ) ?: return out
        val list = result.landmarks()
        if (list.isEmpty()) return out
        val points = list[0]
        for (j in 0 until minOf(25, points.size)) {
            if (j * 4 + 3 >= out.size) break
            out[j * 4] = points[j].x()
            out[j * 4 + 1] = points[j].y()
            out[j * 4 + 2] = points[j].z()
            val vis = readVisibility(points[j])
            out[j * 4 + 3] = if (vis.isNaN()) 1f else vis
        }
        return out
    }

    /**
     * Letters mode used to run the hand landmarker twice per frame — once on the
     * full canvas to find the hand, once on the crop. Measured on device that was
     * ~165ms + ~130ms of a ~350ms frame.
     *
     * The hand barely moves between frames at 3 fps, so we keep the previous crop
     * rectangle and go straight to the second pass, re-acquiring on the full frame
     * only when tracking fails or periodically to stop the box drifting.
     */
    /**
     * Diagnostics for the letters path, reported back to Dart as two extra
     * slots on the landmark array.
     *
     * The letters crop is sized extent / LETTER_TARGET_SPAN and then stretched
     * to CANVAS_W, so the upscale factor is 0.67 * 480 / extent — about 3x for
     * a hand 100px across and 8x for one at 40px. The only guard is
     * `extent < 4f`, which permits a 16x blow-up of a four-pixel blob.
     * MediaPipe returns 21 landmarks on the interpolation regardless and the
     * letters model, which cannot know its input is mush, lands on whichever
     * classes own that degenerate region — reported on device as a stream of
     * P, Q and O that only stops when the hand is very close to the lens.
     *
     * Measured here rather than guessed: [handExtentPx] is the hand's bounding
     * extent in upright pixels and [handFromTracker] says whether the reading
     * came from the reused crop rectangle or a fresh full-frame acquisition.
     */
    private var handExtentPx = 0f
    private var handFromTracker = false

    private val REACQUIRE_EVERY_N = 6
    private var trackedRect: Rect? = null
    private var trackedFrame = -1L

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(
            flutterEngine.dartExecutor.binaryMessenger,
            CHANNEL
        ).setMethodCallHandler { call, result ->
            when (call.method) {
                "initMediaPipe" -> {
                    executor.execute {
                        try {
                            initMediaPipe(applicationContext)
                            mainHandler.post { result.success(true) }
                        } catch (e: Exception) {
                            Log.e("MediaPipe", "Init error: ${e.message}", e)
                            mainHandler.post {
                                result.error("INIT_ERROR", e.message, null)
                            }
                        }
                    }
                }

                "processFrame" -> {
                    val yBytes = call.argument<ByteArray>("yBytes")
                    val uBytes = call.argument<ByteArray>("uBytes")
                    val vBytes = call.argument<ByteArray>("vBytes")
                    val width = call.argument<Int>("width") ?: 0
                    val height = call.argument<Int>("height") ?: 0
                    val yRowStride = call.argument<Int>("yRowStride") ?: width
                    val uvRowStride = call.argument<Int>("uvRowStride") ?: width
                    val uvPixelStride = call.argument<Int>("uvPixelStride") ?: 1
                    val rotation = call.argument<Int>("rotation") ?: 270
                    val mirror = call.argument<Boolean>("mirror") ?: true
                    val lettersMode = call.argument<Boolean>("lettersMode") ?: false
                    val handBlock = call.argument<String>("handBlock") ?: "auto"
                    val debug = call.argument<Boolean>("debug") ?: false

                    if (yBytes == null || !isInitialized || width <= 0 || height <= 0) {
                        result.success(DoubleArray(226))
                        return@setMethodCallHandler
                    }

                    executor.execute {
                        val out = try {
                            extractLandmarks(
                                yBytes, uBytes, vBytes,
                                width, height,
                                yRowStride, uvRowStride, uvPixelStride,
                                rotation, mirror, lettersMode, handBlock, debug
                            )
                        } catch (e: Exception) {
                            Log.e("MediaPipe", "processFrame error: ${e.message}", e)
                            DoubleArray(226)
                        }
                        mainHandler.post { result.success(out) }
                    }
                }


                /**
                 * App-private files directory, for small persistent flags.
                 *
                 * Used to remember that the onboarding tour has been seen.
                 * A marker file rather than shared_preferences: the channel
                 * already exists and works, and adding a dependency days
                 * before a defence is a risk with no matching benefit for one
                 * boolean.
                 */
                "appFilesDir" -> {
                    result.success(filesDir.absolutePath)
                }

                /**
                 * App-specific external storage, for the debug tensor dump.
                 *
                 * Offline evaluation predicts the ASL model should recognise
                 * roughly half of signs under simulated device conditions, and
                 * live it recognises almost none. Everything cheap has been
                 * ruled out by measurement -- frame rate, segmentation,
                 * canonicalisation, zero-hand frames -- so the remaining step is
                 * to capture what the device actually feeds the model and diff
                 * it against the corpus, dimension by dimension.
                 *
                 * This path needs no permission and is reachable over adb:
                 *   adb pull /sdcard/Android/data/<pkg>/files/tensor_dumps
                 */
                /**
                 * App-specific external storage root, for the diagnostic trace.
                 *
                 * Deliberately NOT appFilesDir, which returns internal storage:
                 * plain `adb pull` cannot read /data/user/0/<pkg> on a stock
                 * device, so a trace written there is only reachable through
                 * `run-as` -- which works on a debuggable build and stops
                 * working on the release one, exactly when a field failure
                 * would matter most.
                 *
                 *   adb pull /sdcard/Android/data/<pkg>/files/trace.log
                 *
                 * Falls back to internal storage rather than failing: a trace
                 * that is awkward to retrieve still beats no trace at all.
                 */
                "traceDir" -> {
                    val dir = getExternalFilesDir(null)
                    result.success((dir ?: filesDir).absolutePath)
                }

                "dumpDir" -> {
                    val dir = getExternalFilesDir(null)
                    if (dir == null) {
                        result.success(null)
                    } else {
                        val out = java.io.File(dir, "tensor_dumps")
                        if (!out.exists()) out.mkdirs()
                        result.success(out.absolutePath)
                    }
                }

                else -> result.notImplemented()
            }
        }
    }

    private fun initMediaPipe(context: Context) {
        if (isInitialized) return

        val handModelFile = copyAssetToFile(context, "assets/models/hand_landmarker.task")
        val poseModelFile = copyAssetToFile(context, "assets/models/pose_landmarker_full.task")

        // Try the GPU delegate first — these two models dominate per-frame cost and
        // CPU-only inference measured about 1 fps on the test device. Falls back to
        // CPU wherever the GPU path is unavailable.
        for (delegate in listOf(Delegate.GPU, Delegate.CPU)) {
            try {
                handLandmarker = HandLandmarker.createFromOptions(
                    context,
                    HandLandmarker.HandLandmarkerOptions.builder()
                        .setBaseOptions(
                            BaseOptions.builder()
                                .setModelAssetPath(handModelFile.absolutePath)
                                .setDelegate(delegate)
                                .build()
                        )
                        .setRunningMode(RunningMode.IMAGE)
                        .setNumHands(2)
                        .setMinHandDetectionConfidence(0.5f)
                        .setMinHandPresenceConfidence(0.5f)
                        .setMinTrackingConfidence(0.5f)
                        .build()
                )

                poseLandmarker = PoseLandmarker.createFromOptions(
                    context,
                    PoseLandmarker.PoseLandmarkerOptions.builder()
                        .setBaseOptions(
                            BaseOptions.builder()
                                .setModelAssetPath(poseModelFile.absolutePath)
                                .setDelegate(delegate)
                                .build()
                        )
                        .setRunningMode(RunningMode.VIDEO)
                        .setMinPoseDetectionConfidence(0.5f)
                        .setMinPosePresenceConfidence(0.5f)
                        .setMinTrackingConfidence(0.5f)
                        .build()
                )

                Log.d("MediaPipe", "Initialized successfully on $delegate")
                isInitialized = true
                return
            } catch (e: Throwable) {
                Log.w("MediaPipe", "$delegate delegate unavailable: ${e.message}")
                handLandmarker?.close()
                poseLandmarker?.close()
                handLandmarker = null
                poseLandmarker = null
            }
        }

        throw IllegalStateException("Could not create MediaPipe landmarkers")
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Frame conversion
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * YUV_420_888 -> ARGB_8888. Fixed-point BT.601 (full range), coefficients x1024.
     *
     * MediaPipe's palm detector leans on skin-tone chroma, so handing it a greyscale
     * image (the Y plane replicated into all three channels) measurably degrades
     * landmark precision.
     */
    private fun yuvToArgb(
        y: ByteArray, u: ByteArray?, v: ByteArray?,
        width: Int, height: Int,
        yRowStride: Int, uvRowStride: Int, uvPixelStride: Int
    ): IntArray {
        var buf = argbBuffer
        if (buf == null || buf.size != width * height) {
            buf = IntArray(width * height)
            argbBuffer = buf
        }

        val hasChroma = u != null && v != null

        for (row in 0 until height) {
            val yBase = row * yRowStride
            val uvBase = (row shr 1) * uvRowStride
            val outBase = row * width

            for (col in 0 until width) {
                val yi = yBase + col
                if (yi >= y.size) continue
                val yy = y[yi].toInt() and 0xFF

                var uu = 0
                var vv = 0
                if (hasChroma) {
                    val uvi = uvBase + (col shr 1) * uvPixelStride
                    if (uvi < u!!.size) uu = (u[uvi].toInt() and 0xFF) - 128
                    if (uvi < v!!.size) vv = (v[uvi].toInt() and 0xFF) - 128
                }

                val r = yy + ((1436 * vv) shr 10)
                val g = yy - ((352 * uu + 731 * vv) shr 10)
                val b = yy + ((1815 * uu) shr 10)

                buf[outBase + col] =
                    (0xFF shl 24) or (clamp8(r) shl 16) or (clamp8(g) shl 8) or clamp8(b)
            }
        }
        return buf
    }

    private fun clamp8(v: Int): Int = if (v < 0) 0 else if (v > 255) 255 else v

    /**
     * Rotates the sensor frame upright, optionally mirrors it, then letterboxes the
     * result into the 640x480 training canvas.
     *
     * The mirror reproduces `cv2.flip(frame, 1)` from test_webcam.py. It is not
     * cosmetic: MediaPipe derives its "Left"/"Right" handedness label from the image
     * as given, and swapping the two hand blocks drops letter accuracy from 97.8% to
     * 9.3%, so this flag has to be right.
     */
    private fun buildCanvas(
        pixels: IntArray, width: Int, height: Int,
        rotation: Int, mirror: Boolean
    ): Bitmap {
        var src = srcBitmap
        if (src == null || src.width != width || src.height != height) {
            src?.recycle()
            src = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            srcBitmap = src
        }
        src.setPixels(pixels, 0, width, 0, 0, width, height)

        val matrix = Matrix().apply {
            postRotate(rotation.toFloat())
            // post* applies after the rotation, so this mirrors the upright image.
            if (mirror) postScale(-1f, 1f)
        }
        uprightBitmap?.recycle()
        val upright = Bitmap.createBitmap(src, 0, 0, width, height, matrix, true)
        uprightBitmap = upright

        var canvasBmp = canvasBitmap
        if (canvasBmp == null) {
            canvasBmp = Bitmap.createBitmap(CANVAS_W, CANVAS_H, Bitmap.Config.ARGB_8888)
            canvasBitmap = canvasBmp
        }

        val canvas = Canvas(canvasBmp)
        canvas.drawColor(Color.BLACK)

        // Letterbox geometry, still needed as the fallback and as the frame of
        // reference for translating the last pose back into upright pixels.
        val fitScale = minOf(
            CANVAS_W.toFloat() / upright.width,
            CANVAS_H.toFloat() / upright.height
        )
        val fitW = upright.width * fitScale
        val fitH = upright.height * fitScale
        val fitLeft = (CANVAS_W - fitW) / 2f
        val fitTop = (CANVAS_H - fitH) / 2f

        // Aim the crop using the PREVIOUS frame's pose, which was measured in
        // the previous frame's canvas space, so it has to come back through the
        // box mapping that produced it.
        if (signerShoulder > 0f) {
            val shoulderPx = signerShoulder * CANVAS_W / boxScale
            val cxPx = (signerCx * CANVAS_W - boxLeft) / boxScale
            val cyPx = (signerCy * CANVAS_H - boxTop) / boxScale

            // Width that puts the shoulders at the corpus proportion.
            var wantW = shoulderPx / TARGET_SHOULDER
            // Never zoom past the full frame, and never so tight that a raised
            // hand leaves the crop.
            //
            // minW is clamped to maxW before use. Without that, a stale
            // shoulder measurement makes minW exceed the frame width and
            // coerceIn throws "Cannot coerce value to an empty range" — which
            // extractLandmarks catches and turns into an all-zero vector, so
            // every frame silently reports no hands and no pose from then on.
            // Letters mode reaches that state easily: it never runs pose, so
            // signerShoulder and boxScale drift out of step with each other.
            val maxW = upright.width.toFloat()
            val minW = (shoulderPx * 1.6f).coerceAtMost(maxW)
            wantW = wantW.coerceIn(minW, maxW)

            // Signing space sits above the shoulders more than below, so bias
            // the crop upward rather than centring on the shoulder line.
            val wantCy = cyPx - wantW * 0.12f

            if (cropW <= 0f) {
                cropW = wantW; cropCx = cxPx; cropCy = wantCy
            } else {
                cropW += (wantW - cropW) * CROP_SMOOTH
                cropCx += (cxPx - cropCx) * CROP_SMOOTH
                cropCy += (wantCy - cropCy) * CROP_SMOOTH
            }
        } else if (cropW > 0f && signerLostFrames > CROP_GRACE_FRAMES) {
            // Signer lost for long enough that this is not a blink. Open the
            // box back out and drift it toward the centre, so each frame sees
            // more of the scene than the last and pose gets a chance to
            // re-acquire. Once it reaches the full frame, drop back to the
            // letterbox entirely -- which is the widest view available and the
            // state the pipeline starts in.
            cropW *= CROP_RELAX
            cropCx += (upright.width / 2f - cropCx) * 0.2f
            cropCy += (upright.height / 2f - cropCy) * 0.2f
            if (cropW >= upright.width) {
                cropW = -1f
                // W level deliberately: logcat on this device drops the app's
                // Log.d and every I/flutter line, but app W and E survive
                // (CameraDevice-JV-1 warnings from this very process do). This
                // is the one marker that proves the recovery fired.
                Log.w("MediaPipe",
                    "crop relaxed to full frame after " +
                    "$signerLostFrames lost frames")
            }
        }

        var drew = false
        if (cropW > 0f) {
            val cw = cropW
            val ch = cw * CANVAS_H / CANVAS_W
            if (cw <= upright.width && ch <= upright.height) {
                // Guarded the same way: cw or ch can equal the frame size, so
                // the upper bound can legitimately be 0, and floating point
                // can make it very slightly negative.
                var l = cropCx - cw / 2f
                var t = cropCy - ch / 2f
                l = l.coerceIn(0f, (upright.width - cw).coerceAtLeast(0f))
                t = t.coerceIn(0f, (upright.height - ch).coerceAtLeast(0f))

                canvas.drawBitmap(
                    upright,
                    Rect(l.toInt(), t.toInt(),
                         (l + cw).toInt(), (t + ch).toInt()),
                    RectF(0f, 0f, CANVAS_W.toFloat(), CANVAS_H.toFloat()),
                    canvasPaint
                )
                // Keep canvas -> upright mapping in the same form the overlay
                // and hand-crop code already use: canvas = box + upright*scale.
                boxScale = CANVAS_W / cw
                boxLeft = -l * boxScale
                boxTop = -t * boxScale
                drew = true
            }
        }

        if (!drew) {
            boxScale = fitScale
            boxLeft = fitLeft
            boxTop = fitTop
            canvas.drawBitmap(
                upright, null,
                RectF(fitLeft, fitTop, fitLeft + fitW, fitTop + fitH),
                canvasPaint
            )
        }

        return canvasBmp
    }

    /**
     * Square crop centred on the hand, sized so the hand spans [LETTER_TARGET_SPAN]
     * of it, taken from the full-resolution upright frame and stretched to the
     * training canvas shape. Returns null when no usable crop exists.
     */
    /** Renders an already-known upright-space rectangle into the crop bitmap. */
    private fun cropFromRect(rect: Rect): Bitmap? {
        val upright = uprightBitmap ?: return null
        val left = rect.left.coerceIn(0, upright.width)
        val top = rect.top.coerceIn(0, upright.height)
        val right = rect.right.coerceIn(0, upright.width)
        val bottom = rect.bottom.coerceIn(0, upright.height)
        if (right - left < 8 || bottom - top < 8) return null

        var crop = cropBitmap
        if (crop == null) {
            crop = Bitmap.createBitmap(CANVAS_W, CANVAS_H, Bitmap.Config.ARGB_8888)
            cropBitmap = crop
        }
        val canvas = Canvas(crop)
        canvas.drawColor(Color.BLACK)
        canvas.drawBitmap(
            upright, Rect(left, top, right, bottom),
            RectF(0f, 0f, CANVAS_W.toFloat(), CANVAS_H.toFloat()), canvasPaint
        )
        return crop
    }

    /**
     * Maps landmarks found inside [rect] back to upright-frame pixels and returns
     * the crop rectangle for the next frame.
     */
    private fun rectFromCropResult(result: HandLandmarkerResult, rect: Rect): Rect? {
        val hands = result.landmarks()
        if (hands.isEmpty()) return null
        val w = rect.width().toFloat()
        val h = rect.height().toFloat()

        var minX = Float.MAX_VALUE; var minY = Float.MAX_VALUE
        var maxX = -Float.MAX_VALUE; var maxY = -Float.MAX_VALUE
        for (hand in hands) {
            for (lm in hand) {
                val ux = rect.left + lm.x() * w
                val uy = rect.top + lm.y() * h
                if (ux < minX) minX = ux
                if (uy < minY) minY = uy
                if (ux > maxX) maxX = ux
                if (uy > maxY) maxY = uy
            }
        }
        val extent = maxOf(maxX - minX, maxY - minY)
        if (extent < 4f) return null
        handExtentPx = extent
        val cx = (minX + maxX) / 2f
        val cy = (minY + maxY) / 2f
        val half = (extent / LETTER_TARGET_SPAN) / 2f
        return Rect(
            (cx - half).toInt(), (cy - half).toInt(),
            (cx + half).toInt(), (cy + half).toInt()
        )
    }

    private fun cropAroundHand(result: HandLandmarkerResult?): Bitmap? {
        val upright = uprightBitmap ?: return null
        val hands = result?.landmarks() ?: return null
        if (hands.isEmpty()) return null

        var minX = Float.MAX_VALUE
        var minY = Float.MAX_VALUE
        var maxX = -Float.MAX_VALUE
        var maxY = -Float.MAX_VALUE

        for (hand in hands) {
            for (lm in hand) {
                // Canvas-normalised -> canvas pixels -> upright pixels.
                val cx = lm.x() * CANVAS_W
                val cy = lm.y() * CANVAS_H
                val ux = (cx - boxLeft) / boxScale
                val uy = (cy - boxTop) / boxScale
                if (ux < minX) minX = ux
                if (uy < minY) minY = uy
                if (ux > maxX) maxX = ux
                if (uy > maxY) maxY = uy
            }
        }
        if (minX > maxX || minY > maxY) return null

        val cx = (minX + maxX) / 2f
        val cy = (minY + maxY) / 2f
        val extent = maxOf(maxX - minX, maxY - minY)
        if (extent < 4f) return null
        handExtentPx = extent

        val half = (extent / LETTER_TARGET_SPAN) / 2f
        val left = maxOf(0, (cx - half).toInt())
        val top = maxOf(0, (cy - half).toInt())
        val right = minOf(upright.width, (cx + half).toInt())
        val bottom = minOf(upright.height, (cy + half).toInt())
        if (right - left < 8 || bottom - top < 8) return null
        trackedRect = Rect(left, top, right, bottom)

        var crop = cropBitmap
        if (crop == null) {
            crop = Bitmap.createBitmap(CANVAS_W, CANVAS_H, Bitmap.Config.ARGB_8888)
            cropBitmap = crop
        }
        val canvas = Canvas(crop)
        canvas.drawColor(Color.BLACK)
        canvas.drawBitmap(
            upright,
            Rect(left, top, right, bottom),
            RectF(0f, 0f, CANVAS_W.toFloat(), CANVAS_H.toFloat()),
            canvasPaint
        )
        return crop
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Landmark extraction
    // ─────────────────────────────────────────────────────────────────────────

    private fun extractLandmarks(
        yBytes: ByteArray, uBytes: ByteArray?, vBytes: ByteArray?,
        width: Int, height: Int,
        yRowStride: Int, uvRowStride: Int, uvPixelStride: Int,
        rotation: Int, mirror: Boolean, lettersMode: Boolean,
        handBlock: String, debug: Boolean
    ): DoubleArray {
        val output = DoubleArray(228)
        frameCounter++

        val t0 = System.nanoTime()
        val pixels = yuvToArgb(
            yBytes, uBytes, vBytes, width, height,
            yRowStride, uvRowStride, uvPixelStride
        )
        val t1 = System.nanoTime()
        val frame = buildCanvas(pixels, width, height, rotation, mirror)
        val t2 = System.nanoTime()

        // Pose is launched NOW, concurrently with the hand work below, instead of
        // waiting its turn afterwards.
        //
        // The two landmarkers are separate instances that only read the frame, so
        // running them together is safe, and the cost becomes max(hand, pose)
        // rather than hand + pose. Measured sequentially: hand 72-171ms and pose
        // 95-176ms, so this is worth roughly a third of the frame budget.
        //
        // Frame rate is the binding constraint for ASL in particular: its
        // training sequences move 3.6x as much per timestep as FSL's, so at any
        // given capture rate ASL is undersampled that much worse, and fast signs
        // never accumulate enough distinct frames to classify.
        val poseFuture: java.util.concurrent.Future<FloatArray>? =
            if (!lettersMode) {
                poseExecutor.submit<FloatArray> { detectPose(frame) }
            } else {
                null
            }

        var handResult: HandLandmarkerResult? = null
        var cropped = false
        var tracked = false
        handExtentPx = 0f
        handFromTracker = false

        // Fast path: reuse the previous crop rectangle and run a single detection
        // on it, skipping the full-frame pass entirely.
        if (lettersMode) {
            val rect = trackedRect
            val fresh = rect != null &&
                (frameCounter - trackedFrame) < REACQUIRE_EVERY_N
            if (fresh) {
                val crop = cropFromRect(rect!!)
                if (crop != null) {
                    val r = handLandmarker?.detect(BitmapImageBuilder(crop).build())
                    if (r != null && r.landmarks().isNotEmpty()) {
                        handResult = r
                        cropped = true
                        tracked = true
                        handFromTracker = true
                        rectFromCropResult(r, rect)?.let { trackedRect = it }
                    }
                }
            }
        }

        val t3 = System.nanoTime()

        // Full-frame pass: always for words, and for letters only when tracking is
        // stale or lost.
        if (handResult == null) {
            handResult = handLandmarker?.detect(BitmapImageBuilder(frame).build())

            if (lettersMode) {
                trackedRect = null
                val crop = cropAroundHand(handResult)
                if (crop != null) {
                    val refined =
                        handLandmarker?.detect(BitmapImageBuilder(crop).build())
                    if (refined != null && refined.landmarks().isNotEmpty()) {
                        handResult = refined
                        cropped = true
                        trackedFrame = frameCounter
                        trackedRect?.let { r ->
                            rectFromCropResult(refined, r)?.let { trackedRect = it }
                        }
                    } else {
                        trackedRect = null
                    }
                }
            }
        }

        val t4 = System.nanoTime()

        // Layout must match training: 63 left hand + 63 right hand + 100 pose.
        val lh = FloatArray(63)
        val rh = FloatArray(63)
        var sawLeft = false
        var sawRight = false

        if (handResult != null) {
            val landmarksList = handResult.landmarks()
            val handednessList = handResult.handednesses()

            // The two blocks are not mirror images of one another: a hand in rh gets
            // wrist-normalised downstream, while a hand in lh stays in raw image
            // coordinates. The letters model learned both regimes separately, so the
            // same handshape can decode differently depending on which block it lands
            // in (measured on-device: 'A' reads as S/N in one, A at 100% in the other).
            //
            // MediaPipe's own Left/Right label flips between consecutive frames on the
            // same hand, which made letter output a coin flip. For single-handed
            // fingerspelling we pin the block instead of trusting the label.
            // "left"/"right" pin every hand to one block (single-handed
            // fingerspelling). "swap" keeps both blocks but inverts the label, for
            // two-handed signs: on this device MediaPipe reports the signing hand as
            // "Right" where the training pipeline needed it on the left, so an
            // uninverted mapping hands the LSTM a mirrored pair.
            val forced = handBlock == "left" || handBlock == "right"

            for (i in landmarksList.indices) {
                if (i >= handednessList.size) break
                val label = handednessList[i][0].categoryName()
                val hand = landmarksList[i]

                // Mirrors the extraction scripts: "Left" -> lh, "Right" -> rh.
                val toLeft = when {
                    handBlock == "left" -> true
                    handBlock == "right" -> false
                    handBlock == "swap" -> label == "Right"
                    else -> label == "Left"
                }
                if (label != "Left" && label != "Right" && !forced) continue
                val arr = if (toLeft) {
                    sawLeft = true; lh
                } else {
                    sawRight = true; rh
                }

                for (j in hand.indices) {
                    if (j * 3 + 2 >= arr.size) break
                    arr[j * 3] = hand[j].x()
                    arr[j * 3 + 1] = hand[j].y()
                    arr[j * 3 + 2] = hand[j].z()
                }

                // Pinned mode targets one block, so a second hand would overwrite
                // the first. Fingerspelling is single-handed anyway.
                if (forced) break
            }
        }

        // Pose is skipped entirely in letters mode. Over half the letter training
        // samples had an all-zero pose block anyway (cropped hand stills), and
        // zeroing it costs 0.4% accuracy (97.80% -> 97.42%) while removing the most
        // expensive model in the pipeline.
        var pose = FloatArray(100)
        var poseRan = false
        if (lettersMode) {
            // Letters never runs pose, so anything the crop tracker holds is
            // from an older words session and will not be refreshed. Letting
            // it persist is what allowed the shoulder measurement and the box
            // scale to drift apart.
            signerShoulder = -1f
            cropW = -1f
            signerLostFrames = 0
        }
        if (!lettersMode) {
            poseRan = true
            // Collect the result of the task started before the hand work. By now
            // it has usually finished already, so this costs nothing.
            pose = try {
                // Bounded, not an open-ended get(). An unbounded wait here
                // blocks the worker thread, so result.success() is never
                // posted, so the Dart await never completes, so _isProcessing
                // stays set and every later frame is dropped at the re-entry
                // guard — the camera keeps previewing and detects nothing.
                // A frame is ~80ms, so 1500ms only trips on a real stall, and
                // losing this frame's pose is far better than losing the app.
                poseFuture?.get(1500, java.util.concurrent.TimeUnit.MILLISECONDS)
                    ?: FloatArray(100)
            } catch (e: Exception) {
                Log.e("MediaPipe", "pose task failed: ${e.message}")
                FloatArray(100)
            }
        }

        // Remember where the signer is, so the NEXT frame can crop to them.
        // Landmarks 11 and 12 are the shoulders; visibility is index 3.
        if (poseRan) {
            val lsx = pose[11 * 4]; val lsy = pose[11 * 4 + 1]
            val rsx = pose[12 * 4]; val rsy = pose[12 * 4 + 1]
            if ((lsx != 0f || lsy != 0f) && (rsx != 0f || rsy != 0f)) {
                val w = kotlin.math.hypot((lsx - rsx).toDouble(),
                                          (lsy - rsy).toDouble()).toFloat()
                if (w > 0.01f) {
                    signerCx = (lsx + rsx) / 2f
                    signerCy = (lsy + rsy) / 2f
                    signerShoulder = w
                    signerLostFrames = 0
                }
            } else {
                // Lost the signer. Setting this alone does NOT relax the crop
                // -- it only stops it being updated -- so the counter above is
                // what actually drives it back open in buildCanvas.
                signerShoulder = -1f
                signerLostFrames++
            }
        }

        val t5 = System.nanoTime()

        for (i in 0 until 63) output[i] = lh[i].toDouble()
        for (i in 0 until 63) output[63 + i] = rh[i].toDouble()
        for (i in 0 until 100) output[126 + i] = pose[i].toDouble()
        // Diagnostics, past the 226 the model uses. Dart trims before the
        // tensor is built, so nothing downstream sees them.
        output[226] = handExtentPx.toDouble()
        output[227] = if (handFromTracker) 1.0 else 0.0

        // Logged on a prime-ish cadence so it does not always land on the same
        // phase of the pose cache cycle, which hid the real pose cost last time.
        if (debug && frameCounter % 7L == 0L) {
            fun ms(a: Long, b: Long) = (b - a) / 1_000_000.0
            Log.d(
                "MPPERF",
                "total=%.0fms | yuv=%.0f transform=%.0f track=%.0f full=%.0f pose=%.0f"
                    .format(
                        ms(t0, t5), ms(t0, t1), ms(t1, t2),
                        ms(t2, t3), ms(t3, t4), ms(t4, t5)
                    ) +
                    " | letters=$lettersMode tracked=$tracked poseRan=$poseRan"
            )
        }

        if (debug && frameCounter % 15L == 0L) {
            Log.d(
                "MediaPipe",
                "frame=$frameCounter src=${width}x$height rot=$rotation mirror=$mirror " +
                    "letters=$lettersMode cropped=$cropped left=$sawLeft right=$sawRight " +
                    "span=${"%.3f".format(handSpan(lh, rh))}"
            )
            // Full feature vector, so a captured frame can be replayed against the
            // model offline and compared with the training distribution directly.
            if (sawLeft || sawRight) {
                val sb = StringBuilder("VEC,")
                sb.append(if (lettersMode) "letters," else "words,")
                sb.append("$sawLeft,$sawRight,")
                for (i in 0 until 226) {
                    sb.append("%.5f".format(output[i]))
                    if (i < 225) sb.append(',')
                }
                Log.d("MPVEC", sb.toString())
            }
        }

        return output
    }

    /** Wrist-to-farthest-landmark distance, for comparing against training stats. */
    private fun handSpan(lh: FloatArray, rh: FloatArray): Float {
        var best = 0f
        for (blk in listOf(lh, rh)) {
            if (blk.all { it == 0f }) continue
            for (j in 1 until 21) {
                val dx = blk[j * 3] - blk[0]
                val dy = blk[j * 3 + 1] - blk[1]
                val d = kotlin.math.sqrt(dx * dx + dy * dy)
                if (d > best) best = d
            }
        }
        return best
    }

    private fun readVisibility(point: Any): Float {
        if (!visibilityResolved) {
            visibilityMethod = try {
                point.javaClass.methods.firstOrNull {
                    (it.name == "visibility" || it.name == "getVisibility") &&
                        it.parameterTypes.isEmpty()
                }
            } catch (e: Exception) {
                null
            }
            visibilityResolved = true
        }

        return try {
            when (val v = visibilityMethod?.invoke(point)) {
                is Number -> v.toFloat()
                is java.util.Optional<*> ->
                    (v.orElse(null) as? Number)?.toFloat() ?: 1f
                is String -> v.toFloatOrNull() ?: 1f
                else -> 1f
            }
        } catch (e: Exception) {
            1f
        }
    }

    private fun copyAssetToFile(context: Context, assetPath: String): File {
        val fileName = assetPath.substringAfterLast("/")
        val outFile = File(context.filesDir, fileName)
        if (!outFile.exists()) {
            val flutterAssetPath = "flutter_assets/$assetPath"
            try {
                context.assets.open(flutterAssetPath).use { input ->
                    FileOutputStream(outFile).use { output -> input.copyTo(output) }
                }
            } catch (e: Exception) {
                context.assets.open(assetPath).use { input ->
                    FileOutputStream(outFile).use { output -> input.copyTo(output) }
                }
            }
        }
        return outFile
    }

    override fun onDestroy() {
        executor.shutdown()
        poseExecutor.shutdown()
        handLandmarker?.close()
        poseLandmarker?.close()
        srcBitmap?.recycle()
        canvasBitmap?.recycle()
        uprightBitmap?.recycle()
        cropBitmap?.recycle()
        srcBitmap = null
        canvasBitmap = null
        uprightBitmap = null
        cropBitmap = null
        isInitialized = false
        super.onDestroy()
    }
}
