package com.siklabkumpas.siklab_kumpas

import android.content.Context
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

    // Cached reflective accessor for pose visibility (MediaPipe 0.10.0 exposes it
    // inconsistently across builds). Looked up once instead of 25x per frame.
    private var visibilityMethod: Method? = null
    private var visibilityResolved = false

    private var frameCounter = 0L

    /**
     * Pose is the most expensive model in the pipeline and the body moves far more
     * slowly than the hands, so it runs every Nth frame and the previous result is
     * reused in between. The word buffer is resampled onto a fixed time window
     * anyway, so slightly stale pose costs far less than the frames we'd lose.
     */
    // Measured at 116-176ms per run on this device, so it is worth amortising
    // further. The word buffer is resampled onto a 1s window regardless.
    private val POSE_EVERY_N = 3
    private var cachedPose = FloatArray(100)
    private var cachedPoseFrame = -1L

    /**
     * Letters mode used to run the hand landmarker twice per frame — once on the
     * full canvas to find the hand, once on the crop. Measured on device that was
     * ~165ms + ~130ms of a ~350ms frame.
     *
     * The hand barely moves between frames at 3 fps, so we keep the previous crop
     * rectangle and go straight to the second pass, re-acquiring on the full frame
     * only when tracking fails or periodically to stop the box drifting.
     */
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
                        .setRunningMode(RunningMode.IMAGE)
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

        boxScale = minOf(
            CANVAS_W.toFloat() / upright.width,
            CANVAS_H.toFloat() / upright.height
        )
        val dw = upright.width * boxScale
        val dh = upright.height * boxScale
        boxLeft = (CANVAS_W - dw) / 2f
        boxTop = (CANVAS_H - dh) / 2f
        canvas.drawBitmap(
            upright, null,
            RectF(boxLeft, boxTop, boxLeft + dw, boxTop + dh),
            canvasPaint
        )

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
        val output = DoubleArray(226)
        frameCounter++

        val t0 = System.nanoTime()
        val pixels = yuvToArgb(
            yBytes, uBytes, vBytes, width, height,
            yRowStride, uvRowStride, uvPixelStride
        )
        val t1 = System.nanoTime()
        val frame = buildCanvas(pixels, width, height, rotation, mirror)
        val t2 = System.nanoTime()

        var handResult: HandLandmarkerResult? = null
        var cropped = false
        var tracked = false

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
        if (!lettersMode) {
            val stale = cachedPoseFrame < 0 ||
                (frameCounter - cachedPoseFrame) >= POSE_EVERY_N
            if (stale) {
                poseRan = true
                val poseResult =
                    poseLandmarker?.detect(BitmapImageBuilder(frame).build())
                if (poseResult != null) {
                    val poseLandmarksList = poseResult.landmarks()
                    if (poseLandmarksList.isNotEmpty()) {
                        val points = poseLandmarksList[0]
                        val count = minOf(25, points.size)
                        for (j in 0 until count) {
                            if (j * 4 + 3 >= pose.size) break
                            pose[j * 4] = points[j].x()
                            pose[j * 4 + 1] = points[j].y()
                            pose[j * 4 + 2] = points[j].z()
                            val vis = readVisibility(points[j])
                            pose[j * 4 + 3] = if (vis.isNaN()) 1f else vis
                        }
                    }
                }
                cachedPose = pose.copyOf()
                cachedPoseFrame = frameCounter
            } else {
                pose = cachedPose.copyOf()
            }
        }

        val t5 = System.nanoTime()

        for (i in 0 until 63) output[i] = lh[i].toDouble()
        for (i in 0 until 63) output[63 + i] = rh[i].toDouble()
        for (i in 0 until 100) output[126 + i] = pose[i].toDouble()

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
