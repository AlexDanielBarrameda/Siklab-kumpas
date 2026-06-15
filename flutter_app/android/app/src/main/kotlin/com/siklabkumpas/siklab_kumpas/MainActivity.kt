package com.siklabkumpas.siklab_kumpas

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {

    private val CHANNEL = "com.siklabkumpas/inference"

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(
            flutterEngine.dartExecutor.binaryMessenger,
            CHANNEL
        ).setMethodCallHandler { call, result ->
            when (call.method) {
                "processFrame" -> {
                    try {
                        // Return 226 zeros as placeholder landmarks
                        // MediaPipe integration will replace this
                        val landmarks = FloatArray(226) { 0f }
                        result.success(landmarks.toList())
                    } catch (e: Exception) {
                        result.error("PROCESS_ERROR", e.message, null)
                    }
                }
                else -> result.notImplemented()
            }
        }
    }
}