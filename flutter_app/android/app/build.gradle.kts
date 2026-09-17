import java.io.FileInputStream
import java.util.Properties

plugins {
    id("com.android.application")
    id("kotlin-android")
    id("dev.flutter.flutter-gradle-plugin")
}

// Release signing lives outside the repository. key.properties and the .jks
// are both gitignored; a checkout without them still builds, signed with the
// debug key, so a panel member building from source is not blocked.
val keystorePropertiesFile = rootProject.file("key.properties")
val keystoreProperties = Properties().apply {
    if (keystorePropertiesFile.exists()) load(FileInputStream(keystorePropertiesFile))
}
val hasReleaseKey = keystorePropertiesFile.exists()

android {
    namespace = "com.siklabkumpas.siklab_kumpas"
    compileSdk = 36
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_17.toString()
    }

    defaultConfig {
        applicationId = "com.siklabkumpas.siklab_kumpas"
        minSdk = 24
        targetSdk = 35
        versionCode = flutter.versionCode
        versionName = flutter.versionName

        // Required for TFLite Select TF Ops
        ndk {
            abiFilters += listOf("armeabi-v7a", "arm64-v8a", "x86_64")
        }
        multiDexEnabled = true
    }

    signingConfigs {
        create("release") {
            if (hasReleaseKey) {
                keyAlias = keystoreProperties["keyAlias"] as String
                keyPassword = keystoreProperties["keyPassword"] as String
                storeFile = file(keystoreProperties["storeFile"] as String)
                storePassword = keystoreProperties["storePassword"] as String
            }
        }
    }

    buildTypes {
        release {
            // Deliberately NOT shrunk.
            //
            // Flutter turns R8 on for release by default, and this app has no
            // keep rules of its own. Three of its components — TFLite,
            // MediaPipe and Vosk — are native-backed and reached through JNI
            // and reflection, which is exactly what R8 strips when nothing
            // tells it not to. The failure is silent: the APK builds, installs
            // and opens, and a model then fails to load in front of the panel.
            //
            // The APK carries 66 MB of model assets. Shrinking the Dex saves a
            // megabyte or two against that. Not a trade worth the risk.
            isMinifyEnabled = false
            isShrinkResources = false
            signingConfig = if (hasReleaseKey)
                signingConfigs.getByName("release")
            else
                signingConfigs.getByName("debug")
        }
    }

    // Prevent conflicts with TFLite packaging
    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
            excludes += "META-INF/LICENSE.md"
            excludes += "META-INF/LICENSE-notice.md"
        }
    }

    aaptOptions {
    noCompress += "tflite"
    }
}

dependencies {
    implementation("com.google.mediapipe:tasks-vision:0.10.0")
}

flutter {
    source = "../.."
}