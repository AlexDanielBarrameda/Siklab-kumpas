import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU for conversion

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

MODEL_DIR = "../models"
TFLITE_DIR = "../../assets/models"
os.makedirs(TFLITE_DIR, exist_ok=True)

FSL105_TEST = "../data/processed/fsl_105/X_test.npy"
LETTERS_TEST = "../data/processed/fsl_letters/X_test.npy"


def rebuild_fsl105_on_cpu(gpu_keras_path, output_keras_path):
    """
    Loads a GPU-trained LSTM model, rebuilds identical architecture
    on CPU, copies weights, and saves — produces TFLite-compatible ops.
    """
    print(f"Loading GPU model from {gpu_keras_path}...")
    gpu_model = tf.keras.models.load_model(gpu_keras_path)

    print("Rebuilding architecture on CPU...")
    cpu_model = Sequential(
        [
            LSTM(128, return_sequences=True, input_shape=(30, 226)),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(105, activation="softmax"),
        ]
    )

    cpu_model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Build the model by passing a dummy input
    dummy = tf.zeros((1, 30, 226))
    cpu_model(dummy)

    print("Copying weights from GPU model to CPU model...")
    for cpu_layer, gpu_layer in zip(cpu_model.layers, gpu_model.layers):
        if gpu_layer.get_weights():
            try:
                cpu_layer.set_weights(gpu_layer.get_weights())
                print(f"  Copied weights: {gpu_layer.name}")
            except Exception as e:
                print(f"  Skipped {gpu_layer.name}: {e}")

    cpu_model.save(output_keras_path)
    print(f"CPU-compatible model saved to {output_keras_path}")
    return cpu_model


def convert_to_tflite(keras_path, tflite_name, representative_data=None):
    print(f"\nConverting {keras_path} to TFLite...")

    model = tf.keras.models.load_model(keras_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Required for LSTM ops
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False

    if representative_data is not None:

        def representative_dataset():
            for i in range(min(200, len(representative_data))):
                yield [representative_data[i : i + 1].astype(np.float32)]

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    output_path = os.path.join(TFLITE_DIR, tflite_name)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved: {output_path} ({size_mb:.2f} MB)")
    return output_path


def test_tflite(tflite_path, X_test, y_test):
    print(f"\nTesting {tflite_path}...")

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct = 0
    top3_correct = 0
    total = len(X_test)

    for i in range(total):
        input_data = X_test[i : i + 1].astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])[0]

        pred = np.argmax(output)
        top3 = np.argsort(output)[-3:]

        if pred == y_test[i]:
            correct += 1
        if y_test[i] in top3:
            top3_correct += 1

    print(f"Top-1 Accuracy: {correct/total:.4f} ({correct}/{total})")
    print(f"Top-3 Accuracy: {top3_correct/total:.4f} ({top3_correct}/{total})")


def main():
    X_test_105 = np.load(FSL105_TEST)
    y_test_105 = np.load("../data/processed/fsl_105/y_test.npy")
    X_test_letters = np.load(LETTERS_TEST)
    y_test_letters = np.load("../data/processed/fsl_letters/y_test.npy")

    # Step 1: Rebuild FSL-105 GPU model as CPU-compatible
    cpu_keras_path = os.path.join(MODEL_DIR, "fsl105_cpu_compatible.keras")
    rebuild_fsl105_on_cpu(
        gpu_keras_path=os.path.join(MODEL_DIR, "fsl105_best.keras"),
        output_keras_path=cpu_keras_path,
    )

    # Step 2: Verify accuracy is preserved after weight copy
    print("\nVerifying CPU model accuracy...")
    cpu_model = tf.keras.models.load_model(cpu_keras_path)
    loss, acc = cpu_model.evaluate(X_test_105, y_test_105, verbose=0)
    print(f"CPU model accuracy: {acc:.4f} (should match ~0.7469)")

    # Step 3: Convert both models to TFLite
    tflite_105 = convert_to_tflite(
        keras_path=cpu_keras_path,
        tflite_name="fsl105.tflite",
        representative_data=X_test_105,
    )

    tflite_letters = convert_to_tflite(
        keras_path=os.path.join(MODEL_DIR, "fsl_letters_best.keras"),
        tflite_name="fsl_letters.tflite",
        representative_data=X_test_letters,
    )

    # Step 4: Verify TFLite accuracy
    test_tflite(tflite_105, X_test_105, y_test_105)
    test_tflite(tflite_letters, X_test_letters, y_test_letters)

    print("\nAll done!")
    print(f"TFLite models saved to: {TFLITE_DIR}")


if __name__ == "__main__":
    main()
