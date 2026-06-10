import os
import numpy as np
import tensorflow as tf

# Paths
MODEL_DIR = "../models"
TFLITE_DIR = "../../assets/models"
os.makedirs(TFLITE_DIR, exist_ok=True)

# Test data for quantization calibration
FSL105_TEST = "../data/processed/fsl_105/X_test.npy"
FSL_LETTERS_TEST = "../data/processed/fsl_letters/X_test.npy"


def convert_model(keras_path, tflite_name, representative_data=None):
    print(f"\nConverting {keras_path}...")

    model = tf.keras.models.load_model(keras_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Optimization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Representative dataset for full integer quantization
    if representative_data is not None:
        def representative_dataset():
            for i in range(min(200, len(representative_data))):
                sample = representative_data[i:i+1].astype(np.float32)
                yield [sample]

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    # Save
    output_path = os.path.join(TFLITE_DIR, tflite_name)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    # Report size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved: {output_path}")
    print(f"Size:  {size_mb:.2f} MB")
    return output_path


def test_tflite(tflite_path, X_test, y_test, top_k=1):
    """Run inference on test set to verify TFLite model accuracy."""
    print(f"\nTesting {tflite_path}...")

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct = 0
    top3_correct = 0
    total = len(X_test)

    for i in range(total):
        input_data = X_test[i:i+1].astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        pred = np.argmax(output)
        top3 = np.argsort(output)[-3:]

        if pred == y_test[i]:
            correct += 1
        if y_test[i] in top3:
            top3_correct += 1

    print(f"Top-1 Accuracy: {correct/total:.4f} ({correct}/{total})")
    print(f"Top-3 Accuracy: {top3_correct/total:.4f} ({top3_correct}/{total})")


def main():
    # Load test data
    X_test_105     = np.load(FSL105_TEST)
    y_test_105     = np.load("../data/processed/fsl_105/y_test.npy")
    X_test_letters = np.load(FSL_LETTERS_TEST)
    y_test_letters = np.load("../data/processed/fsl_letters/y_test.npy")

    # Convert FSL-105 model
    tflite_105 = convert_model(
        keras_path=os.path.join(MODEL_DIR, "fsl105_best.keras"),
        tflite_name="fsl105.tflite",
        representative_data=X_test_105
    )

    # Convert FSL Letters model
    tflite_letters = convert_model(
        keras_path=os.path.join(MODEL_DIR, "fsl_letters_best.keras"),
        tflite_name="fsl_letters.tflite",
        representative_data=X_test_letters
    )

    # Verify accuracy is preserved after conversion
    test_tflite(tflite_105, X_test_105, y_test_105)
    test_tflite(tflite_letters, X_test_letters, y_test_letters)

    print("\nAll models converted successfully!")
    print(f"TFLite models saved to: {TFLITE_DIR}")


if __name__ == "__main__":
    main()