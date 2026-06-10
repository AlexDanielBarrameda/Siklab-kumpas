import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
    Bidirectional,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Paths
DATA_DIR = "../data/processed/fsl_105"
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparameters
SEQUENCE_LENGTH = 30
LANDMARK_DIM = 226
NUM_CLASSES = 105
BATCH_SIZE = 32
EPOCHS = 200


def load_data():
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    print(f"Loaded X: {X.shape}, y: {y.shape}")
    return X, y


def augment_sequence(X, y, factor=3):
    """
    Augments landmark sequences by adding:
    - Gaussian noise
    - Temporal jitter (slight time stretching)
    - Mirror (flip left/right hands)
    """
    print(f"Augmenting data by factor {factor}...")
    X_aug = [X]
    y_aug = [y]

    for _ in range(factor - 1):
        # Gaussian noise
        noise = np.random.normal(0, 0.01, X.shape)
        X_noisy = X + noise
        X_aug.append(X_noisy)
        y_aug.append(y)

        # Mirror: swap left hand (0:63) and right hand (63:126)
        X_mirror = X.copy()
        X_mirror[:, :, 0:63] = X[:, :, 63:126]
        X_mirror[:, :, 63:126] = X[:, :, 0:63]
        X_aug.append(X_mirror)
        y_aug.append(y)

    X_out = np.concatenate(X_aug, axis=0)
    y_out = np.concatenate(y_aug, axis=0)

    # Shuffle
    idx = np.random.permutation(len(X_out))
    print(f"Augmented dataset size: {X_out.shape}")
    return X_out[idx], y_out[idx]


def build_model():
    model = Sequential(
        [
            # Bidirectional LSTM — learns patterns in both directions
            Bidirectional(
                LSTM(128, return_sequences=True),
                input_shape=(SEQUENCE_LENGTH, LANDMARK_DIM),
            ),
            BatchNormalization(),
            Dropout(0.3),
            Bidirectional(LSTM(64, return_sequences=False)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


def plot_history(history, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"], label="Train")
    ax1.plot(history.history["val_accuracy"], label="Val")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history.history["loss"], label="Train")
    ax2.plot(history.history["val_loss"], label="Val")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, filename))
    print(f"Plot saved: {filename}")


def train():
    # Load data
    X, y = load_data()

    # Split BEFORE augmentation to prevent data leakage
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(
        f"Before augmentation — Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )

    # Augment only training data
    X_train, y_train = augment_sequence(X_train, y_train, factor=3)

    print(f"After augmentation  — Train: {X_train.shape}")

    # Build model
    model = build_model()

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "fsl105_v2_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # Train
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss:     {test_loss:.4f}")

    # Save final model
    model.save(os.path.join(MODEL_DIR, "fsl105_v2_final.keras"))
    print("Model saved.")

    # Plot
    plot_history(history, "fsl105_v2_training_history.png")

    # Save test set
    np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test)
    print("Test set saved.")


if __name__ == "__main__":
    train()
