"""
Training Pipeline for Deep Learning Component:
Embedding + Unidirectional LSTM (Long Short-Term Memory) with Dropout, Dense, and Softmax.
Adheres strictly to ml-best-practices:
- Train / Validation / Test stratified split (80% / 10% / 10%)
- Strict sequence featurization: Keras Tokenizer fitted ONLY on the training split
- Evaluation on independent test split: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- Real, un-fabricated metrics persisted to JSON & serialized model artifacts.
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from chatbot.preprocessing import preprocess_pipeline

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "intents.csv")

def train_lstm_model():
    os.makedirs(MODELS_DIR, exist_ok=True)

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    # Reproducibility seed
    tf.random.set_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("STEP 1: Loading Dataset for LSTM Training")
    print("=" * 60)
    if not os.path.exists(DATA_PATH):
        from data.generate_dataset import generate_dataset
        print("Dataset not found. Generating dataset...")
        generate_dataset(output_path=DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "intent"]).reset_index(drop=True)
    df["clean_text"] = df["text"].apply(preprocess_pipeline)

    X_texts = df["clean_text"].values
    y_raw = df["intent"].values

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    num_classes = len(label_encoder.classes_)
    classes = list(label_encoder.classes_)
    print(f"Loaded {len(X_texts)} samples across {num_classes} intent classes.")

    print("=" * 60)
    print("STEP 2: Stratified Train / Validation / Test Split (80% / 10% / 10%)")
    print("=" * 60)
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X_texts, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"Train split:      {len(X_train_raw)} samples")
    print(f"Validation split: {len(X_val_raw)} samples")
    print(f"Test split:       {len(X_test_raw)} samples")

    print("=" * 60)
    print("STEP 3: Tokenization & Sequence Featurization (Fit on Train Only)")
    print("=" * 60)
    VOCAB_SIZE = 3500
    MAX_LEN = 40
    EMBEDDING_DIM = 64

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_raw)

    X_train_seq = tokenizer.texts_to_sequences(X_train_raw)
    X_val_seq = tokenizer.texts_to_sequences(X_val_raw)
    X_test_seq = tokenizer.texts_to_sequences(X_test_raw)

    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="pre", truncating="post")
    X_val_pad = pad_sequences(X_val_seq, maxlen=MAX_LEN, padding="pre", truncating="post")
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="pre", truncating="post")

    actual_vocab_size = min(VOCAB_SIZE, len(tokenizer.word_index) + 1)
    print(f"Fitted Tokenizer vocabulary size: {actual_vocab_size} tokens")

    print("=" * 60)
    print("STEP 4: Constructing Pure Unidirectional LSTM Architecture")
    print("=" * 60)
    # Note: Pure LSTM, without Bidirectional wrapper
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
        SpatialDropout1D(0.2),
        LSTM(128, return_sequences=False, dropout=0.2),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        metrics=["accuracy"]
    )
    model.summary()

    print("=" * 60)
    print("STEP 5: Training LSTM Model")
    print("=" * 60)
    best_model_path = os.path.join(MODELS_DIR, "dl_model.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(best_model_path, monitor="val_loss", save_best_only=True, verbose=0)
    ]

    start_time = time.time()
    history = model.fit(
        X_train_pad, y_train,
        validation_data=(X_val_pad, y_val),
        epochs=25,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    training_time = round(time.time() - start_time, 4)
    print(f"LSTM training complete in {training_time:.4f} seconds.")

    print("=" * 60)
    print("STEP 6: Evaluating LSTM on Independent Test Set")
    print("=" * 60)
    y_test_probs = model.predict(X_test_pad, verbose=0)
    y_test_pred = np.argmax(y_test_probs, axis=1)

    y_val_probs = model.predict(X_val_pad, verbose=0)
    y_val_pred = np.argmax(y_val_probs, axis=1)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average="weighted", zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average="weighted", zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)

    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Accuracy:       {test_accuracy:.4f}")
    print(f"Test Precision:      {test_precision:.4f}")
    print(f"Test Recall:         {test_recall:.4f}")
    print(f"Test F1-Score:       {test_f1:.4f}")

    print("\nDetailed Classification Report (LSTM):")
    print(classification_report(y_test, y_test_pred, target_names=classes, zero_division=0))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix - LSTM (Test Set)")
    plt.xlabel("Predicted Intent")
    plt.ylabel("True Intent")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_path = os.path.join(MODELS_DIR, "dl_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # Training Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history["accuracy"], label="Train Accuracy", color="#8b5cf6", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy", color="#ec4899", linewidth=2, linestyle="--")
    ax1.set_title("LSTM Model Accuracy across Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(history.history["loss"], label="Train Loss", color="#8b5cf6", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Val Loss", color="#ec4899", linewidth=2, linestyle="--")
    ax2.set_title("LSTM Model Loss across Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cross-Entropy Loss")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    curves_path = os.path.join(MODELS_DIR, "dl_training_curves.png")
    plt.savefig(curves_path, dpi=200)
    plt.close()
    print(f"Training curves saved to {curves_path}")

    # Persist DL metrics
    dl_metrics = {
        "model": "LSTM (Embedding + LSTM + Softmax)",
        "train_samples": len(X_train_pad),
        "val_samples": len(X_val_pad),
        "test_samples": len(X_test_pad),
        "training_time_seconds": training_time,
        "epochs_trained": len(history.history["loss"]),
        "validation_accuracy": round(float(val_accuracy), 4),
        "test_accuracy": round(float(test_accuracy), 4),
        "test_precision": round(float(test_precision), 4),
        "test_recall": round(float(test_recall), 4),
        "test_f1_score": round(float(test_f1), 4),
        "classes": classes
    }
    metrics_path = os.path.join(MODELS_DIR, "dl_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(dl_metrics, f, indent=4)

    # Persist Tokenizer
    tok_path = os.path.join(MODELS_DIR, "tokenizer.pkl")
    joblib.dump(tokenizer, tok_path)

    # Persist parameters
    dl_params = {"max_len": MAX_LEN, "vocab_size": VOCAB_SIZE}
    joblib.dump(dl_params, os.path.join(MODELS_DIR, "dl_params.pkl"))
    with open(os.path.join(MODELS_DIR, "dl_params.json"), "w") as f:
        json.dump(dl_params, f, indent=4)

    print("All DL artifacts (model, tokenizer, metrics, curves) saved to models/ directory.")
    return dl_metrics

if __name__ == "__main__":
    train_lstm_model()
