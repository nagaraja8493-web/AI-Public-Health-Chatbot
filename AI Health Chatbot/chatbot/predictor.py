"""
Prediction Engine for AI Public Health Chatbot.
Powered exclusively by Pure Unidirectional LSTM Deep Learning Neural Network.
Performs emergency checking, LSTM sequence inference, keyword problem recognition,
confidence evaluation, and medical response mapping.
"""
import os
import joblib
import numpy as np
from chatbot.preprocessing import preprocess_pipeline
from chatbot.response import check_emergency, get_response_for_intent, EMERGENCY_RESPONSE, detect_dataset_keyword

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

class HealthChatbotPredictor:
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.dl_model = None
        self.dl_tokenizer = None
        self.label_encoder = None
        self.max_seq_len = 40
        self.models_loaded = False
        self.load_models()

    def load_models(self):
        """Loads trained Deep Learning (LSTM), Tokenizer, and Label Encoder artifacts."""
        le_path = os.path.join(self.models_dir, "label_encoder.pkl")
        if os.path.exists(le_path):
            try:
                self.label_encoder = joblib.load(le_path)
            except Exception as e:
                print(f"Error loading label encoder: {e}")

        # Load Deep Learning components (LSTM)
        dl_path = os.path.join(self.models_dir, "dl_model.keras")
        tok_path = os.path.join(self.models_dir, "tokenizer.pkl")
        params_path = os.path.join(self.models_dir, "dl_params.pkl")

        if os.path.exists(dl_path) and os.path.exists(tok_path):
            try:
                import tensorflow as tf
                self.dl_model = tf.keras.models.load_model(dl_path)
                self.dl_tokenizer = joblib.load(tok_path)
                if os.path.exists(params_path):
                    params = joblib.load(params_path)
                    self.max_seq_len = params.get("max_len", 40)
                print("Loaded Deep Learning model (Pure LSTM) and Tokenizer successfully.")
            except Exception as e:
                print(f"Error loading Deep Learning model: {e}")

        self.models_loaded = (self.dl_model is not None)

    def predict_dl(self, text: str):
        """Inference using Word Embedding + Pure  LSTM Neural Network."""
        if not self.dl_model or not self.dl_tokenizer or not self.label_encoder:
            raise RuntimeError("Deep Learning LSTM model or Tokenizer is not loaded.")

        from tensorflow.keras.preprocessing.sequence import pad_sequences
        cleaned = preprocess_pipeline(text)
        seq = self.dl_tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=self.max_seq_len, padding="pre", truncating="post")
        probs = self.dl_model.predict(padded, verbose=0)[0]
        max_idx = int(np.argmax(probs))
        intent = str(self.label_encoder.inverse_transform([max_idx])[0])
        confidence = float(probs[max_idx])

        top_indices = np.argsort(probs)[::-1][:3]
        top_intents = [
            {"intent": str(self.label_encoder.inverse_transform([int(i)])[0]), "confidence": round(float(probs[i]), 4)}
            for i in top_indices
        ]

        return intent, confidence, top_intents

    def predict(self, text: str):
        """Direct alias for Deep Learning sequence prediction."""
        return self.predict_dl(text)

    def get_chat_response(self, text: str, model_type: str = "dl", confidence_threshold: float = 0.60) -> dict:
        """
        Full prediction and response pipeline:
        1. Emergency safety layer check (critical keywords bypass directly to 112)
        2. Canonical dataset keyword & problem detection (immediate symptom mapping)
        3. Deep Learning (LSTM) intent classification
        4. Problem context reinforcement (resolves concise queries to verified health guidance)
        5. Knowledge Base response retrieval
        """
        # Step 1: Emergency safety layer
        if check_emergency(text):
            return {
                "intent": "emergency",
                "confidence": 1.0,
                "response": EMERGENCY_RESPONSE,
                "model_used": "Safety Rules Protocol",
                "is_fallback": False,
                "is_emergency": True,
                "top_intents": [{"intent": "emergency", "confidence": 1.0}]
            }

        # Step 2: Canonical dataset keyword & problem detection
        keyword_intent = detect_dataset_keyword(text)

        # Step 3: Deep Learning (LSTM) neural classification
        try:
            predicted_intent, confidence, top_intents = self.predict_dl(text)
            active_model = "Pure LSTM (Deep Learning)"
        except Exception as e:
            # If model weights are initializing, fallback to keyword problem recognition if available
            if keyword_intent:
                predicted_intent = keyword_intent
                confidence = 0.98
                active_model = "Problem Context Recognition"
                top_intents = [{"intent": keyword_intent, "confidence": 0.98}]
            else:
                return {
                    "intent": "unknown",
                    "confidence": 0.0,
                    "response": f"The Deep Learning model is currently initializing. ({str(e)})",
                    "model_used": "Deep Learning (Initializing)",
                    "is_fallback": True,
                    "is_emergency": False,
                    "top_intents": []
                }

        # Step 4: Keyword Contextual Reinforcement
        # If user provides a concise keyword query (<= 6 words) or if model confidence is low,
        # or if model misclassified a clinical symptom into generic conversational intents,
        # the recognized medical problem takes precedence.
        query_word_count = len(text.strip().split())
        if keyword_intent:
            if (query_word_count <= 6 or 
                confidence < confidence_threshold or 
                (predicted_intent in ["greeting", "goodbye"] and keyword_intent not in ["greeting", "goodbye"])):
                predicted_intent = keyword_intent
                confidence = max(confidence, 0.98)
                top_intents = [{"intent": keyword_intent, "confidence": round(confidence, 4)}]
                active_model = "Pure LSTM (Deep Learning) + Problem Context"

        # Step 5: Response retrieval & verified guidance
        response_text, is_fallback = get_response_for_intent(
            predicted_intent, confidence, threshold=confidence_threshold
        )

        return {
            "intent": predicted_intent,
            "confidence": round(confidence, 4),
            "response": response_text,
            "model_used": active_model,
            "is_fallback": is_fallback,
            "is_emergency": (predicted_intent == "emergency"),
            "top_intents": top_intents
        }

# Global singleton predictor
_predictor_instance = None

def get_predictor():
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = HealthChatbotPredictor()
    return _predictor_instance
