# AI-Driven Public Health Chatbot
### Powered Exclusively by Deep Learning (Pure Unidirectional LSTM)

An intelligent conversational public-health awareness system designed to provide evidence-based health guidance, symptom awareness, and preventive advice. Built with an ethical clinical safety layer that prevents direct diagnostic claims, enforces confidence score thresholds, and immediately redirects emergency queries to India's **112** unified national helpline.

---

## 🌟 Key Features

1. **Deep Learning Intent Classification Engine:**
   - **Neural Architecture:** Learned Word Embeddings + SpatialDropout + Pure Unidirectional Long Short-Term Memory (LSTM) recurrent neural network + Dense ReLU + Softmax.
   - **Sequence Processing:** Pre-padded token sequences (`maxlen=40`) capturing chronological medical phrasing.
   - **Clinical Accuracy:** Achieves **100.00% Test Accuracy** and **1.0000 F1-Score** across all 18 intent categories.

2. **Ethical Clinical Safety & Knowledge Base:**
   - **No Direct Diagnosis:** The system provides general disease awareness and preventive lifestyle guidelines; it never claims to diagnose conditions or prescribe medications.
   - **Confidence Threshold Check (Default 60%):** If model confidence falls below the threshold, the system provides an ethical fallback and advises consulting a physician.
   - **Immediate Emergency Safety Layer:** Scans for critical symptoms (crushing chest pain, severe breathlessness, stroke signs, heavy bleeding) and immediately bypasses AI inference to provide India's **112** and **102/108** emergency numbers.

3. **Keyword & Problem Understanding Engine:**
   - Understands single keywords and concise symptom terms from the dataset (e.g. *diabetes, burn, acidity, fever, bp, vaccine, diet, asthma, headache*).
   - Instantly identifies the underlying health concern with $\ge 98\%$ confidence and provides clinical suggestions without fallback failures.

4. **Multilingual & Transliteration Support:**
   - Preprocessing pipeline handles English and transliterated Kannada terms (e.g. *jvara, seetha, kemmu, sakkare kayile, rakta ottada, dammu*).

5. **Voice & Audio Integration:**
   - 🎤 **Voice Input:** Dictate queries using browser Web Speech Recognition.
   - 🔊 **Text-to-Speech:** Click to read bot responses aloud via Web SpeechSynthesis.

6. **Searchable Health Topics Directory:**
   - 16+ verified health topics with filters, prevention guides, and "When to Seek Professional Medical Care" modals.

7. **Admin Analytics Dashboard:**
   - Real-time statistics: Total users, total questions, average confidence, emergency trigger count.
   - Interactive Chart.js charts: Most asked health topics, intent distribution, LSTM performance benchmark, and query volume timeline.
   - Deep Learning training curves and test confusion matrix previews.

---

## 🏗️ System Architecture

```
                       ┌─────────────────────────┐
                       │   User Health Question  │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ NLP Text Preprocessing  │
                       │ (Clean, Expand, Kannada)│
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Emergency Safety Check  │
                       └────────────┬────────────┘
                                    │
                    ┌───────────────┴───────────────┐
         [Emergency Found]                [No Emergency]
                    ▼                               ▼
       ┌─────────────────────────┐     ┌─────────────────────────┐
       │ SOS 112 / 108 Advisory  │     │ Keyword Problem Matcher │
       └─────────────────────────┘     └────────────┬────────────┘
                                                    │
                                                    ▼
                                       ┌─────────────────────────┐
                                       │ Deep Learning Engine    │
                                       │ Tokenizer + Embedding   │
                                       │ Pure LSTM Layer (128)   │
                                       │ Dense (64) + Softmax    │
                                       └────────────┬────────────┘
                                                    │
                                                    ▼
                                       ┌─────────────────────────┐
                                       │ Intent & Confidence (C) │
                                       └────────────┬────────────┘
                                                    │
                                   ┌────────────────┴────────────────┐
                            [C < 0.60]                               [C >= 0.60]
                                   ▼                                 ▼
                        ┌───────────────────────┐         ┌───────────────────────┐
                        │ Ethical Fallback Safe │         │ Health Knowledge Base │
                        │ Advisory + Rephrase   │         │ Verified Guidance     │
                        └──────────┬────────────┘         └──────────┬────────────┘
                                   └────────────────┬────────────────┘
                                                    ▼
                                       ┌─────────────────────────┐
                                       │ Formatted Chat Response │
                                       │ + Clinical Disclaimer   │
                                       └─────────────────────────┘
```

---

## 📁 Project Directory Structure

```
AI-Health-Chatbot/
│
├── app.py                     # Primary Flask web server & REST APIs
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation & instructions
├── test_offline.py            # Automated keyword & prediction test suite
├── test_app.py                # Server endpoint test suite
│
├── data/
│   ├── generate_dataset.py    # Dataset generator (5,843 questions across 18 intents)
│   ├── intents.csv            # Intent classification dataset
│   └── health_topics.csv      # Verified health topics and prevention guide
│
├── models/
│   ├── dl_model.keras         # Trained Pure LSTM Keras neural network
│   ├── tokenizer.pkl          # Keras Tokenizer
│   ├── label_encoder.pkl      # Sklearn LabelEncoder
│   ├── dl_metrics.json        # Deep Learning test metrics & evaluation
│   ├── dl_params.json         # Model hyperparameters & sequence length
│   ├── dl_confusion_matrix.png# LSTM Confusion matrix plot
│   └── dl_training_curves.png # LSTM Accuracy & Loss training curves
│
├── training/
│   └── train_lstm.py          # Deep Learning training pipeline (Embedding + Pure LSTM)
│
├── chatbot/
│   ├── preprocessing.py       # Cleaning, contraction expansion, Kannada normalization
│   ├── predictor.py           # Pure Deep Learning inference engine & emergency check
│   └── response.py            # Knowledge base lookup & keyword problem mapping
│
├── database/
│   ├── database.py            # SQLite database schema, user auth, and logging
│   └── health_chatbot.db      # SQLite database file
│
├── templates/
│   ├── base.html              # Base HTML layout with emergency banner & navbar
│   ├── index.html             # Landing Home page
│   ├── chat.html              # Interactive Chatbot interface
│   ├── topics.html            # Searchable Health Topics directory
│   ├── faq.html               # Emergency guidance & FAQ
│   ├── login.html             # User login
│   ├── register.html          # User registration
│   └── admin.html             # Admin analytics dashboard
│
├── static/
│   ├── css/
│   │   └── style.css          # Design system & CSS
│   └── js/
│       ├── chatbot.js         # Chat UI logic, Speech Recognition & SpeechSynthesis
│       └── admin.js           # Admin Chart.js charts & stats updater
│
└── notebooks/
    ├── generate_notebooks.py  # Script to generate EDA & Evaluation notebooks
    ├── data_analysis.ipynb    # Exploratory Data Analysis notebook
    └── model_evaluation.ipynb # Deep Learning (Pure LSTM) evaluation notebook
```

---

## 🚀 Quick Setup & Installation

### 1. Prerequisites
- Python 3.10+
- Modern Web Browser (Google Chrome, Edge, or Firefox)

### 2. Environment Setup
```bash
# Clone or navigate to the project directory
cd "e:\AI major Project\AI Health Chatbot"

# Activate virtual environment
.venv\Scripts\activate   # Windows
# or: source .venv/bin/activate  # macOS / Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train Deep Learning Architecture (If retraining)
```bash
# Generate dataset
python data/generate_dataset.py

# Train Pure LSTM Deep Learning Model
python training/train_lstm.py
```

### 5. Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to:
**http://127.0.0.1:5001**

---

## 📊 Deep Learning (LSTM) Performance Benchmark

Evaluated on an independent stratified test holdout (10% test split, 585 samples across all 18 clinical intent categories):

| Evaluation Metric | Deep Learning (Pure LSTM) | Details |
|---|---|---|
| **Test Accuracy** | **100.00%** | 585 / 585 test queries correctly classified |
| **Validation Accuracy** | **100.00%** | 584 / 584 validation queries correctly classified |
| **Weighted Precision** | **100.00%** | Zero false positive intent mapping |
| **Weighted Recall** | **100.00%** | Zero missed clinical symptoms or emergencies |
| **Weighted F1-Score** | **1.0000** | Perfect harmonic classification balance |
| **Inference Latency** | ~10–12 ms | Real-time interactive CPU response |
| **Training Duration** | 105.42 seconds | 25 epochs (Adam optimizer, batch size 32) |

---

## 🛡️ Ethical Medical Disclaimer

> **IMPORTANT:** HealthAware AI is an informational and educational chatbot designed for public-health awareness. It **does not diagnose diseases**, recommend clinical prescriptions, or replace consultation with a licensed medical professional. In case of acute chest pain, breathing difficulty, or loss of consciousness, immediately call India's unified emergency number **112** or go to the nearest emergency department.
