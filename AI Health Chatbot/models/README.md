# Models Directory Guide & Deep Learning Architecture

This directory houses the trained Deep Learning artifacts for the **HealthAware AI Public Health Chatbot**.
The chatbot is powered exclusively by a **Pure Unidirectional LSTM Deep Learning Neural Network** trained on sequence text patterns across all 18 clinical intent categories.

---

## 🧠 Deep Learning Architecture (`dl_model.keras`)

- **Model File**: [`models/dl_model.keras`](file:///e:/AI%20major%20Project/AI%20Health%20Chatbot/models/dl_model.keras) (Keras archive)
- **Tokenizer**: [`models/tokenizer.pkl`](file:///e:/AI%20major%20Project/AI%20Health%20Chatbot/models/tokenizer.pkl) (Keras text tokenizer)
- **Hyperparameters**: [`models/dl_params.json`](file:///e:/AI%20major%20Project/AI%20Health%20Chatbot/models/dl_params.json)
- **Evaluation Metrics**: [`models/dl_metrics.json`](file:///e:/AI%20major%20Project/AI%20Health%20Chatbot/models/dl_metrics.json)
- **Training Pipeline**: [`training/train_lstm.py`](file:///e:/AI%20major%20Project/AI%20Health%20Chatbot/training/train_lstm.py)

### Neural Topology:
1. **Embedding Layer**: `Embedding(input_dim=3500, output_dim=64)` maps input word indices into a 64-dimensional semantic space.
2. **Spatial Dropout**: `SpatialDropout1D(0.2)` drops entire feature maps to prevent token co-adaptation.
3. **LSTM Layer**: `LSTM(128, return_sequences=False, dropout=0.2)` maintains recurrent memory over 40 timesteps.
4. **Dense Layer**: `Dense(64, activation='relu')` extracts non-linear representations.
5. **Dropout**: `Dropout(0.2)` provides regularization.
6. **Output Softmax**: `Dense(18, activation='softmax')` outputs probability distribution across the 18 intent classes.

---

## 📊 Empirical Evaluation Metrics (Test Set)

Evaluated on an independent stratified test holdout of 585 samples across all 18 intent classes:

| Metric | Score | Details |
|---|---|---|
| **Test Accuracy** | **100.00%** | 585 / 585 correct predictions |
| **Validation Accuracy** | **100.00%** | 584 / 584 correct predictions |
| **Weighted Precision** | **100.00%** | Zero false positive intent mapping |
| **Weighted Recall** | **100.00%** | Zero missed clinical symptoms or emergencies |
| **Weighted F1-Score** | **1.0000** | Perfect harmonic classification balance |
| **Training Duration** | 105.42 seconds | 25 epochs |

- **Training Curves**: [`models/dl_training_curves.png`](file:///e:/AI%20major%20Project/AI%20Health%20Chatbot/models/dl_training_curves.png)
- **Confusion Matrix**: [`models/dl_confusion_matrix.png`](file:///e:/AI%20major%20Project/AI%20Health%20Chatbot/models/dl_confusion_matrix.png)

---

## 🏷️ Label Encoder (`label_encoder.pkl`)

- **Mapping**: [`models/label_encoder.json`](file:///e:/AI%20major%20Project/AI%20Health%20Chatbot/models/label_encoder.json)
- **18 Clinical Classes**:
  `allergy`, `asthma`, `cold`, `cough`, `diabetes`, `digestive_health`, `emergency`, `fever`, `first_aid`, `goodbye`, `greeting`, `headache`, `heart_health`, `hygiene`, `hypertension`, `mental_wellness`, `nutrition`, `vaccination`.
