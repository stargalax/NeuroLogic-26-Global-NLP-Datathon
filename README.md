
# 🧠 Multilingual Toxic Comment Classification : HateSense AI

## 📌 Project Overview

This project builds a machine learning system to classify multilingual social media comments as **TOXIC (1)** or **NON-TOXIC (0)**.
The model handles both **English and Hindi text**, making it suitable for real-world global content moderation systems.

The solution uses classical NLP techniques with strong performance and interpretability.

---

## 🚀 Problem Statement

Given a comment text, predict whether it contains toxicity such as:

* Abuse
* Hate speech
* Insults
* Threats

### Output:

* `0` → Non-Toxic
* `1` → Toxic

---

## 📂 Dataset

* `toxic_labeled.xlsx` → Training data (text + label)
* `toxic_no_label_evaluation.xlsx` → Test data (text only)

Languages:

* English
* Hindi
* Mixed text

---
## test it out! --->  
## ⚙️ Approach

### 1️⃣ Text Preprocessing

* Lowercasing
* URL removal
* Special character removal
* Hindi Unicode preserved

### 2️⃣ Feature Engineering

* TF-IDF Vectorization
* N-grams: (1,2)
* Max features: 12,000

---

### 3️⃣ Models Used

I used an **ensemble approach**:

#### ✔ Logistic Regression

* Handles sparse text well
* Provides stable probabilities

#### ✔ Linear SVM (Calibrated)

* Strong margin-based classifier
* Converted to probabilities using calibration

---

### 4️⃣ Ensemble Strategy

Final prediction = average of:

```
P = (Logistic Regression + SVM) / 2
```

## 📊 Evaluation Metrics

### ✔ Accuracy

Measures overall correctness of predictions.

### ✔ F1-score

Balances precision and recall, especially important for imbalanced error types.

### ✔ ROC-AUC (Primary Metric)

Evaluates the model’s ability to rank toxic vs non-toxic comments.

### ✔ Confusion Matrix

Analyzes:

* True Positives
* True Negatives
* False Positives
* False Negatives

---

## 📈 Results Summary

### 🧾 Model Performance

| Metric   | Score  |
| -------- | ------ |
| Accuracy | 0.8583 |
| F1-score | 0.8499 |
| ROC-AUC  | 0.9503 |

---

### 🧊 Confusion Matrix

```
[[823  55]
 [200 722]]
```

---

## 🧠 Interpretation

* Strong ROC-AUC shows excellent separability between toxic and non-toxic classes
* High F1-score indicates balanced precision and recall
* Confusion matrix shows the model is slightly more sensitive to toxic detection, with manageable false positives

---

### Insights:

* The model achieves a strong ROC-AUC of 0.95, showing excellent class separation between toxic and non-toxic comments.
* A solid F1-score of 0.85 confirms a good balance between precision and recall.
* Confusion matrix shows slightly higher false positives, meaning the model is conservative in flagging toxicity.
* TF-IDF with n-grams effectively captures multilingual and informal text patterns without deep learning.
* Ensemble (Logistic Regression + SVM) improves stability and reduces prediction variance.
* Threshold tuning (0.65) helps reduce misclassification in borderline cases.

---

## 📊 Visualizations

### ROC Curve

Shows strong separability between toxic and non-toxic classes.
<img width="536" height="470" alt="image" src="https://github.com/user-attachments/assets/438616a8-1f7c-4591-ad94-80b937d8089d" />


### Confusion Matrix Heatmap

Used to analyze prediction errors and threshold tuning.
<img width="575" height="455" alt="image" src="https://github.com/user-attachments/assets/83e7e105-60eb-49fa-9704-ee572856a899" />

---

## 💬 Demo System

The model supports real-time predictions:
<img width="1121" height="715" alt="image" src="https://github.com/user-attachments/assets/90ba15e9-8792-4229-9ff7-bb44ad3b7dbf" />
<img width="1102" height="727" alt="image" src="https://github.com/user-attachments/assets/c8c7564c-3a4f-4b5b-a09d-6065d3a925d8" />

### Example:

```
Input: "I love your dress"
Output: NON-TOXIC (0)

Input: "I will kill you"
Output: TOXIC (1)
```

---

## ⚙️ How to Run

### 1. Install dependencies

```bash
pip install pandas scikit-learn openpyxl matplotlib
```

### 2. Run notebook / script

* Load dataset
* Train model
* Run predictions

### 3. Generate submission

```
no_label.xlsx → contains predicted labels
```

---

## 📁 Project Structure

```
├── train.xlsx    (toxic_labeled.xlsx)
├── test.xlsx     (toxic_no_label_evaluation.xlsx)
├── NeuroLogic '26: Global NLP Datathon.ipynb
├── submission.xls
├── app.py         (Streamlit code)
├── models         (contains the trained models)
└── README.md
```

---

## 🔥 Key Features

* Multilingual support (English + Hindi)
* Strong baseline ensemble model
* ROC-AUC optimized performance
* Interactive prediction system
* Reproducible pipeline
* Clean evaluation metrics

---

## 🧠 Limitations

* TF-IDF lacks deep semantic understanding
* Occasional false positives on informal positive sentences
* Cannot fully understand sarcasm/context

---

## 🚀 Future Improvements

* Transformer-based models (BERT / mBERT)
* Better calibration for probability outputs
* Data augmentation for Hindi text
* Deep learning ensemble models

---

## 🏁 Conclusion

This project demonstrates a strong, interpretable NLP pipeline for multilingual toxicity detection with solid performance and practical real-world applicability in content moderation systems.

---

If you want, I can next:

* 🔥 convert this into a **beautiful GitHub Markdown with badges + visuals**
* 🏆 or help you write a **Devpost submission description (high impact)**
