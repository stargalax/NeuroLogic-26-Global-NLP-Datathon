import streamlit as st
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
st.set_page_config(page_title="HateSense AI", layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.title("🔗 Project Links")

st.sidebar.markdown("""
### 📂 Resources
- 🔗 [GitHub Repo](https://github.com/stargalax/NeuroLogic-26-Global-NLP-Datathon)
- 📓 [Google Colab Notebook](https://colab.research.google.com/drive/1Zr9ayjbNezG_Y3jhnZa1AOW-7KxTAo6J?usp=sharing)
- 🎥 [YouTube Demo](https://youtu.be/s0G42Ze59_o)

---

### 🧠 About
Multilingual Toxic Comment Detection using ML ensemble models.

---

### ⚙️ Tech Stack
- TF-IDF
- Logistic Regression
- Linear SVM
- Streamlit
""")
# LOAD MODELS
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
lr = pickle.load(open("models/lr.pkl", "rb"))
svm_model = pickle.load(open("models/svm.pkl", "rb"))
threshold = 0.65

# -------------------------
# CLEAN TEXT
# -------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\u0900-\u097F\s]", "", text)
    return text

# -------------------------
# PREDICT FUNCTION
# -------------------------
def predict(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])

    lr_p = lr.predict_proba(vec)[0][1]
    svm_p = svm_model.predict_proba(vec)[0][1]

    prob = (lr_p + svm_p) / 2
    label = 1 if prob > threshold else 0

    return label, prob

# -------------------------
# UI
# -------------------------
st.title("🧠 HateSense AI")

tab1, tab2 = st.tabs(["💬 Try Model", "📊 Results"])

# =========================
# TAB 1 - MODEL USE
# =========================
with tab1:
    st.subheader("Enter a sentence")

    user_input = st.text_area("Type here:")

    if st.button("Predict"):
        label, prob = predict(user_input)

        if label == 1:
            st.error("TOXIC (1) 🔴")
        else:
            st.success("NON-TOXIC (0) 🟢")

        st.write("Confidence:", round(prob, 3))

# =========================
# TAB 2 - RESULTS ONLY
# =========================
with tab2:
    st.subheader("Model Performance")

    st.markdown("""
    | Metric | Score |
    |--------|-------|
    | Accuracy | 0.858 |
    | F1 Score | 0.850 |
    | ROC-AUC | 0.950 |
    """)

    st.subheader("Confusion Matrix")

    cm = np.array([[823, 55],
                   [200, 722]])

    fig1, ax1 = plt.subplots()
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["NON-TOXIC", "TOXIC"]
    )
    disp.plot(cmap="Blues", values_format="d", ax=ax1)

    st.pyplot(fig1)

    st.subheader("ROC Curve (Optional Image)")
   

    # If you saved ROC as image:
    st.image("roc curve.png")
