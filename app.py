import streamlit as st
import pandas as pd
import string
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# DOWNLOAD NLTK DATA (SAFE WAY)
# -----------------------------
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords

# -----------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Toxic Comment Detector", page_icon="⚠️")

# -----------------------------
# TITLE
# -----------------------------
st.title("⚠️ Toxic Comment Detection System")
st.markdown("### Detect whether a comment is toxic or not using NLP")

# -----------------------------
# SAMPLE DATASET
# -----------------------------
@st.cache_data
def load_data():
    data = pd.DataFrame({
        'text': [
            "I love this product",
            "You are stupid and useless",
            "Amazing experience, very happy",
            "This is the worst thing ever",
            "Thank you for your help",
            "I hate you"
        ],
        'label': [0, 1, 0, 1, 0, 1]  # 0 = Non-toxic, 1 = Toxic
    })
    return data

data = load_data()

# -----------------------------
# PREPROCESSING FUNCTION
# -----------------------------
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

data['text'] = data['text'].apply(preprocess)

# -----------------------------
# MODEL TRAINING (CACHED)
# -----------------------------
@st.cache_resource
def train_model(data):
    X = data['text']
    y = data['label']

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_vec, y)

    return model, vectorizer

model, vectorizer = train_model(data)

# -----------------------------
# USER INPUT
# -----------------------------
user_input = st.text_area("💬 Enter your comment here:")

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("🔍 Analyze Comment"):
    
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a comment")
    else:
        processed = preprocess(user_input)
        vec = vectorizer.transform([processed])
        
        prediction = model.predict(vec)[0]
        probability = model.predict_proba(vec)[0]

        st.divider()

        # -----------------------------
        # RESULT DISPLAY
        # -----------------------------
        if prediction == 1:
            st.error(f"❌ Toxic Comment\n\nConfidence: {round(probability[1]*100, 2)}%")
        else:
            st.success(f"✅ Non-Toxic Comment\n\nConfidence: {round(probability[0]*100, 2)}%")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📌 About")
st.sidebar.info(
    "This app uses NLP and Machine Learning to detect toxic comments.\n\n"
    "Model: Logistic Regression\n"
    "Technique: TF-IDF"
)
