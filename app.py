import streamlit as st
import pandas as pd
import string

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Toxic Comment Detector", page_icon="⚠️")

st.title("⚠️ Toxic Comment Detection System")

# -----------------------------
# DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.DataFrame({
        'text': [
            "I love this product",
            "You are stupid and useless",
            "Amazing experience, very happy",
            "This is the worst thing ever",
            "Thank you for your help",
            "I hate you"
        ],
        'label': [0,1,0,1,0,1]
    })

data = load_data()

# -----------------------------
# PREPROCESS
# -----------------------------
stop_words = set(ENGLISH_STOP_WORDS)

def preprocess(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

data['text'] = data['text'].apply(preprocess)

# -----------------------------
# MODEL
# -----------------------------
@st.cache_resource
def train():
    X = data['text']
    y = data['label']
    
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)
    
    model = LogisticRegression()
    model.fit(X_vec, y)
    
    return model, vectorizer

model, vectorizer = train()

# -----------------------------
# UI
# -----------------------------
user_input = st.text_area("Enter Comment")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Enter text")
    else:
        processed = preprocess(user_input)
        vec = vectorizer.transform([processed])
        
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        
        if pred == 1:
            st.error(f"❌ Toxic ({round(prob[1]*100,2)}%)")
        else:
            st.success(f"✅ Non-Toxic ({round(prob[0]*100,2)}%)")
