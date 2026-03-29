import streamlit as st
import pickle
import re

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# UI Title
st.title("🧠 Fake News Detector")
st.write("Enter a news article below to check whether it's Fake or Real.")

# Input box
user_input = st.text_area("📰 Enter News Text")

# Predict button
if st.button("Predict"):

    # Handle empty input
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        # Preprocess input
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])

        # Prediction
        prediction = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0]

        # Output result
        if prediction == 0:
            st.error(f"🚨 Fake News (Confidence: {prob[0]*100:.2f}%)")
        else:
            st.success(f"✅ Real News (Confidence: {prob[1]*100:.2f}%)")

        # 🔍 Explanation Feature
        st.subheader("🔍 Important Words Influencing Prediction")

        feature_names = vectorizer.get_feature_names_out()
        vector_dense = vector.toarray()[0]

        # Get top 5 important words
        top_indices = vector_dense.argsort()[-5:][::-1]

        for i in top_indices:
            if vector_dense[i] > 0:
                st.write(f"👉 {feature_names[i]}")