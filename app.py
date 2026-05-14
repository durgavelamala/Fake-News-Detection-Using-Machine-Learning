import streamlit as st
import joblib

# Load your saved files
model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title("📰 Fake News Detection System")
st.write("Enter a news article below to check if it is Real or Fake.")

user_input = st.text_area("Paste News Text Here:", height=200)

if st.button("Analyze News"):
    if user_input.strip() != "":
        # Transform the text using your vectorizer
        transformed_data = vectorizer.transform([user_input])
        prediction = model.predict(transformed_data)
        
        # Display results (Change 0/1 based on your notebook's labels)
        if prediction[0] == 0:
            st.error("🚨 Warning: This news appears to be FAKE!")
        else:
            st.success("✅ Verified: This news appears to be REAL!")
    else:
        st.warning("Please paste some text first.")
