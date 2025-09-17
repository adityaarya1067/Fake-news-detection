import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


user_input = input("Enter the news headline or article:").strip()
transformed_input = vectorizer.transform([user_input])
prediction = model.predict(transformed_input)[0]

        # Show result
if prediction == 1:
              st.success("✅ This news seems **REAL**.")
else:
              st.error("❌ This news seems **FAKE**.")





if st.button("Check News"):
    if user_input.strip():
        # Convert text into vector form
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]

        # Show result
        if prediction == 1:
            st.success("✅ This news seems **REAL**.")
        else:
            st.error("❌ This news seems **FAKE**.")
    else:
        st.warning("Please enter some text above.")
