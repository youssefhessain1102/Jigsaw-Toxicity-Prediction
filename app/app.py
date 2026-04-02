import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/api/v1/predict"

st.title("Toxicity Classifier")

text = st.text_area("Enter text to classify:", height=150)

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Classifying..."):
            try:
                res = requests.post(API_URL, json={"text": text})
                res.raise_for_status()
                predictions = res.json()["predictions"]

                st.subheader("Results")
                for label, value in predictions.items():
                    icon = "🔴" if value == 1 else "🟢"
                    st.write(f"{icon} **{label}**: {'Detected' if value == 1 else 'Clean'}")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API. Is uvicorn running?")
            except Exception as e:
                st.error(f"Error: {e}")