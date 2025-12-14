import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Duplicate Question Detector", layout="centered")

st.title("Duplicate Question Detector (Bi-LSTM Model)")

# ---------------------------
# User Inputs
# ---------------------------
q1 = st.text_input("Enter Question 1")
q2 = st.text_input("Enter Question 2")

# ---------------------------
# Predict Button
# ---------------------------
if st.button("Predict"):

    # ✅ Input validation
    if q1.strip() == "" or q2.strip() == "":
        st.warning("Please enter both questions before predicting.")
    
    else:
        payload = {
            "question1": q1,
            "question2": q2
        }

        try:
            response = requests.post(API_URL, json=payload, timeout=10)

            # ❌ Backend error
            if response.status_code != 200:
                st.error("Backend error")
                st.write("Status Code:", response.status_code)
                st.write("Raw Response:", response.text)

            # ✅ Success
            else:
                result = response.json()

                prediction = result.get("prediction", "Unknown")
                score = result.get("score", 0.0)

                st.subheader("Result")

                if prediction == "Duplicate":
                    st.success("Duplicate Question ✅")
                else:
                    st.info("Not Duplicate ❌")

                st.write(f"**Confidence Score:** {score:.2f}")

        except requests.exceptions.RequestException as e:
            st.error("Could not connect to backend API.")
            st.write(e)
