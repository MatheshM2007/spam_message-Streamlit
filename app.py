import streamlit as st
import pickle

st.set_page_config(page_title="Spam", layout="wide")

@st.cache_resource
def load_objects():
    with open("model.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("spam.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, model
vectorizer, model = load_objects()
st.title("Spam Message Checker")
email_input= st.text_input("Enter the message")
if st.button("Predict"):
    if email_input.strip() == "":
        st.warning("Please enter a message")
    else:
        try:
            email_vector = vectorizer.transform([email_input])
            prediction = model.predict(email_vector)[0]
            if prediction[0] == 1 or str(prediction).lower() == "spam":
                st.error("This is  Spam")
            else:
                st.success("The message is Not Spam")

        except Exception as e:
            st.error("Error during prediction: " + str(e))