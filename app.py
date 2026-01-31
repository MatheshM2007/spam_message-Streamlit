import streamlit as st
import pickle

st.set_page_config(page_title="Spam", layout="wide")

import os

@st.cache_resource
def load_objects():
    """Load model files from disk. Raises FileNotFoundError if missing."""
    if not os.path.exists("model.pkl") or not os.path.exists("spam.pkl"):
        raise FileNotFoundError("Required model files not found")
    with open("model.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("spam.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, model


def create_demo_pickles():
    """Create small demo vectorizer and classifier for testing/deployments."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB

    texts = [
        "win money now",
        "free entry prize",
        "hello friend how are you",
        "meeting schedule for tomorrow",
        "buy cheap meds",
        "offer valid only today",
        "let's catch up over coffee",
        "your invoice is attached",
    ]
    labels = ["spam", "spam", "not spam", "not spam", "spam", "spam", "not spam", "not spam"]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)

    clf = MultinomialNB()
    clf.fit(X, labels)

    with open("model.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("spam.pkl", "wb") as f:
        pickle.dump(clf, f)


# Try loading objects; if missing, show a friendly message and allow creating demo models
try:
    vectorizer, model = load_objects()
except FileNotFoundError:
    st.title("Spam Message Checker")
    st.error("Model files (`model.pkl` or `spam.pkl`) were not found in the app folder.")
    st.info("You can upload the pickles to the app folder, use Git LFS, or create demo models for testing.")
    if st.button("Create demo models (for testing)"):
        create_demo_pickles()
        st.success("Demo models created. Please reload the app to use them.")
    st.stop()

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