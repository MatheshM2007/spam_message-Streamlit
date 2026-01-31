from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Small toy dataset
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

# Create and fit vectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train a simple classifier
clf = MultinomialNB()
clf.fit(X, labels)

# Save the vectorizer as `model.pkl` (matches what `app.py` expects as the vectorizer)
with open("model.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save the classifier as `spam.pkl`
with open("spam.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Demo pickles created: model.pkl (vectorizer) and spam.pkl (classifier)")
