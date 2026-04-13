import pandas as pd
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

data = pd.read_csv("Amazon.csv")

data = data[['reviews.text','reviews.rating','brand','categories']]
data = data.dropna()

print("Sample Data:")
print(data.head())

def clean_text(text):
    text = str(text)
    text = text.lower()                          # lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', text)       # hapus simbol
    text = re.sub(r'\s+', ' ', text).strip()     # hapus spasi berlebih
    return text

data['cleaned_reviews'] = data['reviews.text'].apply(clean_text)

def sentiment_label(rating):
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"

data['sentiment'] = data['reviews.rating'].apply(sentiment_label)

print("\nSentiment Distribution:")
print(data['sentiment'].value_counts())

tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(data['cleaned_reviews'])
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

data['prediction'] = model.predict(X)

data.to_csv("sentiment_output.csv", index=False)

print("\nFile saved as sentiment_output.csv")

data['sentiment'].value_counts().plot(kind='bar')
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()