import pandas as pd
import re

data = pd.read_csv("Amazon.csv")

data = data[['reviews.text','reviews.rating','brand','categories']]
data = data.dropna()

print(data.head())

def clean_text(text):
    text = str(text)                       
    text = text.lower()                    
    text = re.sub(r'[^a-zA-Z]', ' ', text) 
    return text

data['reviews.text'] = data['reviews.text'].astype(str)
data['cleaned_reviews.text'] = data['reviews.text'].apply(clean_text)

print(data['cleaned_reviews.text'])

def sentiment_label(rating):
    if rating > 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"

data['sentiment'] = data['reviews.rating'].apply(sentiment_label)

print(data['sentiment'])

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)
x = tfidf.fit_transform(data['cleaned_reviews.text'])
y = data['sentiment']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

model = LogisticRegression()
model.fit(x_train,y_train)

from sklearn.metrics import classification_report

y_pred = model.predict(x_test)

print(classification_report(y_test,y_pred))

data['prediction'] = model.predict(x)
data.to_csv("sentiment_output.csv", index=False)

data['sentiment'].value_counts

import matplotlib.pyplot as plt

data['sentiment'].value_counts().plot(kind='bar')
plt.show()