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
