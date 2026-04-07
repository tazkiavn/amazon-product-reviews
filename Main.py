import pandas as pd
import re

data = pd.read_csv("Amazon.csv")

data = data[['reviews.text','reviews.rating','brand','categories']]
data = data.dropna()

def clean_text(text):
    text = str(text)                       
    text = text.lower()                    
    text = re.sub(r'[^a-zA-Z]', ' ', text) 
    return text

data['reviews.text'] = data['reviews.text'].astype(str)
data['cleaned_reviews.text'] = data['reviews.text'].apply(clean_text)

print(data['cleaned_reviews.text'])
