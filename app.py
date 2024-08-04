import numpy as np
from tensorflow import keras
from keras.utils import pad_sequences
from keras.models import load_model
from keras.layers import Embedding, SimpleRNN, Dense
from keras.datasets import imdb
import streamlit as st

model = load_model('IMDB.h5')

word_index = imdb.get_word_index()
# reverse the mapping like (value : word)
reverse_word_index = {value:key for key,value in word_index.items()}

def decoded_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.4 else 'Negative'
    return sentiment, prediction[0][0]

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as 'Positive' or 'Negative' : ")

user_input = st.text_area("Movie Review ")

if st.button('Classify'):
    sentiment, confidence = predict_sentiment(user_input)

    st.write(f"Sentiment : {sentiment}")
    st.write(f"Prediction Score : {confidence}")
else :
    st.write("Plese enter a movie review")
