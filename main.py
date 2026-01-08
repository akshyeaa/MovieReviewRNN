import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


word_index=imdb.get_word_index()
reverse_word_index={value: key for key,value in word_index.items()}

model = load_model('simple_rnn_imdb.h5')


## decode the review
def decoded_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3,'?') for i in encoded_review])

def preprocessing_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word, 2)+3 for word in words]
    padded_Review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_Review


def predict_sentiment(review):
    preprocessed_input=preprocessing_text(review)
    prediction=model.predict(preprocessed_input)

    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment,prediction[0][0]


### streamlit app
import streamlit as st
st.title("IMDB Movie Review Classifier")
st.write('Enter the movie name')
user_input=st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input=preprocessing_text(user_input)
    ##make prediction

    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'

    st.write(f'Sentiment:{sentiment}')
    st.write(f'Prediction score:{prediction[0][0]}')
else:
    st.write('please enter movie review') 