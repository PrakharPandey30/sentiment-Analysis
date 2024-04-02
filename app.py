import streamlit as st
from nltk.tokenize import word_tokenize

import pickle

from nltk.corpus import stopwords
def clean_text(text):
    
    text=text.lower()
    tokens=word_tokenize(text)
    
    text=[word for word in tokens if word.isalpha()]
    


    stop_words=set(stopwords.words('english'))
    filtered_words = [word for word in text if word not in stop_words]

    text=" ".join(filtered_words)

    return text
    

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))


st.title('Sentiment Analyser')

input_msg=st.text_input('Enter the message')
if st.button('Predict'):

    # /Preprocess
    transformed_text=clean_text(input_msg)

    # vectorizer
    vector_input=tfidf.transform([transformed_text])
    # predict
    result=model.predict(vector_input)[0]
    # Display
    if result==1:
        st.header("positive")
    else:
        st.header("negative")



