import streamlit as st
import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return ' '.join([word for word in text.split() if word not in stop_words])

@st.cache_resource
def load_model():
    vectorizer = joblib.load('vectorizer.pkl')
    tfidf_matrix = joblib.load('tfidf_matrix.pkl')
    train_df = pd.read_csv(r"C:\Users\hassa\Downloads\train_data_chatbot.csv")
    return vectorizer, tfidf_matrix, train_df

def get_response_tfidf(query, vectorizer, tfidf_matrix, train_df):
    query_clean = clean(query)
    user_vec = vectorizer.transform([query_clean])
    similarity = cosine_similarity(user_vec, tfidf_matrix)
    index = similarity.argmax()
    return train_df.iloc[index]['short_answer']

# --- Streamlit UI ---
st.title("🩺 Medical Chatbot")
st.write("Ask a medical question and get a response from the chatbot.")

user_input = st.text_input("Type your question here:")

vectorizer, tfidf_matrix, train_df = load_model()

if user_input:
    response = get_response_tfidf(user_input, vectorizer, tfidf_matrix, train_df)
    st.markdown("**Chatbot Response:**")
    st.success(response)

