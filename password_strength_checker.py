import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix


model = joblib.load('password_strength_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')


def transform_new_data(password):
    uppercases = sum(1 for c in password if c.isupper())
    specials = sum(1 for c in password if not c.isalnum())
    digits = sum(1 for c in password if c.isdigit())

    new_token = tfidf.transform([password])
    additional = csr_matrix([[len(password), uppercases, specials, digits]])
    features = hstack([new_token, additional])
    return features


st.title("Password Strength Checker")

password_input = st.text_input("Enter your password:")

if password_input:
    features = transform_new_data(password_input)
    prediction = model.predict(features)

    strength_label = {0: 'Weak', 1: 'Medium', 2: 'Strong'}
    strength = strength_label[prediction[0]]

    st.write(f"Password strength: **{strength}**")
