import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
import joblib
import streamlit as st
from sklearn.metrics import accuracy_score


df = pd.read_csv("data.csv",on_bad_lines='skip')
df = df.dropna()
df = df.drop_duplicates(subset=['password'])
df = df.sample(n=50000)

df['length'] = df['password'].apply(len)
df['uppercases'] = df['password'].apply(lambda x: sum(1 for c in x if c.isupper()))
df['digits'] = df['password'].apply(lambda x: sum(1 for c in x if c.isdigit()))
df['specials'] = df['password'].apply(lambda x: sum(1 for c in x if not c.isalnum()))



df_train, df_test = train_test_split(df, test_size=0.2, random_state=30)


tfidf = TfidfVectorizer(token_pattern=r'\b\w+\b')
train_tokens = tfidf.fit_transform(df_train['password'])
test_tokens = tfidf.transform(df_test['password'])

train_additional = csr_matrix(df_train[['length', 'uppercases', 'specials', 'digits']].values)
test_additional = csr_matrix(df_test[['length', 'uppercases', 'specials', 'digits']].values)

x_train = hstack([train_tokens, train_additional])
x_test = hstack([test_tokens, test_additional])

y_train = df_train['strength'].values
y_test = df_test['strength'].values


model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)


joblib.dump(model, 'password_strength_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')


y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy : ",accuracy)


with open("accuracy.txt", "w") as f:
    f.write(str(accuracy))
