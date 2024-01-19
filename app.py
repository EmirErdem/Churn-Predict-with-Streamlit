import streamlit as st
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import joblib


st.title('Did they Churn? :bank:')
def features():
    CustomerId = st.text_input("Customer Id", '1')
    CreditScore = st.number_input("Credit Score")
    Geography = st.sidebar.selectbox('Geography',('France','Germany','Spain'))
    Gender = st.sidebar.selectbox('Gender',('Male','Female'))
    Age = st.sidebar.slider("Choose age", 0, 100,33)
    Tenure = st.sidebar.slider("Choose Tenure", 0, 10,3)
    Balance = st.number_input("Balance")
    NumOfProducts = st.sidebar.slider("Number Of Products", 1,5,3)
    HasCrCard = st.sidebar.slider("Has Credit Card", 0,1)
    IsActiveMember = st.sidebar.slider("Is Active Member?", 0,1)
    EstimatedSalary = st.number_input("Estimated Salary")

    data = {'CustomerId': CustomerId,
            'CreditScore': CreditScore,
            'Geography': Geography,
            'Gender': Gender,
            'Age': Age,
            'Tenure': Tenure,
            'Balance': Balance,
            'NumOfProducts': NumOfProducts,
            'HasCrCard': HasCrCard,
            'IsActiveMember': IsActiveMember,
            'EstimatedSalary': EstimatedSalary }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = features()
st.subheader('User Input parameters')
st.write(input_df)

churn_raw = pd.read_csv('churn.csv')
churn = churn_raw.drop(columns=['Exited','RowNumber','Surname'])
df = pd.concat([input_df,churn],axis=0)

encode = ['Geography','Gender']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1]

model = joblib.load('model.joblib')

prediction = model.predict(df)
prediction_proba = model.predict_proba(df)


st.subheader('Prediction Probability')
st.write(prediction_proba)

if prediction[0] == 1:
    st.error('Churn :thumbsdown:')
else:
    st.success('Not churn :thumbsup:')


