#import modul
import numpy as np
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st 

@st.cache_data()
def load_data():

    df = pd.read_csv('heart-data.csv')

    x = df[['sex','cp','fbs','restecg','exng','slp','caa','thall']]
    y = df[['output']]

    return df, x, y

@st.cache_data()
def train_model(x,y) :
    model= KNeighborsClassifier(n_neighbors=6)
    model.fit(x, y)
    y=model.predict(x)

    score = model. score(x, y)
    return model, score

def predict(x, y, features):
    model, score = train_model(x,y)

    prediction = model.predict(np.array(features).reshape(1,-1))
    return prediction, score