import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st 
import seaborn as sns
import itertools
from sklearn.metrics import confusion_matrix

from web_functions import train_model

def plot_confusion_matrix(model, x, y):
    y = model.predict(x)
    confusion_matrix(y,y)
    sns.heatmap((confusion_matrix(y,y)), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.05)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

def app(df, x, y):
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)


    st.title("Visualisasi")
    model, score = train_model(x, y)
    
    if st.checkbox("Plot"):
        plot_confusion_matrix(model, x, y)
        st.pyplot()