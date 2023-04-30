from PIL import Image
import streamlit as st
from streamlit import pyplot
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import os
from matplotlib import pyplot
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import CategoricalNB
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from tabulate import tabulate
from urllib.request import urlopen
import cloudpickle as cp

st.write("""
# Classification App

#### Web Technologies master's program Program code : MWT

#### Data Exploring (WDE)

#### Class C1

#### Student Name: Meret Khadour

#### Student ID: 189038

Data obtained from the [Used Cars DataSet | Kaggle](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data).

""")

#st.sidebar.header('User Input Features')


## Reads in saved classification model
model_CategoricalNB_clf = pickle.load(open('MeretWDE/model_CategoricalNB.pkl', 'rb'))
model_decision_tree_clf = pickle.load(open('MeretWDE/model_decision_tree.pkl', 'rb'))
model_LogisticRegression_clf = pickle.load(open('MeretWDE/model_LogisticRegression.pkl', 'rb'))

## Apply model to make predictions

def classification(input_data , model):
    #input_data = (5,166,72,19,175,25.8,0.587,51)

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    
    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    prediction = model.predict(input_data_reshaped)
       
    if (prediction[0] == 0):
      return 'SUV'
    elif(prediction[0] == 1):
        return 'bus'
    elif(prediction[0] ==2):
        return 'convertible'
    elif(prediction[0] == 3):
        return 'coupe'
    elif(prediction[0] == 4):
        return 'hatchback' 
    elif(prediction[0] == 5):
        return 'mini-van'
    elif(prediction[0] == 6):
        return 'offroad' 
    elif(prediction[0] == 7):
        return 'other'
    elif(prediction[0] ==8):
        return 'pickup'
    elif(prediction[0] ==9):
        return 'sedan'
    elif(prediction[0] ==10):
        return 'truck'
    elif(prediction[0] ==11):
        return 'van'
    else: 
        return'wagon'


        
def main():
    st.title("Testing ")
    price_num = st.slider("price num", min_value=0, max_value=2556, value=1, step=1)   
    year_num = st.slider("year num", min_value=0, max_value=101, value=1, step=1) 
    manufacturer_num = st.slider("manufacturer num", min_value=0, max_value=40, value=1, step=1) 
    model_num = st.slider("model num", min_value=0, max_value=11528, value=1, step=1) 
    condition_num = st.slider("condition num", min_value=0, max_value=5, value=1, step=1) 
    cylinders_num = st.slider("cylinders num", min_value=0, max_value=7, value=1, step=1) 
    fuel_num = st.slider("fuel num", min_value=0, max_value=4, value=1, step=1) 
    odometer_num = st.slider("odometer num", min_value=0, max_value=37714, value=1, step=1) 
    title_status_num = st.slider("title num", min_value=0, max_value=5, value=1, step=1) 
    transmission_num = st.slider("transmission num", min_value=0, max_value=2, value=1, step=1) 
    drive_num = st.slider("drive num", min_value=0, max_value=2, value=1, step=1) 
    paint_color_num = st.slider("paint_color num", min_value=0, max_value=11, value=5, step=1) 
    
    input_data = [int(price_num),int(year_num),int(manufacturer_num),int(model_num),int(condition_num),int(cylinders_num),int(fuel_num),
             int(odometer_num),int(title_status_num),int(transmission_num),int(drive_num),int(paint_color_num)]
    #input_data = [1,1,1,1,1,1,1,1,1,1,1,1]
    statue = ''
    if st.button("test using model_CategoricalNB"):     
        statue = classification(input_data, model_CategoricalNB_clf)
    if st.button("test using model_decision_tree"):     
        statue = classification(input_data, model_decision_tree_clf)
    if st.button("test using model_LogisticRegression"):     
        statue = classification(input_data, model_LogisticRegression_clf)    
    st.success(statue)
        
if __name__ == '__main__':
    main()


