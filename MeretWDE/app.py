from PIL import Image
import streamlit as st
import numpy as np

import pandas as pd
import pickle

from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE



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




df1 = pd.read_csv('https://raw.githubusercontent.com/MeretKhadour/ClassificationWDE/main/MeretWDE/vehicles_new1.csv')


df_clean = pd.DataFrame(df1)

X =df_clean[['price_num', 'year_num', 'manufacturer_num', 'model_num', 'condition_num', 'cylinders_num',
       'fuel_num', 'odometer_num', 'title_status_num', 'transmission_num', 'drive_num','paint_color_num']]
y =  df_clean['type_num']

count_values = y.value_counts()
#print(count_values)
labels = count_values.index.to_list()



oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
count_values = y.value_counts()
#print(count_values)
labels = count_values.index.to_list()





X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

## Reads in saved classification model
 
       
model_CategoricalNB_clf = pickle.load(open('D:\pklfiles\model_CategoricalNB.pkl', 'rb'))
model_decision_tree_clf = pickle.load(open('D:\pklfiles\model_decision_tree.pkl', 'rb'))
model_LogisticRegression_clf = pickle.load(open('D:\pklfiles\model_LogisticRegression.pkl', 'rb'))
## Apply model to make predictions

y_pred_decision_tree=model_decision_tree_clf.predict(X_test)
confusion_decision_tree = confusion_matrix(y_test, y_pred_decision_tree)
disp_decision_tree = ConfusionMatrixDisplay(confusion_matrix=confusion_decision_tree )



y_pred_CategoricalNB=model_CategoricalNB_clf.predict(X_test)
confusion_CategoricalNB = confusion_matrix(y_test, y_pred_CategoricalNB)
disp_CategoricalNB = ConfusionMatrixDisplay(confusion_matrix=confusion_CategoricalNB )



y_pred_LogisticRegression=model_LogisticRegression_clf.predict(X_test)
confusion_LogisticRegression = confusion_matrix(y_test, y_pred_LogisticRegression)
disp_LogisticRegression = ConfusionMatrixDisplay(confusion_matrix=confusion_LogisticRegression )
                                                






dict_DecisionTree = {' Decision Tree Classification':['Accuracy Score', 'Precision Score', 'Recall Score', 'F1 Score'],
        'weighted':[accuracy_score(y_test, y_pred_decision_tree), precision_score(y_test, y_pred_decision_tree,average = 'weighted'), recall_score(y_test, y_pred_decision_tree,average = 'weighted'), f1_score(y_test, y_pred_decision_tree,average = 'weighted')],
        'micro':[accuracy_score(y_test, y_pred_decision_tree), precision_score(y_test, y_pred_decision_tree,average = 'micro'), recall_score(y_test, y_pred_decision_tree,average = 'micro'), f1_score(y_test, y_pred_decision_tree,average = 'micro')],
        'macro':[accuracy_score(y_test, y_pred_decision_tree), precision_score(y_test, y_pred_decision_tree,average = 'macro'), recall_score(y_test, y_pred_decision_tree,average = 'macro'), f1_score(y_test, y_pred_decision_tree,average = 'macro')]}
df_DecisionTree = pd.DataFrame(dict_DecisionTree)


dict_LogisticRegression = {' Logistic Regression Classification':['Accuracy Score', 'Precision Score', 'Recall Score', 'F1 Score'],
        'weighted':[accuracy_score(y_test, y_pred_LogisticRegression), precision_score(y_test, y_pred_LogisticRegression,average = 'weighted'), recall_score(y_test, y_pred_LogisticRegression,average = 'weighted'), f1_score(y_test, y_pred_LogisticRegression,average = 'weighted')],
        'micro':[accuracy_score(y_test, y_pred_LogisticRegression), precision_score(y_test, y_pred_LogisticRegression,average = 'micro'), recall_score(y_test, y_pred_LogisticRegression,average = 'micro'), f1_score(y_test, y_pred_LogisticRegression,average = 'micro')],
        'macro':[accuracy_score(y_test, y_pred_LogisticRegression), precision_score(y_test, y_pred_LogisticRegression,average = 'macro'), recall_score(y_test, y_pred_LogisticRegression,average = 'macro'), f1_score(y_test, y_pred_LogisticRegression,average = 'macro')]}
df_LogisticRegression = pd.DataFrame(dict_LogisticRegression)


dict_CategoricalNB = {' Logistic CategoricalNB':['Accuracy Score', 'Precision Score', 'Recall Score', 'F1 Score'],
        'weighted':[accuracy_score(y_test, y_pred_CategoricalNB), precision_score(y_test, y_pred_CategoricalNB,average = 'weighted'), recall_score(y_test, y_pred_CategoricalNB,average = 'weighted'), f1_score(y_test, y_pred_CategoricalNB,average = 'weighted')],
        'micro':[accuracy_score(y_test, y_pred_CategoricalNB), precision_score(y_test, y_pred_CategoricalNB,average = 'micro'), recall_score(y_test, y_pred_CategoricalNB,average = 'micro'), f1_score(y_test, y_pred_CategoricalNB,average = 'micro')],
        'macro':[accuracy_score(y_test, y_pred_CategoricalNB), precision_score(y_test, y_pred_CategoricalNB,average = 'macro'), recall_score(y_test, y_pred_CategoricalNB,average = 'macro'), f1_score(y_test, y_pred_CategoricalNB,average = 'macro')]}
df_CategoricalNB = pd.DataFrame(dict_CategoricalNB)


st.write(''' 
### Show Confusion Matrix  ''')

ConfusionMatrixDecisionTree_button = st.button('Decision Tree ')
ConfusionMatrixLogisticRegression_button = st.button('Logistic Regression ')
ConfusionMatrixCategoricalNB_button = st.button('CategoricalNB ')
 


if ConfusionMatrixLogisticRegression_button==True:
    st.write(''' Confusion Matrix Logistic Regression ''')
    st.write(confusion_LogisticRegression)
    



if ConfusionMatrixCategoricalNB_button==True:
    st.write(''' Confusion Matrix CategoricalNB ''')
    st.write(confusion_CategoricalNB)
    

if ConfusionMatrixDecisionTree_button==True:
    st.write(''' Confusion Matrix Decision Tree ''')
    st.write(confusion_decision_tree)
    





st.write(''' 
### Calculation Accuracy, Precision , Recall  , F1 ''')
DisplayDecisionTree_button = st.button("Decision Tree Classification ")
DisplayLogisticRegression_button = st.button("Logistic Regression Classification ")

DisplayCategoricalNB_button = st.button("CategoricalNB Classification ")

if DisplayDecisionTree_button:
    st.write(df_DecisionTree)

if DisplayLogisticRegression_button:
    st.write(df_LogisticRegression)
    
if DisplayCategoricalNB_button:
    st.write(df_CategoricalNB)




st.write(''' 
### Show Decistion Tree  ''')

image = Image.open('D:\pklfiles\decistion_tree.png')
st.image(image, caption='Decistion Tree')
