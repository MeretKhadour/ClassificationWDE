from streamlit import pyplot
import pandas as pd
import pickle
from matplotlib import pyplot
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import CategoricalNB
from sklearn.linear_model import LogisticRegression



df1 = pd.read_csv('vehicles_new1.csv')
df_clean = pd.DataFrame(df1)


X =df_clean[['price_num', 'year_num', 'manufacturer_num', 'model_num', 'condition_num', 'cylinders_num',
       'fuel_num', 'odometer_num', 'title_status_num', 'transmission_num', 'drive_num','paint_color_num']]
y =  df_clean['type_num']


oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model_CategoricalNB = CategoricalNB(alpha=0,force_alpha=True).fit(X_train,y_train )
model_decision_tree = DecisionTreeClassifier().fit(X_train,y_train)
model_LogisticRegression = LogisticRegression(random_state=0).fit(X_train,y_train )
                                                
features_names = ['price_num', 'year_num', 'manufacturer_num', 'model_num', 'condition_num', 'cylinders_num',
       'fuel_num', 'odometer_num', 'title_status_num', 'transmission_num', 'drive_num','paint_color_num']
target_names = ['0','1','2','3','4','5','6','7','8','9','10','11','12']

fig = pyplot.figure(figsize=(100,100))
_=tree.plot_tree(model_decision_tree,feature_names = features_names, class_names =target_names, filled = True )
fig.savefig("decistion_tree.png")

import pickle
pickle.dump(model_decision_tree,open('model_decision_tree.pkl','wb'))
pickle.dump(model_LogisticRegression,open('model_LogisticRegression.pkl','wb'))
pickle.dump(model_CategoricalNB,open('model_CategoricalNB.pkl','wb'))