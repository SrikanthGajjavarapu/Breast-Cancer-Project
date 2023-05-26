#Breast cancer detection/prediction using logistic regression
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
breast_cancer=sklearn.datasets.load_breast_cancer()
print(breast_cancer)
x=breast_cancer.data
y=breast_cancer.target
print(x,y)
print(x.shape,y.shape)
data=pd.DataFrame(breast_cancer.data,columns=breast_cancer.feature_names)
print(data)
data.describe()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=1)
print(y_train)
#LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)
prediction_on_training_data=classifier.predict(x_train)
accracy_of_training_data=accuracy_score(y_train,prediction_on_training_data)
print(accracy_of_training_data)
prediction_on_test_data=classifier.predict(x_test)
accuracy_of_test_data=accuracy_score(y_test,prediction_on_test_data)
print(accuracy_of_test_data)
input_data=(13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)
input_data_as_array=np.asarray(input_data)
print(input_data)
input_data_reshaped=input_data_as_array.reshape(1,-1)
prediction=classifier.predict(input_data_reshaped)
print(prediction)
if (prediction[0]==0):
  print("the cancer is Benign")
else:
  print("the cancer is Malignant")