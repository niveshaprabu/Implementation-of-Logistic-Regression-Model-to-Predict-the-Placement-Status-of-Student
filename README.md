# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student
# AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student. Equipments Required:

Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
# Algorithm
Import the standard libraries.
Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
LabelEncoder and encode the dataset.
Import LogisticRegression from sklearn and apply the model on the dataset.
Predict the values of array.
Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
Apply new unknown values
# Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Nivesha.P
RegisterNumber:  212222040108
*/
import pandas as pd
data=pd.read_csv('/Placement_Data(1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test,y_pred)#Accuracy Score = (TP+TN)/(TP+FN+TN+FP)
#accuracy_score(y_true,y_pred,normalize=False)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion #11+24=35 -correct predictions,5+3=8 incorrect predictions

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
# Output:
# Placement data
![image](https://github.com/niveshaprabu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122986499/ada71815-6be3-4329-ad09-bf10dcae964b)


# Salary data
![image](https://github.com/niveshaprabu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122986499/1d73f93d-70c7-41f7-8a32-402d9d456d85)


# Checking the null() function
![image](https://github.com/niveshaprabu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122986499/2bff798d-e82b-42dc-8893-770dd19e0bc2)


# Data Duplicate
![image](https://github.com/niveshaprabu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122986499/7267e759-db45-4e73-a005-1dd918988dba)


# Print data
![image](https://github.com/niveshaprabu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122986499/4bf82ed3-6d7e-4ce0-93ed-0eee29d4853c)


# Data-status
![image](https://github.com/niveshaprabu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122986499/1a9f7b0a-aa17-4137-89d9-0629d465581f)


# y_prediction array
![image](https://github.com/niveshaprabu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122986499/6911fa22-34f8-4de5-93ea-e9b9a13ddb82)


# Accuracy value
![image](https://github.com/niveshaprabu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122986499/5842b86d-528f-4dbc-a17d-e9de74f0eb37)


# Confusion array
![image](https://github.com/niveshaprabu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122986499/cbc127df-90e2-42a3-af5d-2d3dfdec95af)


# Classification report
![image](https://github.com/niveshaprabu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122986499/8811b421-c018-4b7b-8def-bb604d516037)


# Prediction of LR
![image](https://github.com/niveshaprabu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122986499/a5b0ad03-9291-4828-b638-4dfc7f516d88)


# Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
