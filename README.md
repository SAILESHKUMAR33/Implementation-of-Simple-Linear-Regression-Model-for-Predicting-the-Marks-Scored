# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
## AIM :
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required :
1.Hardware – PCs

2.Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm :
## STEP 1 :
Import the needed packages
## STEP 2 :
Assigning hours To X and Scores to Y
## STEP 3 :
Plot the scatter plot
## STEP 4 :
Use mse,rmse,mae formmula to find

# Program :

## Program to implement the simple linear regression model for predicting the marks scored.
## Developed by: SAILESHKUMAR A
## RegisterNumber: 212222230126
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("/content/student_scores.csv")
#displaying the content in datafile
df.head()
df.tail()
X = df.iloc[:,:-1].values
X  
Y = df.iloc[:,1].values
Y
Y_pred
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="black")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,regressor.predict(X_test),color="skyblue")
plt.title("Hours vs Scores (Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```
# Output :
## df.head() :
![image](https://github.com/shara56/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497104/10f2e6a4-624a-4600-a0b3-483a943cae62)
## df.tail() :
![image](https://github.com/shara56/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497104/58c07cf7-d966-426c-a80c-dcf248d94798)
## Array value of X :
![image](https://github.com/shara56/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497104/13ff22d1-3be3-42e0-b371-3c72bfe05ff2)
## Array value of Y :
![image](https://github.com/shara56/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497104/3124e313-f372-44dd-86d9-beb85bf84ece)
## Values of Y prediction :
![image](https://github.com/shara56/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497104/066bcdd3-4000-4b4a-8f04-3684c3bf731d)
## Array values of Y test :
![image](https://github.com/shara56/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497104/f9597eb4-7dd1-4e21-a1f0-034cf58346a6)
## Training Set Graph :
![image](https://github.com/shara56/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497104/3a8bc460-ddc9-419b-8fc2-36c6f03ab3b3)
## Test Set Graph :
![image](https://github.com/shara56/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497104/159621e9-114e-456d-9e50-5c42155d6747)
## Values of MSE,MAE AND RMSE :
![image](https://github.com/shara56/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497104/fdc45ed3-bd9b-44fb-8e81-53bcf701f69d)
## Result :
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
