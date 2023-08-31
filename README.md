# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python.
2. Set variables for assigning dataset values.
3. Import LinearRegression from the sklearn.
4. Assign the points for representing the graph.
5. Predict the regression for marks by using the representation of graph.
6. Compare the graphs and hence we obtain the LinearRegression for the given datas.

 
## Program:
```python
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Saileshkumar a
RegisterNumber:  212222230126
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("student_scores.csv")
dataset.head()

X = dataset.iloc[:,:-1].values
X
y = dataset.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 1/3,random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
y_pred
y_test
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title("Hours vs scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
```

## Output:
![image](https://github.com/vinushcv/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113975318/dfe92074-3592-4798-83a2-f65c330ff18c)


![image](https://github.com/vinushcv/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113975318/c62d4eb2-1092-479d-b9ef-eec8c2ead530)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
