# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.
2.Write a function computeCost to generate the cost function.
3.Perform iterations og gradient steps with learning rate.
4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: GIRITHICK ROHAN N
RegisterNumber: 212223230063

# Import required package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("ex1.txt",header=None)
data
data.shape
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Vs Prediction")
def computeCost(X,y,theta):
    m=len(y)
    h=X.dot(theta)
    square_err=(h-y)**2

    return 1/(2*m) * np.sum(square_err)
data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)
theta.shape
y.shape
X.shape
def gradientDescent(X,y,theta,alpha,num_iters):
  
  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions - y))
    descent=alpha * 1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))

  return theta, J_history
  
theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")
plt.plot(J_history)
plt.xlabel("Iternations")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Polpulation of City (10,000s)")
plt.ylabel("Profit (10,000s)")
plt.title("Profit Prediction")
def predict(x,theta):
  predictions= np.dot(theta.transpose(),x)
  return predictions[0]
  
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```

## Output:
## Read CSV File:
![image](https://github.com/Girithickrohan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138849207/07484ac4-dc78-4c45-9b91-94a1f97a3f3b)

## Dataset Shape:
![image](https://github.com/Girithickrohan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138849207/1f1737df-a371-45c3-9e4d-7a6c04baeab3)

## Profit Vs Prediction graph:
![image](https://github.com/Girithickrohan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138849207/d372333b-5673-461d-846a-ce1678f23090)

## x,y,theta value:
![image](https://github.com/Girithickrohan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138849207/7f0fbd01-27c8-4cc0-91dd-060ff180ffe3)

## Gradient descent:
![image](https://github.com/Girithickrohan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138849207/6c360394-a5e6-4b17-a7de-76acaff65a78)

## Cost function using Gradient Descent Graph:
![image](https://github.com/Girithickrohan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138849207/37ea0db1-5a24-42eb-9493-1c75cfa88bff)

## Profit Prediction Graph:
![image](https://github.com/Girithickrohan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138849207/bcb1b476-5559-42ae-9f59-38df603ff09a)

## Profit Prediction:
![image](https://github.com/Girithickrohan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138849207/b3110a61-8025-4b35-9ed7-7074b8dc2678)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

