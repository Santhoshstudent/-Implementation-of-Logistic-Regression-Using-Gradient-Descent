# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: santhosh kumar B
RegisterNumber: 212223230193
*/
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
'''
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: santhosh kumar B
RegisterNumber: 212223230193
*/
```

## Output:
![logistic regression using gradient descent](sam.png)
![image ml 01](https://github.com/Santhoshstudent/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145446853/19a25a4e-1103-4085-9f2b-2b515c2d6d5f)
![image ml 02](https://github.com/Santhoshstudent/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145446853/1e1481d5-c84b-4340-8017-1453c2872a11)
![image ml 03](https://github.com/Santhoshstudent/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145446853/c857a61d-370d-4c50-a576-3887888b7111)
![image ml 04](https://github.com/Santhoshstudent/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145446853/f2793f1b-50f7-479a-ba6b-2ce389b5a742)
![image ml 05](https://github.com/Santhoshstudent/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145446853/fbfc52c6-1514-4444-992a-f093c99d9337)
![image ml 06](https://github.com/Santhoshstudent/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145446853/bfd47bb5-d7f1-440c-a88a-1d6df1ea7564)
![image ml 07](https://github.com/Santhoshstudent/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145446853/ae72af17-8a64-4e55-8ed0-89ed8c3c5887)
![image ml 08](https://github.com/Santhoshstudent/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145446853/51044ff0-f81c-4764-b28a-b133004ddcea)
![image 09](https://github.com/Santhoshstudent/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145446853/66f5a2b5-a9a1-462e-a3b1-608cb6beac81)
![image ml 09](https://github.com/Santhoshstudent/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145446853/76b27f50-7563-444a-91d8-c66b6bf8c415)










## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

