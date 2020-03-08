---
layout: post
title:  "Multivariate Linear Regression from scratch"
description: Implementing Simple Linear Regression in Python from scratch
date:   2020-02-25 21:03:36 +0530
categories: Python MachineLearning LinearRegression
---

The following code implements Multivariate Linear Regression from scratch. The code builds up from where we left in Simple Linear Regression. Representing our hypothesis as a function of multiple parameters is both powerful and necessary !

Importing necessary packages
- numpy: for representing vectors,
- pandas: for reading csv files and
- sklearn: we use it for splitting the dataset into train and test, preprocessing, encoding and for calculating errors.

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
```

Loading the csv file and deleting columns not needed
```python
df = pd.read_csv("FuelConsumptionCo2.csv")
```

<img src = "https://raw.githubusercontent.com/SurajSubramanian/SurajSubramanian.github.io/master/_posts/images/scatterplot.png" width="400" height="300" /> 

```python
df['MODELYEAR'].unique()
```

There is only 1 MODELYEAR, hence it makes no sense to include it as a parameter

```python
df = df.drop(['MODELYEAR'], axis=1)
```

We use labelencoder to assign unique values to all the categorical data types as we cant operate on variables that are string.

```
df['MAKE'], df['MODEL'], df['VEHICLECLASS'], df['TRANSMISSION'], df['FUELTYPE'] = labelencoder.fit_transform(df['MAKE']), labelencoder.fit_transform(df['MODEL']), labelencoder.fit_transform(df['VEHICLECLASS']), labelencoder.fit_transform(df['TRANSMISSION']), labelencoder.fit_transform(df['FUELTYPE'])
```

Next, we filter out the variables that have less correlation with the target parameter - CO2EMISSIONS

```python
cor = df.corr()
cor_target = abs(cor["CO2EMISSIONS"])
relevant_features = cor_target[cor_target>0.5]
relevant_features
```

```python
df = df.drop(['MAKE', 'MODEL', 'VEHICLECLASS', 'FUELTYPE', 'TRANSMISSION'], axis=1)
```

```python

df=(df-df.min())/(df.max()-df.min())
```


```python
y,x = df["CO2EMISSIONS"], df.drop(["CO2EMISSIONS"], axis=1)

```

```python
x_, y_ = x.to_numpy(), y.to_numpy()
x_ = np.append(arr = np.ones((len(x_), 1)).astype(int), values = x, axis = 1) 

x_train,x_test,y_train,y_test = train_test_split(x_,y_, test_size = 0.2, random_state=21)
```

```python
a=0.01
X = []
for row in x_train:
    r = [1]
    for item in row:
        r.append(item)
    X.append(r)
    
X = np.asmatrix(X)
```

```python
theta = np.zeros(((X[0].size), 1))

Y = y_train.reshape(-1,1)
temp = np.zeros(theta.shape)
```

The gradientDescent function and the subsequent code are similar to the ones we used for Linear Regression, except that here we represent theta as an array.

One thing I usually do is to examine and play with the shapes of all matrices and arrays and see how they could be combined to produce the required resultant matrix

```python
def gradientDescent(theta, X):
    h = np.dot(X, theta)
    cost = np.sum(np.sum((h-Y)**2))*(1/(2*X.shape[0]))
    temp = theta - np.dot(X.T, h-Y) * (a/X.shape[0])
    theta = temp
    return(theta, X, cost)
```

```python
oldCost = 0
theta = np.ones(theta.shape)
X = np.ones(X.shape)
for i in range(0, 10000):
    (theta, X, cost) = gradientDescent(theta, X)
    if i%1000 == 0:
        print(cost)
        print(theta)
```

```python
X_test = []
for row in x_test:
    r = [1]
    for item in row:
        r.append(item)
    X_test.append(r)
```python
	
```python
mean_squared_error(np.dot(X_test, theta), y_test)
````