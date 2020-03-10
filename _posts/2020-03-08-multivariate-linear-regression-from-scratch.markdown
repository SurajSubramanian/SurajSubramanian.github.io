The following code implements Multivariate Linear Regression from scratch. The code builds up from where we left in Simple Linear Regression. Representing our hypothesis as a function of multiple parameters is both powerful and necessary !

Importing necessary packages
- numpy: for representing vectors,
- pandas: for reading csv files and
- sklearn: we use it for splitting the dataset into train and test, preprocessing, encoding and for calculating errors.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
```

loading the csv file and deleting columns not needed


```python
df = pd.read_csv("FuelConsumptionCo2.csv")
```


```python
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MODELYEAR</th>
      <th>MAKE</th>
      <th>MODEL</th>
      <th>VEHICLECLASS</th>
      <th>ENGINESIZE</th>
      <th>CYLINDERS</th>
      <th>TRANSMISSION</th>
      <th>FUELTYPE</th>
      <th>FUELCONSUMPTION_CITY</th>
      <th>FUELCONSUMPTION_HWY</th>
      <th>FUELCONSUMPTION_COMB</th>
      <th>FUELCONSUMPTION_COMB_MPG</th>
      <th>CO2EMISSIONS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2014</td>
      <td>ACURA</td>
      <td>ILX</td>
      <td>COMPACT</td>
      <td>2.0</td>
      <td>4</td>
      <td>AS5</td>
      <td>Z</td>
      <td>9.9</td>
      <td>6.7</td>
      <td>8.5</td>
      <td>33</td>
      <td>196</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2014</td>
      <td>ACURA</td>
      <td>ILX</td>
      <td>COMPACT</td>
      <td>2.4</td>
      <td>4</td>
      <td>M6</td>
      <td>Z</td>
      <td>11.2</td>
      <td>7.7</td>
      <td>9.6</td>
      <td>29</td>
      <td>221</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2014</td>
      <td>ACURA</td>
      <td>ILX HYBRID</td>
      <td>COMPACT</td>
      <td>1.5</td>
      <td>4</td>
      <td>AV7</td>
      <td>Z</td>
      <td>6.0</td>
      <td>5.8</td>
      <td>5.9</td>
      <td>48</td>
      <td>136</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2014</td>
      <td>ACURA</td>
      <td>MDX 4WD</td>
      <td>SUV - SMALL</td>
      <td>3.5</td>
      <td>6</td>
      <td>AS6</td>
      <td>Z</td>
      <td>12.7</td>
      <td>9.1</td>
      <td>11.1</td>
      <td>25</td>
      <td>255</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2014</td>
      <td>ACURA</td>
      <td>RDX AWD</td>
      <td>SUV - SMALL</td>
      <td>3.5</td>
      <td>6</td>
      <td>AS6</td>
      <td>Z</td>
      <td>12.1</td>
      <td>8.7</td>
      <td>10.6</td>
      <td>27</td>
      <td>244</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['MODELYEAR'].unique()
```




    array([2014])



There is only 1 MODELYEAR, hence it makes no sense to include it as a parameter


```python
df = df.drop(['MODELYEAR'], axis=1)
```

We use labelencoder to assign unique values to all the categorical data types as we cant operate on variables that are string.


```python
df['MAKE'], df['MODEL'], df['VEHICLECLASS'], df['TRANSMISSION'], df['FUELTYPE'] = \
labelencoder.fit_transform(df['MAKE']), labelencoder.fit_transform(df['MODEL']), labelencoder.fit_transform(df['VEHICLECLASS']), \
labelencoder.fit_transform(df['TRANSMISSION']), labelencoder.fit_transform(df['FUELTYPE'])
```

Next, we filter out the variables that have less correlation with the target parameter - CO2EMISSIONS


```python
cor = df.corr()
cor_target = abs(cor["CO2EMISSIONS"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
relevant_features
```


    ENGINESIZE                  0.874154
    CYLINDERS                   0.849685
    FUELCONSUMPTION_CITY        0.898039
    FUELCONSUMPTION_HWY         0.861748
    FUELCONSUMPTION_COMB        0.892129
    FUELCONSUMPTION_COMB_MPG    0.906394
    CO2EMISSIONS                1.000000
    Name: CO2EMISSIONS, dtype: float64


dropping the columns with low correlation, ones below 0.5

```python
df = df.drop(['MAKE', 'MODEL', 'VEHICLECLASS', 'FUELTYPE', 'TRANSMISSION'], axis=1)
```

```python
df.head()
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENGINESIZE</th>
      <th>CYLINDERS</th>
      <th>FUELCONSUMPTION_CITY</th>
      <th>FUELCONSUMPTION_HWY</th>
      <th>FUELCONSUMPTION_COMB</th>
      <th>FUELCONSUMPTION_COMB_MPG</th>
      <th>CO2EMISSIONS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2.0</td>
      <td>4</td>
      <td>9.9</td>
      <td>6.7</td>
      <td>8.5</td>
      <td>33</td>
      <td>196</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2.4</td>
      <td>4</td>
      <td>11.2</td>
      <td>7.7</td>
      <td>9.6</td>
      <td>29</td>
      <td>221</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.5</td>
      <td>4</td>
      <td>6.0</td>
      <td>5.8</td>
      <td>5.9</td>
      <td>48</td>
      <td>136</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3.5</td>
      <td>6</td>
      <td>12.7</td>
      <td>9.1</td>
      <td>11.1</td>
      <td>25</td>
      <td>255</td>
    </tr>
    <tr>
      <td>4</td>
      <td>3.5</td>
      <td>6</td>
      <td>12.1</td>
      <td>8.7</td>
      <td>10.6</td>
      <td>27</td>
      <td>244</td>
    </tr>
  </tbody>
</table>
</div>

We use min-max normalization. This step is really important. You could skip the following cell and try out all other cells to  get some absurd value as the MSE error


```python
df=(df-df.min())/(df.max()-df.min())
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENGINESIZE</th>
      <th>CYLINDERS</th>
      <th>FUELCONSUMPTION_CITY</th>
      <th>FUELCONSUMPTION_HWY</th>
      <th>FUELCONSUMPTION_COMB</th>
      <th>FUELCONSUMPTION_COMB_MPG</th>
      <th>CO2EMISSIONS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.135135</td>
      <td>0.111111</td>
      <td>0.207031</td>
      <td>0.115385</td>
      <td>0.180095</td>
      <td>0.448980</td>
      <td>0.231579</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.189189</td>
      <td>0.111111</td>
      <td>0.257812</td>
      <td>0.179487</td>
      <td>0.232227</td>
      <td>0.367347</td>
      <td>0.297368</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.067568</td>
      <td>0.111111</td>
      <td>0.054688</td>
      <td>0.057692</td>
      <td>0.056872</td>
      <td>0.755102</td>
      <td>0.073684</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.337838</td>
      <td>0.333333</td>
      <td>0.316406</td>
      <td>0.269231</td>
      <td>0.303318</td>
      <td>0.285714</td>
      <td>0.386842</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.337838</td>
      <td>0.333333</td>
      <td>0.292969</td>
      <td>0.243590</td>
      <td>0.279621</td>
      <td>0.326531</td>
      <td>0.357895</td>
    </tr>
  </tbody>
</table>
</div>

```python
y,x = df["CO2EMISSIONS"], df.drop(["CO2EMISSIONS"], axis=1)
```
splitting the dataset into train and test functions

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

h = np.dot(X, theta)
h.shape
```
    (853, 1)

```python
temp = np.zeros(theta.shape)
cost = np.sum (np.dot(np.transpose(h-Y), (h-Y)))*(1/(2*X.shape[0]))
```

```python
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

    28.955263880454513
    [[0.92391923]
     [0.92391923]
     [0.92391923]
     [0.92391923]
     [0.92391923]
     [0.92391923]
     [0.92391923]
     [0.92391923]]
    0.013847910580828569
    [[0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]]
    0.013847910580828569
    [[0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]]
    0.013847910580828569
    [[0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]]
    0.013847910580828569
    [[0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]]
    0.013847910580828569
    [[0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]]
    0.013847910580828569
    [[0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]]
    0.013847910580828569
    [[0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]]
    0.013847910580828569
    [[0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]]
    0.013847910580828569
    [[0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]
     [0.04899041]]

```python
theta
```
    array([[0.04899041],
           [0.04899041],
           [0.04899041],
           [0.04899041],
           [0.04899041],
           [0.04899041],
           [0.04899041],
           [0.04899041]])

predicting the y values for the test dataset and evaluating our model with the MSE function

```python
X_test = []
for row in x_test:
    r = [1]
    for item in row:
        r.append(item)
    X_test.append(r)
```

```python
mean_squared_error(np.dot(X_test, theta), y_test)
```

    0.05530668773441035

You can achieve way better results using the built-in sklearn function !
Thanks for reading :)
