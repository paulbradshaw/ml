# Machine learning - notes on Support Vector Machine Regression (SVR or SVM)

You can [find out more about Support Vector Machine Regression here](https://uk.mathworks.com/help/stats/understanding-support-vector-machine-regression.html):

> "Support vector machine (SVM) analysis is a popular machine learning tool for classification and regression... SVM regression is considered a nonparametric technique because it relies on kernel functions."

This file contains notes from the [Udemy course Machine Learning A-Z™: Hands-On Python & R In Data Science](https://www.udemy.com/machinelearning/).

[Download the data and files from here](https://www.superdatascience.com/machine-learning/)

## Setting the working directory

To set the working directory you can use the following code:

```py
#Import the os library
import os
#Replace with the path
newpath = "/Users/paul/Dropbox/workInProgress/ML/Machine Learning A-Z/Part 2 - Regression/Section 5 - Multiple Linear Regression"
#Change the working directory to the new path
os.chdir(newpath)
```

## Import the libraries

We need 3 basic libraries first: `numpy` to do calculations; `matplotlib` to plot charts and `pandas` to do data importing and analysis. Later we will also import parts of the `sklearn` library, for the regression.

```py
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

## Import the data and split it into dependent and independent variables

The data is in the same working directory as your Python file. We use `read_csv` from the `pandas` library (renamed `pd` when imported above).

We then use `iloc` to grab the values of the independent variable (1 column) and put in a new variable `X`; and to grab the values of the dependent variable(s) and put in a new variable called `y`.

Here is the simple linear regression code - note that there's only one column (variable) for both X and y:

```py
# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
#put the independent variables in one matrix
#iloc takes all rows (:) and column 1 (the last index in the range is not included)
#then we extract the values and put in new variable X
X = dataset.iloc[:, 1:2].values
#put the dependent variables in another matrix
#as above, but all rows and just column index 2
y = dataset.iloc[:, 2].values #may need different number if more dependent variables
```

## Use feature scaling

The library we are going to use doesn't have feature scaling built in, so we need to uncomment any feature scaling lines and apply those to the data.

```py
# Feature Scaling
from sklearn.preprocessing import StandardScaler
#create a scaler object for X and for y
sc_X = StandardScaler()
sc_y = StandardScaler()
#Use the fit_transform method of each to create a scaled version
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
```


## Using the `sklearn.svm` library for Support Vector Regression (SVR)

The `sklearn.svm` library (the SVM is for Support Vector Machine) is what we use to create a Support Vector Regression (SVR). To simplify things we call it `SVR` when we import it.

```py
# Fitting SVR to the dataset
from sklearn.svm import SVR
#Create a SVR object - specify what type of kernel we want
regressor = SVR(kernel = 'rbf')
#Fit our regressor to the data
regressor.fit(X, y)
```

We create a regressor object which is an `sklearn.svm` object. We need to specify which **kernel** type to use.

A kernel is a set of mathematical functions used by the SVM algorithm. In this case `rbf` is specified - it is the most common type - but we could choose others depending on the problem. This [introduction to SVM kernel types](https://data-flair.training/blogs/svm-kernel-functions/) goes into more detail: for example a `linear` kernel might be used with a linear problem, while a **gaussian** kernel (of which `rbf` is one type) is used "when there is no prior knowledge about the data".












## Plotting the results for simple linear regression

With a simple linear regression we can plot the results too. First the training set, and then the predictions for the test set compared against the actual y values:

```py
#Visualise the training set results
#Use scatter to draw a scatterplor; specify the x and y coordinates
plt.scatter(X_train, y_train, color='red')
#Add a plotted line
plt.plot(X_train, regressor.predict(X_train), color='blue')
#Add labels
plt.title('Salary vs experience (training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
#Now show the results
plt.show()

#The red dots will show the real values, the blue line will show the regression results

#Visualise the test set results
#Use scatter to draw a scatterplor; specify the x and y coordinates
plt.scatter(X_test, y_test, color='red')
#Add a plotted line - this is STILL the training data
plt.plot(X_train, regressor.predict(X_train), color='blue')
#Add labels
plt.title('Salary vs experience (test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
#Now show the results
plt.show()

#The red dots will show the real values, the blue line will show the regression results
```

## Checking the results of a multiple regression

We can show the difference between the actual y values in the test set and the predicted y values by using some simple `median` and `mean` functions from `numpy` (named `np` in our code).

```py
#Can we show the difference?
y_diff = y_test - y_pred
np.median(y_diff)
np.mean(y_diff)
```

## Creating a polynomial linear regression

A polynomial linear regression is used where data points don't trend in a straight line but instead curve in some way, e.g. increasing exponentially.

Much of the code is the same: for example, importing the same libraries, importing the data, and splitting it into the dependent and independent variables:

```py
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

In this case each only takes up one column - we don't need the level name and level number both importing because they both refer to the same property. So we just import the numerical 'level' column.

```py
# Importing the dataset
#Plotting level against salary shows a curved, parabolic relationship
#Predict if an applicant's claimed £160k salary on a 6.5 level is likely or not
dataset = pd.read_csv('Position_Salaries.csv')
#independent variable is level: index 1
#To make sure it is a matrix not a vector we use 1:2 which means it is a matrix of one column
X = dataset.iloc[:, 1:2].values
#dependent variable is salary: index 2
y = dataset.iloc[:, 2].values
```

We skip splitting into training and test dataset in this example because a) the dataset is too small and we need all of it to be comprehensive (covering all 10 levels) and b) we're trying to predict something outside of this dataset anyway (a claim).

We start with the same linear regression - importing the `LinearRegression` library and then creating a `LinearRegression` *object* using the function `LinearRegression()`. That object is then *fitted* to the dependent and independent variables (matrices) that we extracted:

```py
# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
#Create the first regressor
lin_reg = LinearRegression()
#Fit the regressor to the X and y variables (no training/test version)
lin_reg.fit(X, y)
```

Next we repeat the process with the `PolynomialFeatures` library and create a `PolynomialFeatures` *object* using `PolynomialFeatures()` - this function has a `degree =` parameter to specify how many powers you want to operate to, e.g. `degree = 1` would do nothing but replicate the same information; `degree = 2` would square all the numbers and create new values for those, meaning two columns in total; `degree = 3` would add cubed values, three columns in total, and so on.

Then, `X_poly = poly_reg.fit_transform(X)` fits that model to the independent variable, and creates a new matrix, `X_poly`, containing the new values (the original independent variables, transformed to the 2nd, 3rd and 4th power).

```py
# Fitting Polynomial Regression to the dataset
#imported from preprocessing library
from sklearn.preprocessing import PolynomialFeatures
#create a PolynomialFeatures 'object'
#This will create NEW independent variables based on the original indepdendent variable
#To the power of 2, 3, and so on - in this case up to 4
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
```

**You can change this code and re-run the fitting and graphing below to test different models.**

Now that we have *more* independent variables, we can create a *polynomial linear regression* fitted to that matrix.

```py
#fit the object to the new matrix and the dependent variable
poly_reg.fit(X_poly, y)
```

But we also need to create a *new* linear regression

```py
#create a new linear regression object
lin_reg_2 = LinearRegression()
#Fit that object to the new matrix and dependent variable too, to compare
lin_reg_2.fit(X_poly, y)
```

### Visualising the predictions

First we plot the results against the predictions of the linear regression model. The key code here is `lin_reg.predict(X)` which runs that linear regression object's `predict` method on `X`:

```py
# Visualising the Linear Regression results
#plot the real results
plt.scatter(X, y, color = 'red')
#plot the linear regression model predictions
plt.plot(X, lin_reg.predict(X), color = 'blue')
#Add title, labels etc.
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
#Show the plot
plt.show()
```

Next we do the same for the polynomial linear regression model. Likewise the key part is `lin_reg_2.predict(poly_reg.fit_transform(X)` which runs that linear regression object's `predict` method (remember `lin_reg_2` is a model fitted to `X_poly`) on the results of using `poly_reg`'s `fit_transform` method on `X` (which is how we originally created `X_poly` itself):

```py
# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
#Note that in the most deeply nested part of this line
#we are repeating what we did with X_poly = poly_reg.fit_transform(X)
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
```

Try changing the line `poly_reg = PolynomialFeatures(degree = 2)` to different degrees and then re-run the code underneath to test different powers.

### Predicting against a single value

We've already used `.predict` to predict based on a matrix, but we can use it to predict based on one value - in this case 6.5. First, with normal linear progression:

```py
# Predicting a new result with Linear Regression
lin_reg.predict(6.5)
```

...And then, with the `poly_reg` object which is using polynomial regression and its `fit_transform` method. Note that this code is the same as that used to draw, too: `lin_reg_2.predict(poly_reg.fit_transform(X))`

```py
# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))
```
