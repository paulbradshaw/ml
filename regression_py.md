# Machine learning - notes

This file contains notes from the [Udemy course Machine Learning A-Z™: Hands-On Python & R In Data Science](https://www.udemy.com/machinelearning/).

[Download the data and files from here](https://www.superdatascience.com/machine-learning/)

## Setting the working directory

To set the working directory you can use the following code:

```
#Import the os library
import os
#Use the getcwd function from that library to store the current working directory
cwd = os.getcwd()
#Store the path to the new directory
newpath = "/Users/paul/Dropbox/workInProgress/ML/Machine Learning A-Z/Part 2 - Regression/Section 5 - Multiple Linear Regression"
#Change the working directory to the new path
os.chdir(newpath)
#Check it
os.getcwd()
#Remove it
del(cwd)
```

## Import the libraries

We need 3 basic libraries first: `numpy` to do calculations; `matplotlib` to plot charts and `pandas` to do data importing and analysis. Later we will also import parts of the `sklearn` library, for the regression.

```
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

## Import the data and split it into dependent and independent variables

The data is in the same working directory as your Python file. We use `read_csv` from the `pandas` library (renamed `pd` when imported above).

We then use `iloc` to grab the values of the independent variable (1 column) and put in a new variable `X`; and to grab the values of the dependent variable(s) and put in a new variable called `y`.

Here is the simple linear regression code - note that there's only one column (variable) for both X and y:

```
# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
#put the independent variables in one matrix - experience
#iloc takes rows (: means all)
#and columns (all up to but not including the last one)
#then we extract the values and put in new variable X
X = dataset.iloc[:,:-1].values
#put the dependent variables - wage - in another     matrix
#as above, but all rows and just column index 1
y = dataset.iloc[:, 1].values #may need different number if more dependent variables
```

Here is the multiple linear regression code - note that there's only one column (variable) for X but 4 for y (i.e. multiple):

```
# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
#Select the last column as the dependent variable
X = dataset.iloc[:, :-1].values
#How many columns are the independent variables? From the first to the 4th
y = dataset.iloc[:, 4].values
```

## Encoding categorical data

In one of the examples (multiple regression) some of the data is categorical, so we need to convert it into numeric data. To do this we import the `sklearn` library - specifically two *classes* of encoder: `from sklearn.preprocessing import LabelEncoder, OneHotEncoder`.

We then specify the categorical column and run the `fit_transform` method on it.

```
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#create a LabelEncoder object
labelencoder = LabelEncoder()
#The categories we want to encode are in column 3
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
#When this code is run, X should now have 3 columns instead of the category column
```

Because we now have 3 new numerical columns that represent the previous categorical column, we also need to remove *one* of those to avoid the **Dummy Variable Trap**:

```
# Avoiding the Dummy Variable Trap
X = X[:, 1:] #get rid of one of the columns (the first)
```

## Split the independent and dependent variables into a training and test set

Having split one dataset into another 2, now we split those again to get another 4. To do this we import the `sklearn` library - specifically: `from sklearn.cross_validation import train_test_split`.

When using `train_test_split` we specify how big the test set is going to be. Typically it is 0.2 but in smaller datasets we may use larger proportions.

```
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
```

For the multiple regression it's almost the same, but we have chosen a different test size because the dataset is larger.

```
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

## Running the regressor

Now that we have a training and a test set, we can run a regressor on the training set (keeping the test set for later, to... test).

We import another function from `sklearn` to do this: `from sklearn.linear_model import LinearRegression`

```
# Fitting simple linear regression to the training set
#import the library
from sklearn.linear_model import LinearRegression
#create object in that class
regressor = LinearRegression()
#Use the .fit method from that class
#First parameter is X, then y - calculate (learn) the correlation
regressor.fit(X_train,y_train)
```

For multiple regression the code is the same:

```
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
#Create a LinearRegression object
regressor = LinearRegression()
#Use the .fit method from that object on the two training variables
regressor.fit(X_train, y_train)
```

## Testing the regression on the test set

Now we can use the `predict` method of the regressor on the *test* dataset for X to predict the results of Y based on the relationship that it calculated with the training set. The results of this prediction are stored in `y_pred`. So we now have these variables:

* The full dataset
* 2 separate datasets for the dependent variable (y) and the independent variable(s) (X) - any categorical values will have been encoded, and one of those results removed to avoid the dummy variable trap
* Separate training and test datasets for both the dependent and independent variable (4 in total)
* A *prediction* dataset showing what the dependent variable (y) *should* be, given new independent variable values (the test set for y), based on a regression analysis of the training set.

```
# Predicting the Test set results
#Use the .predict method of the object on the test set variable
y_pred = regressor.predict(X_test)
```

Again the code is the same for simple and multiple linear regression.

## Plotting the results for simple linear regression

With a simple linear regression we can plot the results too. First the training set, and then the predictions for the test set compared against the actual y values:

```
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

```
#Can we show the difference?
y_diff = y_test - y_pred
np.median(y_diff)
np.mean(y_diff)
```
