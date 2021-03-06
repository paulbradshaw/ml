# Machine learning - notes

This file contains notes from the [Udemy course Machine Learning A-Z™: Hands-On Python & R In Data Science](https://www.udemy.com/machinelearning/).

[Download the data and files from here](https://www.superdatascience.com/machine-learning/)

## Setting the working directory

To set the working directory you can use the following code:

```py
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

```py
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

```py
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

```py
# Avoiding the Dummy Variable Trap
X = X[:, 1:] #get rid of one of the columns (the first)
```

## Split the independent and dependent variables into a training and test set

Having split one dataset into another 2, now we split those again to get another 4. To do this we import the `sklearn` library - specifically: `from sklearn.cross_validation import train_test_split`.

When using `train_test_split` we specify how big the test set is going to be. Typically it is 0.2 but in smaller datasets we may use larger proportions.

```py
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
```

For the multiple regression it's almost the same, but we have chosen a different test size because the dataset is larger.

```py
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

## Running the regressor

Now that we have a training and a test set, we can run a regressor on the training set (keeping the test set for later, to... test).

We import another function from `sklearn` to do this: `from sklearn.linear_model import LinearRegression`

```py
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

```py
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

```py
# Predicting the Test set results
#Use the .predict method of the object on the test set variable
y_pred = regressor.predict(X_test)
```

Again the code is the same for simple and multiple linear regression.

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
