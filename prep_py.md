# Machine learning - notes

This file contains notes from the [Udemy course Machine Learning A-Z™: Hands-On Python & R In Data Science](https://www.udemy.com/machinelearning/).

[Download the data and files from here](https://www.superdatascience.com/machine-learning/)

## Data preprocessing

Some stages:

* Distinguish between **independent and dependent variables**
* Replace missing values with **imputation**
* Turn **categorical data** into numerical data
* Split into a **training set and test set**
* Get values on the same scale with **feature scaling**

Distinguish between **independent and dependent variables**: in the example dataset, demographics (country, age, earnings) are the independent variables and actions (Purchased: yes or no) are the depending variables.

Once you've imported that data into a variable, you may need to separate independent and dependent variables. In Python's `pandas` package, you access the columns and rows using `.iloc` with the rows specified first, then the columns. If you want *all* rows (or columns) use the colon `:`.

```py
# Importing the dataset
dataset = pd.read_csv('Data.csv')
#put the independent variables in one matrix
#iloc takes rows (: means all)
#and columns (all up to but not including the last one)
#then we extract the values and put in new variable X
X = dataset.iloc[:,:-1].values
#put the dependent variables in another     matrix
#as above, but all rows and just column index 3
y = dataset.iloc[:, 3].values
```

### Classes, objects and methods

[From the course](https://www.udemy.com/machinelearning/learn/v4/t/lecture/5859706?start=0):

> "A class is the model of something we want to build. For example, if we make a house construction plan that gathers the instructions on how to build a house, then this construction plan is the class.

> "An object is an instance of the class. So if we take that same example of the house construction plan, then an object is simply a house. A house (the object) that was built by following the instructions of the construction plan (the class). And therefore there can be many objects of the same class, because we can build many houses from the construction plan.

> "A method is a tool we can use on the object to complete a specific action. So in this same example, a tool can be to open the main door of the house if a guest is coming. A method can also be seen as a function that is applied onto the object, takes some inputs (that were defined in the class) and returns some output.""

## Replacing missing values - imputation

Missing values cause problems, so typically they are replaced with the mean, median or mode value for the dataset as a whole, a process called **imputation**.

*Note: I came across this [chapter on Missing-data imputation ](http://www.stat.columbia.edu/~gelman/arm/missing.pdf) which tackles the topic in more depth*

### Imputation using Python's `scikit` library

[The `scikit` library](http://scikit-learn.org/stable/) (imported as `sklearn`) has a class `Imputer` with properties that can be used to identify what defines `missing_values` and what `strategy` should be used to replace them (mean, median or mode (`most_frequent`)). That is used to create a variable which is then used to transform missing values using the code below:

```py
#import the Imputer class from the sklearn library to handle missing data
from sklearn.preprocessing import Imputer
#specify what the missing values are (Nan)...
#...what strategy to use (apply the mean - alternatives are 'median' or 'most_frequent')
#Press CMD+I in front of Imputer to see Help documentation in window
#...and what axis
imputer = Imputer(missing_values = "NaN", strategy = 'mean', axis = 0)
#use that imputer to fit values from X: all rows (:) and columns index 1-2
imputer = imputer.fit(X[:, 1:3])
#This does the actual transformation based on the settings above
X[:,1:3] = imputer.transform(X[:, 1:3])
```



## Categories - encoding categorical variables

We need to encode text (categories) into numbers in order to include them in formulae.

### Encoding categorical data (encoding labels) in Python using `scikit`

In Python you again use `scikit` - [specifically `LabelEncoder`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html). First, to import it:

```py
#import the LabelEncoder class from scikit:
from sklearn.preprocessing import LabelEncoder
```

Next, to use it:

```py
#Converting labels into values so they can be used in formulae
#labelencoder_X is an arbitrary name for the variable we created as a LabelEncoder object
labelencoder_X = LabelEncoder()
#Because it's an LE object we can use methods for that
#the fit_transform takes a (categories) column from our data object X
#this is then reapplied to the same column to replace it
X[:,0] =labelencoder_X.fit_transform(X[:,0])
#Output: array([0, 2, 1, 2, 1, 0, 2, 0, 1, 0])
#This is encoding countries as numbers: 0, 1 and 2 for the 3 values Germany, France and Spain
```

The result transforms the countries into numbers - 0, 1 and 2 because there are 3 different countries. But this is problematic because these **numbers have quantitative relationships**: 2 is greater than 1, and so on.

Instead, then, **dummy encoding** can be used to represent *each* country with the same number - 1 - but in *separate* columns. In other words, the 1 or 0 becomes a true/false marker against each country column.

## Dummy encoding with `OneHotEncoder`

First, to import it:

```py
#import the LabelEncoder AND OneHotEncoder classes from scikit:
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
```

The [documentation for OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) specifies its parameters: `categorical_features` needs to "Specify what features are treated as categorical". In this case we are specifying the first column, index 0, so we add `categorical_features = [0]`

```py
#create a OneHotEncoder object - called onehotencoder
#This will convert the one column of 0, 1 and 2 into 3 columns each with 1 or 0 = T or F
onehotencoder = OneHotEncoder(categorical_features = [0])
#transform X using that object, to an array, and assign back to X
#index 0 will be used because we already specified it in the line above
X = onehotencoder.fit_transform(X).toarray()
```

Note that this is *after* X has had its first column transformed into numbers(0,1,2) - otherwise it throws an error.

Now the X dataset has 3 columns where there was just one previously: columns 0, 1 and 2 contain either 0 or 1 depending on whether that row contained the relevant value.

For a *dependent* variable we don't need to do this. Here's the code:

```py
#Now transform the Yes & No column into numerical labels
#First create the LabelEncoder object
labelencoder_y = LabelEncoder()
#Now use the .fit_transform method on it to transform y and assign the transformation back to y
y = labelencoder_y.fit_transform(y)
```

## Splitting into training set and test set - Python

The code below uses a library to split our full dataset into a training and a test dataset.

The training set will be used to 'learn' a relationship between x (independent) and y (dependent).

This 'learned' relationship will then be tested against the test dataset - how accurately does it predict the results in the data which was left out?

It sets the `test_size` parameter to 0.2 or 20%, meaning 20% of the full data will go into the test dataset and 80% into the training. This is quite typical, but can be 0.25 or 0.3

```py
#import train_test_split library from sklearn
from sklearn.cross_validation import train_test_split
#Create 4 new variables for x and y (independent and dependent) - 2 for test and 2 for train
#use train_test_split - this takes arrays (X and y here), sets the size of the test set, i.e. 20% of the data will go into each set
#random_state is used to ensure our results are the same as the person running the MOOC
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```


## Feature scaling

Feature scaling is a solution to the problem of having data on different scales. Without this, machine learning would be dominated by data points within bigger ranges (e.g. thousands of pounds) having a bigger influence that those within smaller ranges (e.g. ages, weights).

There are two broad types of feature scaling: standardisation, and normalisation. Some [literature on these methods can be found here](http://sebastianraschka.com/Articles/2014_about_feature_scaling.html).

```py
# Import the class that we need to use
from sklearn.preprocessing import StandardScaler
#Create an object of that class - we're calling it X to indicate the independent variable
sc_X = StandardScaler()
#Call the method fit_transform which will fit it AND transform it
X_train = sc_X.fit_transform(X_train)
#Repeat for test dataset - this time no need for fitting because sc_X was fitted in the line above and we need to fit it on the same basis
X_test = sc_X.transform(X_test)
```

We will need to apply feature scaling to the y (dependent) variable as well in regression analysis.
