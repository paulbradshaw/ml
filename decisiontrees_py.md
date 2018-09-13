# Decision Trees and `sklearn.tree` in Python: notes

*This file contains notes from the [Udemy course Machine Learning A-Z™: Hands-On Python & R In Data Science](https://www.udemy.com/machinelearning/) - specifically [this video](https://www.udemy.com/machinelearning/learn/v4/t/lecture/5846962?start=0). [Download the data and files from here](https://www.superdatascience.com/machine-learning/)*


## Setting the working directory

As always, make sure the right working directory is set so that files are imported without error:

```py
#Set working directory or files won't be found
import os
os.chdir("/Users/paul/Dropbox/workInProgress/ML/Machine Learning A-Z/Part 2 - Regression/Section 8 - Decision Tree Regression")
```

## Using the `sklearn.tree` library

The [DecisionTreeRegressor class](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) from `sklearn.tree` can be used to create a decision tree regressor.

You can find [more about this here](https://cambridgespark.com/content/tutorials/from-simple-regression-to-multiple-regression-with-decision-trees/index.html).

The `fit` method from that class is used to fit the data to the regressor.

```py
# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
#default = "mse" - mean squared error
#random_state specified so we get same result as tutor
regressor = DecisionTreeRegressor(random_state = 0)
#Fit the regression to the dataset X and y (dependent variable vector)
regressor.fit(X, y)
```

A prediction for a data point of `6.5` is produced using:

```py
# Predicting a new result
# Based on independent variable we have: a level of 6.5
y_pred = regressor.predict(6.5)
```

## Visualising the results: intervals, not slopes

We make sure that the results are shown at higher resolution to prevent lines being drawn between prediction points, which would make curves or slopes (not appropriate for a non-liner and non-continuous models).

Instead we should see **intervals**, shown as *steps* in the chart.

```py
# Visualising the Decision Tree Regression results (higher resolution)
#Steps indicate single values (averages) rather than slopes, which would be shown at lower resolution
#It is a NON CONTINUOUS regression model
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
```

## Other notes

In the example we are *not* creating a training set because the dataset is too small.

We are not using a *linear* method because the data is *not linear* (it increases much more as it goes up levels).

We are not using feature scaling because decision trees are not based on Euclidean distances but rather on conditions in the independent variables.
