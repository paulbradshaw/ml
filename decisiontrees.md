# Decision trees

The two types of decision trees are **Classification Trees** and **Regression Trees**. The acronym **CART** helps remember them (Classification And Regression Tree).

## Regression trees

Regression Trees are more complex. They split your variables (the scatterplot) into different parts (*leaves*), using an algorithm looking at **information entropy**: the key concept behind this is whether a split *adds* information; the algorithm stops when it cannot add any more meaningful information with any more splits.

Each split creates a *branch* in the decision tree.

For example if your first split is at the 20 point on one axis (X1), the branch for that would be X1<20 (TRUE or FALSE). In other words, the first branch splits the data into whether they fall before or after that line.

Next, the second split is made *within* one of those leaves so that second branch is *only* placed on the appropriate branch. So for example if the split line is drawn for the >20 area and it is drawn where X2 is 170, then the test in that branch would be X2<170.

Here are those splits shown on a scatterplot:

![](/svr_scatter.png)

And then how those are represented as a series of branches:

![](/svr_tree.png)

The last leaf that a datapoint falls in (after following branches) is called a *terminal leaf*.

## Making the prediction

Remember at this point that our objective is to predict a value - the dependent variable - when given other values (in this case X1 and X2, two independent variables).

This prediction is based on an *average* of dependent variable values *within the terminal leaf*. So if the average of Y for X1 and X2 values within terminal leaf 4 is 32, then that's what prediction will be made.

In the completed decision tree below, then, you can see the average values of Y given as predictions in response to the answers to the series of questions, which place a value within a leaf in order to allocate the right predicted value (the average for Y values based on previous X values):

![](/svr_predictions.png)

Note that the scatterplot is just a conceptual aid for this example: we may have more than 2 independent variables so it's best to imagine this as a multi-dimensional space (2 dimensions for 2 independent variables, 3-dimensional when the dependent variable is factored in, and more if there are more independent variables).

## Visualising

Decision tree regression does not produce a continuous prediction (unlike linear regression etc.): it is **non-continuous**. When plotted at high resolution, for example, you should see a series of steps (one step for each average value) rather than a curved line. If you get a curved line it may be worth increasing the resolution.

In the code below the line `X_grid = np.arange(min(X), max(X), 0.01)` is where this happens - `0.01` is the resolution, which is better than 0.1 for example.

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

This code only works in 2D regression - not for more variables.
