# Decision Trees and `rpart` in R: notes

*This file contains notes from the [Udemy course Machine Learning A-Z™: Hands-On Python & R In Data Science](https://www.udemy.com/machinelearning/) - specifically [this video](https://www.udemy.com/machinelearning/learn/v4/t/lecture/5846964?start=0). [Download the data and files from here](https://www.superdatascience.com/machine-learning/)*

[More on tree-based models in R can be found here](https://www.statmethods.net/advstats/cart.html)

## Setting the working directory

As always, make sure the right working directory is set so that files are imported without error:

`setwd("/Users/paul/Dropbox/workInProgress/ML/Machine Learning A-Z/Part 2 - Regression/Section 8 - Decision Tree Regression")`

## Using the `rpart` package

The [`rpart` *package*](https://cran.r-project.org/web/packages/rpart/rpart.pdf) is designed to do "Recursive partitioning for classification, regression and survival trees".

You can find a 62-page [Introduction to Recursive Partitioning Using the RPART Routines](https://cran.r-project.org/web/packages/rpart/vignettes/longintro.pdf) which provides much more guidance on the package.

It contains a *function* (also) called `rpart`. We use that to create a regressor:

```r
# Fitting Decision Tree Regression to the dataset
# install.packages('rpart')
library(rpart)
regressor = rpart(formula = Salary ~ .,
                  data = dataset,
                  control = rpart.control(minsplit = 1))
```

The `formula = Salary ~ .` parameter specifies that `Salary` is the dependent variable and all the others (indicated by a period `.`) are independent variables, with the two separated by a tilde: `~`.

The `control` parameter is added to specify a minimum number of splits in the model. Without this, the code just generates one overall average (249500 in this case). But with `rpart.control` (a function from rpart) and the argument `(minsplit = 1)` we ensure it has at least one split.

A prediction for a data point of `6.5` is produced using:

```r
# Predicting a new result with Decision Tree Regression
y_pred = predict(regressor, data.frame(Level = 6.5))
```

## Visualising the results: intervals, not slopes

We make sure that the results are shown at higher resolution to prevent lines being drawn between prediction points, which would make curves or slopes (not appropriate for a non-liner and non-continuous models).

Instead we should see **intervals**, shown as *steps* in the chart.

```r
# Visualising the Decision Tree Regression results (higher resolution)
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Decision Tree Regression)') +
  xlab('Level') +
  ylab('Salary')
```

## Other notes

In the example we are *not* creating a training set because the dataset is too small.

We are not using a *linear* method because the data is *not linear* (it increases much more as it goes up levels).

We are not using feature scaling because decision trees are not based on Euclidean distances but rather on conditions in the independent variables.
