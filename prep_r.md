*See the readme.md file for intro and Python*

### Imputation using R and `ifelse`

Replacing missing values in R is simpler: you just need to use [the `ifelse` function](https://www.datamentor.io/r-programming/ifelse-function) to specify that *if* a cell contains a missing value such as `NA` it should be replaced by a mean (or median or mode). Here's the code:

```r
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN= function(x) mean(x, na.rm = TRUE)),
                     dataset$Age
                     )
```

Like the `IF` function in Excel, `ifelse` takes 3 parameters: a test which will return true or false; what you want to do if the test returns true; and what to do if it returns false.

In the code above the test is `is.na(dataset$Age)`. In other words: if (any) value in the `dataset$Age` field is `NA` (again, there's an Excel equivalent function here, `ISNA`).

If that is true, the code executes the code `ave(dataset$Age, FUN= function(x) mean(x, na.rm = TRUE)`. Back to this in a second.

If `is.na(dataset$Age)` is false, we simply fetch `dataset$Age`.

This needs breaking down. The function `ave` calculates an average. The code `ave(dataset$Age)` would calculate an average of that column. *However*, if the column contains any `NA` errors then the average will also return an `NA` error.

To stop this from happening, we need to specify an *extra* parameter, `FUN = `. This specifies a function to run as part of the calculation.

This function takes `(x)` as its argument: `x` [is the variables being handled by the function](https://www.rdocumentation.org/packages/stats/versions/3.4.3/topics/ave) (the ages, in this case). Next it calculates a `mean` for those variables, but with an important extra parameter: `na.rm = TRUE`. This means *remove any `NA` values* from the calculation (I'm not sure if 'rm' is short for 'remove' but it is at least easy to remember that way).

This prevents the basic `ave()` function from returning `NA`. [A more detailed exploration of NA handling can be found here](https://thomasleeper.com/Rcourse/Tutorials/NAhandling.html)

The results of the `ifelse` function are put in `dataset$Age`. Or, put another way, *for each value in dataset$Age*, the `ifelse` function runs. If the value is an `NA` it is replaced with the results of that `ave()` function which ignores NA values; if the value is *not* (false) then it is replaced with... itself. In other words there is no change; it will only change the NA values.

## Categories - encoding categorical variables

We need to encode text (categories) into numbers in order to include them in formulae.


### Encoding categorical data (encoding labels) in R using factors

In R we don't need to transform one column into 3 TRUE/FALSE columns as in Python. Instead we can transform one column to contain *factors* which we specify.

Here's the code - it uses the `factor` function, which takes 3 parameters:

* the column or vector it is converting;
* a vector specifying the different 'levels' in that - in the example below we specify it manually but this could be generated using `unique(dataset$Country)`.
* a vector of labels to replace those levels with - again below these are entered manually but a longer list could be generated using `seq` with the upper limit being the `length` of the vector containing the levels, e.g. `seq(1,length(unique(dataset$Country)))`

Here's the simple version:

```r
#Encoding categorical data
dataset$Country = factor(dataset$Country,
                         levels = c('France','Spain','Germany'),
                         labels = c(1,2,3)
                         )
dataset$Purchased = factor(dataset$Purchased,
                        levels = c('No','Yes'),
                        labels = c(0,1)
                        )
```

## Splitting into training set and test set - in R

We need to install [the caTools package](https://cran.r-project.org/package=caTools) first.

```r
# Splitting the dataset into the Training set and Test set
install.packages('caTools')
library(caTools)
```

Where in Python we set `random_state = 0` to ensure we get the same result as the person sending the code to us (in that case, the MOOC instructor), in R we *set the seed* to do the same - this is a number (chosen by you) which is used as a seed - as long as both parties use the same number they should get the same split:

```r
#This number is the same as the person running the MOOC, so we get the same results
set.seed(123)
```

Next to split the data into training and test sets. We do this using a method from caTools called [`sample.split`](https://www.rdocumentation.org/packages/caTools/versions/1.17.1/topics/sample.split). It only requires the specification of Y (dependent variable), not X. In this case our dependent variable is `dataset$Purchased` (we are trying to find out how purchase decisions *depend* on the other variables).

And as before it needs to know the split between test set and training set - but this time the `SplitRatio` refers to the *training* set. So whereas in Python we specified the test size at 0.2, here we are specifying the training size - which would be 0.8.

```r
#
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
```

The result is a vector of TRUE/FALSE values - TRUE if it has been selected for the training set, and FALSE if it is for the test set. That can then be used to create a `subset` for either set - or both:

```r
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)
```

## Feature scaling in R

Feature scaling is a solution to the problem of having data on different scales. Without this, machine learning would be dominated by data points within bigger ranges (e.g. thousands of pounds) having a bigger influence that those within smaller ranges (e.g. ages, weights).

To do this in R you use the `scale` function. This will return an error if any columns are non-numeric, including *factors* that look like numbers. Because our dataset contains factors (0, 1, 2 for countries and purchases) we exclude those columns 1 and 4 by only scaling the second and third columns using `[,2:3]` - note this must be done either side of the equals operator so that the new scaled columns are reassigned to the same columns.

```r
training_set[,2:3] = scale(training_set[,2:3])
test_set[,2:3] = scale(test_set[,2:3])
```
