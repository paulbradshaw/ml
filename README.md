# Machine learning - notes

This file contains notes from:
* The [Udemy course Machine Learning A-Z™: Hands-On Python & R In Data Science](https://www.udemy.com/machinelearning/)
* The [Coursera course on Machine Learning](https://www.coursera.org/learn/machine-learning/) with Andrew Ng.
* *Interpretable Machine Learning*, an ebook by Christoph Molnar.

[Download the data and files for the former course from here](https://www.superdatascience.com/machine-learning/)

## The 3 types of machine learning

I've [written a blog post here](https://onlinejournalismblog.com/2017/12/14/data-journalisms-ai-opportunity-the-3-different-types-of-machine-learning-how-they-have-already-been-used/) explaining the 3 types of machine learning with examples of each being used in journalism. These are:

* **Supervised learning**: whereby an algorithm is given *training data* to establish a *relationship* between variables to predict values, or to *classify*/categorise. The point here is that there is a 'right answer' that the algorithm is trained on.
* **Unsupervised learning** is when you don't know what the answer is, or at least want to see what answer the algorithm comes up with on its own. This is often used to classify things when you think an algorithm might come up with better classifications. The question being asked is "Can you find some structure or patterns in this data?" The results involve **clustering** (find patterns) and **non-clustering** (extract information from 'noise' that shares a pattern).
* **Reinforcement learning** is about letting an algorithm discover the *optimal* approach to a task by learning through trial and error (the reinforcement).

[A discussion of non-clustering here tries to tease out the distinction](https://www.reddit.com/r/learnmachinelearning/comments/7zuu73/difference_between_clustering_and_nonclustering/)

> "The way I see it we are finding the commonalities between the 2 voices [in different aural datasets] and then categorizing it so that one voice goes to one category and the other voice goes to another category which is basically clustering as it grouping on the basis of some criterion (the voice)."

[Different clustering techniques are explained in this presentation](http://www.mit.edu/~9.54/fall14/slides/Class13.pdf)

### Classification or regression

If the output is numerical, this is called **regression**; if the output is categorical, this is **classification**. The algorithm might estimate parameters (weighting those) or learning structures (trees) (Molnar 2019)

## Different models

Each of these types represents a different **model** of machine learning, but you can also talk about models in the following terms:

* Linear regression models (**univariate** linear regression means *one variable*)
* Non-linear regression models (e.g. when values increase exponentially)
* Non-linear and non-continous regression models (e.g. decision trees)

Different models fit different **problems**.

Models are more specifically represented in a formula, explained below:

## The 'hypothesis'

A machine learning algorithm will typically generate a **hypothesis** which can be applied to new data in order to generate *predictions* or some other result.

For example, an algorithm trained on data on housing might produce the (overly simplified) hypothesis that "House price is equal to square footage times 31.56", or in algebraic terms: `hp = sf * 31.56`. Given any new `sf` (square footage) we can predict the `hp` (house price). Of course in reality the hypothesis is likely to be much more complex and involve many more variables, but you get the idea. It is also often represented in formulaic terms using `h` for *hypothesis* with the theta character like so:

![](https://qph.fs.quoracdn.net/main-qimg-f671fc96001a43560adac1bd8bc87fda)

Of course the term hypothesis is used because this is not a fact, and we can *test* the hypothesis as new data emerges, to determine its accuracy or effectiveness.

### The formula/expression

Some conventions:

* `x` is used to denote input
* `h` is used to denote hypothesis
* `y` is used to denote output/target prediction

## How effective? The cost function

The **cost function** "takes an average difference (actually a fancier version of an average) of all the results of the hypothesis with inputs from x's and the actual output y's" ([source](https://www.coursera.org/learn/machine-learning/supplement/nhzyF/cost-function?errorCode=invalidCredential)). Or, as [this guide puts it](https://towardsdatascience.com/machine-learning-fundamentals-via-linear-regression-41a5d11f5220):

> "In ML, cost functions are used to estimate how badly models are performing. Put simply, a cost function is a measure of how wrong the model is in terms of its ability to estimate the relationship between X and y. This is typically expressed as a difference or distance between the predicted value and the actual value. The cost function (you may also see this referred to as loss or error.) can be estimated by iteratively running the model to compare estimated predictions against “ground truth” — the known values of y.

>"The objective of a ML model, therefore, is to find parameters, weights or a structure that minimises the cost function."

A **gradient descent** is a way of exploring the distances calculated by a cost function. As [one cheatsheet puts it](https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html):

> "Gradient descent is an optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient. In machine learning, we use gradient descent to update the parameters of our model."

This leads us to a *local minimum* - not necessarily the lowest in the whole scope, but the one that is closest (imagine a 3D landscape with dips and peaks).

In a gradient descent the **learning rate** (*alpha*) is how big a step the algorithm takes each time it looks to find the optimal 'next point' as it searches for the ultimate optimal point. If that is too small, it may slow the gradient descent; if it is too large, it may overshoot the optimal point (or fail to converge, or diverge).

> "We should adjust our parameter [alpha] to ensure that the gradient descent algorithm converges in a reasonable time. Failure to converge or too much time to obtain the minimum value imply that our step size is wrong."

Gradient descent will take smaller steps *anyway* as you converge on a local minimum, because the slope will be less steep.

If you happen to start at the optimal point it will go nowhere - it will stay with that point because the *[tangent point](https://en.wikipedia.org/wiki/Tangent)* 'slope' will be perfectly horizontal, the **derivative point** will be 0.

**'Batch' gradient descent** uses all the training examples at each stage. In contrast **stochastic gradient descent** (SGD) [uses one example at a time](https://towardsdatascience.com/difference-between-batch-gradient-descent-and-stochastic-gradient-descent-1187f1291aa1) and **mini-batch gradient descent** involves smaller batches. [A Gentle Introduction to Mini-Batch Gradient Descent and How to Configure Batch Size](https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/) explains some of the pros and cons of each method, alongside descriptions and further reading.



## Data preprocessing

Some stages:

* Distinguish between **independent and dependent variables**
* Replace missing values with **imputation**
* Turn **categorical data** into numerical data - typically this is done by creating a different column for each category, and using 0 and 1 in that column to represent its presence of absence. These are called **dummy variables**
* Split into a **training set and test set**
* Get values on the same scale with **feature scaling**.

See [prep_py](/prep_py.md) for those steps in Python and [prep_r](/prep_r.md) for steps in R.


## Linear regression

Linear regression attempts to establish a relationship - and the strength of that relationship - between a dependent variable (the thing affected) and one or more independent variables (the things affecting it).

A linear regression has some assumptions:

* Linearity
* [Homoscedasticity](https://en.wikipedia.org/wiki/Homoscedasticity)
* Multivariate normality
* Independence of errors
* Lack of [multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity) - the phenomenon where one or more independent variables in a linear regression predict another.

> "Make sure you don't just blindly repeat the steps of the course but go back to the basic assumptions first."

### Categories and dummy variables

Because we cannot create a calculation based on a *word* we create a different column for each category, and use 0 and 1 in that column to represent its presence of absence. These are called **dummy variables**.

However, if there are two categories only *one* of those dummy columns is used for the calculation (because 0 becomes the default). The column becomes the *difference* between the two categories (presence or absence)

The **dummy variable trap** is when both columns are used - the model cannot then distinguish between their effects (dummy variable B is equal to dummy variable A minus 1)

When using dummy variables **always omit one dummy variable**, irrespective of the number of dummy variables.


### P-values

Explanations of the p-value can be [found on MathBootCamps](https://www.mathbootcamps.com/what-is-a-p-value/) and [on WikiHow](https://www.wikihow.com/Calculate-P-Value). The former emphasises that it is:

> "the probability of getting a sample like ours, or more extreme than ours IF the null hypothesis is true. So, we assume the null hypothesis is true and then determine how “strange” our sample really is. If it is not that strange (a large p-value) then we don’t change our mind about the null hypothesis. As the p-value gets smaller, we start wondering if the null really is true and well maybe we should change our minds (and reject the null hypothesis).

> "...A small p-value indicates that by pure luck alone, it would be unlikely to get a sample like the one we have if the null hypothesis is true."

[Video](https://youtu.be/eyknGvncKLw)

This is similar to the statistical significance of a result, where we might say that, for example, this result could only occur by chance 1 in every 20 times.

## Building a model

5 methods of building a model:

* All-in: when you use *all* your variables
* Backward elimination: start with all and recursively eliminate the one with the highest p-value until that is below a significant threshold
* Forward selection: start with the one with the lowest p-value then recursively add those that, in combination with that, have the lowest p-value - until the lowest p-value is above a significance threshold
* Bidirectional elimination: start with the first steps of forward selection but also perform backward elimination to remove variables, and repeat in an iterative process whereby variables enter and leave until you are left with a set
* Score comparison: a massive comparison of every possible model on every possible column

**Stepwise regression** refers to the middle 3 methods above: backward, forward and bidirectional, and sometimes just bidirectional.

### All-in models

These might be used because you already know all the variables are relevant, or because you are instructed to. Or as part of preparation for Backward elimination.

### Backward elimination step-by-step

1. Select a **significance level** (SL) (e.g. 0.05) that any variable must meet to *stay* in the model - in other words, a threshold for being included
2. Fit the model with all possible predictors (variables) - i.e. use All-In
3. Consider the predictor with the *highest p-value*. If P > SL, go to the next step (4). Otherwise stop.
4. Remove that predictor
5. Re-Fit the model *without* this variable (this will change the p-values)
6. Repeat step 3 onwards and continue until the highest is > SL and you stop. At this point the model is ready - any 'insignificant' variables have been *eliminated* from the model.

### Forward selection

1. Select a **significance level** (SL) (e.g. 0.05) that any variable must meet to *enter* in the model - in other words, a threshold for being included
2. Fit all simple regression models - select the one with the *lowest p-value*.
3. Keep this variable - and *fit all possible models* with *one extra predictor* added to that one. So if we had 10 variables, we would pick one and then try 9 two-variable regressions that combine our best variable with each of the other 9 in turn.
4. Consider the predictor (the second value in our two-value regression) with the *lowest* p-value. If P < SL repeat step 3, this time with one more variable (so a 3-value regression next, and so on). If not, stop. The model is finished.

### Bidirectional elimination

1. Select *two* **significance levels** (SL) that any variable must meet to *enter and stay* in the model (e.g. SLENTER = 0.05, SLSTAY = 0.05)
2. Perform the next step of *forward selection*: new variables must have P < SLENTER to enter
3. Perform *all* of the steps of *backward selection*: old variables must have P < SLSTAY to stay
4. Repeat step 3 - until no new variables can enter, and no old variables can exit: stop.

### Score comparison

All possible models:

1. Select a criterion of **goodness of fit**
2. Construct all possible regressions - 10 columns would mean 1023 models!
3. Pick the model with the best criterion

### Polynomial linear regression

* Used to fit patterns which may be *curved*, rather than linear, e.g. the points increase more towards the end. An example might be epidemics which spread slowly at the start and then increase later.
* Uses variables to the *power of* something rather than simply variable times x. This gives it the *parabolic effect* (the curving)
* You can try both linear and polynomial to see which fits better.
* A version of multiple linear regression
* Still called *linear* because the whole formula is on one line. If division was involved, other coefficients, it would be on multiple lines and not linear.

### Multivariate linear regression (linear regression with multiple variables)

Multivariate simply means 'multiple features' - in other words, we are not just calculating a regression based on one feature (e.g. the age of a house has X correlation to its price), but more than one (the age, number of rooms, crime rate, etc.)

## Support Vector Regression (SVR)

[See the file in this repo](/svr.md)

## Linear algebra

For a primer on concepts such as matrices and vectors, notification conventions etc. [see the file in this repo](/linearalgebra.md)

## Feature scaling and mean normalization

Get values on the same scale with **feature scaling**.

This is because widely varying scales (e.g. 1-5 and 0-5000) make for very skewed ellipses for traversing with gradient descent (which takes longer). In that case 0-5000 might be scaled to 0-50 etc.

> "We can speed up gradient descent by having each of our input values in roughly the same range. This is because θ will descend quickly on small ranges and slowly on large ranges, and so will oscillate inefficiently down to the optimum when the variables are very uneven." [Gradient Descent in Practice I - Feature Scaling](https://www.coursera.org/learn/machine-learning/supplement/CTA0D/gradient-descent-in-practice-i-feature-scaling)

In fact, specifically the advice is to get every feature in this situation into a range between -1 and 1.

The same applies if ranges are too small.

**Mean normalisation** can be used to change the range so that the mean is 0.

> "Mean normalization involves subtracting the average value for an input variable from the values for that input variable resulting in a new average value for the input variable of just zero." [Gradient Descent in Practice I - Feature Scaling](https://www.coursera.org/learn/machine-learning/supplement/CTA0D/gradient-descent-in-practice-i-feature-scaling)

## Convergence tests

You can check gradient descent is working properly by plotting the value of theta against the number of iterations. When the value converges over a large number of iterations (i.e. it doesn't change much even after 100 or 1000 more iterations) then it's probably not going to change much more. This can be codified as 'Declare convergence if J(theta) decreases by less than X in Y iterations'.

If theta is *increasing* over iterations then it is not working. Try a smaller learning rate (alpha).
