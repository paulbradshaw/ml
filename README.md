# Machine learning - notes

This file contains notes from the [Udemy course Machine Learning A-Z™: Hands-On Python & R In Data Science](https://www.udemy.com/machinelearning/) and the [Coursera course on Machine Learning](https://www.coursera.org/learn/machine-learning/) with Andrew Ng.

[Download the data and files for the former course from here](https://www.superdatascience.com/machine-learning/)

## The 3 types of machine learning

I've [written a blog post here](https://onlinejournalismblog.com/2017/12/14/data-journalisms-ai-opportunity-the-3-different-types-of-machine-learning-how-they-have-already-been-used/) explaining the 3 types of machine learning with examples of each being used in journalism. These are:

* **Supervised learning**: whereby an algorithm is given *training data* to establish a *relationship* between variables to predict values, or to *classify*/categorise. The point here is that there is a 'right answer' that the algorithm is trained on.
* **Unsupervised learning** is when you don't know what the answer is, or at least want to see what answer the algorithm comes up with on its own. This is often used to classify things when you think an algorithm might come up with better classifications. The question being asked is "Can you find some structure or patterns in this data?"
* **Reinforcement learning** is about letting an algorithm discover the *optimal* approach to a task by learning through trial and error (the reinforcement).

### Different models

Each of these types represents a different **model** of machine learning, but you can also talk about models in the following terms:

* Linear regression models (**univariate** linear regression means *one variable*)
* Non-linear regression models (e.g. when values increase exponentially)
* Non-linear and non-continous regression models (e.g. decision trees)

Different models fit different **problems**.

### The 'hypothesis'

A machine learning algorithm will typically generate a **hypothesis** which can be applied to new data in order to generate *predictions* or some other result.

For example, an algorithm trained on data on housing might produce the (overly simplified) hypothesis that "House price is equal to square footage times 31.56", or in algebraic terms: `hp = sf * 31.56`. Given any new `sf` (square footage) we can predict the `hp` (house price). Of course in reality the hypothesis is likely to be much more complex and involve many more variables, but you get the idea. It is also often represented in formulaic terms using `h` for *hypothesis* with the theta character like so:

![](https://qph.fs.quoracdn.net/main-qimg-f671fc96001a43560adac1bd8bc87fda)

Of course the term hypothesis is used because this is not a fact, and we can *test* the hypothesis as new data emerges, to determine its accuracy or effectiveness.

## Data preprocessing

Some stages:

* Distinguish between **independent and dependent variables**
* Replace missing values with **imputation**
* Turn **categorical data** into numerical data - typically this is done by creating a different column for each category, and using 0 and 1 in that column to represent its presence of absence. These are called **dummy variables**
* Split into a **training set and test set**
* Get values on the same scale with **feature scaling**

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

## Support Vector Regression (SVR)

[See the file in this repo](/svr.md)
