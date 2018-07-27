# Linear regression - notes

This file contains notes from the [Udemy course Machine Learning A-Z™: Hands-On Python & R In Data Science](https://www.udemy.com/machinelearning/).

[Download the data and files from here](https://www.superdatascience.com/machine-learning/)

## Jargon

* Dependent variable = what is being affected (salary)
* Independent variable = what is affecting that (experience)
* Observations = pieces of data, points on a scatterplot against x and y (experience and salary), that we use to generate the regression.
* Trend line = a line that best describes the path taken by all the points. This is the line which is least distant (in aggregate) from all the observations
* Constant = starting point, where the line crosses the y axis (zero experience = £30k)
* Coefficient = what the dependent variable changes by, based on a change in the independent variable(s)

In the dataset to show numbers as normal click *Format* and change `3g` to `0f`

## Assumptions

Linear regression attempts to establish a relationship - and the strength of that relationship - between a dependent variable (the thing affected) and one or more independent variables (the things affecting it).

A linear regression has some assumptions:

* Linearity
* [Homoscedasticity](https://en.wikipedia.org/wiki/Homoscedasticity)
* Multivariate normality
* Independence of errors
* Lack of [multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity) - the phenomenon where one or more independent variables in a linear regression predict another.

> "Make sure you don't just blindly repeat the steps of the course but go back to the basic assumptions first."
