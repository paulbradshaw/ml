# Support Vector Regression (SVR)

> "SVR has a different goal compared to linear regression. In linear regression we are trying to *minimise the error* between the prediction and the data. In SVR our goal is to make sure that errors *do not exceed the [error] threshold* [epsilon]." - Kirill Eremenko, [Machine Learning A-Z, Udemy](https://www.udemy.com/machinelearning/learn/v4/t/lecture/10459548?start=0)

### Decision trees

The two types of decision trees are **Classification Trees** and **Regression Trees**. The acronym **CART** helps remember them (Classification And Regression Tree).

Regression Trees are more complex. They split your variables (the scatterplot) into different parts (*leaves*), using an algorithm looking at **information entropy**: the key concept behind this is whether a split *adds* information; the algorithm stops (with a *terminal leaf*) when it cannot add any more meaningful information with any more splits.

Each split creates a *branch* in the decision tree.

For example if your first split is at the 20 point on one axis (X1), the branch for that would be X1<20 (TRUE or FALSE). In other words, the first branch splits the data into whether they fall before or after that line.

Next, the second split is made *within* one of those leaves so that second branch is *only* placed on the appropriate branch. So for example if the split line is drawn for the >20 area and it is drawn where X2 is 170, then the test in that branch would be X2<170.

Here are those splits shown on the scatterplot:

![](/svr_scatter.png)

And then how those are represented as a series of branches:

![](/svr_tree.png)
