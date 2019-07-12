# Linear algebra - notes

## Matrices and vectors

A **matrix** is a 2-dimensional array - in other words it has both multiple rows (or both rows and columns); it is a table.

The matrix is normally shown inside square brackets like so:

![](https://i.stack.imgur.com/04gqV.png)

It might be created with code like so:

`A = [1, 2, 3; 10, 15, 20; 7, 8, 9; 400, 211, 512]`

Note the comma to separate each item, and the semi-colon to separate each row.

In pandas, it would be created like this:

```py
examplematrix = ([1, 2, 3],
                 [3, 4, 5],
                 [7, 6, 4])
```

Note the square brackets containing each row.

Entries might be indicated by the matrix name followed by two digits (indices) indicating the row, and column, position (index) of the data (normally in subscript).

So,`A12` is matrix A, row 1 column 2.

Conventionally capital letters are used to refer to matrices.

A **vector** is a matrix with one column.

The number of rows/items in that vector is referred to as **dimensions**, e.g. a vector with 5 items would be called a "5-dimensional vector"

It might be created with code like so:

`v = [1;2;3] `

Note that a semi-colon is used to separate each item (just as it separates each column in a matrix).

Vector items are referred to with indices too, e.g. `b2` would be the second item in the B vector.

Conventionally lower case letters are used to refer to matrices.

Indexes can start from 0 or 1. For example Python uses zero-based indexing and R uses 1-based indexes.

### Calculations with matrices

You can perform addition or subtraction across more than one matrix *as long as they have the same dimensions*. If we had two 2x4 matrices called `A` and `B`, for example, and wrote `A+B` then the result would add `A11` to `B11` (the numbers in the first row and column), and so on until each number was added to the other number in the same position. The result would be a new 2x4 matrix containing the results of those 8 additions. This sort of calculation is called **element-wise** because it is done by the element's position.

Calculations can be done across *all* numbers in a matrix. E.g. `5xA` would multiply each of the 8 numbers in `A` by 5, resulting in a new 2x4 matrix. This is called **scalar multiplication** (or division, etc.). The single value being used as the basis for the calculation (in this example, 5) is called the **scalar value**.

### Multiplying a matrix by a vector

You can multiply (or divide etc.) a matrix by a vector - the process is quite confusing, but here it is:

* The vector has to have the same number of items (dimensions) as the *columns* in the matrix
* This is because the first vector item will be multiplied by the matrix item's first column value, and so on (it may be easier to mentally rotate the vector 180 degrees so that it lies on top of the matrix)
* The result of the calculation will be a vector
* That vector will have the same number of dimensions (rows/items) as the matrix has *rows*
* The calculation will repeat across each item (i.e. each item in the first row of the matrix will be multiplied by each item in the respective position in the vector) and then the results *added together*.

Here's a matrix:

```
[1, 2, 3;
  10, 20, 30;
  100,150,200
]
```

Here's a vector (remember it has to have as many dimensions as the matrix has columns):

`[10; 30; 20]`

Multiplying the two you get:

```
[10, 60, 60;
  100, 600, 600;
  1000,4500,4000
]
```

Which are then added in rows like so:

```
[10+ 60+ 60;
  100+ 600+ 600;
  1000+4500+4000
]
```

To result in a vector which is:

`[130; 1300; 9500]`

### Using vector-matrix calculations in machine learning

This is used in machine learning because we may have some (training or test) data in a matrix, and we want to multiply it by a list (vector) of parameters that correspond to the 'weight' of each column when making a prediction. It will look like this

`prediction = DataMatrix * parametersvector`

So, for example, we have a table with 4 columns: population, murder rate, distance from school, average wage.

Those are being used to predict something else - house price, for example.

The vector would then have 4 dimensions, which correspond to the weighting of each: `[0.1; 0.3; 2.1; 2.5]`

By multiplying the matrix by the vector we get a *weighted result* of each value, which are then added up to give a single digit (representing, say, the prediction of a single house price given the relevant properties).

This is more computationally efficient than looping through the figures and performing the calculations on each in turn.

### Multiplying a matrix by a matrix

When multiplying one matrix by another the same process occurs as when multiplying by a vector explained above - this time, however, this process occurs for *each column* in the second matrix, and the result is a series of vectors, that make up a matrix.

It's easier to explain with an example:

* The second matrix again has to have the same number of *rows* as the number of *columns* in the first matrix
* This is because the first matrix item will be multiplied by the second matrix item's first column value, and so on
* The result of the calculation will be a matrix
* That matrix will have the same number of rows as the first matrix, and the same number of columns as the second matrix has.
* The calculation will repeat across each item (i.e. each item in the first row of the matrix will be multiplied by each item in the respective position in the matrix's first column) and then the results *added together*.
* The totals will then be *nested together* with each resulting vector making up a column, to create a new matrix


### Using matrix-matrix calculations in machine learning

Just as a vector-matrix calculation is used to make a single prediction, matrix-matrix calculations are used in machine learning to test *multiple* predictions (i.e. multiple vectors = a matrix).

Again, we may have some (training or test) data in a matrix, and we want to multiply it by more than one list (vector) of parameters that correspond to the 'weight' of each column when making a prediction. It will look like this

`predictionresults = DataMatrix * PredictionsMatrix`

So, for example, we still have a table with 4 columns: population, murder rate, distance from school, average wage.

Those are being used to predict something else - house price, for example.

The other matrix would then have, say, 5 columns, which correspond to the 5 different weightings predicted.

By multiplying the data matrix by the predictions matrix we get a *weighted results matrix* of each prediction. In other words, a collection of results.

### The identity matrix

A special type of matrix to be aware of is the identity matrix. [Briefly described](https://www.coursera.org/learn/machine-learning/supplement/Xl0xT/matrix-multiplication-properties), it is:

> "The identity matrix, when multiplied by any matrix of the same dimensions, results in the original matrix. It's just like multiplying numbers by 1. The identity matrix simply has 1's on the diagonal (upper left to lower right diagonal) and 0's elsewhere."
