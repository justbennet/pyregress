# Writing a naive regression in Python

## Introducing NumPy

Before we can use any NumPy functions, we have to import.  It is conventional to import NumPy so that it can be referred to as `np` from within your Python program, which means that functions begin `np.` instead of `numpy.` which is shorter and less susceptible to typographical error.  To do that,

```
>>> import numpy as np
```

where the `>>>` indicates the Python prompt and should not be typed.  What follows, should be, and you should press Return to get the `>>>` prompt back.  If a Python command continues to another line, the prompt on the second and subsequent lines will be `...` to indicate continuation.  Lines without a prompt are expected output.

Assuming you have some prior experience with Python, you will know that

```
Y = [ 1, 2, 3 ]
```

creates a list of numbers.  NumPy has its own way of storing those, so you can convert an existing list of numbers with

```
>>> Y = np.array(Y)
```

We say that a matrix has a _shape_, which means how many rows by how many columns, and we obtain that with

```
>>> Y.shape
(3,)
```

Our simple array has three elements, and they are on the first dimension (or axis).

We can confirm that Y is now a NumPy _array_ by its display.

```
>>> Y
array([1, 2, 3])
```

However, if you have data in, say, a `.csv` file, NumPy will read it directly into an array to save you the bother (and extra memory of storing two copies) of converting.  So, if we have a file, `data.csv`, that contains

```
x,y
4.0,33
4.5,42
5.0,45
5.5,51
6.0,53
6.5,61
7.0,62
```

we can read that with

```
>>> data = np.loadtxt('data.csv', delimiter=',', skiprows=1)
```

Note that

```
>>> data.shape
(7, 2)
```

Our equation for regression wants ***X*** and ***Y*** in separate arrays (arrays, matrices, and vectors are all stored as NumPy arrays).  That means we will have to separate the two colums we read into `data`,

```
>>> data
array([[  4. ,  33. ],
       [  4.5,  42. ],
       [  5. ,  45. ],
       [  5.5,  51. ],
       [  6. ,  53. ],
       [  6.5,  61. ],
       [  7. ,  62. ]])
```

into ***X*** and ***Y***.  That's done using something called a _slice_.  A slice is a selection from an array.  We have here a two-dimensional array.  In case you forgot or did not know, Python likes to index (count) array elements starting from 0 not from one.  So, our first slice looks at the 0th row,

```
>>> data[0]
array([  4.,  33.])
```

From that, we can add something to select just the first (0th) column,

```
>>> data[0,0]
4.0
```

We might obtain the first two rows, then select column 1.


```
>>> data[0:,0]
array([ 4. ,  4.5,  5. ,  5.5,  6. ,  6.5,  7. ])
```

which would be the vector of values for our independent variable; or we could chose the second column to get the vector of response values,

```
>>> data[0:,1]
array([ 33.,  42.,  45.,  51.,  53.,  61.,  62.])
```

When there is nothing on one side of the colon, it means "the end"; so, `0:` means from the first element to the last, and `:6` means from the end starting at the first element up to the sixth element.

Slicing gets complex and you should definitely spend some time reading about them and trying things with them until you get comfortable with them.  In our simple case, we want to get the two columns into arrays of their own.

```
>>> X = data[:,0]
>>> Y = data[:,1]
>>> X
array([ 4. ,  4.5,  5. ,  5.5,  6. ,  6.5,  7. ])
>>> Y
array([ 33.,  42.,  45.,  51.,  53.,  61.,  62.])
```

We also need to have a column of ones in the matrix of response variable values so that the constant (<em>&beta;</em><sub>0</sub>) will be calculated.  NumPy has two functions to help when you need to create arrays with constant value, `np.zeros()` and `np.ones()`.  We can create an array of 1s that is the same length as our `X` this way

```
>>> constant = np.ones(X.shape[0])
```

We are using the `[0]` to get just the row length.  If you look above, when we did `data.shape`, it came back with `(7,2)`.  That's a _tuple_ (a thing that has multiple elements and can't be changed) with two elements.  If we do `shape.X`, it will come back with `(7,)`, which is still a tuple, but this time it only has one element.  We want to make sure we get just the number of rows, so we are going to explicitly ask for that by taking just the first number from the tuple, even if there is only one.  (Try for yourself to see what happens if you do `np.ones(data.shape)`.)

For matrix operations, like multiplying, we want to see our data look like this,

```
              1  x1                    e1
              1  x2                    e2
predicted = [ 1  x3 ] * [ b1, b2 ] + [ e3 ]
              1  x4                    e4
              1  x5                    e5
```

where everything is in columns.  All of our variables are in rows right now, and we need to change that if we wish to use the formula for regression that we find in our book.  And we need to create the ***X*** matrix from our current `constant` and `X` variables.

```
>>> X
array([ 4. ,  4.5,  5. ,  5.5,  6. ,  6.5,  7. ])
>>> Y
array([ 33.,  42.,  45.,  51.,  53.,  61.,  62.])
>>> constant
array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.])

```

First, let's create the matrix for the independent variable.  We do this by creating a new array,

```
>>> X = np.array([constant, X])
>>> X
array([[ 1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ],
       [ 4. ,  4.5,  5. ,  5.5,  6. ,  6.5,  7. ]])
```

but we still need to tip that over so that the ones are in the first column and the data values are in the second.  We do that by taking the _transpose_, that is making the first row the first column, the second row the second column, etc.

```
>>> X = X.T
>>> X
array([[ 1. ,  4. ],
       [ 1. ,  4.5],
       [ 1. ,  5. ],
       [ 1. ,  5.5],
       [ 1. ,  6. ],
       [ 1. ,  6.5],
       [ 1. ,  7. ]])
```

Now we have to fix up the independent values.  We can try

```
>>> Y.T
array([ 33.,  42.,  45.,  51.,  53.,  61.,  62.])
```

but that doesn't work because there is only one dimension, that is, only a row, so the transpose doesn't work.  That's why there is no second element in `Y.shape`.

```
>>> Y.shape
(7,)
```

We can add a new, but empty dimension for the columns like this.

```
>>> Y = Y[:, np.newaxis]
>>> Y
array([[ 33.],
       [ 42.],
       [ 45.],
       [ 51.],
       [ 53.],
       [ 61.],
       [ 62.]])
```

So, we now have our data arranged the way we want it to calculate the regression equation.

```
>>> X
array([[ 1. ,  4. ],
       [ 1. ,  4.5],
       [ 1. ,  5. ],
       [ 1. ,  5.5],
       [ 1. ,  6. ],
       [ 1. ,  6.5],
       [ 1. ,  7. ]])
>>> Y
array([[ 33.],
       [ 42.],
       [ 45.],
       [ 51.],
       [ 53.],
       [ 61.],
       [ 62.]])
```

In the next section, we will look at how we manipulate those to get the matrix of beta coefficients and our vector of errors (the residuals).
