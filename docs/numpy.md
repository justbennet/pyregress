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

and we know that Y is now a NumPy _array_ by its display.

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

into ***X*** and ***Y***.  That's done using something called a _slice_.
