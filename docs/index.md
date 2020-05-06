# Writing a naive linear regression in Python

## Introduction to regression

Linear regression is probably the most commonly used statistical model on earth that attempts to show a relationship between two or more variables (correlation is just a special case of linear regression).  The equation for linear regression can be written and implemented using vectors and matrices.  NumPy is written to help make dealing with matrices and vectors easier, so linear regression makes a good example to show how. 

Variables shown in bold, like **Y** are vectors or matrices; that is, they contain more than one value.  Variables that represent only a single number, like *y*, are sometimes called _scalars_.  When it matters that a vector have only one column but multiple rows, it will be called a _column vector_, and similarly when it has only one row with multiple columns, it will be called a _row vector_.  That matters in multiplication because the length of a column, or the width of a row, has to match up with thing they are being multiplied with.

The basic form of the regression model, when written in matrix notation is,

**Y** = **X&beta;** + **&epsilon;**,

where **Y** is the vector of dependent variable values, **X** is the vector (or matrix) of independent variable values, **&beta;** is the vector (or matrix) of regression coefficients, and **&epsilon;** is the vector of independent errors (residuals).

If our sample size is _n_, then the **Y** vector is a single column of numbers that is _n_ rows tall.  For our case, we will only have a single value for **X**, so it, too, is a single column of numbers that is _n_ rows tall.  If there were more variables, the number of columns would increase, but the number of rows in **Y** and **X** must be the same.

The **&beta;** is two columns, which is the number of columns in both **X** and **Y**, but only one row.  Finally, the **&epsilon;** is one column with _n_ rows.

It is most common for data to be organized so that the values for **X** and **Y** are in a single file.  The comma-separated values (`.csv`) format is very common.  Let's say we have some `.csv` data in a file called `data.csv` that looks like this.

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

All we know are **X** and **Y**.  We have to estimate (calculate) **&beta;** and **&epsilon;** ourselves.

If we look the formula up in a statistics book or online, we will find that the estimate of **&beta;** is typically called **b**, which can be calculated this way,


**b** = (**X**'**X**)<sup>-1</sup>**X'Y**

Note that **X**'**X** means '**X** transpose' times **X** and the transpose is sometimes indicated by **X**<sup>`T`</sup>.  Multiplication with matrices can be tricky because there is more than one meaning for "multiply" and because, usually, **A** * **B** does _not_ equal **B** * **A** the way it does with numbers.  You'll sometimes see or hear something about "premultiply" or "postmultiply" and those are used to make clear which comes first.

That will produce a regression equation, into which we can plug each value of _x_ from the rows of **X** to produce a _predicted value_ value for _y_, which is usually different from the actual value of _y_ from each of the rows of **Y**.  The estimated value for **Y** is calculated as

**Y&#770;** = **Xb** = **X**(**X'X**)<sup>-1</sup>**X**'**Y**

The vector **&epsilon;** of residuals, which are the differences between the predicted value for a _y_ and the actual value that we have, is calculated as

**&epsilon;** = **Y** - **Xb**

Those will the things we will be calculating with NumPy and Python from the data above to generate the regression equation.

In this example, we will create a Python program that will calculate that simple version of the regression equation in the most straightforward way as a learning exercise.  To do that, we will introduce NumPy, the library that is most used to do linear algebra in Python.  The point of this exercise is to provide a relatively simple example of going from what you might see written in an article or mathematics book to a working bit of computer code.

We will break down the quantities above and calculate the simplest components first, then combine them into the more complex, and along the way we will introduce NumPy concepts, structures, and functions as we need them.

1. Read the data from `data.csv` and create the ***X*** and ***Y*** vectors.
1. Calculate **X**'
1. Calculate **X**'**X**

Calculate (**X**-**X**)<sup>-1</sup>

1. Calculate **X**'**Y**
1. Calculate **b** = (**X**-**X**)<sup>-1</sup>**X**'**Y**
1. Calculate **Y&#770;** = **Xb** - (**X**-**X**)<sup>-1</sup>**X**'**Y**

Before we get to calculating those quantities, though, we need to introduce you to NumPy and get some data to work with.

## Introduction to NumPy

NumPy is a Python _library_ for doing linear algebra, that is, computations with matrices and vectors.  It does need to be installed.  There are many ways to install Python and NumPy, and we will assume you have done so.  In what follows, we will show examples either as Python commands that you type at a Python prompt, as in

```
>>> import numpy
>>> print("Hello, NumPy")
Hello, NumPy
```
If you use IPython, then your prompt will look more like
```
In [1]: import numpy
In [2]: print("Hello, NumPy")
Hello, NumPy
```
Where `In[1]` means the first line of input, and output will be shown 


We will put some commands into a file, such as `welcome.py`, and then show the results of running those commands from a command line (we assume Linux, but it may be Mac or Windows), as in the following example.
<hr>
Create a file called `welcome.py` with the following contents.

```
print("Welcome to regression with NumPy")
```

Then run it using Python, in the following way.

```
$ python welcome.py
Welcome to regression with NumPy
```
<hr noshade>

In the example above, we showed you how to `import numpy`, which is to say, make it available to the current Python program or script.

Let's start through our list of tasks and introduce the concepts that we need as we encounter them.  Our first task, as with many numerical and scientific programs is to get some data to work with.  Often there will be libraries (like NumPy) that need to be imported, and it is conventional to do so at the top of the script.  You can put _comments_ into your Python scripts that are not interpreted as a commands, and those begin with the `#` character.

So, make sure you have created the `data.csv` file as shown above, and let's get to it.  We start `python` the program from a command prompt, which we are indicating with the `$` at the beginning of the line.  Once `python` starts, it will change the prompt to `>>>`.

```
$ python
>>> import numpy
>>>
```
When we want to work with data, we have to put into a variable with a name so we can refer to it when we want to calculate something using it.  That's called _assignment_, and the variable that is being assigned a value is on the left of an equals sign and the value that it should take is on the right.

In the case of reading data, we have a _function_ that will read data from a `.csv` file, and it will produce (or _return_) the values contained in the columns.  Functions from libraries have to have the name of the library as a prefix.  The function name is `loadtxt`, and it takes _arguments_, that is values of some kind that will give the function information or change its behavior, inside the parentheses that follow the name.

`loadtxt` takes one required argument, the name of the file in quotes, and we are going to use two optional arguments.  The first will say what character (delimiter) separates (delimits) the values from each other in the file, and the second that it should not read values from the first row of the file.  Once we have read the values, we will display them to confirm that we got what we think we should.
```
>>> data = numpy.loadtxt('data.csv', delimiter=',', skiprows=1)
>>> data
array([[ 4. , 33. ],
       [ 4.5, 42. ],
       [ 5. , 45. ],
       [ 5.5, 51. ],
       [ 6. , 53. ],
       [ 6.5, 61. ],
       [ 7. , 62. ]])
```
We now have some data -- seven rows of two numbers -- in a variable called `data`.  When we display the values by typing the variable name and pressing Return, NumPy tells us that it is an _array_, which is the basic data type for NumPy.  You'll see that there are a lot of brackets and parentheses in that display.  Before we go much further, we should look at what those mean and what is stored in an array.

Let's start by looking at the middle row: `[ 5.5, 51. ],`.  It is enclosed in brackets, which indicates that it is an array in its own right, and in this case it has one _dimension_, which looks here like a row, and it contains two numbers, 5.5 and 51.  The numbers inside the array are separated from each other when the array is printed by commas, just as you can see that the arrays that make up data are separated from each other by commas.

We can construct an array of our own using the `array` function, which takes a Python list as an argument, as in this example.
```
>>> numpy.array([1, 2, 3])
array([1, 2, 3])
```
Note that we did not assign that array to any variable, so we will not be able to use it in any subsequent calculations.  Also note that there is only one set of brackets.  The brackets enclose the values in a dimension.  Let's do that again, only this time, let's assign it to a variable.
```
>>> a = numpy.array([1, 2, 3])
>>> a.shape
(3,)
>>> a
array([1, 2, 3])
```
Arrays have _attributes_ that tell us about information about them, and the `shape` attribute tells us how many rows, columns, layers (and whatever you want to call the fourth and subsequent dimensions).  The `(3,)` says that it has 3 columns and no rows.  No rows?!

It's a little weird, but whether it is a row or a column doesn't matter until we define them both.  Right now, it is one dimensional.  If we really want to make it have rows and columns, we need to create a second dimension.  When we count the number of elements in an array, we start at 1; when we refer to their _position_ in the array, we start at 0.  So, the first element in an array is at position 0, the second at position 1, etc.  When we want to refer to the elements in an array, we use the colon character, and it is preceded by the first element and it is followed by the length it would be if it started from 0.  The second number is also, therefore, one less than the position of the last element.

We all found that confusing at first, so here are a couple of examples to help you see what we mean.  The brackets following the variable name enclose the range of values, separated by the colon as above.  The first example asks for all the elements starting with that which occupies position one and would make an array of length one.  That is, just the first element.
```
>>> a[0:1]
array([1])
```
We can continue the example, by changing the one to a 2, and then to a 3, and then to a 4.

```
>>> a[0:2]
array([1, 2])
>>> a[0:3]
array([1, 2, 3])
>>> a[0:4]
array([1, 2, 3])
```
Do note that in the last example, we wanted four elements, but only got the three that were there.  We did _not_ get an error.  So, in what follows, you must keep track of the lengths and sizes of things because sometimes you can get unexpected results.

Now let's see what happens if we start increasing the number on the left of the colon.
```
>>> a[0:3]
array([1, 2, 3])
>>> a[1:3]
array([2, 3])
>>> a[2:3]
array([3])
>>> a[3:3]
array([], dtype=int64)
```
The first line we've seen before: Start at position 0 and show an array that would have 3 elements starting at 0.  The second example is still saying that the array produced should stop at element 3, but now it should start at position 1, which is really the second element, so we get `[2, 3]`.  The third example gives us just the final element because we start and end at position 3.  The last example looks odd, and that is telling us that we got an array with nothing in it, but if we did put something in it, the type thing would be a number of type `int64`, which we can ignore for now.

Now you can see a bit how ranges of values along one dimension can be shortened at either end.  Dimensions are separated from each other within the brackets by a comma.  A colon by itself with no numbers means 'from the beginning to the end'.

Armed with all that information, we will use the `newaxis` function with our array, `a`, to add a second dimension.

```
>>> a.shape
(3,)
>>> b = a[:, numpy.newaxis]
>>> b.shape
(3,1)
>>> b
array([[1],
       [2],
       [3]])
```
So, we defined a new dimension, and when we ask about the shape, we now see that there are three elements on the first dimension and one on the second, that is, we have a column vector.  We got a colum vector because we put all the elements of the already existing `a` on the left of the comma.  Here's what we get if we put them on the right.

```
>>> b = a[numpy.newaxis, :]
>>> b.shape
(1, 3)
>>> b
array([[1, 2, 3]])
```
Now we have two dimensions, but it's only got one element along the first (the row) and three along the second (the columns), so this is a row vector.

Note that when there is only a one-dimensional variable being display inside `array()`'s parentheses, the outer set of square brackets aren't used.  Once we add a second dimension, each dimension gets its own set of brackets.  To show that, here is an example of creating a two-dimensional array.  Compare its output to that of the `b` row vector above.
```
>>> numpy.array([[1, 2, 3], [4,5,6]])
array([[1, 2, 3],
       [4, 5, 6]])
```
Finally, we have arrived at the shape of our current `data` variable, and we now understand that it has two dimensions, which is seven rows and two columns in an array.

Looking back at our list of things to calculate, we need to create variables for **X** and for **Y**, and those need to be column vectors.  We will create those using a _slice_, which is a way to take a subset of the values from a variable that contains more than one value.  We will use the colon the same way we've just been doing to create the **Y** vector.

```
>>> Y = data[:, 1]
>>> Y
array([33., 42., 45., 51., 53., 61., 62.])
>>> Y.shape
(7,)
```
To review the notation above, `[:, 1]` means take all the values from beginning to end of the first dimension -- those will be values from the all the rows, and only do that for the values in column position 1, that is the second column.  So now we have a one-dimensional **Y**, but we are going to need a column vector, so using the technique we showed above, we can do so with this, and do note that it is safe to have the same variable name on the left of the equal as on the right!
```
>>> Y = Y[:, numpy.newaxis]
>>> Y.shape
(7,1)
>>> Y
array([[33.],
       [42.],
       [45.],
       [51.],
       [53.],
       [61.],
       [62.]])
```

