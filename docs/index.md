# Writing a naive linear regression in Python

## Introduction

Linear regression is probably the most commonly used statistical model on earth that attempts to show a relationship between two or more variables (correlation is just a special case of linear regression).  The equation for linear regression can be written and implemented using vectors and matrices.  NumPy is written to help make dealing with matrices and vectors easier, so linear regression makes a good example to show how. The basic form of the regression model, when written in matrix notation is

***Y*** = ***X&beta;*** + ***&epsilon;***,

where ***Y*** is the vector of dependent variable values, ***X*** is the vector (or matrix) of independent variable values, ***&beta;*** is the vector (or matrix) of regression coefficients, and ***&epsilon;*** is the vector of independent errors (residuals).

If our sample size is _n_, then the ***Y*** vector is a single column of numbers that is _n_ rows tall.  For our case, we will only have a single value for ***X***, so it, too, is a single column of numbers that is _n_ rows tall.  If there were more variables, the number of columns would increase, but the number of rows in ***Y*** and ***X*** must be the same.

The ***&beta;*** is two columns, which is the number of columns in both ***X*** and ***Y***, but only one row.  Finally, the ***&epsilon;*** is one column with _n_ rows.

It is most common for data to be organized so that the values for ***X*** and ***Y*** are in a single file.  The comma-separated values (`.csv`) format is very common.  Let's say we have some `.csv` data in a file called `data.csv` that looks like this.

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

All we know are ***X*** and ***Y***.  We have to estimate (calculate) ***&beta;*** and ***&epsilon;*** ourselves.

If we look the formula up in a statistics book or online, we will find that the estimate of ***&beta;*** is typically called ***b***, which can be calculated this way,


***b*** = (***X***'***X***)<sup>-1</sup>***X'Y***

Note that ***X***'***X*** means '***X*** transpose' times ***X*** and the transpose is sometimes indicated by ***X***<sup>`T`</sup>.  Multiplication with matrices can be tricky because there is more than one meaning for "multiply" and because, usually, ***A*** * ***B*** does _not_ equal ***B*** * ***A*** the way it does with numbers.  You'll sometimes see or hear something about "premultiply" or "postmultiply" and those are used to make clear which comes first.

That will produce a regression equation, into which we can plug each value of _x_ from the rows of ***X*** to produce a _predicted value_ value for _y_, which is usually different from the actual value of _y_ from each of the rows of ***Y***.  The estimated value for ***Y*** is calculated as

***Y***_est = ***Xb*** = ***X***(***X'X***)<sup>-1</sup>***X***'***Y***

The vector ***&epsilon;*** of residuals, which are the differences between the predicted value for a _y_ and the actual value that we have, is calculated as

***&epsilon;*** = ***Y*** - ***Xb***

Those will the things we will be calculating with NumPy and Python from the data above to generate the regression equation.

In this example, we will create a Python program that will calculate that simple version of the regression equation in the most straightforward way as a learning exercise.  To do that, we will introduce NumPy, the library that is most used to do linear algebra in Python.  The point of this exercise is to provide a relatively simple example of going from what you might see written in an article or mathematics book to a working bit of computer code.

We will break down the quantities above and calculate the simplest components first, then combine them into the more complex, and along the way we will introduce NumPy concepts, structures, and functions as we need them.

1. Read the data from `data.csv` and create the ***X*** and ***Y*** vectors.
1. Calculate ***X***'
1. Calculate ***X***'***X***
1. Calculate (***X***-***X***)<sup>-1</sup>
1. Calculate ***X***'***Y***
1. Calculate ***b*** = (***X***-***X***)<sup>-1</sup>***X***'***Y***
1. Calculate ***Y***_est = ***Xb*** - (***X***-***X***)<sup>-1</sup>***X***'***Y***

Before we get to calculating those quantities, though, we need to introduce you to NumPy and get some data to work with.

Please click on the menu to the right to go to the next section, Reading Data.
