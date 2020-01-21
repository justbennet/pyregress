# Writing a naive regression in Python

## Introduction

As you know, the regression model can be written and implemented
using matrices.  The basic form of the model is

***Y*** = ***X&beta;*** + ***&epsilon;***

where ***Y*** is the vector of dependent variable values, ***X*** is the vector (or matrix) of independent variable values, ***&beta;*** is the vector (or matrix) of regression coefficients, and ***&epsilon;*** is the vector of independent errors (residuals).  Let's say we have some data that looks like

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

All we know are ***X*** and ***Y***.  We have to calculate ***&beta;*** and ***&epsilon;*** ourselves.  If we look that up in book or online, we will find that the estimate of ***&beta;*** is ***b***, which can be calculated this way,


***b*** = (***X***'***X***)<sup>-1</sup>***X'Y***

Note that `X'X` means "X transpose times X" and the transpose is sometimes indicated by `X<sup>T</sup>`.  Multiplication with matrices can be tricky because there is more than one meaning for "multiply" and because, usually, `A*B` does _not_ equal `B*A` the way it does with numbers.  You'll sometimes see or hear something about "premultiply" or "postmultiply" and that is why.

and that the estimated value for ***Y*** is

***Y***_est = ***Xb*** = ***X***(***X'X***)<sup>-1</sup>***X***'***Y***


and finally the ***&epsilon;*** (resduals) is calculated as

***&epsilon;*** = ***Y*** - ***Xb***

Those will be what we are really calculating with NumPy and Python from the data above.

In this example, we will try to create a Python program that will calculate that simple version of the regression equation in the most straightforward way as a learning exercise.  To do that, we will introduce NumPy, the library that is most used to do linear algebra in Python.  The point of this exercise is to provide a relatively simple example of going from what you might see written in an article or mathematics book to a working bit of computer code.

Before we get to the regression equation, though, we need to introduce you to NumPy and get some data to work with.  Please click on the menu to the right to go to the next section.
