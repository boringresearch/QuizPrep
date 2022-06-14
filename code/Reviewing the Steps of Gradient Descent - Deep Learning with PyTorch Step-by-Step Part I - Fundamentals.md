# Reviewing the Steps of Gradient Descent - Deep Learning with PyTorch Step-by-Step: Part I - Fundamentals

> An overview of the steps involved in gradient descent.

An overview of the steps involved in gradient descent.

We'll cover the following

Simple linear regression[#](#Simple-linear-regression)
------------------------------------------------------

Most tutorials start with some nice and pretty image classification problems to illustrate how to use PyTorch. It may seem cool, but I believe it distracts you from learning how PyTorch works.

For this reason, in this first example, we will stick with a simple and familiar problem: a **linear regression** with a single feature `x`! It does not get much simpler than that. It has the following equation:

y\=b+wx+ϵy = b + w x + \\epsilon

It is also possible to think of it as the simplest neural network possible: one input, one output, and no activation function (that is, linear).

> **Note:** This lesson serves as a review for the previous chapter.

Data generation[#](#Data-generation)
------------------------------------

Let us start generating some synthetic data. We start with a vector of 100 (`N`) points for our feature (`x`) and create our labels (`y`) using `b = 1`, `w = 2`, and some [**Gaussian noise**](https://en.wikipedia.org/wiki/Gaussian_noise) **(epsilon)**.

### Synthetic data generation[#](#Synthetic-data-generation)

The following code generates our synthetic data:

Enter to Rename, ⇧Enter to Preview

Data generation

### Splitting data[#](#Splitting-data)

Next, let us split our synthetic data into train and validation sets, shuffling the array of indexes and using the first 80 shuffled points for training.

Enter to Rename, ⇧Enter to Preview

Splitting synthetic dataset into train and validation sets for linear regression

The following figure shows the subplots of both the training (`x_train`, `y_train`) and validation sets (`x_val`, `y_val`) of generated data:

We know that `b = 1` and `w = 2`, but now let us see how close we can get to the true values by using gradient descent and the 80 points in the training set (for training: `N = 80`).

Gradient descent[#](#Gradient-descent)
--------------------------------------

We will be covering the five basic steps you would need to go through to use gradient descent and the corresponding Numpy code.

### Step 0 - Random initialization[#](#Step-0---Random-initialization)

For training a model, you need to randomly initialize the parameters/weights. In our case, we have only two: `b`, and `w`.

Enter to Rename, ⇧Enter to Preview

Step 0 - random initialisation

### Step 1 - Compute model’s predictions[#](#Step-1---Compute-model’s-predictions)

This is the forward pass. It simply computes the model’s predictions using the current values of the parameters/weights.

At the very beginning, we will be producing really bad predictions, as we started with random values from Step 0.

Enter to Rename, ⇧Enter to Preview

You can see the values of these predictions by running the following code:

Enter to Rename, ⇧Enter to Preview

Displaying the bad predictions (uses random values)

### Step 2 - Compute the loss[#](#Step-2---Compute-the-loss)

For a regression problem, the loss is given by the Mean Squared Error (MSE). As a reminder, MSE is the average of all squared errors, meaning the average of all squared differences between labels (`y`) and predictions (`b + wx`).

Let us now compute the loss using Python.

> In the code below, we are using all data points of the training set to compute the loss, so `n = N = 80`. Meaning, we are performing batch gradient descent.

Enter to Rename, ⇧Enter to Preview

Step 2 - computing the loss

![](chrome-extension://cjedbglnccaioiolemnfhjncicchinao/udata/kkzLMn56Y0v/setting-gears.png)

> **Gradient descent types:**  
> 
> *   If we use all points in the training set (`n = N`) to compute the loss, we are performing a batch gradient descent.  
>     
> *   If we were to use a single point (`n = 1`) each time, it would be a stochastic gradient descent  
>     
> *   Anything else (`n`) in-between 1 and `N` characterizes a mini-batch gradient descent

### Step 3 - Compute the gradients[#](#Step-3---Compute-the-gradients)

A gradient is a **partial derivative**. Why partial? Because one computes it with respect to (w.r.t.) a single parameter. Since we have two parameters of, `b`, and `w`, therefore, we must compute two partial derivatives.

A derivative tells you how much a given quantity changes when you slightly vary some other quantity. In our case, how much does our MSE loss change when we vary each one of our two parameters separately?

![](chrome-extension://cjedbglnccaioiolemnfhjncicchinao/udata/w6kGyp62dp6/idea.svg)

> Gradient = how much the loss changes if one parameter changes a little bit!

Using the equations above, we will now compute the gradients with respect to the `b` and `w` coefficients.

Enter to Rename, ⇧Enter to Preview

Step 3 - computing the gradient

### Step 4 - Update the parameters[#](#Step-4---Update-the-parameters)

In the final step, we use the gradients to update the parameters. Since we are trying to minimize our losses, we reverse the sign of the gradient for the update.

There is still another hyperparameter to consider; the learning rate, denoted by the Greek letter eta (that looks like the letter `n`), is the **multiplicative** factor that we need to apply to the gradient for the parameter update.

In our example, let us start with a value of 0.1 for the learning rate (which is a relatively big value as far as learning rates are concerned!).

Enter to Rename, ⇧Enter to Preview

Step 4 - updating the parameters

### Step 5 - Rinse and repeat![#](#Step-5---Rinse-and-repeat!)

Now we use the updated parameters to go back to Step 1 and restart the process.

![](chrome-extension://cjedbglnccaioiolemnfhjncicchinao/udata/kkzLMn56Y0v/setting-gears.png)

  

> **Definition of epoch:**
> 
> An epoch is complete whenever every point in the training set (`N`) has already been used in all steps: forward pass, computing loss, computing gradients, and updating parameters.

> During one epoch, we perform at least one update, but no more than `N` updates.
> 
> The number of updates (`N/n`) will depend on the type of gradient descent being used:
> 
> *   For batch (`n = N`) gradient descent, this is trivial, as it uses all points for computing the loss. One epoch is the same as one update.
> *   For stochastic (`n = 1`) gradient descent, one epoch means `N` updates since every individual data point is used to perform an update.  
>     
> *   For mini-batch (of size `n`), one epoch has `N/n` updates since a mini-batch of `n` data points is used to perform an update.

Repeating this process over and over for many epochs is training a model in a nutshell.

Practice[#](#Practice)
----------------------

Try to solve this short quiz to test your understanding of the concepts explained in this lesson:

Check all that apply. Which steps are executed inside the gradient descent training loop?

###### B)

Making predictions (forward pass)

###### C)

Randomly initializing parameters

Linear Regression in Numpy


[Source](https://www.educative.io/courses/deep-learning-pytorch-fundamentals/YMYrNvpo9rW#Data-generation)