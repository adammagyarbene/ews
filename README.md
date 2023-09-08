We will be using Google Colaboratory. Colaboratory is a research tool for machine learning education and research. It’s a Jupyter notebook environment that requires no setup to use. Colaboratory works with most major browsers. And what more, it is completely free to use. Before, you ask me Colaboratory is really reliable and fast too.



## Introduction - Slide 7

Coding has been the bread and butter for developers since the dawn of computing. We’re used to creating applications by breaking down requirements into composable problems that can then be coded against. So for example, if we have to write an application that figures out a stock analytic, maybe the price divided by the ratio, we can usually write code to get the values from a data source, do the calculation and then return the result. Or if we’re writing a game we can usually figure out the rules. For example, if the ball hits the brick then the brick should vanish and the ball should rebound. But if the ball falls off the bottom of the screen then maybe the player loses their life.

We can represent that with this diagram. Rules and data go in answers come out. Rules are expressed in a programming language and data can come from a variety of sources from local variables up to databases. Machine learning rearranges this diagram where we put answers in data in and then we get rules out. So instead of us as developers figuring out the rules when should the brick be removed, when should the player’s life end, or what’s the desired analytic for any other concept, what we will do is we can get a bunch of
examples for what we want to see and then have the computer figure out the rules.

## Activity recognition example - Slide 

So consider this example, activity recognition. If I’m building a device that detects if somebody is say walking and I have data about their speed, I might write code like this and if they’re running well that’s a faster speed so I could adapt my code to this and if they’re biking, well that’s not too bad either. I can adapt my code like this. But then I have to do golf recognition too, now my concept becomes broken. But not only that, doing it by speed alone of course is quite naive. We walk and run at different speeds uphill and downhill and other people walk and run at different speeds to us.

The new paradigm is that I get lots and lots of examples and then I have labels on those examples and I use the data to say this is what walking looks like, this is what running looks like, this is what biking looks like and yes, even this is what golfing looks like. So, then it becomes answers and data in with rules being inferred by the machine. A machine learning algorithm then figures out the specific patterns in each set of data that determines the distinctiveness of each. That’s what’s so powerful and exciting about this programming paradigm. It’s more than just a new way of doing the same old thing. It opens up new possibilities that were infeasible to do before.

So, now I am going to show you the basics of creating a neural network for doing this type of pattern recognition. A neural network is just a slightly more advanced implementation of machine learning and we call that deep learning. But fortunately it's actually very easy to code. So, we're just going to jump straight into deep learning. We'll start with a simple one and then we'll move on to one that can classify benign and malignant skin diseases.


## Hello Neural Networks - Slide

To show how that works, let’s take a look at a set of numbers and see if you can determine the pattern between them. Okay, here are the numbers.

You easily figure this out y = 2x — 1

You probably tried that out with a couple of other values and see that it fits. Congratulations, you’ve just done the basics of machine learning in your head.

Check tensorflow version
```
import tensorflow as tf

print(tf.__version__)
```

Okay, here’s our first line of code. This is written using Python and TensorFlow and an API in TensorFlow called keras. Keras makes it really easy to define neural networks. A neural network is basically a set of functions which can learn patterns.

```
import numpy as np
from tensorflow import keras

# Dense = define layer of connected neurons
layer = keras.layers.Dense(units = 1, input_shape=[1])
model = keras.Sequential(layer)

```

The simplest possible neural network is one that has only one neuron in it, and that’s what this line of code does. In keras we use the word Dense to define a layer of connected neurons. There is only one Dense here means that there is only one layer and there is only single unit in it so there is only one neuron. Successive layers in keras are defined in a sequence so the word Sequential . You define the shape of what's input to the neural network in the first and in this case the only layer, and you can see that our input shape is super simple. It's just one value. You've probably seen that for machine learning, you need to know and use a lot of math, calculus probability and the like. It's really good to understand that as you want to optimize your models but the nice thing for now about TensorFlow and keras is that a lot of that math is implemented for you in functions. There are two function roles that you should be aware of though and these are loss functions and optimizers.

```
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')
```

This lines defines that for us. So, lets understand what it is.

Understand it like this the neural network has no idea of the relation between X and Y so it makes a random guess say y=x+3 . It will then use the data that it knows about, that’s the set of Xs and Ys that we’ve already seen to measure how good or how bad its guess was. The loss function measures this and then gives the data to the optimizer which figures out the next guess. So the optimizer thinks about how good or how badly the guess was done using the data from the loss function. Then the logic is that each guess should be better than the one before. As the guesses get better and better, an accuracy approaches 100 percent, the term convergence is used.

Here we have used the loss function as mean squared error and optimizer as SGD or stochactic gradient descent.

Our next step is to represent the known data. These are the Xs and the Ys that you saw earlier. The np.array is using a Python library called numpy that makes data representation particularly enlists much easier. So here you can see we have one list for the Xs and another one for the Ys. The training takes place in the fit command. Here we’re asking the model to figure out how to fit the X values to the Y values. The epochs equals 500 value means that it will go through the training loop 500 times. This training loop is what we described earlier. Make a guess, measure how good or how bad the guesses with the loss function, then use the optimizer and the data to make another guess and repeat this. When the model has finished training, it will then give you back values using the predict method.

Here’s the code of what we talked about.


```
xs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))
```

So, what output do you except — 19, right?

But when you try this in the workbook yourself you will see it gives me a value close to 19 not 19.

Why do you think this happens because the equation is y = 2x-1 .

There are two main reasons:

The first is that you trained it using very little data. There’s only six points. Those six points are linear but there’s no guarantee that for every X, the relationship will be Y equals 2X minus 1. There’s a very high probability that Y equals 19 for X equals 10, but the neural network isn’t positive. So it will figure out a realistic value for Y. The the second main reason. When using neural networks, as they try to figure out the answers for everything, they deal in probability. You’ll see that a lot and you’ll have to adjust how you handle answers to fit. Keep that in mind as you work through the code. Okay, enough talking. Now let’s get hands-on and write the code that we just saw and then we can run it.


## Excercise 2

```
def house_model(y_new):
    xs=[]
    ys=[]
    for i in range(1,10):
        xs.append(i)
        ys.append((1+float(i))*50)
    
    xs=np.array(xs,dtype=float)
    ys=np.array(ys, dtype=float)
model = keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')    
    
    model.fit(xs, ys, epochs = 4500)
    return (model.predict(y_new))
```




# Section 2

Skin cancer is an abnormal growth of skin cells, it is one of the most common cancers and unfortunately, it can become deadly. The good news though is when caught early, your dermatologist can treat it and eliminate it entirely.

Using deep learning and neural networks, we'll be able to classify benign and malignant skin diseases, which may help the doctor diagnose cancer at an earlier stage. In this tutorial, we will make a skin disease classifier that tries to distinguish between benign (nevus and seborrheic keratosis) and malignant (melanoma) skin diseases from only photographic images using TensorFlow framework in Python.

## Keras

Keras is the high-level API of the TensorFlow platform. It provides an approachable, highly-productive interface for solving machine learning (ML) problems, with a focus on modern deep learning. Keras covers every step of the machine learning workflow, from data processing to hyperparameter tuning to deployment. It was developed with a focus on enabling fast experimentation.

With Keras, you have full access to the scalability and cross-platform capabilities of TensorFlow. You can run Keras on a TPU Pod or large clusters of GPUs, and you can export Keras models to run in the browser or on mobile devices. You can also serve Keras models via a web API.
 
## NumPy

NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. 

## Pandas

Pandas is a software library written for the Python programming language for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series.

## MatPlotLib

Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy. It provides an object-oriented API for embedding plots into applications using general-purpose GUI toolkits

## Block2
For this tutorial, we'll be using only a small part of ISIC archive dataset, the below function downloads and extract the dataset into a new data folder.

This will take several minutes depending on your connection, after that, the data folder will appear that contains the training, validation and testing sets. Each set is a folder that has three categories of skin disease images.

## Block4

Now that we have the dataset in our machine, let's find a way to label these images, remember we're going to classify only benign and malignant skin diseases, so we need to label nevus and seborrheic keratosis as the value 0 and melanoma 1.

The below cell generates a metadata CSV file for each set, each row in the CSV file corresponds to a path to an image along with its label (0 or 1)

The generate_csv() function accepts 2 arguments, the first is the path of the set.

The second parameter is a dictionary that maps each skin disease category to its corresponding label value (again, 0 for benign and 1 for malignant).

The reason I did a function like this is the ability to use it on other skin disease classifications (such as melanocytic classification), so you can add more skin diseases and use it for other problems as well.

## Block5

After we successfully ran the previous cell, we can notice that 3 CSV files will appear in our current directory. Now let's load our data into DataSets.

Now we have loaded the dataset (train_ds and valid_ds), each sample is a tuple of filepath (path to the image file) and label (0 for benign and 1 for malignant).

## Block6

Let's load the images

## Block7

Everything is as expected, now let's prepare this dataset for training

## Block8

As you can see, it's extremely hard to differentiate between malignant and benign diseases, let's see how our model will deal with it.

Great, now our dataset is ready, let's dive into building our model.

## Block9

Notice before, we resized all images to (299, 299, 3), and that's because of what InceptionV3 architecture expects as input, so we'll be using transfer learning with TensorFlow Hub library to download and load the InceptionV3 architecture along with its ImageNet pre-trained weights.

We set trainable to False so we won't be able to adjust the pre-trained weights during our training, we also added a final output layer with 1 unit that is expected to output a value between 0 and 1 (close to 0 means benign, and 1 for malignant).

After that, since this is a binary classification, we built our model using binary crossentropy loss, and used accuracy as our metric (not that reliable metric, we'll see sooner why), here is the output of our model summary:

## Block10

Since fit() method doesn't know the number of samples there are in the dataset, we need to specify steps_per_epoch and validation_steps parameters for the number of iterations (the number of samples divided by the batch size) of the training set and validatiion set respectively.

## Block11

Now that we've trained our model to predict the benign and malignant classes let's make a function that predicts the class of any image passed to it.

## Conclusion






