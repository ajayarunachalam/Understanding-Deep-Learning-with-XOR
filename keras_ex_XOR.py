
# coding: utf-8

# In[1]:

# import libraries
import numpy as np
import pandas as pd
import keras


# In[2]:

# There are two ways to build Keras models: sequential and functional.
# In sequential you create models layer-by-layer & in functional we can connect layers to (literally) any other layer 
# rather than just the previous and next layers
from keras.models import Sequential
from keras.layers.core import Activation, Dense


# In[3]:

# setup predictors & target data according to the functioning of XOR GATE
train_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
target_data = np.array([[0],[1],[1],[0]],"float32")


# # CONSTRUCTING THE MODEL

# In[4]:

# Initialize a sequential model
model= Sequential()


# In[5]:

## create the model & add layers ##
# We use the dense function to define fully connected layers and number of neurons in each layer. 
# The model above consist of an input layer with k neurons (16),and output layer with 1 neuron. 
# In practice, the output layer consist of 1 neuron for a regression and binary classification problem 
# and n neurons for a multi-class classification, where n is the number of classes in the target variable. 
# We also specify the activation function in each layer. 
# Keras support a number of activation functions, including the popular Rectified linear unit (relu) , 
# softmax, sigmoid, tanh, exponential linear unit (elu) among others. 
# Obviously, there are tons of other features and arguments that you can use when configuring, 
# compiling and fitting your model. See keras documentation for more details.
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1,activation='sigmoid'))


# In[6]:

## compile the model
# When compiling the model, we have to specify the objective function.
# Available functions include mean_squared_error for regression problem 
# binary_crossentropy and categorical_crossentropy for binary and multi-class classification, respectively. And many more.
# The second required argument is the optimizer for estimating and updating the model parameters. 
# Keras support several optimizers- Stochastic gradient descent (sgd) optimizer, 
# Adaptive Monument Estimation (adam), Adaptive learning rate (Adadelta) among others. 
# We use accuracy as the metric to assess the performance of the model. Check more details at https://keras.io/metrics/
model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])


# In[7]:

## fit the model
# In the crucial step of fitting the model, we specify the epochs i.e., 
# the number of times the algorithm sees the entire training data
# and the batch size i.e., the size of sample to be passed through the algorithm in each epoch. 
# A training sample of size 10, for example, with batch size =2, will give 5 batches and hence 5 iterations per epoch. 
# If epoch = 4, then we have 20 iterations for training. So, how many epochs do we use? 
# Too few, we risk underfitting and too many risk overfitting. 
# The early stopping function helps the model from overfitting.

# By setting verbose 0, 1 or 2 you just say how do you want to see the training progress for each epoch.
# verbose=0 will show you nothing (silent)
# verbose=1 will show you an animated progress bar
# verbose=2 will just mention the number of epoch 
model.fit(train_data,target_data,nb_epoch=500, verbose=2)


# In[8]:

# create unseen new data to validate our model to examine the functioning of XOR GATE
unseen_data = np.array([[1,0],[1,1],[1,0],[0,1],[0,0]], "float32")


# In[9]:

# prediction
print(model.predict(unseen_data).round())


# In[10]:

# check for different nb_epochs
model.fit(train_data,target_data,nb_epoch=10, verbose=1)


# In[11]:

# prediction
print(model.predict(unseen_data).round())


# In[ ]:



