# =============================================================================
# Chapter 1
# =============================================================================

# Video 2
#forward propogation
import numpy as np
input_data= np.array([2,3])
weights={'node_0': np.array([1,1]), 'node_1': np.array([-1,1]), 'output': np.array([2,-1])}
node_0_value=(input_data * weights ['node_0']).sum()
node_1_value=(input_data * weights ['node_1']).sum()
hidden_layer_values=np.array([node_0_value, node_1_value])
print (hidden_layer_values)
output=(hidden_layer_values*weights['output']).sum()
print(output)

# Video 3
# activation functions
#ReLU (Rectified Linear Activation)
import numpy as np
input_data= np.array([2,3])
weights={'node_0': np.array([1,1]), 'node_1': np.array([-1,1]), 'output': np.array([2,-1])}
node_0_input=(input_data * weights ['node_0']).sum()
node_0_output=np.tanh(node_0_input)
node_1_input=(input_data * weights ['node_1']).sum()
node_1_output=np.tanh(node_1_input)
hidden_layer_values=np.array([node_0_output, node_1_output])
print (hidden_layer_values)
output=(hidden_layer_values*weights['output']).sum()
print(output)


# =============================================================================
# Chapter 2
# =============================================================================

# Video 1
# the need for optimization


# Video 2
# gradient descent
# common learning rate 0.01
# to calculate the slope for a weight we need to multiply:
# >> slope of the loss function w.r.t value at the node we feed into
# >> the value of the node that feeds into out weight
# >> slope of the actvation function w.r.t value we feed into

import numpy as np
weights=np.array([1,2])
input_data=np.array([3,4])
target=6
learning_rate=0.01
preds=(weights* input_data).sum()
error=preds-target
print(error)
gradient=2*input_data*error
gradient
weights_updated=weights-learning_rate*gradient
preds_updated=(weights_updated*input_data).sum()
error_updated=preds_updated-target
print (error_updated)


# Video 3
# Backpropogation 

# Video 4
# Backpropagation in practice


# =============================================================================
# Chapter 3
# =============================================================================

# Video 1
# Creating Keras model

import numpy as np
from keras.layers import Dense
from kras.models import Sequential
predictors= np.loadtxt('predictors_data.csv', delimiter=',')
n_cols=predictors.shape[1]
model=Sequential()
model.add(Dense(100, activation='relu', input_shape=(n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))


# Video 2
# Compiling and fitting a model
# optimizer : "Adam"
# loss function: "mean_squared_error"

import numpy as np
from keras.layers import Dense
from kras.models import Sequential
predictors= np.loadtxt('predictors_data.csv', delimiter=',')
n_cols=predictors.shape[1]
model=Sequential()
model.add(Dense(100, activation='relu', input_shape=(n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
model.compile (optimizer="adam", loss="mean_squared_error")
model.fit (predictors, target)

# Video 3
# Classification models
# in classification the loss function is specified as "categorical_crossentropy"

from keral.utils import to_categorcial
data=pd.read_csv('basketball_shot_log.csv')
predictors=data.drop (['shot_result'], axis=1).as_matrix()
target=to_categorcial(data.shot_result)
model=Sequential()
model.add(Dense(100, activation='relu', input_shape=(n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorcial_crossentropy', metrics=['accuracy'])
model.fit(predictors, target)

# Video 4
# Using models
# >> save
# >> reload
# >> make predictions

from keras.models import load_model
model.save("model_file.h5")
my_model=load_model('my_model.h5')
predictions=my_model.predict (data_to_predict_with)
probablity_true=predictions[:,1]
mu_model.summary()

# =============================================================================
# Chapter 4
# =============================================================================

# Video 1
# Understanding model optimization
def get_new_model(input_shape=input_shape):
    model=Sequential()
    model.add(Dense(100, activation='relu', input_shape=input_shape))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return(model)
lr_to_test=[.000001, 0.01,1]
for lr in lr_to_test:
    model=get_new_model()
    my_optimizer=SGD(lr=lr)
    model.compile(optimizer=my_optimizer, loss='categorical_crossentropy')
    model.fit(predictors, target)


# Video 2
# Model validation

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(predictors, target, valudation_split=0.3) 
#early stopping
from keras.callbacks import EarlyStopping
early_stopping_monitor=EarlyStopping (patience=2)
model.fit(predictions, target, validation_split=0.3, epochs=20, callbacks=[early_stopping_monitor])


# Video 3
# Thinking about model capacity
# underfitting vs overfitting
# >> start with simple network and get the validation score
# >> keep increasing capacity until validation score is no longer improving


# Video 4
# Stepping up to images
# MNIST dataset (handwritten digits)
# Create the model: model
model = Sequential()
# Add the first hidden layer
model.add(Dense(50,activation='relu',input_shape=(784,)))
# Add the second hidden layer
model.add(Dense(50,activation='relu',))
# Add the output layer
model.add(Dense(10,activation='softmax'))
# Compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Fit the model
model.fit(X,y,validation_split=0.3)


# Video 5
# Final thoughts
# keras.io
# if you do not have a CUDE compatible GPU, use this to set up a cloud:
# >> http://bit.ly/2mYQXQb
