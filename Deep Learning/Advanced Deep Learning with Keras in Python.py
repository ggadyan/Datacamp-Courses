# =============================================================================
# Chapter 1
# =============================================================================
 
# Video 1
# Keras input and dense layers
import pandas as pd
games_season=pd.read_csv("games_season.csv")
games_season.head()
games_tourney=pd.read_csv("games_tourney.csv")
games_tourney.head()

from keras.layers import Input
input_tensor=Input(shape=(1,))
print (input_tensor)

from keras.layers import Dense
output_layer=Dense(1)
print(output_layer)
output_tensor=output_layer(input_tensor)
print(output_tensor)


# Video 2
# Keras models

from keras.layers import Input, Dense
input_tensor= Input(shape=(1,))
output_tensor= Dense(1)(input_tensor)

from keras.models import Model
from keras.utils import plot_model
model=Model(input_tensor, output_tensor)
model.compile(optimizer='adam', loss='mae') #mae is mean absolute error
model.summary()

# plot model using keras
input_tensor= Input(shape=(1,))
output_layer= Dense(1, nam="Predicted-Score_Diff")
output_tensor=output_layer(input_tensor)
model=Model (input_tensor, output_tensor)
plot_model(model, to_file='model.png')

from matplotlib import pyplot as plt
img=plt.imread('model.png')
plt.imshow(img)
plt.show()


# Video 3
# Fit and evaluate a model
import pandas as pd
games_tourney=pd.read_csv('games_tourney.csv')
games_tourney.head()

from keras.models import Model
from keras.layers import Input, Dense
input_tensor= Input(shape=(1,))
output_tensor=Dense(1)(input_tensor)
model=Model(input_tensor, output_tensor)
model.compile(optimizer='adam', loss='mae')

from pandas import read_csv
games=read_csv ('games_tourney.csv')
model.fit(games['seed_diff'], games['score_diff'], batch_size=64, validation_split=.20, verbose= True)
model.evaluate(games_test['seed_diff'], games_test['score_diff']) #second check



# =============================================================================
# Chapter 2
# =============================================================================

# Video 1
# Category embeddings
from keras.layers import Embedding
input_tensor=Input(shape=(1,))
n_teams=10887
embed_layer=Embedding(input_dims=n_teams, input_length=1, output_dim=1, name='Team-Strength-Lookup')
embed_tensor=embed_layer(input_tensor)

from keras.layers import Flatten
flatten_tensor=Flatten()(embed_tensor)
model=Model(input_tensor,flatten_tensor)


# Video 2
# Shared layers
#creating two inputs
input_tensor_1=Input((1,))
input_tensor_2=Input((1,))
shared_layer=Dense(1)
output_tensor_1=shared_layer(input_tensor_1)
output_tensor_2=shared_layer(input_tensor_2)

# Video 3
# Merge layers
# >> add, subtract, multiply, concatenate

from keras.layers import Input, Add
in_tensor_1=Input((1,))
in_tensor_2=Input((1,))
out_tensor=Add()[in_tensor_1, in_tensor_2]

from keras.models import Model
model=Model([in_tensor_1, in_tensor_2], out_tensor)
model.compile(optimizer="adam", loss='mean_absolute_error')


# Video 4
# predict from your model

model.fit([data_1, data_2], target)
model.predict([np.array([[1]]), np.array([[2]])])
array([[3.]], dtype=float32)
model.predict([np.array([[42]]), np.array([[119]])])
array([[161.]], dtype=float32)

model.evaluate([np.array([[-1]]), np.array([[-2]])], np.array([[-3]]))


# =============================================================================
# Chapter 3
# =============================================================================
# Video 1
# Three-input models
from keras.layers import Input, Concatenate, Dense
in_tensor_1=Input((1,))
in_tensor_2=Input((1,))
in_tensor_3=Input((1,))
out_tensor=Concatenate()([in_tensor_1, in_tensor_2, in_tensor_3])
output_tensor=Dense(1)(out_tensor)

from keras.models import Model
model=Model([in_tensor_1, in_tensor_2, in_tensor_3], out_tensor)

shared_layer=Dense(1)
shared_tensor_1=shared_layer(in_tensor_1)
shared_tensor_2=shared_layer(in_tensor_2)
out_tensor=Concatenate()([shared_tensor_1, shared_tensor_2, shared_tensor_3 ])
out_tensor=Dense(1)(out_tensor)

from keras.models import Model
model=Model([in_tensor_1, in_tensor_2, in_tensor_3], out_tensor)
model.compile(loss="mae", optimizer="adam")
model.fit([[train['col1'], train['col2'], train['col3']], train_data['target']])
model.evaluate([[train['col1'], train['col2'], train['col3']], train_data['target']])


# Video 2
# Summarizing and plotting models

# Video 3
# Stacking models
from pandas import read_csv
games_season=pd.read_csv("games_season.csv")
games_season.head()
games_tourney=pd.read_csv("games_tourney.csv")
games_tourney.head()

in_data_1=games_tourney['team_1']
in_data_2=games_tourney['team_2']
in_data_3=games_tourney['home']
pred=regular_season_model.predict([in_data_1, in_data_2, in_data_3])
games_tourney['pred']=pred
games_tourney.head()

games_tourney[['home', 'seed_dif', 'pred']].head()


from keras.layers import Input, Dense
in_tensor=Input(shape=(3,))
out_tensor=Dense(1)(in_tensor)

from keras.models import Model
model=Model(in_tensor, out_tensor)
model.compile(optimizer='adam', loss='mae')
train_X=train_data[['home', 'seed_diff', 'pred']]
train_y=train_data['score_diff']
model.fit(train_X, train_y, epochs=10, valudation_split=0.10)
test_X=test_data[['home', 'seed_diff', 'pred']]
test_y=test_data['score_diff']
model.evaluate(test_X, test_y)


# =============================================================================
# Chapter 4
# =============================================================================
# Video 1
# Two-output models

from keras.layers import Input, Concatenate, Dense
input_tensor=Input(shape=(1,))
output_tensor=Dense(2)(input_tensor)

from keras.models import Model
model=Model(input_tensor, output_tensor)
model.compile(optimizer='adam', loss='mean_absolute_error')

# example
games_tourney_train[['seed_diff', 'score_1', 'score_2']].head()
X=games_tourney_train[['seed_diff']]
y=games_tourney_train[['score_1', 'score_2']]
model.fit(X,y, epochs=500)
model.get_weights()
model.evaluate(X,y)


# Video 2
# Single model for classification and regression

from keras.layers import Input, Dense
input_tensor=Input(shape=(1,))
output_tensor_reg=Dense(1)(input_tensor)
output_tensor_class=Dense(1, activation='sigmoid')(output_tensor_reg)

from keras.models import Model
model=Model(input_tensor, [output_tensor_reg, output_tensor_class])
model.compile(loss=['mean_absolute_error', 'binary_crossentropy'], optimizer='adam')

X=games_tourney_train[['seed_diff']]
y_reg=games_tourney_train[['score_diff']]
y_class=games_tourney_train[['won']]
model.fit(X, [y_reg, y_class], epochs=100)
model.get_weights()
from scipy.special import expit as sigmoid
print(sigmoid(1*0.012870609+0.00734114))

X=games_tourney_train[['seed_diff']]
y_reg=games_tourney_train[['score_diff']]
y_class=games_tourney_train[['won']]
model.evaluate(X, [y_reg, y_class])

# Video 3
# Wrap-up
# for input text, you first do the embeding layer then LSTM layer
# for input numerics you directly skip to the next layer (dense)
# for input images, you first do the convolutional later (2) (CNN)
# then for all of them we do concat layer and output layer

#input layer >> regression output(linear) >> classification output (sigmoid)

input_tensor=Input((100,))
hidden_tensor=Dense(256, activation='relu')(input_tensor)
hidden_tensor=Dense(256, activation='relu')(hidden_tensor)
hidden_tensor=Dense(256, activation='relu')(hidden_tensor)
output_tensor=Concatenate()([input_tensor, hidden_tensor])
output_tensor=Dense(256, activation='relu')(output_tensor)
# visualization of loss landscape of neural networks (skip connections)
