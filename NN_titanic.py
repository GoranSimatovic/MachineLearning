import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam, SGD
from keras.metrics import categorical_crossentropy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


model = keras.models.Sequential()
model.add(keras.layers.Dense(100,input_shape=(6,), activation="relu"))
model.add(keras.layers.Dense(360, activation="relu"))
model.add(keras.layers.Dense(360, activation="relu"))
model.add(keras.layers.Dense(2, activation="sigmoid"))

model.compile(loss="sparse_categorical_crossentropy",
optimizer="Adam",
metrics=["accuracy"])

data = pd.read_csv("titanic.csv")

data = data.drop(['Name'], axis=1)
data['Sex'].replace('female',0,inplace=True)
data['Sex'].replace('male',1,inplace=True)

scaler = MinMaxScaler(feature_range=(0,1))
data.astype('float64').dtypes
data = scaler.fit_transform(data)

train_set, test_set = train_test_split(data, test_size=0.2)

train_set_labels = train_set[:,0]
test_set_labels = test_set[:,0]

train_set = train_set[:,1:]
test_set = test_set[:,1:]

history = model.fit(train_set, train_set_labels, batch_size=10,epochs = 250, shuffle = True, verbose = 3)

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
