from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


model = keras.Sequential()
model.add(layers.Dense(units=10, kernel_initializer='uniform', input_shape=(6,)))
model.add(layers.Activation('softmax'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(units=20))
model.add(layers.Activation('relu'))
model.add(layers.Dense(units=1))


opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt)


data = pd.read_csv("titanic.csv")
data = data.drop(['Name'], axis=1)
data['Sex'].replace('female', 0, inplace=True)
data['Sex'].replace('male', 1, inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
data.astype('float64').dtypes
data = scaler.fit_transform(data)

train_set, test_set = train_test_split(data, test_size=0.2)

train_set_labels = train_set[:, 0]
test_set_labels = test_set[:, 0]

train_set = train_set[:, 1:]
test_set = test_set[:, 1:]

# single run, no fancy stuff
history = model.fit(train_set,
                    train_set_labels,
                    batch_size=10,
                    epochs=25,
                    shuffle=True,
                    verbose=3)

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
