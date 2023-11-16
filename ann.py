"

import pandas as pd
import numpy as np
import tensorflow as tf
#Information about the dataset
#Column 1: Temperature (°C)
#Column 2: Pressure (milibar)
#Column 3: Humidity
#Column 4: Exhaust Vacuum (cm Hg)
#Column 5: Net electrical output (MW/h)

"""##Data Preprocessing

### Importing the dataset
"""

dataset = pd.read_excel("Folds5x2_pp.xlsx")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
print(X)
print(y)

"""### Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""##Building the ANN

### Initializing the ANN
"""

ann = tf.keras.models.Sequential()

"""### Adding the input layer and the first hidden layer"""

ann.add(tf.keras.layers.Dense(units = 6, activation = "relu"))
#Activation function Rectifier

"""### Adding the second hidden layer"""

ann.add(tf.keras.layers.Dense(units = 6, activation = "relu"))

"""### Adding the output layer"""

ann.add(tf.keras.layers.Dense(units = 1))
#No activation function because we are doing regression, the AF are more usefull for classification

"""## Training the ANN

### Compiling the ANN
"""

ann.compile(optimizer = "adam", loss = "mean_squared_error")
#Adam performs SGD (Stochastic gradient descent)

"""### Training the ANN model on the Training set"""

ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

"""

### Results of test set prediction
"""

y_pred = ann.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))