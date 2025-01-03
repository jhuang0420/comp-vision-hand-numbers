import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical # type: ignore
from keras.models import Sequential # type: ignore
from keras.layers import Input, Flatten, Dense, Dropout # type: ignore

train = pd.read_csv("./landmark_data.csv")
target = "label"
labels = labels = train[target].unique().size

X = train.drop(columns=[target])
y = train[target]

X_shaped = X.values.reshape(-1,21,3)

X_min = X_shaped.min(axis=(0, 1), keepdims=True)  # Min for x, y, z across all samples
X_max = X_shaped.max(axis=(0, 1), keepdims=True)  # Max for x, y, z across all samples

X_normalized = (X_shaped - X_min) / (X_max - X_min)

y_ohe = to_categorical(y, num_classes = labels) 

X_train, X_val, y_train, y_val = train_test_split(X_normalized, y_ohe, test_size=0.2, random_state=42)

X = train.drop(columns=[target])
y = train[target]

model = Sequential([
    Input(shape=(21,3)),
    
    Dense(32, activation='relu'), 
    Dropout(0.2),

    Dense(64, activation='relu'), 
    Dropout(0.2),
    
    Dense(32, activation='relu'), 
    Dropout(0.2),
        
    Flatten(),                     
    Dense(32, activation='relu'), 
    Dropout(0.5),                  
    Dense(labels, activation='softmax')    
])

model.compile(
    optimizer='adam',   # Adaptive moment estimation
    loss='categorical_crossentropy', # Loss function for categorical prediction
    metrics=['accuracy'] # Classification uses accuracy (correct/total)
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,       # Number of times model will run through training data
                     # More may lead to better prediction but also overfit to training
    batch_size=16    # Number of samples in each batch, smaller generalizes better
                     # Larger speeds up training but may overfit
)

model.save('my_model.keras')

















