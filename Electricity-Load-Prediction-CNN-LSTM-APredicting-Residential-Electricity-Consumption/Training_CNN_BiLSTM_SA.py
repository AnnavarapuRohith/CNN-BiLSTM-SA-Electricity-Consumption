# CNN-BiLSTM-Self-Attention for Residential Electricity Consumption

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Flatten, MaxPooling1D, Conv1D,
    LSTM, Bidirectional, Attention, Input
)

# -----------------------------
# MAPE calculation
# -----------------------------
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# -----------------------------
# Sliding window generator
# -----------------------------
def split(data, max_values):
    X, Y = [], []
    cnt = int(max_values / 8)
    c = 0
    for _ in range(cnt):
        X.append(data[c:c+8])
        Y.append(data[c+8:c+12])
        c += 1
    return np.array(X), np.array(Y)

# -----------------------------
# Load and preprocess dataset
# -----------------------------

# Load dataset
df = pd.read_csv(
    'dataset/IHEPC.csv',
    parse_dates=['datetime'],
    na_values=['nan', '?'],
    usecols=['datetime', 'Global_active_power']
)

df.set_index('datetime', inplace=True)

# Fill missing values
df['Global_active_power'] = df['Global_active_power'].fillna(
    df['Global_active_power'].mean()
)

# Convert to numpy
values = df.values

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
values = scaler.fit_transform(values)

# Flatten
values = values.reshape(-1)

# Generate sequences
max_values = len(values)
X, Y = split(values, max_values)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42
)

# Reshape for CNN input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test  = X_test.reshape((X_test.shape[0],  X_test.shape[1],  1))

# -----------------------------
# CNN-BiLSTM-Self-Attention Model
# -----------------------------

# Input layer
inputs = Input(shape=(X_train.shape[1], 1))

# CNN feature extractor
x = Conv1D(filters=8, kernel_size=3, activation='relu')(inputs)
x = Conv1D(filters=16, kernel_size=3, activation='relu')(x)
x = MaxPooling1D(pool_size=2)(x)

# Bidirectional LSTM
x = Bidirectional(LSTM(32, return_sequences=True))(x)

# Self-Attention
attention = Attention()([x, x])

# Flatten attention output
x = Flatten()(attention)

# Output layer (predict next 4 time steps)
outputs = Dense(4)(x)

# Build model
model = Model(inputs=inputs, outputs=outputs)

# Compile model
model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

# Model summary
print(model.summary())

# -----------------------------
# Train the model
# -----------------------------
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# -----------------------------
# Save trained model
# -----------------------------
model.save("model/model_CNN_BiLSTM_SA.h5")

# -----------------------------
# Test the model
# -----------------------------
predictions = model.predict(X_test)

# Flatten for evaluation
y_test_flat = y_test.flatten()
pred_flat = predictions.flatten()

# -----------------------------
# Evaluation metrics
# -----------------------------
mse = mean_squared_error(y_test_flat, pred_flat)
rmse = sqrt(mse)
mae = mean_absolute_error(y_test_flat, pred_flat)
mape = mean_absolute_percentage_error(y_test_flat, pred_flat)

print("MSE :", mse)
print("RMSE:", rmse)
print("MAE :", mae)
print("MAPE:", mape)

# -----------------------------
# Plot actual vs predicted
# -----------------------------
plt.figure(figsize=(8,4))
plt.plot(y_test_flat[:100], label="Actual", marker='.')
plt.plot(pred_flat[:100], label="Predicted", marker='.')
plt.legend()
plt.title("CNN-BiLSTM-Self-Attention Prediction")
plt.savefig("results/actual_predicted_CNN_BiLSTM_SA.png")
plt.show()
