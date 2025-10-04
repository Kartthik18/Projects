import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocessing import load_data, create_dataset
from model import build_lstm

# Load data
closedf, scaler = load_data("data/btc.csv")

# Train-test split
training_size = int(len(closedf) * 0.60)
train_data, test_data = closedf[0:training_size, :], closedf[training_size:len(closedf), :1]

# Dataset creation
time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape for LSTM [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build and train
model = build_lstm(time_step)
model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse scaling
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Metrics
rmse = np.sqrt(mean_squared_error(y_test_inv, test_predict))
mae = mean_absolute_error(y_test_inv, test_predict)
r2 = r2_score(y_test_inv, test_predict)

print(f"Test RMSE: {rmse}")
print(f"Test MAE: {mae}")
print(f"Test R² Score: {r2}")
