import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

crypto = 'BTC'
aginst_currency = 'USD'

start = dt.datetime(2016, 1, 1)
end = dt.datetime.now()

data = web.DataReader(f'{crypto}-{aginst_currency}', 'yahoo', start, end)

print(data.head())

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days = 60
future_days = 30

x_train, y_train = [], []

for x in range(prediction_days, len(scaled_data)-future_days):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x+future_days, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

models = Sequential()

models.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
models.add(Dropout(0.2))
models.add(LSTM(units=50, return_sequences=True))
models.add(Dropout(0.2))
models.add(LSTM(units=50))
models.add(Dropout(0.2))
models.add(Dense(units=1))

models.compile(optimizer='adam', loss='mean_squared_error')
models.fit(x_train, y_train, epochs=25, batch_size=32)

t_start = dt.datetime(2020,10,8)
t_end = dt.datetime.now()

t_data = web.DataReader(f'{crypto}-{aginst_currency}', 'yahoo', t_start, t_end)
actual_price = t_data['Close'].values

total_dataset = pd.concat((data['Close'], t_data['Close']), axis=0)

model_input = total_dataset[len(total_dataset) - len(t_data) - prediction_days:].values
model_input = model_input.reshape(-1, 1)
model_input = scaler.fit_transform(model_input)

i_test = []

for i in range(prediction_days, len(model_input)):
    i_test.append(model_input[i - prediction_days:i, 0])

i_test = np.array(i_test)
i_test = np.reshape(i_test, (i_test.shape[0], i_test.shape[1], 1))

predict_price = models.predict(i_test)
predict_price = scaler.inverse_transform(predict_price)

plt.plot(actual_price, color='black', label='Actual Price')
plt.plot(predict_price, color='green', label='Predicted Price')
plt.title(f'{crypto} price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()

# predict next day

real_data = [model_input[len(model_input) + 1 - prediction_days:len(model_input) + 1, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = models.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print()