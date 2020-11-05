"""
https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
Predict the pollution for the next day 
based on the weather conditions and pollution over the last 24 hours.
"""
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

n_day = 7  #based on past n_day to predict aqi
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# load dataset
dataset = read_csv(r"D:\workspace\ML\air.csv", header=0, index_col=0)
values = dataset.values
# ensure all data is float
values = values.astype('float32')
# standlize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
min_aqi= min(dataset['AQI'])
max_aqi= max(dataset['AQI'])
# frame as supervised learning
reframed = series_to_supervised(scaled, n_day, 1)
print(reframed.head())
# specify the number of lag hours
n_features = 7

# split into train and test sets
values = reframed.values
n_train = 2000
train = values[:n_train, :]
test = values[n_train:, :]
# split into input and outputs
train_X, train_y = train[:, :n_day*n_features], train[:, -n_features]
test_X, test_y = test[:, :n_day*n_features], test[:, -n_features]
print(train_X.shape, len(train_X), train_y.shape)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((len(train_X), n_day, n_features))
test_X = test_X.reshape((len(test_X), n_day, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = keras.Sequential()
model.add(
  keras.layers.Bidirectional(
    keras.layers.LSTM(
      units=50, 
      input_shape=(train_X.shape[1], train_X.shape[2])
    )
  )
)
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=32, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
# invert scaling for forecast
inv_yhat = yhat*(max_aqi-min_aqi)+min_aqi
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = test_y*(max_aqi-min_aqi)+min_aqi
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)