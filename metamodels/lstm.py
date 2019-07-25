from simulations import simulation, simulation2
from pandas import DataFrame
from pandas import Series
from pandas import concat
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Bidirectional
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy



# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='relu'), batch_input_shape=(batch_size, X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
        print('Epoch {}'.format(i))
    return model


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


# load dataset

sim = simulation2.Simulator(50)
sim.simulate()

s = simulation.Simulation([[1, 1]],  # to_plot, to_report
                          [[0.1, [0.2, 0.1], [15, 2], [30, 2]]],  # interarrivals, demand, replenishment_lead, expiry
                          [[70.0, 110.0, 5.0, 30.0, 100.0, 100.0]],  # purchase price, sales price, handling, backorder, overflow, recycle
                          [[50, 35]])  # storage, reorder point

s.simulate()


#raw_values = sim.stats.inventory_vector
raw_values = s.w.products[0].stats.storage
raw_values = raw_values[0::30]
print(len(raw_values))
diff_values = difference(raw_values, 1)

# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# split data into train and test-sets
train, test = supervised_values[0:-30], supervised_values[-30:]

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

# fit the model
lstm_model = fit_lstm(train_scaled, 1, 10)
# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
#lstm_model.predict(train_reshaped, batch_size=1)

# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, 1, X)
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    # invert differencing
    yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
    # store forecast
    predictions.append(yhat)
    expected = raw_values[len(train) + i + 1]
    print('Time=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, expected))

# report performance
mse = mean_squared_error(raw_values[-30:-2], predictions[1:-1])
rmse = sqrt(mse)
ape = []
real_values = raw_values[-30:-2]
raw_value = raw_values[-30:-2]
predictions = predictions[1:-1]
for i in range(len(predictions)):
    value = abs(predictions[i]-real_values[i])/real_values[i]
    if value < 1:
        ape.append(value)

mape = sum(ape)/len(ape)*100


print('Test RMSE: %.3f' % rmse)
print('Test MSE: %.3f' % mse)
print('Mean absolute percentage error: ', round(mape,2), "%")

# plot
pyplot.plot(raw_values[-30:-2], label='simulation')
pyplot.plot(predictions[1:-1], label='predicted by LSTM neural network')
pyplot.xlabel('time')
pyplot.ylabel('inventory level')
pyplot.grid()
pyplot.legend()
pyplot.show()