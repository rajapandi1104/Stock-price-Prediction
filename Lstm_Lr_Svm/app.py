import warnings
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import streamlit as st
import numpy as np
import pandas as pd
import pandas_datareader as data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
plt.style.use('bmh')


st.title('STOCK MARKET PREDICTION USING LINEAR REGRESSION MODEL')
user_input = st.text_input('Enter The Stock Ticker', 'AAPL')
start = '2019-01-01'
end = '2021-12-15'
df = data.DataReader(user_input, 'yahoo', start, end)
df2 = df
# set the date as the index
# df = df.set_index(data.DatetimeIndex(df['Date'].values))

# describing data
if st.checkbox("Show raw data", False):
    st.subheader('Data From 2010-2021')
    st.write(df.describe())

# df.shape

st.subheader(f'Closing Price vs Time Chart of {user_input}')
fig = plt.figure(figsize=(12, 6))
plt.title(user_input)
plt.xlabel('Year-Month')
plt.ylabel('Close Price USD ($)')
plt.plot(df['Close'])
plt.legend(['Close Price'])
st.pyplot(fig)

df = df[['Close']]
# df.head(4)


future_days = 25
df['Prediction'] = df[['Close']].shift(-future_days)
# df.head(4)

X = np.array(df.drop(['Prediction'], 1))[:-future_days]
# print(X)

y = np.array(df['Prediction'])[:-future_days]
# print(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

lr = LinearRegression().fit(x_train, y_train)

x_future = df.drop(['Prediction'], 1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)
# x_future


# print()
lr_prediction = lr.predict(x_future)
# print(lr_prediction)


# prediction using linear regression
st.subheader(f'Linear Regression Prediction of {user_input}')
predictions = lr_prediction
valid = df[X.shape[0]:]
valid['Predictions'] = predictions
figlr = plt.figure(figsize=(12, 6))
plt.title(user_input)
plt.xlabel('Year-Month')
plt.ylabel('Close Price USD ($)')
plt.plot(df['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Original price', 'test data price', 'Prediction price'])
st.pyplot(figlr)


# LSTM
st.write(' ')
st.write(' ')
st.write(' ')
st.write(' ')
st.write(' ')
st.title('STOCK MARKET PREDICTION USING LSTM MODEL')

# VISU
st.subheader('Closing Price vs Time Chart')
fig1 = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.xlabel('Year-Month')
plt.ylabel('Close Price USD ($)')
plt.legend(['Original Close price'])
st.pyplot(fig1)


# 100ma vs closeprice
st.subheader('Closing Price vs Time Chart With 100MA')
ma100 = df.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(12, 6))
plt.xlabel('Year-Month')
plt.ylabel('Close Price USD ($)')
plt.plot(df.Close)
plt.plot(ma100)
plt.legend(['Original Close Price', '100Days Moving Average'])
st.pyplot(fig2)


# 100ma+200ma+closeprice
st.subheader('Closing Price vs Time Chart With 100MA & 200mMA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(12, 6))
plt.xlabel('Year-Month')
plt.ylabel('Close Price USD ($)')
plt.plot(df.Close, 'b')
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')

plt.legend(['Original Close Price', '100Days Moving Average',
           '200Days Moving Average'])
st.pyplot(fig3)

# spliting data into tarining and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)

# spliting the data

# load my model
model = load_model('keras_model.h5')

# testing part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

# making prediction
y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# final graph
st.subheader('Prediction vs Original')
figlstm = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='predicted Price')
plt.xlabel('Time')
plt.ylabel('Close Price USD ($)')
plt.legend()
st.pyplot(figlstm)





# SVM
st.write(' ')
st.write(' ')
st.write(' ')
st.write(' ')
st.write(' ')
st.title('Support Vector Machine')
# Import data
# Machine learning


st.pyplot(fig)
# For data manipulation

# To plot
plt.style.use('seaborn-darkgrid')

# To ignore warnings
warnings.filterwarnings("ignore")

# Create predictor variables
df2['Open-Close'] = df2.Open - df2.Close
df2['High-Low'] = df2.High - df2.Low

# Store all predictor variables in a variable X
X = df2[['Open-Close', 'High-Low']]

# Target variables
y = np.where(df2['Close'].shift(-1) > df2['Close'], 1, 0)


split_percentage = 0.8
split = int(split_percentage*len(df2))

# Train data set
X_train = X[:split]
y_train = y[:split]

# Test data set
X_test = X[split:]
y_test = y[split:]
cls = SVC().fit(X_train, y_train)
df2['Predicted_Signal'] = cls.predict(X)
df2['Return'] = df2.Close.pct_change()
df2['Strategy_Return'] = df2.Return * df2.Predicted_Signal.shift(1)
df2['Cum_Ret'] = df2['Return'].cumsum()
df2['Cum_Strategy'] = df2['Strategy_Return'].cumsum()

# final graph
st.subheader('Prediction vs Original')
fig3 = plt.figure(figsize=(12, 6))
plt.plot(df2['Cum_Ret'], 'r', label='predicted Price')
plt.plot(df2['Cum_Strategy'], 'b', label='original price')
plt.xlabel('Time')
plt.ylabel('Close Price USD ($)')
plt.legend()
st.pyplot(fig3)


# LSTM VS LR VS SVM

st.write(' ')
st.write(' ')
st.write(' ')
st.write(' ')
st.write(' ')
st.title('Linear Regression vs LSTM vs SVM')
col1, col2, col3 = st.columns(3)

with col1:
    st.header("LR")
    st.pyplot(figlr)
with col2:
    st.header("LSTM")
    st.pyplot(fig2)
with col3:
    st.header("SVM")
    st.pyplot(fig3)
