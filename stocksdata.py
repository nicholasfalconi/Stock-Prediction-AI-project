import time 
import datetime
import pandas as pd 
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
plt.style.use('bmh')
scaler = MinMaxScaler(feature_range=(0, 1))

ticker = input("Hello what stock would you like to look at: ")
period1 = int(time.mktime(datetime.datetime(2019,12,1,23,59).timetuple()))
period2 = int(time.mktime(datetime.datetime(2021,4,7,23,59).timetuple()))
interval = '1d' # Daily : 1D. Montly: 1m

query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'

df = pd.read_csv(query_string)
nf = pd.read_csv(query_string)

df.to_csv("stock.csv")

#Get the closing Price data
df = df[['Close']]


plt.figure(figsize=(16,8))
plt.title(ticker, fontsize = 18)
plt.xlabel('Days', fontsize= 18)
plt.ylabel('Close Price USD ($)', fontsize = 18)
plt.plot(df['Close'])
plt.show()

#Create variable to predict 'X' days put into the future
future_days = 25

#Create a new column (the target or dependent variable) shifted 'x' units/days up
df['Prediction'] = df[['Close']].shift(-future_days)

#Create feature data set
X = np.array(df.drop(['Prediction'], 1))[:-future_days]
y = np.array(df['Prediction'])[:-future_days]


def linear_regression_model(Amat,yvec,df): 
    
    #Split the data into 75% Training and 25% Testing
    x_train, x_test, y_train, y_test =  train_test_split(Amat,yvec,test_size = 0.25)

    #Create the linear regression model
    lr = LinearRegression().fit(x_train, y_train)

    #Get the feature data, 
    #AKA all the rows from the original data set except the last 'x' days
    x_future = df.drop(['Prediction'], 1)[:-future_days]

    #Get the last 'x' rows
    x_future = x_future.tail(future_days)
    
    #Convert the data set into a numpy array
    x_future = np.array(x_future)

    #Show the model linear regression prediction
    lr_prediction = lr.predict(x_future)

    preds = lr.predict(x_test)

    rms=np.sqrt(np.mean(np.power((np.array(y_test)-np.array(preds)),2)))

    #Plot the data
    valid =  df[X.shape[0]:]
    valid['Predictions'] = lr_prediction #Create a new column called 'Predictions' that will hold the predicted prices
    plt.figure(figsize=(16,8))
    plt.title('Linear Regression Model')
    plt.xlabel('Days',fontsize=18)
    plt.ylabel('Close Price USD ($)',fontsize=18)
    plt.plot(df['Close'])
    plt.plot(valid[['Close','Predictions']])
    plt.legend(['Train', 'Val', 'Prediction' ], loc='lower right')
    plt.show()

    #return rmse
    return rms

def decision_tree_prediction(Amat,yvec,df):
    #Split the data into 75% Training and 25% Testing
    x_train, x_test, y_train, y_test =  train_test_split(Amat,yvec,test_size = 0.25)

    #Create the decision tree regressor model
    tree = DecisionTreeRegressor().fit(x_train, y_train)

    #Get the feature data, 
    #AKA all the rows from the original data set except the last 'x' days
    x_future = df.drop(['Prediction'], 1)[:-future_days]

    #Get the last 'x' rows
    x_future = x_future.tail(future_days)
    
    #Convert the data set into a numpy array
    x_future = np.array(x_future)

    #Show the model tree prediction
    tree_prediction = tree.predict(x_future)
    preds = tree.predict(x_test)

    rms=np.sqrt(np.mean(np.power((np.array(y_test)-np.array(preds)),2)))
    

    #Plot the data
    valid =  df[X.shape[0]:]
    valid['Predictions'] = tree_prediction #Create a new column called 'Predictions' that will hold the predicted prices
    plt.figure(figsize=(16,8))
    plt.title('Decision tree Model')
    plt.xlabel('Days',fontsize=18)
    plt.ylabel('Close Price USD ($)',fontsize=18)
    plt.plot(df['Close'])
    plt.plot(valid[['Close','Predictions']])
    plt.legend(['Train', 'Val', 'Prediction' ], loc='lower right')
    plt.show()

    #return rmse
    return rms

def knn_model(Amat,yvec,df): 
    #Split the data into 75% Training and 25% Testing
    x_train, x_test, y_train, y_test =  train_test_split(Amat,yvec,test_size = 0.25)

    #scaling data
    x_train_scaled = scaler.fit_transform(x_train)
    x_train = pd.DataFrame(x_train_scaled)

    x_test_scaled = scaler.fit_transform(x_test)
    x_test = pd.DataFrame(x_test_scaled)

    #using gridsearch to find the best parameter
    params = {'n_neighbors':np.arange(1,25)}
    knn = neighbors.KNeighborsRegressor()
    model = GridSearchCV(knn, params, cv=5)

    #fit the model and make predictions
    model.fit(x_train,y_train)

    #Get the feature data, 
    #AKA all the rows from the original data set except the last 'x' days
    x_future = df.drop(['Prediction'], 1)[:-future_days]

    #Get the last 'x' rows
    x_future = x_future.tail(future_days)
    
    #Convert the data set into a numpy array
    x_future = np.array(x_future)

    x_future_scaled = scaler.fit_transform(x_future)
    x_future = pd.DataFrame(x_future_scaled)

    #Show the knn prediction
    knn_predictions = model.predict(x_future)
    preds = model.predict(x_test)
    
    #Calculating the RMSE
    rms=np.sqrt(np.mean(np.power((np.array(y_test)-np.array(preds)),2)))

    #Plot the data
    valid =  df[X.shape[0]:]
    valid['Predictions'] = knn_predictions #Create a new column called 'Predictions' that will hold the predicted prices
    plt.figure(figsize=(16,8))
    plt.title('KNN Model')
    plt.xlabel('Days',fontsize=18)
    plt.ylabel('Close Price USD ($)',fontsize=18)
    plt.plot(df['Close'])
    plt.plot(valid[['Close','Predictions']])
    plt.legend(['Train', 'Val', 'Prediction' ], loc='lower right')
    plt.show()
    
    #return rmse
    return rms

def LSTM_model1(Amat,yvec,df):
    
    #Creating data set
    data = df.filter(['Close'])
    dataset = data.values

    #size
    training_data_len = math.ceil(len(dataset)*.8)

    #Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:training_data_len,:]

    #Split into x and y train 
    x_train = []
    y_train = []

    for i in range(60,len(train_data)): 
        x_train.append(train_data[i-60:i,0])
        y_train.append(train_data[i,0])
    
    #Testing data set
    test_data = scaled_data[training_data_len-60: , :]

    #Create x and y test
    x_test = []
    y_test = dataset[training_data_len:,:]

    for i in range(60,len(test_data)): 
        x_test.append(test_data[i-60:i,0])

    x_train, y_train, x_test = np.array(x_train), np.array(y_train),np.array(x_test)

    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
    
    #Build the LSTM model
    model = Sequential()

    model.add(LSTM(50, return_sequences= True, input_shape = (x_train.shape[1],1)))
    model.add(LSTM(50, return_sequences= False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer = 'adam',loss='mean_squared_error')

    #Train the model
    model.fit(x_train, y_train, batch_size = 1, epochs=1)

    preds = model.predict(x_test)
    preds = scaler.inverse_transform(preds)

    rmse = np.sqrt(np.mean(np.power((np.array(y_test)-np.array(preds)),2)))

    #Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = preds
    plt.figure(figsize=(16,8))
    plt.title('LSTM Model')
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Close Price USD ($)',fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close','Predictions']])
    plt.legend(['Train', 'Val', 'Prediction' ], loc='lower right')
    plt.show()

    return rmse

def LSTM_model(Amat,yvec,df): 
    #creating dataframe
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

    #size
    training_data_len = math.ceil(len(data)*.8)

    for i in range(0,len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['Close'][i] = data['Close'][i]

    #setting index
    new_data.index = new_data.Date
    new_data.drop('Date', axis=1, inplace=True)

    #creating train and test sets
    dataset = new_data.values

    print(dataset)

    train = dataset[0:training_data_len,:]
    valid = dataset[training_data_len:,:]

    #converting dataset into x_train and y_train
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = [], []
    for i in range(60,len(train)):
        x_train.append(scaled_data[i-60:i,0])
        y_train.append(scaled_data[i,0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

    #predicting 246 values, using past 60 from the train data
    inputs = new_data[len(new_data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs  = scaler.transform(inputs)

    X_test = []
    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i-60:i,0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)

    rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))

    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = closing_price
    plt.figure(figsize=(16,8))
    plt.title('LSTM Model with date')
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Close Price USD ($)',fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close','Predictions']])
    plt.legend(['Train', 'Val', 'Prediction' ], loc='lower right')
    plt.show()

    return rms

#print(linear_regression_model(X,y,df))
print(decision_tree_prediction(X,y,df))
print(knn_model(X,y,df))
print(LSTM_model(X,y,nf))
print(LSTM_model1(X,y,df))