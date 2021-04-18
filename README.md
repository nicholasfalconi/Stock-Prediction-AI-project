# Stock Price predictor (Machine learning project)

This project was made to look at the rmse values of different machine learning models on a stock prices data set. The program is being developed to pick the best model to predict a stocks price and eventually help determine when the right time to buy a stock would be. This project is not completely finished but at the moment allows for the user to enter a stock ticker and plots are displayed to show how accurate the models predictions are.

# Functions

###### linear_regression_model: 

This function is used to perform a linear regression model on the price price of a stock and a trian and test set are used to train the data on the tain set and to test it on the remaining 25% of the data. The function returns the RMSE of the model. 

###### decision_tree_prediction: 

This function is used to perform a decision tree model on the price of a stock and uses a train and test set to get the RMSE values of the model. These values are then returned. 

###### knn_model: 

This function is used to perform a knn nearest neighbor model on the price of a stock and uses a train and test set to get the RMSE values of the model. These values are then returned.

###### LSTM_model:

This function is used to perform a LSTM (Long term short memory) model on the price of a stock and uses a train and test set to get the RMSE values of the model. These values are then returned.

