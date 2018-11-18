# Predicting-Energy-Consumption-oF-Building
1) Problem Statement
    In the competition hosted on 'https://www.drivendata.org', challenge was to predict energy comsumption of building given past consumption history of building. Apart from consumption history other data like temparature, surface area of building are also given.
2) Model structure
    a)  As the input data is in the form of time series I am using LSTM Network
    b)  LSTM layer will be looking at 24 previous instance data to predict current instance energy consumption.
    c)  Included is the Python code to train prepare data, train LSTM network and predict consumption for test data.
3) Results 
    On training data MSE is   :- 0.07
    On validation data MSE is :- 0.13
4) Data :-
    consumption_train.csv contains the training data and cold_start.csv contains the data used for testing.
    Data can be downladed from :- https://www.drivendata.org/competitions/55/schneider-cold-start/
    
