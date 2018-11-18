import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint, Callback
import matplotlib.pyplot as plt

# Important Network varaibles
number_of_lags        = 5
number_of_hidden_size = 5
batch_size            = 128
time_steps            = 24
feature_size          = 5
counter               = 0
minimum_loss_so_far   = np.inf
nan_results           = 0
train_data_mean       = 107623.78564423509
train_data_std        = 162661.10946757084

def restore_prediction_values(num):
    return (num*train_data_std + train_data_mean)

def make_lstm_model(batch_input_shape, num_neurons):
    model = Sequential()
    print(batch_input_shape)
    model.add(LSTM(units = num_neurons, 
              batch_input_shape = batch_input_shape, 
              return_sequences=True))
    model.add(LSTM(units = num_neurons, return_sequences=True))
    model.add(LSTM(units = num_neurons, return_sequences=True))
    model.add(TimeDistributed(Dense(5)))
    model.add(LSTM(units = num_neurons, return_sequences=False))
    model.add(Dense(5))
    model.add(Dense(1))
    return model

# Preprocess data - For now Normalizing data
def preprocess_column(df):
    #df = df/10000
    df = (df - df.mean())/df.std()
    return df

# Shift data
def shift_df(df, lag):
    df['consumption'] = (df['consumption'] - df['consumption'].mean())/df['consumption'].std()
    t = 'consumption_'
    for i in range(1, lag):
        df[t+str(i)] = df['consumption'].shift(i)
        df[t+str(i)] = preprocess_column(df[t+str(i)])
    return df

def calculate_for_test_data(df, model):
    global counter, nan_results
    # Get list of unique series id in training data
    serid = df.series_id.unique()
    # Results is accumulated in below Dataframe and saved at the end for analysis
    error_list = pd.DataFrame(columns=['calculated', 'actual'])
    loop_counter = 0 # Counter for series id
    for i in serid:
        if (np.sum(df.series_id == i) > 24):
            df_slice = df[df.loc[:, 'series_id'] == i]
            df_slice = df_slice.loc[:, (df_slice.columns != 'series_id') & (df_slice.columns != 'consumption')]
            for cntr in range(24, df_slice.shape[0]):
                consumption = df_slice.iloc[cntr, -1]
                input('cosumption location changed, fix above line maybe getloc will help get location of consumption')
                x = model.predict(np.reshape(df_slice.iloc[cntr - 24:cntr, :].values, [1, time_steps, df_slice.shape[1]]))
                # drop nan results
                if (np.isfinite(x)):
                    error_list = error_list.append({'calculated':x[0][0], 'actual':consumption}, ignore_index=True)
                else:
                    nan_results = nan_results + 1
        loop_counter = loop_counter + 1
        if loop_counter%25 == 0:
            # print every 25 series id to know program is preogressing
            print(loop_counter) 
    # Save result for further analysis
    error_list.to_csv("error_list.csv", index=False )

            
    
# Read Data
df = pd.read_csv("..\\data\\cold_start_added_data_from_meta.csv")

df = shift_df(df, number_of_lags)
df.rename( columns = {'Unnamed: 0.1':'Garbage'}, inplace=True )
df.rename( columns = {'Unnamed: 0':'Garbage1'}, inplace=True )
df = df.loc[:, (df.columns != 'Garbage') & (df.columns != 'Garbage1')]

#
# Data from 3 columns not used as features in trainng, but is needed for selecting 
# data out of pandas dataframe, so subtarct 3 for definig number of features in model
#
number_of_hidden_size = df.shape[1] - 3
feature_size          = df.shape[1] - 3
batch_input_shape     = (1, time_steps, feature_size)
# Define model
model                 = make_lstm_model(batch_input_shape, number_of_hidden_size)
# Load pretrained model weights
model.load_weights("..\\data\\energy_forecast_single_lstm_final_model_11_14.h5")

# Match Column sequence in testing data as column sequence of training data
df = df[['series_id', 'consumption', 'temperature', 'IS_HOLIDAY', 'surface', 'base_temperature', 'time', 'day', 'consumption_1', 'consumption_2', 'consumption_3', 'consumption_4']]

# Now calculate for Series ID with more than 24 hrs of data
calculate_for_test_data(df, model)

