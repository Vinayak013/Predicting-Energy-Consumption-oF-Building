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

# Preprocess data - Processing only consumption.
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
        #df[t+str(i)] = (df[t+str(i)] - df[t+str(i)].mean())/df[t+str(i)].std()
        df[t+str(i)] = preprocess_column(df[t+str(i)])
    return df

#
# Assuming data is pre processed and saved in pandas dataframe
# In each loop generator should yield 1 whole batch
#
def lstm_generator_keras_only_one_op_full_batch():
    global batch_size, counter, df

    df = df.sample(frac=1).reset_index(drop=True)
    unique_series_id = df['series_id'].unique()
    unique_series_id_shape = unique_series_id.shape

    seriesIdCntr  = 0
    batch_counter = 0
    counter       = time_steps

    while True:
        # Iterate through all series_id assigned to each building
        df_slice = df[df.loc[:, 'series_id'] == unique_series_id[seriesIdCntr]]
        
        raw_data        = df_slice.loc[:, (df.columns != 'consumption') & (df_slice.columns != 'series_id')].values
        expected_output = df_slice.loc[:,df.columns == 'consumption'].values
        [raw_data_row, raw_data_col] = raw_data.shape
        # Shape of X is (batch_size, number_of_time_steps, feature_length)
        X = np.zeros((batch_size, time_steps, raw_data_col))
        # Shape of Y is (batch_size, number_of_time_steps, 1) as only 1 prediction
        Y = np.zeros((batch_size, 1))
        # iterate through each sample
        for batch_counter in range(batch_size):
            
            if counter < (raw_data_row - time_steps - 1 + 1) :
                X[batch_counter, :, :] = raw_data[counter - time_steps + 1:counter + 1,:]
                Y[batch_counter] = expected_output[counter]
                counter = counter + 1
            else:
                seriesIdCntr = seriesIdCntr + 1
                counter = time_steps  
                   
        if seriesIdCntr == (unique_series_id_shape[0] - 1):
            seriesIdCntr = 0
        yield X, Y
    
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

class TargetLossHistory(Callback):
    def __init__(self, save_after_epoch):
        self.epoch_count = 0
        self.epoch_save = save_after_epoch

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        global minimum_loss_so_far
        self.epoch_count = self.epoch_count + 1
        if logs.get('loss') < minimum_loss_so_far:
            minimum_loss_so_far = logs.get('loss')
            print('saving target.')
            energy_forecst.save_weights('energy_forecast_single_lstm.h5')
        self.losses.append(logs.get('loss'))
        plt.plot(self.losses, 'g')
        plt.pause(0.01)

################# Main execution function #################
#
# Read data from CSV file and remove unwatned columns not used as feature
# like index column, timestamp etc.
#
df_original = pd.read_csv("..\\data\\consumption_tarin_final_usable_with_all_features.csv", index_col=0)
df_original.rename( columns = {'Unnamed: 0.1':'Garbage'}, inplace=True )
df = df_original.loc[:, df_original.columns != 'Garbage']
df  = df.loc[:, df.columns != 'timestamp']
df.rename( columns = {'Unnamed: 0':'Garbage'}, inplace=True )
df = df.loc[:, df.columns != 'Garbage']

#  convert the timeseries problem to a supervised learning problem
df = shift_df(df, number_of_lags)
df.fillna(0, inplace = True)

# Check if any column has NaN elements
for column in df.columns:
    df_new = df[column]
    if(df_new.isna().sum()):
        print(column)

print(df.columns.values)
input('reached_here')
# Data from 2 columns not used as features in trainng so subtarct 2 for definig model
number_of_hidden_size = df.shape[1] - 2
feature_size   = df.shape[1] - 2
batch_input_shape = (batch_size, time_steps, feature_size)

energy_forecst = make_lstm_model(batch_input_shape, number_of_hidden_size)

# Use callbacks to save model during training 
losshistory = TargetLossHistory(save_after_epoch=10)
callbacks_list = [losshistory]

# compile the model
energy_forecst.compile(optimizer='sgd', loss='mse')

# print model summary
print(energy_forecst.summary())

# Train the model
energy_forecst.fit_generator(generator = lstm_generator_keras_only_one_op_full_batch(),
                                 steps_per_epoch=512, epochs=100, callbacks = callbacks_list)

