import numpy as np
import pandas as pd
from keras.layers import CuDNNLSTM #represents the hidden LSTM layers
from keras.models import Sequential #Neural network object representing a sequence of layers
from keras.layers import Dropout #dropout regularization to combat overfitting
from keras.layers import Dense #represents the output layer
from keras.callbacks import TensorBoard
from sklearn.preprocessing import MinMaxScaler


# =============================================================================
# Settings:
time_step = 900
perceptrons = 100
hidden_layers = 4
dropout_rate = 0.2
epoch_num = 100
batch = 64
model_name = 'model6'
stock = 'sp'
#stock 'dow'
#stock = 'hoogle'
#stock = 'amazon'
#stock = 'facebook'
#stock = 'apple'
# =============================================================================




RawData = pd.read_csv(stock + '_train.csv')
TrainingSet = RawData.iloc[:, 1:2].values
# Normalization
#################################################
normalized = MinMaxScaler(feature_range = (0, 1))
ScaledSet = normalized.fit_transform(TrainingSet)
#################################################



# Price attribute
#################################################
TimeStep = []
Price = []
for i in range(time_step, ScaledSet.size):
    TimeStep.append(ScaledSet[i-time_step:i, 0])
    Price.append(ScaledSet[i, 0])
TimeStep, Price = np.array(TimeStep), np.array(Price)
TimeStep = np.reshape(TimeStep, (TimeStep.shape[0], TimeStep.shape[1], 1))
#################################################

model = Sequential()
model.add(CuDNNLSTM(units = perceptrons, 
                   return_sequences = True, 
                   input_shape = (TimeStep.shape[1], 1)))
model.add(Dropout(dropout_rate))
# =============================================================================
# Extra Hidden Layers
for i in range(0, hidden_layers):
    model.add(CuDNNLSTM(units = perceptrons, 
                       return_sequences = True))
    model.add(Dropout(dropout_rate))
 
# =============================================================================
model.add(CuDNNLSTM(units = perceptrons))
model.add(Dropout(dropout_rate))
model.add(Dense(units = 1))
TBoard = TensorBoard(log_dir='./logs', histogram_freq=0,  
          write_graph=True, write_images=False)
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(TimeStep, Price, 
              epochs = epoch_num, 
              batch_size = batch)

sOpen = 'Info_' + model_name + '.txt'
file = open(sOpen, 'w')
file.write(str(time_step))
file.write(',')
file.write(str(perceptrons))
file.write(',')
file.write(str(hidden_layers + 2))
file.write(',')
file.write(str(dropout_rate))
file.write(',')
file.write(str(epoch_num))
file.write(',')
file.write(str(batch))
file.close()

model.save(model_name + '.h5')