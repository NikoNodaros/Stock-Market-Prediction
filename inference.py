import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler



# =============================================================================
# Settings
modelName = 'model5'
stock = 'sp'
#stock 'dow'
#stock = 'google'
#stock = 'amazon'
#stock = 'facebook'
#stock = 'apple'

# =============================================================================



sOpen = 'Info_' + modelName + '.txt' 
file = open(sOpen, 'r')
contents = file.read()
file.close()

step = ''
for c in contents:
    if c != ',':
        step += c
    else:
        break

time_step = int(step)
# =============================================================================
# Set custom timestep
#time_step = 100
# =============================================================================
RawData = pd.read_csv(stock + '_train.csv')
TrainingSet = RawData.iloc[:, 1:2].values
# Normalization
#################################################

normalized = MinMaxScaler(feature_range = (0, 1))
ScaledSet = normalized.fit_transform(TrainingSet)
#################################################
dataset_test = pd.read_csv(stock + '_test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values
dataset_total = pd.concat((RawData['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - time_step:].values
inputs = inputs.reshape(-1,1)
inputs = normalized.transform(inputs)


X_test = []
for i in range(time_step, real_stock_price.size):
    X_test.append(inputs[i-time_step:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
model = load_model(modelName + '.h5')
predicted_stock_price = model.predict(X_test)
predicted_stock_price = normalized.inverse_transform(predicted_stock_price)


truncated_real_stock_price = []
for i in range (time_step, real_stock_price.size):
    truncated_real_stock_price.append(real_stock_price[i - time_step, 0])

plt.plot(truncated_real_stock_price, color = 'red', 
         label = 'Real Price')
plt.plot(predicted_stock_price, color = 'blue', 
         label = 'Predicted Price')
plt.title('Price Trend')
plt.xlabel('Day')
plt.ylabel('Price')
plt.legend()
plt.show()