import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model (assuming it is saved as 'stock_price_prediction_model.h5')
model = load_model('stock_price_prediction_model50.h5')

# Load dataset B into a DataFrame
df_B = pd.read_csv('teset.csv')
df_B['date'] = pd.to_datetime(df_B['date'])

# Normalize dataset B using the MinMaxScaler from dataset A
# Note: It's crucial to use the same scaler object that was fitted on dataset A to ensure consistency
# If the scaler was not saved from the last session, refit it on the training dataset of A and save it
scaler_B = MinMaxScaler(feature_range=(0, 1))
# Assuming that df_A is your original training set and you are re-fitting the scaler
df_A = pd.read_csv('bigblackcock.csv')
scaler_B.fit(df_A[['price', 'PE_ratio', 'PEG_ratio']])
scaled_data_B = scaler_B.transform(df_B[['price', 'PE_ratio', 'PEG_ratio']])

def to_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, 0])  # Price is the first column and our target
    return np.array(x), np.array(y)

# Create sequences for dataset B
sequence_length = 50  # Same sequence length that was used for dataset A
sequences_B, _ = to_sequences(scaled_data_B, sequence_length)

# Make predictions using the trained model on the new dataset B sequences
predicted_B = model.predict(sequences_B)

# Invert normalization for predictions
dummy_array_B = np.zeros(shape=(len(predicted_B), scaled_data_B.shape[1]))
dummy_array_B[:,0] = predicted_B[:,0]  # We are only interested in the first feature (price)
denormalized_predictions_B = scaler_B.inverse_transform(dummy_array_B)[:,0]

true_prices_B = df_B['price'].values[sequence_length:]


# Retrieve corresponding dates for sequences in dataset B
# The date for each prediction should be the date following the end of the sequence it was predicted on
# Hence, we get dates starting from the 'sequence_length' + 1 to the end of the DataFrame
dates_B = df_B['date'].values[sequence_length:]

# Plot predicted and true prices for dataset B
plt.figure(figsize=(15,7))
plt.plot(dates_B, denormalized_predictions_B, label='Predicted Price')
plt.plot(dates_B, true_prices_B, label='True Price', alpha=0.7)  # Alpha for transparency
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Predicted vs True Stock Prices for Dataset B')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()