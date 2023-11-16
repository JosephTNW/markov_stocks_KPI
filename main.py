import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Read the Excel file
df = pd.read_csv('bigblackcock.csv')

pd.set_option('display.max_rows', None)

# Select the columns of interest
df = df[['date', 'price', 'stock_name', 'PE_ratio', 'PEG_ratio']]

df['date'] = pd.to_datetime(df['date'])

# Normalize the features using MinMaxScaler (excluding 'stock_name' and 'date'columns)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['price', 'PE_ratio', 'PEG_ratio']])

# Convert to sequences (assuming we want to look at 10 previous days to predict the future price)
sequence_length = 3

generator = TimeseriesGenerator(scaled_data, scaled_data[:, 0], length=sequence_length, batch_size=1)


def to_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, 0])  # Price is the first column and our target
    return np.array(x), np.array(y)

# Split the sequences into training and testing sets 
# This is just a placeholder since the sample provided is not enough to create sequences
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

# Output the shapes of our training and test sets to confirm
print("Training set (X_train):", X_train.shape)
print("Training set (y_train):", y_train.shape)
print("Test set (X_test):", X_test.shape)
print("Test set (y_test):", y_test.shape)