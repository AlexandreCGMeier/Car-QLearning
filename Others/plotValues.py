import pandas as pd
import matplotlib.pyplot as plt

# Load data from a text file
data = pd.read_csv('minDQ.txt', header=None, names=['Values'])

# Calculate the moving average with a window of 5
data['Moving Average'] = data['Values'].rolling(window=2).mean()

# Plot the moving average
plt.figure(figsize=(10, 6))
plt.plot(data['Moving Average'], label='Moving Average (20 measurements)')
plt.title('Moving Average Plot')
plt.xlabel('Measurement Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
