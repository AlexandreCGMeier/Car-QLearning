import matplotlib.pyplot as plt
import time
with open("score.txt", "r") as f:
    values = [float(line.strip()) for line in f]
plt.figure(figsize=(8, 4))
plt.plot(values[1:], label="Time series")
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("One-dimensional Time Series")
plt.legend()
plt.show()
plt.grid(True)
