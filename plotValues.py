import polars as pl
import matplotlib.pyplot as plt

# Define the function to read and process the data
def process_data(file_path, variable, window_size):
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Initialize lists to hold the extracted data
    episodes = []
    values = []
    
    # Extract the relevant data
    for line in lines:
        parts = line.split('\t')
        episode = int(parts[0].split(': ')[1])
        value = float([part.split(': ')[1] for part in parts if part.startswith(variable)][0])
        
        episodes.append(episode)
        values.append(value)
    
    # Create a Polars DataFrame
    df = pl.DataFrame({
        "Episode": episodes,
        variable: values
    })
    
    # Compute the rolling average
    df = df.with_columns([
        pl.col(variable).rolling_mean(window_size).alias('Rolling Average')
    ])
    
    return df

# Define the function to plot the data
def plot_data(df, variable):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Episode'], df[variable], label=variable)
    plt.plot(df['Episode'], df['Rolling Average'], label='Rolling Average', color='orange')
    plt.xlabel('Episode')
    plt.ylabel(variable)
    plt.title(f'{variable} and its Rolling Average')
    plt.legend()
    plt.grid(True)
    plt.show()

# Path to your file
file_path = 'QLearningFromOldMate_Input.txt'
variable = 'lifespan'  # Change this to the variable you want to visualize
window_size = 100  # Adjust the window size for the rolling average

# Process the data
df = process_data(file_path, variable, window_size)

# Plot the data
plot_data(df, variable)
