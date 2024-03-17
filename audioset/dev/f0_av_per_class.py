import pandas as pd

# Load the data
data = pd.read_csv('mean_f0_target_clean.csv')

# Calculate the average F0 for each class
average_f0_per_class = data.groupby('Target')['MeanF0'].mean()

print(average_f0_per_class)